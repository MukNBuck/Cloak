"""
History-aware RAG chat pipeline. All models run locally (Ollama + on-disk HF).

Query flow:
  question → heuristic follow-up rewrite (no LLM)
  → exact-term scan + hybrid search (BM25 + vector)
  → cross-encoder rerank → answer from retrieved chunks
"""

import json
import re
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

from cloak_config import (
    CHARS_PER_TOKEN,
    HISTORY_MESSAGES,
    LLM_MODEL,
    MAX_CHUNK_CHARS,
    MAX_CONTEXT_CHARS,
    MAX_PACKED_DOCS,
    NUM_CTX,
    NUM_PREDICT,
    OFFLINE,
    PROMPT_RESERVE_TOKENS,
    RERANK_CANDIDATES,
    RERANK_MODEL,
    TOP_DOCS,
    torch_device,
)
from ingestion_pipeline import load_chunks_from_vectorstore

_DEVICE = torch_device()

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.0,
    num_ctx=NUM_CTX,
    num_predict=NUM_PREDICT,
)

try:
    reranker = CrossEncoder(RERANK_MODEL, device=_DEVICE, local_files_only=OFFLINE)
except TypeError:
    reranker = CrossEncoder(RERANK_MODEL, device=_DEVICE)

APPENDIX_HEADING_RE = re.compile(
    r"(?im)^\s*(appendix|annex)(\s+[a-z0-9]+)?\b"
)
REFUSAL_RE = re.compile(
    r"^\s*(i\s+don'?t\s+know|i\s+do\s+not\s+know|unknown|n/?a)\b",
    re.I,
)
FOLLOWUP_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|the same|previous|above)\b",
    re.I,
)
TABLE_QUERY_RE = re.compile(r"\b(table|tables|figure|figures|chart|schedule)\b", re.I)
STOPWORDS = frozenset(
    """
    a an and are as at be by can could did do does for from how i if in into is it
    its me of on or please should show tell than that the these this those to was
    were what when where which who why with would you your
    """.split()
)


def _tokens(text):
    return [token for token in re.findall(r"[A-Za-z][A-Za-z0-9_./+-]{1,}", text or "")]


def _quoted_phrases(text):
    phrases = []
    for match in re.finditer(r'"([^"]{2,80})"|“([^”]{2,80})”', text or ""):
        phrase = (match.group(1) or match.group(2) or "").strip()
        if phrase:
            phrases.append(phrase)
    return phrases


def _is_strict_term(token):
    if any(character.isdigit() or character == "_" for character in token):
        return True
    return token.isupper() and len(token) >= 3


def _dedupe(items):
    seen = set()
    unique = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _key_terms(text):
    terms = list(_quoted_phrases(text))
    for token in _tokens(text):
        lowered = token.lower()
        if lowered in STOPWORDS:
            continue
        if _is_strict_term(token) or token[0].isupper() or len(token) >= 6:
            terms.append(token)
    return _dedupe(terms)


def _strict_terms(text):
    terms = list(_quoted_phrases(text))
    for token in _tokens(text):
        if _is_strict_term(token):
            terms.append(token)
    return _dedupe(terms)


def _keeps_key_terms(text, original):
    keys = _key_terms(original)
    if not keys:
        original_tokens = {token.lower() for token in _tokens(original) if len(token) > 2}
        new_tokens = {token.lower() for token in _tokens(text) if len(token) > 2}
        return bool(original_tokens and original_tokens & new_tokens)
    lowered = (text or "").lower()
    return any(key.lower() in lowered for key in keys)


def needs_history_rewrite(query, chat_history):
    if not chat_history:
        return False
    if _key_terms(query) and not FOLLOWUP_RE.search(query):
        return False
    return bool(FOLLOWUP_RE.search(query))


def sanitize_search_question(rewritten, original):
    text = re.sub(r"</?[^>]+>", " ", rewritten or "")
    text = re.sub(r"\s+", " ", text).strip().strip('"')
    if not text or REFUSAL_RE.match(text) or len(text) < 4:
        return original
    if not _keeps_key_terms(text, original) and not _key_terms(text):
        return original
    if len(text) > 220:
        return original
    return text


def rewrite_with_history(query, chat_history):
    """Resolve pronouns from the last human question. No LLM call."""
    if not needs_history_rewrite(query, chat_history):
        return query
    last_human = None
    for message in reversed(chat_history):
        if isinstance(message, HumanMessage):
            last_human = message.content
            break
    if not last_human:
        return query
    keys = _key_terms(last_human)[:6]
    if not keys or _key_terms(query):
        return query
    rewritten = f"{' '.join(keys)} {query}"
    return sanitize_search_question(rewritten, query)


def search_queries(original_query):
    """Keyword variants only. Local LLMs invent terms if asked to rewrite searches."""
    queries = [original_query]
    keys = _key_terms(original_query)
    if not keys:
        return queries
    joined = " ".join(keys)
    if joined.lower() != original_query.lower():
        queries.append(joined)
    return queries[:2]


def lexical_hits(query, chunks, limit=8):
    """Boost chunks that literally contain names, IDs, or quoted phrases."""
    keys = _strict_terms(query)
    if not keys or not chunks:
        return []
    scored = []
    for chunk in chunks:
        text = chunk.page_content or ""
        text_l = text.lower()
        hits = 0
        for key in keys:
            count = text_l.count(key.lower())
            if count:
                hits += 2 + min(count, 5)
        if not hits:
            continue
        density = hits / max(len(text) / 400.0, 1.0)
        scored.append((hits + density, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:limit]]


def reciprocal_rank_fusion(chunk_lists, k=60, query_weights=None, verbose=False):
    if verbose:
        print("\n" + "=" * 60)
        print("APPLYING RRF")
        print("\n" + "=" * 60)

    rrf_scores = defaultdict(float)
    all_unique_chunks = {}
    chunk_id_map = {}
    chunk_counter = 1

    for query_idx, result_chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Proccessing Query {query_idx} results:")
        weight = 1.0
        if query_weights and query_idx - 1 < len(query_weights):
            weight = query_weights[query_idx - 1]

        for position, chunk in enumerate(result_chunks, 1):
            chunk_content = chunk.page_content

            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1

            chunk_id = chunk_id_map[chunk_content]
            all_unique_chunks[chunk_content] = chunk
            position_score = weight / (k + position)
            rrf_scores[chunk_content] += position_score

            if verbose:
                print(
                    f"Position {position}: {chunk_id} + {position_score:.4f} "
                    f"(running total: {rrf_scores[chunk_content]:.4f})"
                )

    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [all_unique_chunks[content] for content, score in sorted_chunks]


def prepend_unique(preferred, existing):
    seen = {doc.page_content for doc in preferred}
    merged = list(preferred)
    for doc in existing:
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)
        merged.append(doc)
    return merged


def get_answer_context(doc: Document) -> str:
    raw = doc.metadata.get("original_content")
    if not raw:
        return doc.page_content

    try:
        original = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return doc.page_content

    parts = []
    raw_text = (original.get("raw_text") or "").strip()
    if raw_text:
        parts.append(raw_text)

    tables = original.get("tables_html") or []
    for i, table in enumerate(tables, 1):
        if table:
            parts.append(f"Table {i}:\n{table}")

    return "\n\n".join(parts) if parts else doc.page_content


def _clip(text, limit):
    text = re.sub(r"[ \t]+", " ", (text or "").strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _chars_from_tokens(tokens):
    return int(max(0, tokens) * CHARS_PER_TOKEN)


def _history_block(chat_history):
    if not chat_history:
        return ""
    lines = []
    for message in chat_history[-HISTORY_MESSAGES:]:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        content = _clip(getattr(message, "content", None) or "", 360)
        if content:
            lines.append(f"{role}: {content}")
    if not lines:
        return ""
    return "Recent conversation:\n" + "\n".join(lines)


def source_char_budget(query, history_block=""):
    """Leave room for prompt, question, history, and the answer. Fill the rest with sources."""
    window = _chars_from_tokens(NUM_CTX)
    reserved = _chars_from_tokens(NUM_PREDICT + PROMPT_RESERVE_TOKENS)
    leftover = window - reserved - len(query or "") - len(history_block or "")
    leftover = max(_chars_from_tokens(160), leftover)
    if MAX_CONTEXT_CHARS > 0:
        leftover = min(leftover, MAX_CONTEXT_CHARS)
    return leftover


def _source_block(index, doc, clip_limit):
    meta = doc.metadata or {}
    loc = f"{meta.get('source') or 'document.pdf'} p.{meta.get('page', '?')} {meta.get('type') or 'text'}"
    body = _clip(get_answer_context(doc), clip_limit)
    return f"[Source {index} | {loc}]\n{body}"


def pack_context(docs, char_budget):
    """Fit as many reranked chunks as the leftover window allows."""
    if not docs:
        return "", []
    target = max(TOP_DOCS, 1)
    clip_limit = min(MAX_CHUNK_CHARS, max(320, char_budget // target))
    parts = []
    packed = []
    used = 0
    for doc in docs:
        index = len(packed) + 1
        block = _source_block(index, doc, clip_limit)
        extra = len(block) + (2 if parts else 0)
        if packed and used + extra > char_budget:
            break
        if not packed and extra > char_budget:
            tighter = max(200, char_budget - 80)
            block = _source_block(1, doc, tighter)
            extra = len(block)
        parts.append(block)
        packed.append(doc)
        used += extra
    print(
        f"Packed {len(packed)} sources using {used}/{char_budget} chars "
        f"(ctx={NUM_CTX}, predict={NUM_PREDICT})"
    )
    return "\n\n".join(parts), packed


def citation_excerpt(doc: Document, max_len=420) -> str:
    return _clip(doc.page_content or "", max_len)


def appendix_start_page(chunks):
    pages = []
    for chunk in chunks or []:
        text = chunk.page_content or ""
        page = (chunk.metadata or {}).get("page")
        section = (chunk.metadata or {}).get("section")
        if page is None:
            continue
        if section == "appendix" or APPENDIX_HEADING_RE.search(text[:400]):
            try:
                pages.append(int(page))
            except (TypeError, ValueError):
                pass
    return min(pages) if pages else None


def chunk_is_appendix(doc: Document, appendix_start=None) -> bool:
    meta = doc.metadata or {}
    if meta.get("section") == "appendix":
        return True
    text = doc.page_content or ""
    if APPENDIX_HEADING_RE.search(text[:400]):
        return True
    page = meta.get("page")
    if appendix_start is not None and page is not None:
        try:
            return int(page) >= appendix_start
        except (TypeError, ValueError):
            return False
    return False


def select_top_docs(scored_docs, limit=None, per_page=2):
    limit = limit or TOP_DOCS
    picked = []
    page_counts = defaultdict(int)
    for _score, doc in scored_docs:
        page = (doc.metadata or {}).get("page")
        key = page if page is not None else id(doc)
        if page_counts[key] >= per_page:
            continue
        picked.append(doc)
        page_counts[key] += 1
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        seen = {id(doc) for doc in picked}
        for _score, doc in scored_docs:
            if id(doc) in seen:
                continue
            picked.append(doc)
            if len(picked) >= limit:
                break
    return picked


def extract_display_answer(raw_text: str) -> str:
    if not raw_text:
        return ""
    match = re.search(r"<final_answer>(.*?)</final_answer>", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?final_answer>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?reasoning>", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip() or raw_text.strip()


def extract_reasoning(raw_text: str) -> str:
    if not raw_text:
        return ""
    match = re.search(r"<reasoning>(.*?)</reasoning>", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def _build_citations(top_docs):
    citations = []
    for index, doc in enumerate(top_docs, 1):
        meta = doc.metadata or {}
        page = meta.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        excerpt = citation_excerpt(doc)
        print(f"Citation {index}: {meta.get('source')} p.{page} {meta.get('type')} {excerpt[:80]!r}")
        citations.append(
            {
                "id": index,
                "source": meta.get("source") or "document.pdf",
                "content_hash": meta.get("content_hash") or "",
                "type": meta.get("type") or "text",
                "page": page,
                "excerpt": excerpt,
            }
        )
    return citations


def _complete(answer, citations=None):
    citations = citations or []
    return {
        "type": "complete",
        "answer": answer,
        "context": [item.get("excerpt") or "" for item in citations],
        "citations": citations,
        "reasoning": "",
    }


def _trim_history(chat_history):
    if len(chat_history) > HISTORY_MESSAGES:
        del chat_history[:-HISTORY_MESSAGES]


def _remember(chat_history, query, answer):
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer or ""))
    _trim_history(chat_history)


def _identifier_boost(text_l, keys):
    if not keys:
        return 0.0
    if all(key.lower() in text_l for key in keys):
        return 1.5
    if any(key.lower() in text_l for key in keys):
        return 0.5
    return 0.0


def _run_llm(prompt):
    assembled = []
    try:
        for chunk in llm.stream(prompt):
            piece = getattr(chunk, "content", None) or ""
            if piece:
                assembled.append(piece)
                yield piece, "".join(assembled)
        return
    except Exception as exc:
        print(f"LLM stream failed, using invoke: {exc}")
    response = llm.invoke(prompt)
    text = response.content or ""
    yield text, text


def ask_question_events(query, session):
    chat_history = session.chat_history

    if not session.chunks and session.vectorstore is not None:
        session.chunks = load_chunks_from_vectorstore(session.vectorstore)
        session.rebuild_retrievers()

    print(f"Session has {len(session.chunks)} chunks")
    if not session.chunks or session.hybrid_retriever is None:
        answer = "I don't know"
        _remember(chat_history, query, answer)
        yield _complete(answer)
        return

    search_question = rewrite_with_history(query, chat_history)
    if search_question != query:
        print(f"Follow-up resolved to: {search_question}")
        yield {
            "type": "stage",
            "id": "understanding",
            "label": "Understanding the question",
            "detail": search_question[:120],
        }
    else:
        if chat_history:
            print("Skipping history rewrite; question is already standalone.")
        yield {
            "type": "stage",
            "id": "understanding",
            "label": "Understanding the question",
            "detail": search_question[:120],
        }

    print(f"Searching for {search_question}...")

    yield {
        "type": "stage",
        "id": "fetching",
        "label": "Fetching chunks",
        "detail": "Keyword scan and hybrid search",
    }
    query_variations = search_queries(search_question)
    for i, variation in enumerate(query_variations, 1):
        print(f"{i}. {variation}")
    query_weights = [2.0] + [1.2] * (len(query_variations) - 1)

    exact_docs = lexical_hits(search_question, session.chunks)
    print(f"Exact-term scan found {len(exact_docs)} matching chunks")

    all_retrieval_results = []
    fetched = 0
    for i, variation in enumerate(query_variations, 1):
        print(f"RESULTS FOR QUERY {i}: {variation}")
        docs = session.hybrid_retriever.invoke(variation)
        fetched += len(docs or [])
        all_retrieval_results.append(docs)
        yield {
            "type": "stage",
            "id": "fetching",
            "label": "Fetching chunks",
            "detail": f"Query {i}/{len(query_variations)} · {fetched} hits so far",
        }

    print(f"\nFusing {len(all_retrieval_results)} query result lists with RRF...")
    fused_docs = reciprocal_rank_fusion(
        all_retrieval_results, k=60, query_weights=query_weights
    )
    fused_docs = prepend_unique(exact_docs, fused_docs)
    print(f"RRF produced {len(fused_docs)} unique candidates")

    if not fused_docs:
        answer = "I don't know"
        _remember(chat_history, query, answer)
        print(f"Answer: {answer}")
        yield _complete(answer)
        return

    yield {
        "type": "stage",
        "id": "reranking",
        "label": "Reranking sources",
        "detail": f"{len(fused_docs)} unique chunks",
    }
    print("Reranking fused candidates with cross-encoder...")
    rerank_docs = fused_docs[:RERANK_CANDIDATES]
    pairs = [[search_question, doc.page_content] for doc in rerank_docs]
    scores = reranker.predict(pairs)
    question_l = search_question.lower()
    keys = _strict_terms(search_question)
    wants_appendix = "appendix" in question_l or "annex" in question_l
    boost_tables = bool(TABLE_QUERY_RE.search(search_question))
    appendix_start = None if wants_appendix else appendix_start_page(session.chunks)
    scored_docs = []
    for score, doc in zip(scores, rerank_docs):
        adjusted = float(score)
        text_l = (doc.page_content or "").lower()
        adjusted += _identifier_boost(text_l, keys)
        if boost_tables and doc.metadata.get("type") == "table":
            adjusted += 0.35
        if not wants_appendix and chunk_is_appendix(doc, appendix_start):
            adjusted -= 0.4
        scored_docs.append((adjusted, doc))
    scored_docs.sort(key=lambda item: item[0], reverse=True)

    candidate_docs = select_top_docs(scored_docs, limit=MAX_PACKED_DOCS)
    for index, (score, doc) in enumerate(scored_docs[:8], 1):
        meta = doc.metadata or {}
        preview = re.sub(r"\s+", " ", (doc.page_content or ""))[:90]
        print(
            f"  rank {index}: score={score:.3f} type={meta.get('type')} "
            f"page={meta.get('page')} section={meta.get('section')} {preview!r}"
        )

    history_block = _history_block(chat_history)
    char_budget = source_char_budget(query, history_block)
    combined_docs, top_docs = pack_context(candidate_docs, char_budget)
    citations = _build_citations(top_docs)
    source_bits = []
    for item in citations:
        bit = item.get("source") or "document.pdf"
        if item.get("page"):
            bit += f" p.{item['page']}"
        source_bits.append(bit)
    yield {
        "type": "stage",
        "id": "reranking",
        "label": "Reranking sources",
        "detail": f"Using {len(citations)} sources · {', '.join(source_bits[:3])}",
        "citations": citations,
    }

    history_section = f"{history_block}\n\n" if history_block else ""
    yield {
        "type": "stage",
        "id": "reasoning",
        "label": "Reasoning",
        "detail": "Reading the selected chunks",
        "citations": citations,
    }

    prompt = f"""Answer strictly from the numbered sources. If the sources do not contain the answer, respond with "I don't know".

{history_section}Sources:
{combined_docs}

Question: {query}

Tables are written as "Row: Header is value". Do not mix values across rows.
Do not invent names, numbers, dates, or quotes that are not in the sources.
First write a short <reasoning> of 2-4 sentences naming the source numbers you used.
Then put a short direct answer inside <final_answer> tags.

Answer:"""

    assembled = ""
    last_detail = None
    for _piece, assembled in _run_llm(prompt):
        detail = extract_reasoning(assembled) or "Writing the answer"
        if detail == last_detail:
            continue
        last_detail = detail
        yield {
            "type": "stage",
            "id": "reasoning",
            "label": "Reasoning",
            "detail": detail,
            "citations": citations,
        }

    display_answer = extract_display_answer(assembled)
    reasoning = extract_reasoning(assembled)
    _remember(chat_history, query, display_answer)
    print(f"Answer: {display_answer}")

    yield {
        "type": "complete",
        "answer": display_answer,
        "context": [item["excerpt"] for item in citations],
        "citations": citations,
        "reasoning": reasoning,
    }


def ask_question(query, session):
    result = {"answer": "I don't know", "context": [], "citations": []}
    for event in ask_question_events(query, session):
        if event.get("type") == "complete":
            result = {
                "answer": event.get("answer") or "",
                "context": event.get("context") or [],
                "citations": event.get("citations") or [],
                "reasoning": event.get("reasoning") or "",
            }
    return result
