"""
History-aware RAG chat pipeline.

Query flow:
  question → (optional history rewrite) → multi-query
  → hybrid search (BM25 + vector) → RRF across queries
  → cross-encoder rerank → answer from original_content
"""

import json
import re
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from ingestion_pipeline import load_chunks_from_vectorstore

load_dotenv()

llm = ChatOllama(
    model="llava",
    temperature=0.0,
    num_ctx=4096,
    num_predict=1024,
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


class QueryVariations(BaseModel):
    queries: list[str] = Field(
        min_length=2,
        max_length=2,
        description="Two search queries that keep the original technical terms",
    )


SPEC_HINTS = (
    "dbm", "ma", "gpio", "pin", "sensitivity", "touch", "capacitive",
    "mhz", "voltage", "current", "table", "mapping", "restriction",
)
APPENDIX_HEADING_RE = re.compile(
    r"(?im)^\s*(appendix|annex)(\s+[a-z0-9]+)?\b"
)
REFUSAL_RE = re.compile(
    r"^\s*(i\s+don'?t\s+know|i\s+do\s+not\s+know|unknown|n/?a)\b",
    re.I,
)
FOLLOWUP_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|the pin|the rail|the table|same|previous)\b",
    re.I,
)


def _tokens(text):
    return [token for token in re.findall(r"[A-Za-z][A-Za-z0-9_./+-]{1,}", text or "")]


def _key_terms(text):
    terms = []
    for token in _tokens(text):
        if any(character.isdigit() or character == "_" for character in token):
            terms.append(token)
        elif token.isupper() and len(token) >= 3:
            terms.append(token)
    return terms


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
    if not _keeps_key_terms(text, original):
        return original
    if len(text) > 220:
        return original
    return text


def _fallback_variations(original_query):
    keys = _key_terms(original_query)
    if not keys:
        return [f"{original_query} definition", f"{original_query} pin voltage"]
    primary = keys[0]
    return [
        f"what is {primary}",
        f"{primary} pin power rail voltage",
    ]


def create_query_variations(original_query: str, llm) -> list[str]:
    keys = _key_terms(original_query)
    keep = ", ".join(keys) if keys else original_query
    prompt = f"""Generate 2 alternative SEARCH queries for a PDF datasheet.

Original query: {original_query}

Rules:
Keep these terms exactly: {keep}
Do not invent parts, metrics, or features that are not in the original query.
Do not answer the question. Return only search queries."""

    try:
        llm_queries = llm.with_structured_output(QueryVariations, method="json_schema")
        query_variations = llm_queries.invoke(prompt)
        if isinstance(query_variations, dict):
            variations = query_variations.get("queries", [])
        else:
            variations = getattr(query_variations, "queries", [])
    except Exception as exc:
        print(f"Query variation LLM failed: {exc}")
        variations = []

    cleaned = []
    for variation in variations[:2]:
        text = sanitize_search_question(str(variation or ""), original_query)
        if text != original_query and text not in cleaned:
            cleaned.append(text)

    if len(cleaned) < 2:
        for fallback in _fallback_variations(original_query):
            if fallback not in cleaned and fallback != original_query:
                cleaned.append(fallback)
            if len(cleaned) >= 2:
                break

    for i, variation in enumerate(cleaned[:2], 1):
        print(f"{i}. {variation}")
    print("Generated Query variations")
    return cleaned[:2]


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


def citation_excerpt(doc: Document, max_len=420) -> str:
    text = re.sub(r"\s+", " ", (doc.page_content or "").strip())
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"


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


def select_top_docs(scored_docs, limit=5, per_page=2):
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
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
        yield _complete(answer)
        return

    if needs_history_rewrite(query, chat_history):
        yield {
            "type": "stage",
            "id": "understanding",
            "label": "Understanding the question",
            "detail": "Rewriting a follow-up into a search query",
        }
        messages = [
            SystemMessage(
                content=(
                    "Rewrite the user's follow-up into one standalone SEARCH query. "
                    "Do not answer it. Never reply with I don't know. "
                    "Keep pin names, rail names, part numbers, and units exactly. "
                    "Return only the rewritten query."
                )
            )
        ] + chat_history[-6:] + [
            HumanMessage(content=f"Standalone search query for: {query}")
        ]
        result = llm.invoke(messages)
        search_question = sanitize_search_question(result.content, query)
        if search_question != query:
            print(f"Rewrote follow-up to: {search_question}")
        else:
            print("History rewrite ignored; using the original question.")
    else:
        search_question = query
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
        "detail": "Planning search queries",
    }
    query_variations = [search_question] + create_query_variations(search_question, llm)
    query_weights = [2.0] + [1.0] * (len(query_variations) - 1)

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
    print(f"RRF produced {len(fused_docs)} unique candidates")

    if not fused_docs:
        answer = "I don't know"
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
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
    rerank_docs = fused_docs[:24]
    pairs = [[search_question, doc.page_content] for doc in rerank_docs]
    scores = reranker.predict(pairs)
    question_l = search_question.lower()
    wants_appendix = "appendix" in question_l or "annex" in question_l
    boost_tables = any(hint in question_l for hint in SPEC_HINTS)
    appendix_start = None if wants_appendix else appendix_start_page(session.chunks)
    scored_docs = []
    for score, doc in zip(scores, rerank_docs):
        adjusted = float(score)
        text_l = (doc.page_content or "").lower()
        if boost_tables and doc.metadata.get("type") == "table":
            if any(hint in text_l for hint in SPEC_HINTS if hint in question_l):
                adjusted += 0.35
        if not wants_appendix and chunk_is_appendix(doc, appendix_start):
            adjusted -= 1.2
        scored_docs.append((adjusted, doc))
    scored_docs.sort(key=lambda item: item[0], reverse=True)

    top_docs = select_top_docs(scored_docs)
    for index, (score, doc) in enumerate(scored_docs[:8], 1):
        meta = doc.metadata or {}
        preview = re.sub(r"\s+", " ", (doc.page_content or ""))[:90]
        print(
            f"  rank {index}: score={score:.3f} type={meta.get('type')} "
            f"page={meta.get('page')} section={meta.get('section')} {preview!r}"
        )

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

    combined_docs = "\n\n".join(get_answer_context(doc) for doc in top_docs)
    yield {
        "type": "stage",
        "id": "reasoning",
        "label": "Reasoning",
        "detail": "Reading the selected chunks",
        "citations": citations,
    }

    prompt = f"""Answer the following question based strictly on the provided context. If the answer is not contained within the context, respond with "I don't know".

    Context:
    {combined_docs}

    Question: {query}

    Instructions:
    Tables are written as sentences like "Row: Header is value". Treat each Row as one complete record; do not mix values across rows.
    Rely on explicit tables and lists before inferring answers from partial acronyms or variable names. CAP in SENSOR_CAPP is not capacitive touch.
    If the context has a related metric but not the asked quantity or unit (for example mA when the question asks for dBm), respond with "I don't know".
    First write a short <reasoning> of 2-4 sentences naming the pages or tables you used.
    Then put a short direct answer inside <final_answer> tags. Do not put reasoning in the final answer.

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
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=display_answer))
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
            }
    return result
 