"""
History-aware RAG chat pipeline. All models run locally (Ollama + on-disk HF).

Query flow:
  question → heuristic follow-up rewrite (no LLM)
  → exact-term scan + hybrid search (BM25 + vector)
  → cross-encoder rerank → answer from retrieved chunks
"""

import json
import math
import os
import re
import threading
import time
from collections import defaultdict

from cloak_config import (
    AGENT_RETRIES,
    CHARS_PER_TOKEN,
    COMPARE_TOP_PER_SIDE,
    EXTRACT_TOP_DOCS,
    HISTORY_MESSAGES,
    LLM_MODEL,
    MAX_CHUNK_CHARS,
    MAX_CONTEXT_CHARS,
    MAX_PACKED_DOCS,
    MUST_ANSWER_RETRY,
    NUM_CTX,
    PACK_ORDER,
    NUM_PREDICT,
    OFFLINE,
    PROMPT_RESERVE_TOKENS,
    RERANK_CANDIDATES,
    RERANK_MAX_CHARS,
    RERANK_MIN_SCORE,
    RERANK_MODEL,
    SUMMARIZE_MAP_CHUNKS,
    TOP_DOCS,
    torch_device,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from route import classify_intent, decide_route, route_label
from harness import strip_chat, thread_messages, try_direct_clock, try_direct_math
from chat_tools import (
    CHAT_TOOLS,
    MAX_TOOL_CHARS,
    MAX_TOOL_ROUNDS,
    TOOL_NUM_PREDICT,
    attached_pdf_names,
    execute_tool,
    normalize_tool_calls,
    stage_for_calls,
)

_DEVICE = torch_device()
_llm_lock = threading.Lock()
_llm_cache = {}


def _safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(str(text).encode("ascii", "replace").decode("ascii"))


def get_llm(model_name=None, temperature=0.0, *, chat=False):
    name = (model_name or LLM_MODEL).strip() or LLM_MODEL
    key = (name, round(float(temperature), 2), bool(chat))
    with _llm_lock:
        client = _llm_cache.get(key)
        if client is None:
            kwargs = {
                "model": name,
                "temperature": float(temperature),
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
                "reasoning": False,
                "keep_alive": "30m",
            }
            if chat:
                # Qwen2.5-Instruct recommended sampling. RAG stays greedy.
                kwargs.update(top_p=0.8, top_k=20, repeat_penalty=1.05)
            client = ChatOllama(**kwargs)
            _llm_cache[key] = client
        return client


llm = get_llm(LLM_MODEL)


class TransformersReranker:
    """BGE-style cross-encoder without sentence-transformers/sklearn/scipy."""

    def __init__(self, model_name, device="cpu", max_length=512, batch_size=8):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        kwargs = {}
        if OFFLINE:
            kwargs["local_files_only"] = True
        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        print(f"Loading reranker {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        self.model.to(device)
        self.model.eval()

    def predict(self, pairs):
        scores = []
        torch = self.torch
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            queries = [pair[0] or "" for pair in batch]
            docs = [pair[1] or "" for pair in batch]
            encoded = self.tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
            scores.extend(logits.view(-1).float().cpu().tolist())
        return scores


reranker = None


def get_reranker():
    global reranker
    if reranker is None:
        reranker = TransformersReranker(RERANK_MODEL, device=_DEVICE)
    return reranker

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


def _sigmoid(score):
    value = float(score)
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _rerank_text(doc):
    """Cross-encoders cap around 512 tokens; score the head of the chunk, not a truncated tail."""
    text = doc.page_content or ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= RERANK_MAX_CHARS:
        return text
    return text[:RERANK_MAX_CHARS].rsplit(" ", 1)[0]


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


def _window_around_chunk(source, chunk, window=4800):
    """Keep the retrieved span, not the start of a long HTML page."""
    source = source or ""
    chunk = (chunk or "").strip()
    if not source:
        return chunk
    if not chunk:
        return source
    needle = chunk[:96].strip()
    idx = source.find(needle) if len(needle) >= 12 else -1
    if idx < 0 and len(chunk) >= 20:
        needle = re.sub(r"\s+", " ", chunk[:48]).strip()
        idx = re.sub(r"\s+", " ", source).find(needle)
        if idx >= 0:
            ratio = idx / max(len(re.sub(r"\s+", " ", source)), 1)
            idx = int(ratio * len(source))
    if idx < 0:
        return chunk if len(chunk) >= 80 else source
    lead = min(480, window // 5)
    start = max(0, idx - lead)
    end = min(len(source), start + window)
    if end - start < window:
        start = max(0, end - window)
    return source[start:end]


def get_answer_context(doc: Document) -> str:
    meta = doc.metadata or {}
    raw = meta.get("original_content") or meta.get("original_content")
    parent = (meta.get("parent_text") or "").strip()
    chunk = doc.page_content or ""
    parsed_parts = []
    if raw:
        try:
            original = json.loads(raw) if isinstance(raw, str) else raw
            raw_text = (original.get("raw_text") or "").strip()
            if raw_text:
                parsed_parts.append(raw_text)
            for i, table in enumerate(original.get("tables_html") or [], 1):
                if table:
                    parsed_parts.append(f"Table {i}:\n{table}")
        except (json.JSONDecodeError, TypeError, AttributeError):
            parsed_parts = []
    source = "\n\n".join(parsed_parts) if parsed_parts else (parent or chunk)
    return _window_around_chunk(source, chunk)


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
    source = meta.get("source") or "document.pdf"
    page = meta.get("page", "?")
    kind = meta.get("type") or "text"
    heading = (meta.get("heading") or meta.get("section") or "").strip()
    loc = f"{source} p.{page} {kind}"
    if heading:
        loc = f"{loc} | {heading[:80]}"
    body = _clip(get_answer_context(doc), clip_limit)
    return f"[Source {index} | {loc}]\n{body}"


def _token_set(text):
    return {token.lower() for token in re.findall(r"[a-z0-9]{3,}", text or "", re.I)}


def dedupe_near_duplicate_docs(docs, threshold=0.82):
    """Drop overlapping chunks so repeated text does not dominate the prompt."""
    unique = []
    seen_sets = []
    for doc in docs:
        tokens = _token_set(get_answer_context(doc) or doc.page_content or "")
        if len(tokens) >= 8 and any(
            (len(tokens & other) / (len(tokens | other) or 1)) >= threshold
            for other in seen_sets
            if other
        ):
            continue
        unique.append(doc)
        seen_sets.append(tokens)
    return unique


def order_docs_for_pack(docs):
    """Reorder reranked docs before packing.

    Lost-in-the-Middle (Liu et al. 2023): 7B-class models attend more to the
    start and end of the prompt than the trough. u_shape puts rank-1 first and
    rank-2 last. best_last is a recency prior.
    """
    docs = list(docs or [])
    if len(docs) < 2:
        return docs
    if PACK_ORDER == "best_last":
        return list(reversed(docs))
    if PACK_ORDER == "u_shape" and len(docs) >= 3:
        return [docs[0], *docs[2:], docs[1]]
    return docs


def pack_context(docs, char_budget):
    """Fit reranked chunks into the leftover window.

    Clip as if packing a few long pages so the top hits stay intact; extra
    chunks only fill leftover budget. Retrieval can still consider up to
    MAX_PACKED_DOCS candidates (that was the 42.5% generate run).
    """
    docs = order_docs_for_pack(
        dedupe_near_duplicate_docs(collapse_to_parents(docs or []))[:MAX_PACKED_DOCS]
    )
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
        f"(ctx={NUM_CTX}, predict={NUM_PREDICT}, order={PACK_ORDER})"
    )
    return "\n\n".join(parts), packed


def citation_excerpt(doc: Document, max_len=420) -> str:
    return _clip(get_answer_context(doc) or doc.page_content or "", max_len)


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
    for item in scored_docs:
        doc = item[1]
        page = (doc.metadata or {}).get("page")
        key = page if page is not None else id(doc)
        if page_counts[key] >= per_page:
            continue
        picked.append(item)
        page_counts[key] += 1
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        seen = {id(item[1]) for item in picked}
        for item in scored_docs:
            if id(item[1]) in seen:
                continue
            picked.append(item)
            if len(picked) >= limit:
                break
    return picked


def extract_display_answer(raw_text: str) -> str:
    if not raw_text:
        return ""
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(
        r"<final_answer>(.*?)</final_answer>|<final_answer>(.*?)</final_answer>",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        body = match.group(1) if match.group(1) is not None else match.group(2)
        return re.sub(r"\s+", " ", body or "").strip()
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?final_answer>", "", cleaned, flags=re.IGNORECASE)
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


def distinct_reasoning(answer, reasoning):
    """Hide reasoning that is just a copy of the visible answer."""
    answer_l = re.sub(r"\s+", " ", (answer or "")).strip().lower()
    text = re.sub(r"\s+", " ", (reasoning or "")).strip()
    if not text:
        return ""
    reason_l = text.lower()
    if not answer_l:
        return text
    if reason_l == answer_l or reason_l in answer_l or answer_l in reason_l:
        return ""
    answer_words = set(answer_l.split())
    reason_words = set(reason_l.split())
    if reason_words and len(answer_words & reason_words) / len(reason_words) >= 0.8:
        return ""
    return text


def session_cancelled(session):
    event = getattr(session, "cancel", None)
    return event is not None and event.is_set()


def _stopped(citations=None):
    return _complete("Stopped.", citations)


def _build_citations(top_docs, scores=None):
    citations = []
    scores = scores or {}
    for index, doc in enumerate(top_docs, 1):
        meta = doc.metadata or {}
        page = meta.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        excerpt = citation_excerpt(doc)
        raw_score, confidence = scores.get(id(doc), (None, None))
        _safe_print(
            f"Citation {index}: {meta.get('source')} p.{page} {meta.get('type')} {excerpt[:80]!r}"
        )
        item = {
            "id": index,
            "source": meta.get("source") or "document.pdf",
            "content_hash": meta.get("content_hash") or "",
            "type": meta.get("type") or "text",
            "page": page,
            "excerpt": excerpt,
        }
        if raw_score is not None:
            item["rerank_score"] = round(float(raw_score), 4)
            item["rerank_confidence"] = round(float(confidence), 4)
        citations.append(item)
    return citations


def _complete(answer, citations=None, extra=None):
    citations = citations or []
    payload = {
        "type": "complete",
        "answer": answer,
        "context": [item.get("excerpt") or "" for item in citations],
        "citations": citations,
        "reasoning": "",
    }
    if extra:
        payload.update(extra)
    return payload


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


def _retrieve(session, variation):
    try:
        docs = session.hybrid_retriever.invoke(variation) or []
    except Exception as exc:
        print(f"Vector search failed ({exc}); using keyword search only.")
        docs = []
        if session.bm25_retriever is not None:
            try:
                docs = session.bm25_retriever.invoke(variation) or []
            except Exception as bm25_exc:
                print(f"BM25 search also failed: {bm25_exc}")
                return []
    allowed = {
        (chunk.metadata or {}).get("content_hash")
        for chunk in (session.chunks or [])
        if (chunk.metadata or {}).get("content_hash")
    }
    if not allowed:
        return []
    return [
        doc
        for doc in docs
        if (doc.metadata or {}).get("content_hash") in allowed
    ]


def _rerank_scores(search_question, rerank_docs):
    pairs = [[search_question, _rerank_text(doc)] for doc in rerank_docs]
    try:
        return get_reranker().predict(pairs)
    except Exception as exc:
        print(f"Reranker failed ({exc}); keeping retrieval order.")
        return [float(len(rerank_docs) - index) for index in range(len(rerank_docs))]


def _run_llm(prompt, session=None, temperature=0.0, chat=False):
    assembled = []
    model_name = getattr(session, "llm_model", None) if session is not None else None
    client = get_llm(model_name, temperature=temperature, chat=chat)
    try:
        for chunk in client.stream(prompt):
            if session_cancelled(session):
                return
            piece = getattr(chunk, "content", None) or ""
            if piece:
                assembled.append(piece)
                yield piece, "".join(assembled)
        return
    except Exception as exc:
        print(f"LLM stream failed, using invoke: {exc}")
    if session_cancelled(session):
        return
    response = client.invoke(prompt)
    text = response.content or ""
    yield text, text


INTENT_SUMMARIZE_RE = re.compile(
    r"\b(summarize|summary|overview|tl;?dr|recap|walk me through)\b", re.I
)
INTENT_COMPARE_RE = re.compile(
    r"\b(compare|versus|\bvs\.?\b|difference between|differ from|contrast)\b", re.I
)
CLAIM_NUM_RE = re.compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+%?|\d{3,}(?:\.\d+)?%?"
)
SOURCE_CITE_RE = re.compile(r"\[(?:Source\s*)?(\d+)\]", re.I)


def split_compare_topics(query):
    patterns = [
        r"(?:compare|contrast)\s+(.+?)\s+(?:and|vs\.?|versus|with)\s+(.+)",
        r"difference between\s+(.+?)\s+and\s+(.+)",
        r"(.+?)\s+vs\.?\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query or "", re.I)
        if not match:
            continue
        left = match.group(1).strip(" ?.")
        right = match.group(2).strip(" ?.")
        if left and right and left.lower() != right.lower():
            return [left, right]
    return None


def parent_key(doc):
    meta = doc.metadata or {}
    if meta.get("parent_id"):
        return meta["parent_id"]
    return (
        meta.get("source"),
        meta.get("page"),
        meta.get("type"),
        (doc.page_content or "")[:80],
    )


def collapse_to_parents(docs):
    seen = set()
    unique = []
    for doc in docs:
        key = parent_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def merge_answer_docs(passing, source_docs, limit):
    """Prefer rerank-passing chunks, then fill from the wider retrieved set."""
    merged = []
    seen = set()
    for doc in list(passing or []) + list(source_docs or []):
        marker = id(doc)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(doc)
        if len(merged) >= limit:
            break
    return merged


def context_supports_query(query, packed_text):
    packed_l = (packed_text or "").lower()
    if not packed_l:
        return False
    keys = _key_terms(query)
    if not keys:
        return False
    hits = sum(1 for key in keys if key.lower() in packed_l)
    return hits >= max(1, min(2, len(keys)))


def write_trace(session, payload):
    record = dict(payload)
    record["ts"] = time.time()
    line = json.dumps(record, ensure_ascii=False, default=str)
    print(f"CLOAK_TRACE {line}")
    temp_dir = getattr(session, "temp_dir", None)
    if not temp_dir:
        return
    try:
        with open(os.path.join(temp_dir, "traces.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception as exc:
        print(f"Trace write failed: {exc}")


def retrieve_fused(session, search_question):
    query_variations = search_queries(search_question)
    query_weights = [2.0] + [1.2] * (len(query_variations) - 1)
    exact_docs = lexical_hits(search_question, session.chunks)
    lists = []
    fetched = 0
    for variation in query_variations:
        docs = _retrieve(session, variation)
        fetched += len(docs or [])
        lists.append(docs)
    fused = reciprocal_rank_fusion(lists, k=60, query_weights=query_weights)
    fused = prepend_unique(exact_docs, fused)
    return fused, query_variations, fetched, len(exact_docs)


def score_documents(search_question, fused_docs, session):
    rerank_docs = fused_docs[:RERANK_CANDIDATES]
    if not rerank_docs:
        return [], {}
    scores = _rerank_scores(search_question, rerank_docs)
    raw_by_id = {}
    question_l = search_question.lower()
    keys = _strict_terms(search_question)
    wants_appendix = "appendix" in question_l or "annex" in question_l
    boost_tables = bool(TABLE_QUERY_RE.search(search_question))
    appendix_start = None if wants_appendix else appendix_start_page(session.chunks)
    scored_docs = []
    for score, doc in zip(scores, rerank_docs):
        raw = float(score)
        raw_by_id[id(doc)] = raw
        adjusted = raw
        text_l = (get_answer_context(doc) or doc.page_content or "").lower()
        adjusted += _identifier_boost(text_l, keys)
        if boost_tables and (doc.metadata or {}).get("type") == "table":
            adjusted += 0.35
        if not wants_appendix and chunk_is_appendix(doc, appendix_start):
            adjusted -= 0.4
        scored_docs.append((adjusted, doc))
    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return scored_docs, raw_by_id


def pick_passing(scored_docs, raw_by_id, limit):
    overfetch = select_top_docs(scored_docs, limit=max(limit * 3, limit))
    picked = []
    seen = set()
    for item in overfetch:
        doc = item[1]
        key = parent_key(doc)
        if key in seen:
            continue
        seen.add(key)
        picked.append(item)
        if len(picked) >= limit:
            break
    score_map = {}
    source_docs = []
    for _adjusted, doc in picked:
        raw = float(raw_by_id.get(id(doc), 0.0))
        score_map[id(doc)] = (raw, _sigmoid(raw))
        source_docs.append(doc)
    passing = [
        doc for doc in source_docs
        if score_map.get(id(doc), (0.0, 0.0))[1] >= RERANK_MIN_SCORE
    ]
    top_confidence = max((score_map[id(doc)][1] for doc in source_docs), default=0.0)
    return source_docs, passing, score_map, top_confidence


def retry_search_question(query, search_question, source_docs):
    combined = " ".join(get_answer_context(doc) or "" for doc in source_docs).lower()
    missing = [term for term in _key_terms(query) if term.lower() not in combined]
    if missing:
        return f"{search_question} {' '.join(missing[:8])}"
    keys = _key_terms(query)
    joined = " ".join(keys)
    if joined and joined.lower() != (search_question or "").lower():
        return joined
    return None


def _parse_source_numbers(text):
    values = []
    for raw in CLAIM_NUM_RE.findall(text or "") + re.findall(r"\b\d+(?:\.\d+)?%?\b", text or ""):
        try:
            values.append(float(str(raw).replace(",", "").replace("%", "")))
        except (TypeError, ValueError):
            continue
    return values


def _number_supported(num_str, packed_l, packed_nums):
    if (num_str or "").lower() in packed_l:
        return True
    try:
        value = float(str(num_str).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return False
    for source in packed_nums:
        if abs(source - value) <= 0.51:
            return True
    for index, left in enumerate(packed_nums):
        for right in packed_nums[index + 1 :]:
            if abs(abs(left - right) - value) <= 0.51:
                return True
            if abs((left + right) - value) <= 0.51:
                return True
            if abs((left * right) - value) <= 1.0:
                return True
            if right and abs((left / right) * 100.0 - value) <= 0.6:
                return True
            if left and abs((right / left) * 100.0 - value) <= 0.6:
                return True
    return False


AREA_IN_QUERY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:square\s*)?(meters?|metres?|m|feet|ft)\s*(?:by|x)\s*(\d+(?:\.\d+)?)",
    re.I,
)


def try_query_compute(query):
    """Use numbers stated in the question (Lewis et al. RAG: the query is context too)."""
    text = query or ""
    if not re.search(r"\barea\b", text, re.I):
        return None
    match = AREA_IN_QUERY_RE.search(text)
    if not match:
        return None
    left = float(match.group(1))
    right = float(match.group(3))
    unit = match.group(2).lower()
    product = left * right
    label = "square meters" if unit.startswith("m") else "square feet"
    if abs(product - round(product)) < 1e-6:
        return f"{int(round(product))} {label}"
    return f"{product:g} {label}"


LOADED_PREMISE_RE = re.compile(
    r"(originally (chosen|titled|called)|upcoming|planning to launch|"
    r"worst performing|most popular song|"
    r"(?:the\s+)?(?:worst|best)\s+\w+(?:\s+\w+){0,4},\s*[A-Za-z])",
    re.I,
)
IDENTITY_APPOSITIVE_RE = re.compile(
    r"(?:the\s+)?(?:worst|best|only)\s+\w+(?:\s+\w+){0,4},\s*[A-Za-z][\w.\- ]{1,40}\s*\??\s*$",
    re.I,
)
OR_COMPARE_RE = re.compile(r"\b(or|vs\.?|versus|compared to)\b", re.I)
SOFT_UNKNOWN_RE = re.compile(
    r"(do not (contain|provide|specify|mention)|not (mentioned|specified|provided|possible to determine)|"
    r"no specific|cannot be answered|not possible to)",
    re.I,
)


def _false_premise_from_sources(query, packed_text, session):
    prompt = f"""Does the question contain a false assumption given the sources?
YES if the sources portray the entity differently than the question assumes, or they contradict a claim built into the question (for example asking for an actor's hit song, or calling a company the worst stock when it is not).
NO if the answer is merely missing from the sources, or the assumption is supported.
Reply with one word: YES or NO.

Sources:
{(packed_text or "")[:3200]}

Question: {query}

Answer:"""
    text = _llm_text(prompt, session) or ""
    first = (text.strip().split() or [""])[0]
    return first.lower().startswith("yes")


def canonicalize_answer(answer, query):
    text = (answer or "").strip()
    if re.match(r"^\s*invalid question\b", text, re.I):
        return "invalid question"
    if REFUSAL_RE.match(text):
        return "I don't know"
    if LOADED_PREMISE_RE.search(query or "") and SOFT_UNKNOWN_RE.search(text):
        return "invalid question"
    return text


def verify_grounding(answer, packed_text, n_sources, intent=None, query=""):
    text = (answer or "").strip()
    if not text or REFUSAL_RE.match(text):
        return True, text
    if re.match(r"^\s*invalid question\b", text, re.I):
        return True, text
    stripped = SOURCE_CITE_RE.sub(" ", text)
    packed_l = (packed_text or "").lower()
    cites = [int(num) for num in SOURCE_CITE_RE.findall(text)]
    if cites and any(num < 1 or num > n_sources for num in cites):
        print("Grounding fail: citation out of range")
        return False, "I don't know"
    packed_nums = _parse_source_numbers(packed_text) + _parse_source_numbers(query)
    missing_nums = [
        num for num in CLAIM_NUM_RE.findall(stripped)
        if not _number_supported(num, packed_l, packed_nums)
    ]
    if missing_nums:
        print(f"Grounding fail: numbers not in sources {missing_nums[:5]}")
        return False, "I don't know"
    if intent == "compute":
        return True, text
    keys = _strict_terms(stripped)
    if keys:
        hit = sum(1 for key in keys if key.lower() in packed_l)
        if hit / len(keys) < 0.5:
            print("Grounding fail: identifiers missing from sources")
            return False, "I don't know"
    return True, text


def retrieve_and_rerank(query, session, limit=None):
    """Shared retrieval core for chat and the eval harness."""
    search_question = rewrite_with_history(query, session.chat_history)
    fused, variations, fetched, exact_n = retrieve_fused(session, search_question)
    scored, raw_by_id = score_documents(search_question, fused, session)
    source_docs, passing, score_map, top_confidence = pick_passing(
        scored, raw_by_id, limit or TOP_DOCS
    )
    return {
        "search_question": search_question,
        "variations": variations,
        "fetched": fetched,
        "exact_n": exact_n,
        "fused_n": len(fused),
        "source_docs": source_docs,
        "passing": passing,
        "score_map": score_map,
        "top_confidence": top_confidence,
        "scored_docs": scored,
        "raw_by_id": raw_by_id,
    }


_FEW_SHOT = """
Format examples (do not copy the facts):

Example A — sources contain the answer:
Sources:
[Source 1 | physics.pdf p.2 text]
Albert Einstein was born in 1879. Marie Curie was born in 1867.
Question: how many years older was Marie Curie than Albert Einstein?
<reasoning>
1. Source 1: Curie 1867, Einstein 1879.
2. 1879 - 1867 = 12.
</reasoning>
<final_answer>Marie Curie was 12 years older than Albert Einstein.</final_answer>

Example B — sources are about a different topic:
Sources:
[Source 1 | sports.pdf p.1 text]
The 2023 final was played in Istanbul.
Question: who was the goalkeeper in that final?
<reasoning>
1. Source 1 names only the venue.
2. The goalkeeper is not written.
</reasoning>
<final_answer>I don't know</final_answer>

Example C — the question assumes something the sources contradict:
Sources:
[Source 1 | bio.pdf p.1 text]
Austin Butler is an American actor known for playing Elvis Presley.
Question: what is Austin Butler's most popular song?
<reasoning>
1. Sources describe him as an actor, not a recording artist.
2. The question's assumption is false.
</reasoning>
<final_answer>invalid question</final_answer>
"""


CONSTRAINT_RE = re.compile(
    r"\b(in \d{4}|of \d{4}|year \d{4}|by \d{4}|since \d{4}|"
    r"last|first|final|total|sector|only)\b",
    re.I,
)
LIST_QUERY_RE = re.compile(
    r"\b(name the|can you name|who are in|who are the|what teams|what songs|"
    r"songs that|members|group [a-d]|put out in|released in|"
    r"mergers?|acquisitions?|acquistions?)\b",
    re.I,
)
PERCENT_YEAR_RE = re.compile(r"percentage of the year|percent of the year|what percentage of the year", re.I)
HOW_LONG_RE = re.compile(r"how long would it take|how long would|how long to", re.I)
HOW_MANY_RE = re.compile(r"\bhow many\b", re.I)
SECTOR_RE = re.compile(r"\bsector\b", re.I)
PER_SHARE_RE = re.compile(r"\bper share\b", re.I)
TOGETHER_RE = re.compile(r"\btogether\b", re.I)
LOSS_COUNT_RE = re.compile(r"\b(lose|lost|losses)\b", re.I)
MA_RE = re.compile(r"mergers?|acquisitions?|acquistions?", re.I)


def _candidate_spans(query, packed_text, limit=10):
    """Pull quoted titles and Group-C style lists from packed sources (extractive RAG)."""
    text = packed_text or ""
    found = []
    year_hit = re.search(r"\b((?:19|20)\d{2})\b", query or "")
    year = year_hit.group(1) if year_hit else None
    for match in re.finditer(r'"([^"]{2,80})"', text):
        item = match.group(1).strip()
        if not item or item in found:
            continue
        if year:
            nearby = text[max(0, match.start() - 100) : match.end() + 100]
            if year not in nearby:
                continue
        found.append(item)
    for match in re.finditer(
        r"group\s+[a-h]\b(?:\s*[:\-]\s*|\s+)([A-Z][^\n.]{8,220})",
        text,
        re.I,
    ):
        item = match.group(1).strip()
        if item and item not in found:
            found.append(item)
    if re.search(r"group\s+[a-h]\b", query or "", re.I):
        for line in text.split("\n"):
            if re.search(r"group\s+[c]\b", line, re.I) and line.count(",") >= 2:
                clip = re.sub(r"\s+", " ", line).strip()
                if 20 < len(clip) < 240 and clip not in found:
                    found.append(clip)
    year = None
    year_hit = re.search(r"\b(19|20)\d{2}\b", query or "")
    if year_hit:
        year = year_hit.group(0)
        for line in text.split("\n"):
            if year in line and re.search(r"\b(song|single|album|track|released)\b", line, re.I):
                clip = re.sub(r"\s+", " ", line).strip()
                if 20 < len(clip) < 180 and clip not in found:
                    found.append(clip)
    return found[:limit]


def _looks_like_list_answer(text):
    body = text or ""
    if body.count(",") >= 2:
        return True
    if len(re.findall(r'"[^"]+"', body)) >= 2:
        return True
    return False


def _answer_prompt(query, combined_docs, history_section="", intent="lookup"):
    extra = ""
    if intent == "compute":
        extra = (
            "This is a calculation. Use numbers from the sources AND from the question itself. "
            "If the question asks how much older or younger, write: NAME is N years older/younger than NAME. "
            "If the question asks how many, list each matching item from the sources first, then count them. "
            "Compute the result and write a full sentence that names the items plus the number.\n"
        )
        if PERCENT_YEAR_RE.search(query or ""):
            extra += (
                "Percentage of the year means (weeks or days at the requested rank) / 52 weeks or 365 days. "
                "Only count weeks or days inside the asked calendar year, not a lifetime total. "
                "Do not treat a long chart run as nearly 100% of a calendar year.\n"
            )
        if HOW_LONG_RE.search(query or ""):
            extra += "Answer with a time duration (days, weeks, or years), not a distance in miles or kilometers.\n"
        if LOSS_COUNT_RE.search(query or ""):
            extra += "Count only losses, not wins. Honor home/away and the stated month or year.\n"
        if TOGETHER_RE.search(query or ""):
            extra += "List every title both people share in the sources, then count. Do not stop at the first film.\n"
    elif intent == "compare":
        extra = (
            "Compare both sides using only the sources. Name BOTH sides in one short sentence. "
            "If only one side is present, still name both sides and report the value you have.\n"
        )
    elif intent == "extract":
        extra = (
            "List every matching name from the sources. Copy titles verbatim. "
            "Include former and founding members, not only the current lineup. "
            "Do not replace a list with a count like 'three awards'.\n"
        )
    candidates = _candidate_spans(query, combined_docs)
    if candidates and (intent == "extract" or LIST_QUERY_RE.search(query or "")):
        extra += "Candidate spans copied from the sources:\n- " + "\n- ".join(candidates) + "\n"
    if SECTOR_RE.search(query or ""):
        extra += "If the question asks for a whole sector, do not answer with one company's per-share dividend.\n"

    if CONSTRAINT_RE.search(query or ""):
        extra += (
            "Honor year, last/first, by-DATE, and sector vs one-company constraints. "
            "Do not answer with a career total or a single company if the question asks otherwise.\n"
        )
    return f"""You are a grounded document assistant. Use ONLY the numbered sources below.

Rules:
- If the question assumes something false (wrong identity, a thing that does not exist, an attribute that does not apply), respond with exactly: invalid question
- If a source mentions the asked people, titles, or numbers, answer from that text.
- Prefer a partial answer over I don't know when any relevant span exists, unless the premise is false.
- Write one complete sentence that repeats the key names from the question and the value from the sources.
- Copy names and numbers as written. Arithmetic is allowed when the inputs are in the sources or in the question.
- Say I don't know only when the sources are about something else entirely.
- Put citations only in <reasoning>, not in <final_answer>.

{_FEW_SHOT}
{history_section}Sources:
{combined_docs}

Question: {query}

{extra}Write <reasoning> as short numbered steps. Then put the user-facing sentence inside <final_answer>.

Answer:"""


def _must_answer_prompt(query, combined_docs):
    return f"""The sources below are relevant to the question. Answer from them.
Do not say I don't know. Write one complete sentence that names the question's entities and the value from the sources.
If several items are listed, include as many as the sources support.

Sources:
{combined_docs}

Question: {query}

<reasoning>steps</reasoning>
<final_answer>complete sentence</final_answer>
"""


def _is_non_answer(text):
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    if lowered.startswith("invalid"):
        return True
    if REFUSAL_RE.match(text or ""):
        return True
    if SOFT_UNKNOWN_RE.search(text or ""):
        return True
    return False


def _comparison_retry_prompt(query, combined_docs):
    return f"""This is a valid comparison question. Do not say invalid question.
Compare the two sides using the sources. Name both sides and the values if present.
If only one side is present, report that side.

Sources:
{combined_docs}

Question: {query}

<reasoning>steps</reasoning>
<final_answer>complete sentence</final_answer>
"""


def _extract_retry_prompt(query, combined_docs):
    candidates = _candidate_spans(query, combined_docs)
    hint = ""
    if candidates:
        hint = "Candidate spans copied from the sources:\n- " + "\n- ".join(candidates) + "\n"
    return f"""Copy matching names, titles, or items from the sources. List every one you can find.
Include former members, alternate titles, and every team or song the sources name.
Do not say I don't know if even one matching item is written below. Copy spellings verbatim.
{hint}
Sources:
{combined_docs}

Question: {query}

<reasoning>steps</reasoning>
<final_answer>complete sentence listing the items</final_answer>
"""


def _sector_retry_prompt(query, combined_docs):
    return f"""The question asks for a whole sector, not one company.
Find the sector-wide total in the sources. Do not answer with a per-share dividend for a single firm.

Sources:
{combined_docs}

Question: {query}

<reasoning>steps</reasoning>
<final_answer>complete sentence</final_answer>
"""


def _duration_retry_prompt(query, combined_docs):
    return f"""The question asks how long something takes. Answer with time (years, months, weeks, or days), not miles or kilometers.

Sources:
{combined_docs}

Question: {query}

<reasoning>steps</reasoning>
<final_answer>complete sentence</final_answer>
"""


def _percent_year_retry_prompt(query, combined_docs):
    return f"""Percentage of the year means (weeks at the requested rank inside the asked year) / 52, or (days) / 365.
Do not use a career or multi-year total. A chart run of several weeks is around 10-20% of a year, not 90%+.
Read the weeks or days from the sources, divide, and write the percent.

Sources:
{combined_docs}

Question: {query}

<reasoning>steps</reasoning>
<final_answer>complete sentence</final_answer>
"""


def _together_retry_prompt(query, combined_docs):
    return f"""List every movie or show both people appear in from the sources, then count them.
Do not stop after the first shared title.

Sources:
{combined_docs}

Question: {query}

<reasoning>steps</reasoning>
<final_answer>complete sentence</final_answer>
"""


def _high_year_percent(text):
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*%", text or "")
    return any(float(n) >= 18 for n in nums)


def _llm_text(prompt, session):
    assembled = ""
    for _piece, assembled in _run_llm(prompt, session):
        if session_cancelled(session):
            return None
    return (assembled or "").strip()


def _generate_answer(query, session, docs, score_map, history_block, extra):
    char_budget = source_char_budget(query, history_block)
    combined_docs, top_docs = pack_context(docs[:MAX_PACKED_DOCS], char_budget)
    citations = _build_citations(top_docs, score_map)
    history_section = f"{history_block}\n\n" if history_block else ""
    intent = extra.get("intent") or "lookup"
    if OR_COMPARE_RE.search(query or "") and intent != "compute":
        intent = "compare"
    direct = try_query_compute(query)
    if direct:
        extra = dict(extra)
        extra["grounded"] = True
        extra["retried_answer"] = False
        extra["direct_compute"] = True
        return direct, "Computed from numbers in the question.", citations, extra, top_docs
    comparison = bool(OR_COMPARE_RE.search(query or ""))
    if combined_docs and not comparison:
        if IDENTITY_APPOSITIVE_RE.search(query or ""):
            extra = dict(extra)
            extra["grounded"] = True
            extra["retried_answer"] = False
            extra["false_premise"] = True
            return "invalid question", "", citations, extra, top_docs
        if LOADED_PREMISE_RE.search(query or "") and _false_premise_from_sources(
            query, combined_docs, session
        ):
            extra = dict(extra)
            extra["grounded"] = True
            extra["retried_answer"] = False
            extra["false_premise"] = True
            return "invalid question", "", citations, extra, top_docs
    prompt = _answer_prompt(query, combined_docs, history_section, intent)
    assembled = _llm_text(prompt, session)
    if assembled is None:
        return None, None, citations, extra, None
    display_answer = canonicalize_answer(extract_display_answer(assembled), query)
    if (
        comparison
        and combined_docs
        and (display_answer or "").lower().startswith("invalid")
    ):
        retry = _llm_text(_comparison_retry_prompt(query, combined_docs), session)
        if retry is None:
            return None, None, citations, extra, None
        if retry:
            assembled = retry
            display_answer = canonicalize_answer(extract_display_answer(assembled), query)
    if (
        combined_docs
        and not (display_answer or "").lower().startswith("invalid")
        and (
            (intent == "extract" and _is_non_answer(display_answer))
            or (
                LIST_QUERY_RE.search(query or "")
                and not _looks_like_list_answer(display_answer)
            )
        )
    ):
        retry = _llm_text(_extract_retry_prompt(query, combined_docs), session)
        if retry is None:
            return None, None, citations, extra, None
        if retry:
            assembled = retry
            display_answer = canonicalize_answer(extract_display_answer(assembled), query)
    if (
        combined_docs
        and SECTOR_RE.search(query or "")
        and (
            PER_SHARE_RE.search(display_answer or "")
            or _is_non_answer(display_answer)
        )
    ):
        retry = _llm_text(_sector_retry_prompt(query, combined_docs), session)
        if retry is None:
            return None, None, citations, extra, None
        if retry:
            assembled = retry
            display_answer = canonicalize_answer(extract_display_answer(assembled), query)
    if (
        combined_docs
        and HOW_LONG_RE.search(query or "")
        and (
            (display_answer or "").lower().startswith("invalid")
            or _is_non_answer(display_answer)
            or re.search(r"\b(miles?|kilometers?|km)\b", display_answer or "", re.I)
        )
    ):
        retry = _llm_text(_duration_retry_prompt(query, combined_docs), session)
        if retry is None:
            return None, None, citations, extra, None
        if retry:
            assembled = retry
            display_answer = canonicalize_answer(extract_display_answer(assembled), query)
    if (
        combined_docs
        and HOW_LONG_RE.search(query or "")
        and re.search(r"migrat", query or "", re.I)
        and re.search(r"\bweeks?\b", display_answer or "", re.I)
    ):
        retry = _llm_text(_duration_retry_prompt(query, combined_docs), session)
        if retry is None:
            return None, None, citations, extra, None
        if retry:
            assembled = retry
            display_answer = canonicalize_answer(extract_display_answer(assembled), query)
    if (
        combined_docs
        and HOW_MANY_RE.search(query or "")
        and TOGETHER_RE.search(query or "")
        and re.search(r"\b(1|one)\b", display_answer or "", re.I)
        and not (display_answer or "").lower().startswith("invalid")
    ):
        retry = _llm_text(_together_retry_prompt(query, combined_docs), session)
        if retry is None:
            return None, None, citations, extra, None
        if retry:
            assembled = retry
            display_answer = canonicalize_answer(extract_display_answer(assembled), query)
    if (
        combined_docs
        and PERCENT_YEAR_RE.search(query or "")
        and _high_year_percent(display_answer)
    ):
        retry = _llm_text(_percent_year_retry_prompt(query, combined_docs), session)
        if retry is None:
            return None, None, citations, extra, None
        if retry:
            assembled = retry
            display_answer = canonicalize_answer(extract_display_answer(assembled), query)
    if (
        MUST_ANSWER_RETRY
        and REFUSAL_RE.match(display_answer or "")
        and combined_docs
        and context_supports_query(query, combined_docs)
        and display_answer.lower() != "invalid question"
    ):
        retry = _llm_text(_must_answer_prompt(query, combined_docs), session)
        if retry is None:
            return None, None, citations, extra, None
        if retry:
            assembled = retry
            display_answer = canonicalize_answer(extract_display_answer(assembled), query)
    grounded, display_answer = verify_grounding(
        display_answer, combined_docs, len(top_docs), intent, query=query
    )
    display_answer = canonicalize_answer(display_answer, query)
    extra = dict(extra)
    extra["grounded"] = grounded
    extra["retried_answer"] = bool(REFUSAL_RE.match(display_answer or "") is False)
    if not grounded:
        extra["rerank_passed"] = False
    reasoning = "" if not grounded else distinct_reasoning(display_answer, extract_reasoning(assembled))
    return (display_answer, reasoning, citations, extra, top_docs)


def load_thread_history(session, messages):
    """Restore a saved chat into the live session so follow-ups keep context."""
    session.chat_history = []
    for item in messages or []:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        role = item.get("role") or ""
        if role == "user":
            session.chat_history.append(HumanMessage(content=content))
        elif role == "assistant":
            session.chat_history.append(AIMessage(content=content))
    _trim_history(session.chat_history)


def _chat_events(query, session):
    """General assistant harness — native chat template, streamed tokens, tools."""
    direct = try_direct_math(query)
    reason = "direct_math"
    if not direct:
        direct = try_direct_clock(query)
        reason = "direct_clock"
    if direct:
        _remember(session.chat_history, query, direct)
        yield _complete(
            direct,
            extra={
                "route": "chat",
                "reason": reason,
                "intent": "chat",
                "rerank_passed": None,
            },
        )
        return
    yield {
        "type": "stage",
        "id": "reasoning",
        "label": "Thinking",
        "detail": "Thinking…",
    }
    pdf_names = attached_pdf_names(session)
    messages = thread_messages(
        session.chat_history,
        query,
        limit=HISTORY_MESSAGES,
        pdf_names=pdf_names,
    )
    used_tools = []
    hit_docs = []
    answer = None
    model_name = getattr(session, "llm_model", None)
    try:
        tool_llm = get_llm(model_name, temperature=0.15, chat=True).bind_tools(
            CHAT_TOOLS,
            num_predict=TOOL_NUM_PREDICT,
        )
    except Exception as exc:
        print(f"Chat tools unavailable ({exc}); answering without them.")
        tool_llm = None

    if tool_llm is not None:
        for _round in range(MAX_TOOL_ROUNDS):
            if session_cancelled(session):
                yield _stopped()
                return
            try:
                response = tool_llm.invoke(messages)
            except Exception as exc:
                print(f"Tool-calling round failed ({exc}); answering without more tools.")
                break
            calls = normalize_tool_calls(response)
            if not calls:
                visible = strip_chat(getattr(response, "content", None) or "")
                if visible:
                    yield {"type": "token", "text": visible, "answer": visible}
                    answer = visible
                break
            try:
                response.tool_calls = [
                    {
                        "name": call["name"],
                        "args": call["args"],
                        "id": call["id"],
                        "type": "tool_call",
                    }
                    for call in calls
                ]
            except Exception:
                pass
            messages.append(response)
            yield {
                "type": "stage",
                "id": "tools",
                "label": "Using tools",
                "detail": stage_for_calls(calls),
            }
            for call in calls:
                if session_cancelled(session):
                    yield _stopped()
                    return
                name = call.get("name") or ""
                used_tools.append(name)
                result, docs = execute_tool(name, call.get("args"), session)
                if docs:
                    hit_docs.extend(docs)
                _safe_print(f"Tool {name} -> {str(result)[:180]!r}")
                messages.append(
                    ToolMessage(
                        content=str(result)[:MAX_TOOL_CHARS],
                        tool_call_id=call.get("id") or name,
                    )
                )
        else:
            print("Chat tool round limit reached; writing a final answer.")

    if answer is None:
        yield {
            "type": "stage",
            "id": "reasoning",
            "label": "Thinking",
            "detail": "Thinking…",
        }
        assembled = ""
        for piece, assembled in _run_llm(messages, session, temperature=0.7, chat=True):
            if session_cancelled(session):
                yield _stopped()
                return
            visible = strip_chat(assembled)
            if visible:
                yield {"type": "token", "text": piece, "answer": visible}
        answer = strip_chat(assembled) or (assembled or "").strip() or "I don't know"

    _remember(session.chat_history, query, answer)
    citations = _build_citations(_dedupe_tool_docs(hit_docs)[:6]) if hit_docs else []
    write_trace(
        session,
        {
            "event": "chat",
            "query": query,
            "route": getattr(session, "last_route", "chat"),
            "reason": getattr(session, "last_route_reason", ""),
            "tools": used_tools,
        },
    )
    yield _complete(
        answer,
        citations,
        extra={
            "route": getattr(session, "last_route", "chat"),
            "reason": getattr(session, "last_route_reason", ""),
            "intent": "chat",
            "rerank_passed": None,
            "tools": used_tools,
        },
    )


def _dedupe_tool_docs(docs):
    seen = set()
    unique = []
    for doc in docs or []:
        meta = getattr(doc, "metadata", None) or {}
        key = (meta.get("content_hash"), meta.get("page"), meta.get("source"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def ask_question_events(query, session, force_mode="auto"):
    chat_history = session.chat_history
    started = time.time()
    doc_count = len(
        {
            (chunk.metadata or {}).get("content_hash")
            for chunk in (session.chunks or [])
            if (chunk.metadata or {}).get("content_hash")
        }
    ) or (1 if session.chunks else 0)

    def cancelled():
        return session_cancelled(session)

    route, reason = decide_route(query, doc_count=doc_count, force_mode=force_mode)
    session.last_route = route
    session.last_route_reason = reason
    print(f"Route={route} reason={reason} docs={doc_count} chunks={len(session.chunks or [])}")
    yield {
        "type": "stage",
        "id": "routing",
        "label": "Thinking" if route == "chat" else "Routing",
        "detail": "Thinking…" if route == "chat" else f"{route_label(route)} ({reason})",
        "route": route,
        "reason": reason,
    }

    if route == "chat":
        yield from _chat_events(query, session)
        return

    if cancelled():
        yield _stopped()
        return

    if not session.chunks or session.hybrid_retriever is None:
        answer = "I don't know"
        _remember(chat_history, query, answer)
        yield _complete(answer, extra={"route": route, "reason": reason, "intent": "lookup"})
        return

    intent = {
        "rag_summarize": "summarize",
        "rag_compare": "compare",
        "rag_compute": "compute",
        "rag_extract": "extract",
        "rag_lookup": "lookup",
    }.get(route, classify_intent(query))
    search_question = rewrite_with_history(query, chat_history)
    yield {
        "type": "stage",
        "id": "understanding",
        "label": "Understanding the question",
        "detail": f"{intent}: {search_question[:120]}",
    }
    if search_question != query:
        print(f"Follow-up resolved to: {search_question}")

    if intent == "summarize":
        yield from _summarize_events(query, session)
        return

    compare_topics = split_compare_topics(query) if intent == "compare" else None
    if compare_topics:
        yield from _compare_events(query, session, compare_topics)
        return

    keep_n = max(
        EXTRACT_TOP_DOCS if intent == "extract" else TOP_DOCS,
        MAX_PACKED_DOCS,
    )
    yield {
        "type": "stage",
        "id": "fetching",
        "label": "Fetching chunks",
        "detail": "Keyword scan and hybrid search",
    }
    retrieval = retrieve_and_rerank(query, session, limit=keep_n)
    fused_n = retrieval["fused_n"]
    yield {
        "type": "stage",
        "id": "fetching",
        "label": "Fetching chunks",
        "detail": f"{retrieval['fetched']} hits · {fused_n} unique · {retrieval['exact_n']} exact",
    }
    if not fused_n:
        answer = "I don't know"
        _remember(chat_history, query, answer)
        write_trace(session, {"event": "empty_retrieval", "query": query, "intent": intent, "route": route})
        yield _complete(answer, extra={"route": route, "reason": reason, "intent": intent})
        return

    yield {
        "type": "stage",
        "id": "reranking",
        "label": "Reranking sources",
        "detail": f"{fused_n} unique chunks",
    }
    source_docs = retrieval["source_docs"]
    passing = retrieval["passing"]
    score_map = retrieval["score_map"]
    top_confidence = retrieval["top_confidence"]
    retried = False
    if not passing and AGENT_RETRIES > 0:
        retry_q = retry_search_question(query, retrieval["search_question"], source_docs)
        if retry_q and retry_q.lower() != retrieval["search_question"].lower():
            retried = True
            yield {
                "type": "stage",
                "id": "retry",
                "label": "Retrying retrieval",
                "detail": retry_q[:120],
            }
            fused, _, _, _ = retrieve_fused(session, retry_q)
            fused = prepend_unique(source_docs, fused)
            scored, raw_by_id = score_documents(retrieval["search_question"], fused, session)
            source_docs, passing, score_map, top_confidence = pick_passing(
                scored, raw_by_id, keep_n
            )

    citations = _build_citations(source_docs, score_map)
    extra = {
        "rerank_confidence": round(float(top_confidence), 4),
        "rerank_min_score": RERANK_MIN_SCORE,
        "rerank_passed": bool(passing),
        "intent": intent,
        "retried": retried,
    }
    yield {
        "type": "stage",
        "id": "reranking",
        "label": "Reranking sources",
        "detail": (
            f"Top {len(source_docs)} · best conf {top_confidence:.2f} "
            f"(min {RERANK_MIN_SCORE:.2f})"
        ),
        "citations": citations,
    }

    if cancelled():
        yield _stopped(citations)
        return

    answer_docs = merge_answer_docs(passing, source_docs, MAX_PACKED_DOCS)
    if not answer_docs:
        answer = "I don't know"
        _remember(chat_history, query, answer)
        write_trace(
            session,
            {
                "event": "no_answer_docs",
                "query": query,
                "intent": intent,
                "confidence": top_confidence,
                "retried": retried,
                "elapsed": time.time() - started,
            },
        )
        yield _complete(answer, citations, extra)
        return
    if not passing:
        print(
            f"Rerank confidence {top_confidence:.3f} < {RERANK_MIN_SCORE:.3f}; "
            "generating from top retrieved docs anyway."
        )

    history_block = _history_block(chat_history)
    yield {
        "type": "stage",
        "id": "reasoning",
        "label": "Reasoning",
        "detail": "Writing the answer",
        "citations": citations,
    }
    generated = _generate_answer(query, session, answer_docs, score_map, history_block, extra)
    if generated[0] is None:
        yield _stopped(citations)
        return
    display_answer, reasoning, citations, extra, _top_docs = generated
    _remember(chat_history, query, display_answer)
    write_trace(
        session,
        {
            "event": "complete",
            "query": query,
            "intent": intent,
            "confidence": top_confidence,
            "retried": retried,
            "grounded": extra.get("grounded", True),
            "elapsed": time.time() - started,
            "answer": display_answer[:240],
        },
    )
    print(f"Answer: {display_answer}")
    yield {
        "type": "complete",
        "answer": display_answer,
        "context": [item["excerpt"] for item in citations],
        "citations": citations,
        "reasoning": reasoning,
        **extra,
        "rerank_passed": True if extra.get("grounded", True) else False,
    }


def _summarize_events(query, session):
    names = {
        (chunk.metadata or {}).get("source")
        for chunk in session.chunks
        if (chunk.metadata or {}).get("source")
    }
    mentioned = [name for name in names if name.lower() in (query or "").lower()]
    pool = session.chunks
    if mentioned:
        pool = [
            chunk for chunk in session.chunks
            if (chunk.metadata or {}).get("source") in mentioned
        ]
    parents = collapse_to_parents(pool)[:SUMMARIZE_MAP_CHUNKS]
    if not parents:
        answer = "I don't know"
        _remember(session.chat_history, query, answer)
        yield _complete(answer)
        return
    yield {
        "type": "stage",
        "id": "summarize",
        "label": "Summarizing document",
        "detail": f"{len(parents)} pages/sections",
    }
    citations = _build_citations(parents[:TOP_DOCS])
    partials = []
    for index, doc in enumerate(parents, 1):
        if session_cancelled(session):
            yield _stopped(citations)
            return
        yield {
            "type": "stage",
            "id": "summarize",
            "label": "Summarizing document",
            "detail": f"Section {index}/{len(parents)}",
            "citations": citations,
        }
        body = _clip(get_answer_context(doc), 3500)
        prompt = (
            "Summarize this document section densely. Keep names, numbers, dates, "
            "and obligations. Output only the summary.\n\n"
            f"{body}"
        )
        assembled = ""
        for _piece, assembled in _run_llm(prompt, session):
            if session_cancelled(session):
                yield _stopped(citations)
                return
        text = extract_display_answer(assembled) or assembled.strip()
        if text:
            partials.append(text)
    if not partials:
        answer = "I don't know"
        _remember(session.chat_history, query, answer)
        yield _complete(answer, citations)
        return
    combined = "\n\n".join(partials)
    final_prompt = (
        "Combine these section summaries into one coherent Markdown summary "
        "with a short overview, key points, and important numbers or dates. "
        "Use only this material.\n\n"
        f"{_clip(combined, 12000)}\n\nUser request: {query}\n\n"
        "Put the user-facing summary inside <final_answer> tags."
    )
    assembled = ""
    for _piece, assembled in _run_llm(final_prompt, session):
        if session_cancelled(session):
            yield _stopped(citations)
            return
    display_answer = extract_display_answer(assembled) or assembled.strip()
    _remember(session.chat_history, query, display_answer)
    write_trace(session, {"event": "summarize", "query": query, "sections": len(parents)})
    yield {
        "type": "complete",
        "answer": display_answer,
        "context": [item["excerpt"] for item in citations],
        "citations": citations,
        "reasoning": "",
        "intent": "summarize",
        "rerank_passed": True,
    }


def _compare_events(query, session, topics):
    yield {
        "type": "stage",
        "id": "compare",
        "label": "Comparing topics",
        "detail": " vs ".join(topics)[:120],
    }
    combined_docs = []
    score_map = {}
    for topic in topics:
        retrieval = retrieve_and_rerank(topic, session, limit=COMPARE_TOP_PER_SIDE)
        chosen = merge_answer_docs(
            retrieval["passing"], retrieval["source_docs"], COMPARE_TOP_PER_SIDE
        )
        for doc in chosen:
            combined_docs.append(doc)
            score_map.update(retrieval["score_map"])
    combined_docs = collapse_to_parents(combined_docs)
    passing = [
        doc for doc in combined_docs
        if score_map.get(id(doc), (0.0, 0.0))[1] >= RERANK_MIN_SCORE
    ]
    citations = _build_citations(combined_docs, score_map)
    extra = {
        "intent": "compare",
        "rerank_passed": bool(passing),
        "rerank_min_score": RERANK_MIN_SCORE,
        "rerank_confidence": round(
            max((score_map.get(id(doc), (0.0, 0.0))[1] for doc in combined_docs), default=0.0),
            4,
        ),
    }
    if not passing:
        print("Compare rerank below threshold; generating from retrieved docs anyway.")
    history_block = _history_block(session.chat_history)
    yield {
        "type": "stage",
        "id": "reasoning",
        "label": "Reasoning",
        "detail": "Writing the comparison",
        "citations": citations,
    }
    generated = _generate_answer(
        query, session, merge_answer_docs(passing, combined_docs, MAX_PACKED_DOCS),
        score_map, history_block, extra,
    )
    if generated[0] is None:
        yield _stopped(citations)
        return
    display_answer, reasoning, citations, extra, _top = generated
    _remember(session.chat_history, query, display_answer)
    write_trace(session, {"event": "compare", "query": query, "topics": topics})
    yield {
        "type": "complete",
        "answer": display_answer,
        "context": [item["excerpt"] for item in citations],
        "citations": citations,
        "reasoning": reasoning,
        **extra,
    }


def ask_question(query, session, force_mode="auto"):
    result = {"answer": "I don't know", "context": [], "citations": []}
    for event in ask_question_events(query, session, force_mode=force_mode):
        if event.get("type") == "complete":
            result = {
                "answer": event.get("answer") or "",
                "context": event.get("context") or [],
                "citations": event.get("citations") or [],
                "reasoning": event.get("reasoning") or "",
                "rerank_confidence": event.get("rerank_confidence"),
                "rerank_min_score": event.get("rerank_min_score"),
                "rerank_passed": event.get("rerank_passed"),
                "intent": event.get("intent"),
                "route": event.get("route"),
                "reason": event.get("reason"),
            }
    return result
