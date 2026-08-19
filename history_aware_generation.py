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


def create_query_variations(original_query: str, llm) -> list[str]:
    prompt = f"""Generate 2 alternative search queries for a hardware datasheet.

    Original query: {original_query}

    Rules:
    Keep exact technical terms, units, and metric names from the original (for example: sensitivity, dBm, capacitive touch, GPIO).
    Do not replace a requested metric with a related one (do not change sensitivity into current consumption, or touch into CAP/SENSOR).
    Only rephrase around those preserved terms."""

    llm_queries = llm.with_structured_output(QueryVariations, method="json_schema")
    query_variations = llm_queries.invoke(prompt)

    if isinstance(query_variations, dict):
        variations = query_variations.get("queries", [])
    else:
        variations = getattr(query_variations, "queries", [])

    variations = variations[:2]
    for i, variation in enumerate(variations, 1):
        print(f"{i}. {variation}")

    print("Generated Query variations")
    return variations


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


def ask_question(query, session):
    chat_history = session.chat_history

    if not session.chunks and session.vectorstore is not None:
        session.chunks = load_chunks_from_vectorstore(session.vectorstore)
        session.rebuild_retrievers()

    print(f"Session has {len(session.chunks)} chunks")
    if not session.chunks or session.hybrid_retriever is None:
        answer = "I don't know"
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
        return {"answer": answer, "context": [], "citations": []}

    if chat_history:
        messages = [
            SystemMessage(
                content="Given the chat history, rewrite the question to be a standalone question."
            )
        ] + chat_history + [
            HumanMessage(content=f"New question: {query}")
        ]

        result = llm.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for {search_question}...")
    else:
        search_question = query

    all_retrieval_results = []
    query_variations = [search_question] + create_query_variations(search_question, llm)
    query_weights = [2.0] + [1.0] * (len(query_variations) - 1)

    for i, variation in enumerate(query_variations, 1):
        print(f"RESULTS FOR QUERY {i}: {variation}")
        docs = session.hybrid_retriever.invoke(variation)
        all_retrieval_results.append(docs)

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
        return {"answer": answer, "context": []}

    print("Reranking fused candidates with cross-encoder...")
    rerank_docs = fused_docs[:24]
    pairs = [[search_question, doc.page_content] for doc in rerank_docs]
    scores = reranker.predict(pairs)
    boost_tables = any(hint in search_question.lower() for hint in SPEC_HINTS)
    scored_docs = []
    for score, doc in zip(scores, rerank_docs):
        adjusted = float(score)
        if boost_tables and doc.metadata.get("type") == "table":
            adjusted += 1.0
        scored_docs.append((adjusted, doc))
    scored_docs.sort(key=lambda item: item[0], reverse=True)

    top_docs = [doc for score, doc in scored_docs[:5]]

    combined_docs = "\n\n".join(get_answer_context(doc) for doc in top_docs)

    prompt = f"""Answer the following question based strictly on the provided context. If the answer is not contained within the context, respond with "I don't know".

    Context:
    {combined_docs}

    Question: {query}

    Instructions:
    Tables are written as sentences like "Row: Header is value". Treat each Row as one complete record; do not mix values across rows.
    Rely on explicit tables and lists before inferring answers from partial acronyms or variable names. CAP in SENSOR_CAPP is not capacitive touch.
    If the context has a related metric but not the asked quantity or unit (for example mA when the question asks for dBm), respond with "I don't know".
    Reply with a short direct answer only. Put it inside <final_answer> tags. Do not write reasoning, lists, or extra commentary.

    Answer:"""

    response = llm.invoke(prompt)
    display_answer = extract_display_answer(response.content)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=display_answer))

    print(f"Answer: {display_answer}")

    citations = []
    for index, doc in enumerate(top_docs, 1):
        meta = doc.metadata or {}
        page = meta.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        citations.append(
            {
                "id": index,
                "source": meta.get("source") or "document.pdf",
                "content_hash": meta.get("content_hash") or "",
                "type": meta.get("type") or "text",
                "page": page,
                "excerpt": get_answer_context(doc),
            }
        )

    return {
        "answer": display_answer,
        "context": [item["excerpt"] for item in citations],
        "citations": citations,
    }
 