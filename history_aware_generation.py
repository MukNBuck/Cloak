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
        min_length=3,
        max_length=3,
        description="A list of 3 standalone search queries based on the conversation",
    )


def create_query_variations(original_query: str, llm) -> list[str]:
    prompt = f"""Generate 3 different variations of the query that would help retrieve relevant documents:

    Original query: {original_query}

    Return 3 alternative queries that rephrase or approach the same question from different angles."""

    llm_queries = llm.with_structured_output(QueryVariations, method="json_schema")
    query_variations = llm_queries.invoke(prompt)

    if isinstance(query_variations, dict):
        variations = query_variations.get("queries", [])
    else:
        variations = getattr(query_variations, "queries", [])

    variations = variations[:3]
    for i, variation in enumerate(variations, 1):
        print(f"{i}. {variation}")

    print("Generated Query variations")
    return variations


def reciprocal_rank_fusion(chunk_lists, k=60, verbose=False):
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

        for position, chunk in enumerate(result_chunks, 1):
            chunk_content = chunk.page_content

            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1

            chunk_id = chunk_id_map[chunk_content]
            all_unique_chunks[chunk_content] = chunk
            position_score = 1 / (k + position)
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

    if not session.chunks or session.hybrid_retriever is None:
        answer = "I don't know"
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
        return {"answer": answer, "context": []}

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

    for i, variation in enumerate(query_variations, 1):
        print(f"RESULTS FOR QUERY {i}: {variation}")
        docs = session.hybrid_retriever.invoke(variation)
        all_retrieval_results.append(docs)

    print(f"\nFusing {len(all_retrieval_results)} query result lists with RRF...")
    fused_docs = reciprocal_rank_fusion(all_retrieval_results, k=60)
    print(f"RRF produced {len(fused_docs)} unique candidates")

    if not fused_docs:
        answer = "I don't know"
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
        print(f"Answer: {answer}")
        return {"answer": answer, "context": []}

    print("Reranking fused candidates with cross-encoder...")
    pairs = [[search_question, doc.page_content] for doc in fused_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, fused_docs), key=lambda x: x[0], reverse=True)

    top_docs = [doc for score, doc in scored_docs[:3]]

    combined_docs = "\n\n".join(get_answer_context(doc) for doc in top_docs)

    prompt = f"""Answer the following question based strictly on the provided context. If the answer is not contained within the context, respond with "I don't know".

    Context:
    {combined_docs}

    Question: {query}

    Instructions:
    Tables are written as sentences like "Row: Header is value". Treat each Row as one complete record; do not mix values across rows.
    Reply with a short direct answer only. Put it inside <final_answer> tags. Do not write reasoning, lists, or extra commentary.

    Answer:"""

    response = llm.invoke(prompt)
    display_answer = extract_display_answer(response.content)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=display_answer))

    print(f"Answer: {display_answer}")

    return {
        "answer": display_answer,
        "context": [get_answer_context(doc) for doc in top_docs],
    }
