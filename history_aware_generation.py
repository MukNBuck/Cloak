"""
History-aware RAG chat pipeline.

Query flow:
  question → (optional history rewrite) → multi-query
  → hybrid search (BM25 + vector) → RRF across queries
  → cross-encoder rerank → answer from original_content
"""

import json
from collections import defaultdict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

load_dotenv()

# ---------------------------------------------------------------------------
# Vector store + BM25 corpus 
# ---------------------------------------------------------------------------
persist_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"}, 
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_name="rag_collection",
    collection_metadata={"hnsw:space": "cosine"},
)

print(f"Total vectors stored in database: {db._collection.count()}")

_raw = db.get(include=["documents", "metadatas"])
chunks = [
    Document(page_content=content, metadata=metadata or {})
    for content, metadata in zip(_raw["documents"] or [], _raw["metadatas"] or [])
]

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
llm = ChatOllama(
    model="qwen2.5vl:7b",
    temperature=0.0,
    num_gpu=99,
    num_ctx=16384,
    num_predict=200,
)

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------
vector_retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8},
)

if chunks:
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 8
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever]
    )
else:
    hybrid_retriever = vector_retriever

chat_history = []

class QueryVariations(BaseModel):
    queries: list[str] = Field(
        description="A list of 3 standalone search queries based on the conversation"
    )

def create_query_variations(original_query: str, llm) -> list[str]:
    prompt = f"""Generate 3 different variations of the query that would help retrieve relevant documents:

    Original query: {original_query}

    Return 3 alternative queries that rephrase or approach the same question from different angles."""

    llm_queries = llm.with_structured_output(QueryVariations, method="json_schema")
    query_variations = llm_queries.invoke(prompt)
    for i, variation in enumerate(query_variations.queries, 1):
        print(f"{i}. {variation}")

    print("Generated Query variations")
    return query_variations.queries

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

def ask_question(query):
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
        docs = hybrid_retriever.invoke(variation)
        all_retrieval_results.append(docs)

    print(f"\nFusing {len(all_retrieval_results)} query result lists with RRF...")
    fused_docs = reciprocal_rank_fusion(all_retrieval_results, k=60)
    print(f"RRF produced {len(fused_docs)} unique candidates")

    if not fused_docs:
        answer = "I don't know"
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
        print(f"Answer: {answer}")
        
        # Return empty context array if nothing is found
        return {"answer": answer, "context": []}

    print("Reranking fused candidates with cross-encoder...")
    pairs = [[search_question, doc.page_content] for doc in fused_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, fused_docs), key=lambda x: x[0], reverse=True)

    top_docs = [doc for score, doc in scored_docs[:5]]

    combined_docs = "\n\n".join(get_answer_context(doc) for doc in top_docs)

    prompt = f"""Answer the following question based strictly on the provided context. If the answer is not contained within the context, respond with "I don't know".

    Context:
    {combined_docs}

    Question: {query}
    Answer:"""

    response = llm.invoke(prompt)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))

    print(f"Answer: {response.content}")
    
    # Return both the LLM's generated answer and the source document excerpts
    return {
        "answer": response.content,
        "context": [get_answer_context(doc) for doc in top_docs]
    }

