import os
import base64
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_PACK_VERSION = 1
EXPORTABLE_METADATA_KEYS = ("source", "type", "content_hash")


def get_image_base64(image_path):
    """Convert an image to base64 so Ollama can read it."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def summarize_image_with_ollama(image_path):
    """Send the extracted image to your GTX 1060 vision model."""
    print(f"Sending {os.path.basename(image_path)} to qwen2.5vl:7b for visual parsing...")

    vision_llm = ChatOllama(model="llava", temperature=0.0)
    image_b64 = get_image_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image, chart, or diagram in meticulous detail. Include any numbers, dimensions, or text visible in the image. This will be used in a searchable engineering database.",
                },
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            ],
        }
    ]

    response = vision_llm.invoke(messages)
    return response.content


@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_ephemeral_vectorstore(collection_name="rag_collection"):
    client = chromadb.EphemeralClient()
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        collection_metadata={"hnsw:space": "cosine"},
    ), client


def _clean_metadata(metadata):
    cleaned = {}
    for key in EXPORTABLE_METADATA_KEYS:
        value = (metadata or {}).get(key)
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
    return cleaned


def _partition_pdf(file_path, output_image_dir):
    kwargs = dict(
        filename=file_path,
        strategy="fast",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )
    try:
        return partition_pdf(**kwargs, extract_image_block_output_dir=output_image_dir)
    except TypeError:
        return partition_pdf(**kwargs)


def load_documents_with_vision(file_path, output_image_dir, source_filename=None, content_hash=None):
    """Extract text AND images from a single PDF, summarize images, and combine into documents."""
    filename = source_filename or os.path.basename(file_path)
    print(f"Loading and partitioning {filename}...")
    os.makedirs(output_image_dir, exist_ok=True)

    print(f"\nProcessing {filename}...")
    raw_pdf_elements = _partition_pdf(file_path, output_image_dir)

    all_documents = []
    base_metadata = {"source": filename, "type": "text"}
    if content_hash:
        base_metadata["content_hash"] = content_hash

    for element in raw_pdf_elements:
        if "CompositeElement" in str(type(element)):
            all_documents.append(
                Document(page_content=element.text, metadata=dict(base_metadata))
            )

    for image_name in os.listdir(output_image_dir):
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(output_image_dir, image_name)
        image_summary = summarize_image_with_ollama(img_path)
        image_metadata = {
            "source": filename,
            "type": "image",
            "image_path": img_path,
        }
        if content_hash:
            image_metadata["content_hash"] = content_hash
        all_documents.append(
            Document(
                page_content=f"Image Description: {image_summary}",
                metadata=image_metadata,
            )
        )

    return all_documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]):
    print(f"Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:2]):
            print(f"\nChunk {i+1}:")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:\n{chunk.page_content[:200]}...\n" + "-" * 50)

    return chunks


def add_chunks_to_vectorstore(vectorstore, chunks, content_hash):
    """Embed chunks into an existing ephemeral Chroma collection."""
    if not chunks:
        return []

    ids = [f"{content_hash}-{i}" for i in range(len(chunks))]
    for chunk in chunks:
        chunk.metadata = _clean_metadata(chunk.metadata)
        chunk.metadata["content_hash"] = content_hash

    print("Creating embeddings and storing in session ChromaDB...")
    vectorstore.add_documents(documents=chunks, ids=ids)
    print(f"Finished embedding {len(chunks)} chunks for {content_hash[:12]}...")
    return ids


def collection_has_hash(vectorstore, content_hash):
    existing = vectorstore.get(where={"content_hash": content_hash})
    return len(existing.get("ids") or []) > 0


def embedding_pack_is_valid(pack, content_hash):
    if not isinstance(pack, dict):
        return False
    if pack.get("model") != EMBEDDING_MODEL_NAME:
        return False
    if pack.get("content_hash") != content_hash:
        return False
    chunks = pack.get("chunks")
    return isinstance(chunks, list) and len(chunks) > 0


def import_embedding_pack(vectorstore, pack):
    """Load precomputed embeddings into Chroma without re-running the embedding model."""
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    langchain_docs = []

    content_hash = pack["content_hash"]
    filename = pack.get("filename") or "document.pdf"

    for i, chunk in enumerate(pack["chunks"]):
        text = chunk.get("text") or ""
        embedding = chunk.get("embedding")
        if not text or not isinstance(embedding, list) or not embedding:
            continue

        chunk_id = chunk.get("id") or f"{content_hash}-{i}"
        metadata = _clean_metadata(chunk.get("metadata"))
        metadata["content_hash"] = content_hash
        metadata.setdefault("source", filename)
        metadata.setdefault("type", "text")

        ids.append(str(chunk_id))
        embeddings.append([float(x) for x in embedding])
        documents.append(text)
        metadatas.append(metadata)
        langchain_docs.append(Document(page_content=text, metadata=metadata))

    if not ids:
        raise ValueError("Embedding pack contained no usable chunks.")

    print(f"Restoring {len(ids)} precomputed embeddings for {filename}...")
    vectorstore._collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return langchain_docs


def _embedding_to_list(embedding):
    if embedding is None:
        return []
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return [float(x) for x in embedding]


def export_embedding_pack(vectorstore, content_hash, filename):
    """Dump one PDF's chunks + vectors as a portable JSON pack for the client."""
    data = vectorstore._collection.get(
        where={"content_hash": content_hash},
        include=["embeddings", "documents", "metadatas"],
    )

    chunks = []
    ids = data.get("ids") or []
    embeddings = data.get("embeddings") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    for i, chunk_id in enumerate(ids):
        chunks.append(
            {
                "id": chunk_id,
                "text": documents[i] if i < len(documents) else "",
                "embedding": _embedding_to_list(embeddings[i] if i < len(embeddings) else None),
                "metadata": _clean_metadata(metadatas[i] if i < len(metadatas) else {}),
            }
        )

    return {
        "version": EMBEDDING_PACK_VERSION,
        "model": EMBEDDING_MODEL_NAME,
        "content_hash": content_hash,
        "filename": filename,
        "chunks": chunks,
    }
