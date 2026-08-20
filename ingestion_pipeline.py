import os
import math
import base64
import re
from collections import defaultdict
from functools import lru_cache
from html.parser import HTMLParser

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
EXPORTABLE_METADATA_KEYS = ("source", "type", "content_hash", "page", "section")
APPENDIX_HEADING_RE = re.compile(
    r"(?im)^\s*(appendix|annex)(\s+[a-z0-9]+)?\b"
)


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
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
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


class _HTMLTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self._table = None
        self._row = None
        self._cell = None
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._in_cell = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._in_cell = False
            self._cell = None
        elif tag == "tr" and self._table is not None and self._row is not None:
            self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data):
        if self._in_cell and self._cell is not None:
            self._cell.append(data)


def _clean_cell(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_table_rows(raw_rows):
    if not raw_rows:
        return []
    width = max(len(row) for row in raw_rows)
    rows = []
    for raw in raw_rows:
        row = [_clean_cell(cell) for cell in raw]
        if len(row) < width:
            row.extend([""] * (width - len(row)))
        if any(row):
            rows.append(row)
    return rows


def _parse_html_tables(html):
    if not html or "<table" not in html.lower():
        return []
    parser = _HTMLTableParser()
    try:
        parser.feed(html)
    except Exception:
        return []
    return [_normalize_table_rows(table) for table in parser.tables if table]


def _table_fingerprint(rows):
    cells = []
    for row in rows[:6]:
        for cell in row:
            if cell:
                cells.append(cell.lower())
    return "|".join(cells)[:800]


def _extract_tables_pdfplumber(file_path):
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber not installed; using unstructured table elements only.")
        return []

    tables = []
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            extracted = page.extract_tables() or []
            for raw_table in extracted:
                rows = _normalize_table_rows(raw_table)
                if len(rows) >= 2:
                    tables.append({"page": page_number, "rows": rows})
    print(f"pdfplumber found {len(tables)} tables in {os.path.basename(file_path)}.")
    return tables


def _extract_tables_unstructured(elements):
    tables = []
    for element in elements:
        type_name = type(element).__name__
        if "Table" not in type_name:
            continue
        html = getattr(getattr(element, "metadata", None), "text_as_html", None)
        parsed = _parse_html_tables(html) if html else []
        rows = parsed[0] if parsed else _normalize_table_rows(
            [line.split("\t") for line in (element.text or "").splitlines() if line.strip()]
        )
        if len(rows) >= 2:
            tables.append({"page": None, "rows": rows})
    return tables


def _headers_from_row(row):
    headers = []
    for index, cell in enumerate(row, 1):
        headers.append(cell if cell else f"Column {index}")
    return headers


def _page_from_element(element):
    page = getattr(getattr(element, "metadata", None), "page_number", None)
    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None


def _section_from_text(text):
    head = (text or "")[:400]
    if APPENDIX_HEADING_RE.search(head):
        return "appendix"
    return None


def _documents_from_element(element, base_metadata):
    metadata_obj = getattr(element, "metadata", None)
    orig = getattr(metadata_obj, "orig_elements", None) if metadata_obj else None
    grouped = defaultdict(list)
    if orig:
        for sub in orig:
            text = (getattr(sub, "text", None) or "").strip()
            if text:
                grouped[_page_from_element(sub)].append(text)
    else:
        text = (getattr(element, "text", None) or "").strip()
        if text:
            grouped[_page_from_element(element)].append(text)

    documents = []
    for page, parts in grouped.items():
        text = "\n".join(parts).strip()
        if not text:
            continue
        meta = dict(base_metadata)
        if page is not None:
            meta["page"] = page
        section = _section_from_text(text)
        if section:
            meta["section"] = section
        documents.append(Document(page_content=text, metadata=meta))
    return documents


def _appendix_start_page(documents):
    pages = [
        int(doc.metadata["page"])
        for doc in documents
        if doc.metadata.get("section") == "appendix" and doc.metadata.get("page") is not None
    ]
    return min(pages) if pages else None


def verbalize_table(rows, source, table_index, page=None):
    headers = _headers_from_row(rows[0])
    location = f" from {source}"
    if page:
        location += f" page {page}"
    lines = [f"Table {table_index}{location}."]
    body = rows[1:] if len(rows) > 1 else []
    for row in body:
        pairs = []
        for header, value in zip(headers, row):
            if value:
                pairs.append(f"{header} is {value}")
        if pairs:
            lines.append("Row: " + "; ".join(pairs) + ".")
    return "\n".join(lines)


def _table_documents(tables, source_filename, content_hash, max_rows=12, appendix_start=None):
    documents = []
    seen = set()
    table_index = 0
    for table in tables:
        rows = table["rows"]
        fingerprint = _table_fingerprint(rows)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        table_index += 1
        headers = [rows[0]]
        body = rows[1:]
        if not body:
            continue
        page = table.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        for start in range(0, len(body), max_rows):
            chunk_rows = headers + body[start:start + max_rows]
            metadata = {
                "source": source_filename,
                "type": "table",
            }
            if content_hash:
                metadata["content_hash"] = content_hash
            if page is not None:
                metadata["page"] = page
            if appendix_start and page is not None and page >= appendix_start:
                metadata["section"] = "appendix"
            documents.append(
                Document(
                    page_content=verbalize_table(
                        chunk_rows,
                        source_filename,
                        table_index,
                        page=page,
                    ),
                    metadata=metadata,
                )
            )
    print(f"Created {len(documents)} verbalized table chunks for {source_filename}.")
    return documents


def load_documents_with_vision(file_path, output_image_dir, source_filename=None, content_hash=None):
    """Extract text, tables, and images; verbalize tables for a small language model."""
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
            all_documents.extend(_documents_from_element(element, base_metadata))

    appendix_start = _appendix_start_page(all_documents)
    plumber_tables = _extract_tables_pdfplumber(file_path)
    unstructured_tables = _extract_tables_unstructured(raw_pdf_elements)
    all_documents.extend(
        _table_documents(
            plumber_tables + unstructured_tables,
            filename,
            content_hash,
            appendix_start=appendix_start,
        )
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

    preserved = [doc for doc in documents if doc.metadata.get("type") == "table"]
    splittable = [doc for doc in documents if doc.metadata.get("type") != "table"]
    chunks = text_splitter.split_documents(splittable) + preserved

    if chunks:
        for i, chunk in enumerate(chunks[:2]):
            print(f"\nChunk {i+1}:")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:\n{chunk.page_content[:200]}...\n" + "-" * 50)

    return chunks


def _as_json_float(value):
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return number


def _chroma_field(data, key, default=None):
    if isinstance(data, dict):
        value = data.get(key, default)
    else:
        value = getattr(data, key, default)
    return default if value is None else value


def add_chunks_to_vectorstore(vectorstore, chunks, content_hash, filename="document.pdf"):
    """Embed chunks into an existing ephemeral Chroma collection and return a client pack."""
    if not chunks:
        return [], {
            "version": EMBEDDING_PACK_VERSION,
            "model": EMBEDDING_MODEL_NAME,
            "content_hash": content_hash,
            "filename": filename,
            "chunks": [],
        }

    ids = [f"{content_hash}-{i}" for i in range(len(chunks))]
    texts = []
    metadatas = []
    for chunk in chunks:
        chunk.metadata = _clean_metadata(chunk.metadata)
        chunk.metadata["content_hash"] = content_hash
        texts.append(chunk.page_content)
        metadatas.append(chunk.metadata)

    print("Creating embeddings and storing in session ChromaDB...")
    raw_embeddings = get_embedding_model().embed_documents(texts)
    embeddings = [[_as_json_float(value) for value in row] for row in raw_embeddings]
    vectorstore._collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Finished embedding {len(chunks)} chunks for {content_hash[:12]}...")
    pack = {
        "version": EMBEDDING_PACK_VERSION,
        "model": EMBEDDING_MODEL_NAME,
        "content_hash": content_hash,
        "filename": filename,
        "chunks": [
            {
                "id": ids[i],
                "text": texts[i],
                "embedding": embeddings[i],
                "metadata": metadatas[i],
            }
            for i in range(len(ids))
        ],
    }
    return ids, pack


def collection_has_hash(vectorstore, content_hash):
    existing = vectorstore.get(where={"content_hash": content_hash})
    ids = _chroma_field(existing, "ids", []) or []
    return len(ids) > 0


def load_chunks_from_vectorstore(vectorstore, content_hash=None):
    kwargs = {"include": ["documents", "metadatas"]}
    if content_hash:
        kwargs["where"] = {"content_hash": content_hash}
    data = vectorstore.get(**kwargs)
    documents = list(_chroma_field(data, "documents", []) or [])
    metadatas = list(_chroma_field(data, "metadatas", []) or [])
    chunks = []
    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        chunks.append(Document(page_content=text or "", metadata=_clean_metadata(metadata)))
    return chunks


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
    return [_as_json_float(x) for x in embedding]


def export_embedding_pack(vectorstore, content_hash, filename):
    """Dump one PDF's chunks + vectors as a portable JSON pack for the client."""
    data = vectorstore._collection.get(
        where={"content_hash": content_hash},
        include=["embeddings", "documents", "metadatas"],
    )

    chunks = []
    ids = list(_chroma_field(data, "ids", []) or [])
    embeddings = list(_chroma_field(data, "embeddings", []) or [])
    documents = list(_chroma_field(data, "documents", []) or [])
    metadatas = list(_chroma_field(data, "metadatas", []) or [])

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
