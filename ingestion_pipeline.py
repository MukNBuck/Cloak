import os
import math
import json
import base64
import re
import sys
from collections import defaultdict
from functools import lru_cache
from html.parser import HTMLParser
from types import SimpleNamespace

from cloak_config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    ENABLE_VISION,
    OFFLINE,
    PARENT_TEXT_CHARS,
    PDF_PARSE_STRATEGY,
    VISION_MODEL,
    torch_device,
)

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

EMBEDDING_PACK_VERSION = 1
EXPORTABLE_METADATA_KEYS = (
    "source",
    "type",
    "content_hash",
    "page",
    "section",
    "heading",
    "parent_id",
    "parent_text",
    "original_content",
)
APPENDIX_HEADING_RE = re.compile(
    r"(?im)^\s*(appendix|annex)(\s+[a-z0-9]+)?\b"
)
MIN_VISION_IMAGE_BYTES = 12_000
MIN_VISION_IMAGE_PX = 80


def get_image_base64(image_path):
    """Convert an image to base64 so Ollama can read it."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _image_worth_describing(image_path):
    try:
        if os.path.getsize(image_path) < MIN_VISION_IMAGE_BYTES:
            return False
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
        return width >= MIN_VISION_IMAGE_PX and height >= MIN_VISION_IMAGE_PX
    except Exception:
        return True


def summarize_image_with_ollama(image_path):
    """Describe a figure with the local vision model. Never leaves the machine."""
    print(f"Describing {os.path.basename(image_path)} with local {VISION_MODEL}...")

    vision_llm = ChatOllama(model=VISION_MODEL, temperature=0.0)
    image_b64 = get_image_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe this image for later search. Include visible text, "
                        "numbers, labels, and what the chart or diagram shows."
                    ),
                },
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            ],
        }
    ]

    response = vision_llm.invoke(messages)
    return (response.content or "").strip()


class TransformersEmbeddings(Embeddings):
    """Dense embeddings via transformers only (no sentence-transformers/scipy)."""

    def __init__(self, model_name, device="cpu", max_length=1024, batch_size=8):
        import torch
        from transformers import AutoModel, AutoTokenizer

        kwargs = {}
        if OFFLINE:
            kwargs["local_files_only"] = True
        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.to(device)
        self.model.eval()

    def _encode(self, texts):
        vectors = []
        torch = self.torch
        for start in range(0, len(texts), self.batch_size):
            batch = [text or "" for text in texts[start : start + self.batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).type_as(hidden)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
        return vectors

    def embed_documents(self, texts):
        return self._encode(list(texts))

    def embed_query(self, text):
        return self._encode([text or ""])[0]


@lru_cache(maxsize=1)
def get_embedding_model():
    print(f"Loading embeddings {EMBEDDING_MODEL_NAME} on {torch_device()}...")
    return TransformersEmbeddings(EMBEDDING_MODEL_NAME, device=torch_device())


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


def _pdf_strategies():
    requested = PDF_PARSE_STRATEGY
    if requested in ("hi_res", "ocr_only", "fast"):
        fallbacks = [s for s in ("hi_res", "ocr_only", "fast") if s != requested]
        return [requested] + fallbacks
    return ["hi_res", "ocr_only", "fast"]


class NarrativeText:
    """Minimal page element so pdfplumber output fits the unstructured ingest path."""

    def __init__(self, text, page):
        self.text = text
        self.metadata = SimpleNamespace(page_number=page)


def _partition_pdfplumber(file_path):
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "PDF ingest needs pdfplumber in the same Python that runs the server: "
            f"{sys.executable}. Install with: \"{sys.executable}\" -m pip install pdfplumber"
        ) from exc
    elements = []
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                elements.append(NarrativeText(text, page_number))
    print(
        f"pdfplumber extracted text from {len(elements)} pages in "
        f"{os.path.basename(file_path)}."
    )
    return elements


def _partition_pdf(file_path, output_image_dir):
    """Layout-aware parse with OCR fallback, then pdfplumber if unstructured is missing."""
    partition_pdf = None
    try:
        from unstructured.partition.pdf import partition_pdf as _partition
        partition_pdf = _partition
    except Exception as exc:
        print(f"unstructured PDF import failed ({type(exc).__name__}: {exc}); using pdfplumber.")
        return _partition_pdfplumber(file_path)

    last_error = None
    for strategy in _pdf_strategies():
        kwargs = dict(
            filename=file_path,
            strategy=strategy,
            extract_images_in_pdf=ENABLE_VISION,
            infer_table_structure=True,
        )
        print(f"Partitioning PDF with strategy={strategy}...")
        try:
            if ENABLE_VISION:
                try:
                    elements = partition_pdf(
                        **kwargs, extract_image_block_output_dir=output_image_dir
                    )
                except TypeError:
                    elements = partition_pdf(**kwargs)
            else:
                elements = partition_pdf(**kwargs)
            print(f"PDF strategy={strategy} produced {len(elements or [])} elements.")
            if elements:
                return elements
        except Exception as exc:
            last_error = exc
            print(f"PDF strategy={strategy} failed ({exc}); trying next.")
    if last_error:
        print(f"All unstructured PDF strategies failed; last error: {last_error}")
    print("Falling back to pdfplumber for page text.")
    return _partition_pdfplumber(file_path)


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
    headings = {}
    if orig:
        for sub in orig:
            text = (getattr(sub, "text", None) or "").strip()
            if not text:
                continue
            page = _page_from_element(sub)
            grouped[page].append(text)
            if "Title" in type(sub).__name__ and page not in headings:
                headings[page] = text[:200]
    else:
        text = (getattr(element, "text", None) or "").strip()
        if text:
            page = _page_from_element(element)
            grouped[page].append(text)
            if "Title" in type(element).__name__:
                headings[page] = text[:200]

    documents = []
    for page, parts in grouped.items():
        text = "\n".join(parts).strip()
        if not text:
            continue
        meta = dict(base_metadata)
        if page is not None:
            meta["page"] = page
        heading = headings.get(page)
        if heading:
            meta["heading"] = heading
        section = _section_from_text(text)
        if section:
            meta["section"] = section
        documents.append(Document(page_content=text, metadata=meta))
    return documents


def _documents_from_pdf_elements(elements, base_metadata):
    """Group raw unstructured elements into page-level parent documents."""
    skip = ("Table", "Image", "Figure", "Header", "Footer", "PageBreak")
    grouped = defaultdict(list)
    headings = {}
    documents = []
    used_composite = False
    for element in elements:
        name = type(element).__name__
        if any(token in name for token in skip):
            continue
        if "CompositeElement" in name:
            used_composite = True
            documents.extend(_documents_from_element(element, base_metadata))
            continue
        text = (getattr(element, "text", None) or "").strip()
        if not text:
            continue
        page = _page_from_element(element)
        grouped[page].append(text)
        if "Title" in name and page not in headings:
            headings[page] = text[:200]

    if used_composite and documents:
        return documents

    for page, parts in grouped.items():
        text = "\n".join(parts).strip()
        if not text:
            continue
        meta = dict(base_metadata)
        if page is not None:
            meta["page"] = page
        heading = headings.get(page)
        if heading:
            meta["heading"] = heading
        section = _section_from_text(text)
        if section:
            meta["section"] = section
        documents.append(Document(page_content=text, metadata=meta))
    return documents


def _clip_meta(text, limit):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def stamp_parent_fields(doc, parent_id):
    """Store page-level text for small-to-big retrieval and dual-channel tables."""
    raw = doc.page_content or ""
    payload = {"raw_text": raw, "tables_html": []}
    if (doc.metadata or {}).get("type") == "table":
        payload["tables_html"] = [raw]
    meta = dict(doc.metadata or {})
    meta["parent_id"] = parent_id
    meta["parent_text"] = _clip_meta(raw, PARENT_TEXT_CHARS)
    meta["original_content"] = json.dumps(payload, ensure_ascii=False)
    doc.metadata = meta
    return doc


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


def load_pdf_documents(file_path, output_image_dir, source_filename=None, content_hash=None):
    """Extract text and tables from a PDF. Optional local vision for figures."""
    filename = source_filename or os.path.basename(file_path)
    print(f"Loading and partitioning {filename}...")
    os.makedirs(output_image_dir, exist_ok=True)

    print(f"\nProcessing {filename}...")
    raw_pdf_elements = _partition_pdf(file_path, output_image_dir)

    all_documents = []
    base_metadata = {"source": filename, "type": "text"}
    if content_hash:
        base_metadata["content_hash"] = content_hash

    all_documents.extend(_documents_from_pdf_elements(raw_pdf_elements, base_metadata))

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

    if ENABLE_VISION:
        described = 0
        skipped = 0
        failed = 0
        for image_name in os.listdir(output_image_dir):
            if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(output_image_dir, image_name)
            if not _image_worth_describing(img_path):
                skipped += 1
                continue
            try:
                image_summary = summarize_image_with_ollama(img_path)
            except Exception as exc:
                failed += 1
                print(f"Vision failed for {image_name} ({exc}); skipping figure.")
                continue
            if not image_summary:
                failed += 1
                continue
            image_metadata = {
                "source": filename,
                "type": "image",
            }
            if content_hash:
                image_metadata["content_hash"] = content_hash
            all_documents.append(
                Document(
                    page_content=f"Image Description: {image_summary}",
                    metadata=image_metadata,
                )
            )
            described += 1
        print(
            f"Vision described {described} figures; skipped {skipped} tiny images; "
            f"{failed} failed."
        )
    else:
        print("Vision ingest off (CLOAK_VISION=0); text and tables only.")

    hashed = content_hash or filename
    for index, doc in enumerate(all_documents):
        page = (doc.metadata or {}).get("page")
        kind = (doc.metadata or {}).get("type") or "text"
        parent_id = f"{hashed}-{kind}-{page if page is not None else 'x'}-{index}"
        stamp_parent_fields(doc, parent_id)

    print(f"Parent documents for {filename}: {len(all_documents)}")
    return all_documents


def load_documents(file_path, output_image_dir, source_filename=None, content_hash=None):
    """Load any supported file. More types can be added here without changing RAG."""
    suffix = os.path.splitext(file_path)[1].lower()
    if suffix == ".pdf":
        return load_pdf_documents(
            file_path,
            output_image_dir,
            source_filename=source_filename,
            content_hash=content_hash,
        )
    raise ValueError(f"Unsupported file type: {suffix or '(none)'}")


# Keep the old name so existing imports keep working.
load_documents_with_vision = load_pdf_documents


def _split_recursive(text, chunk_size, chunk_overlap, separators):
    """Character splitter; avoids langchain_text_splitters (nltk/scipy on import)."""
    text = text or ""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    seps = list(separators) if separators else ["\n\n", "\n", " ", ""]
    if seps[-1] != "":
        seps = seps + [""]

    def split_once(body, remaining):
        if len(body) <= chunk_size:
            return [body] if body else []
        sep, *more = remaining
        if sep == "":
            step = max(1, chunk_size - chunk_overlap)
            return [body[i : i + chunk_size] for i in range(0, len(body), step)]
        parts = body.split(sep)
        merged = []
        current = ""
        for part in parts:
            candidate = part if not current else current + sep + part
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                merged.extend(split_once(current, more))
            current = part
        if current:
            merged.extend(split_once(current, more))
        return merged

    pieces = split_once(text, seps)
    return [p for p in pieces if p]


def split_documents(documents, chunk_size=None, chunk_overlap=None, separators=["\n\n", "\n", " ", ""]):
    """Embed small children; keep parent page text on metadata for answering."""
    chunk_size = chunk_size or CHILD_CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHILD_CHUNK_OVERLAP
    print(
        f"Splitting into child chunks of size {chunk_size} "
        f"with overlap {chunk_overlap} (small-to-big)..."
    )

    preserved = [
        doc for doc in documents
        if doc.metadata.get("type") in ("table", "image")
    ]
    splittable = [
        doc for doc in documents
        if doc.metadata.get("type") not in ("table", "image")
    ]
    children = []
    for parent in splittable:
        heading = (parent.metadata or {}).get("heading")
        pieces = [
            Document(page_content=text, metadata=dict(parent.metadata or {}))
            for text in _split_recursive(
                parent.page_content or "", chunk_size, chunk_overlap, separators
            )
        ]
        if not pieces:
            children.append(parent)
            continue
        for piece in pieces:
            meta = dict(parent.metadata or {})
            meta.update(piece.metadata or {})
            body = piece.page_content or ""
            if heading and heading.lower() not in body[:200].lower():
                body = f"{heading}\n\n{body}"
            piece.page_content = body
            piece.metadata = meta
            children.append(piece)

    chunks = children + preserved

    if chunks:
        for i, chunk in enumerate(chunks[:2]):
            print(f"\nChunk {i+1}:")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"Parent: {chunk.metadata.get('parent_id', '')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:\n{chunk.page_content[:200]}...\n" + "-" * 50)

    print(f"Child chunks: {len(chunks)} from {len(documents)} parents.")
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


def delete_chunks_by_hash(vectorstore, content_hash):
    """Remove all Chroma rows for a document so vector search cannot resurrect them."""
    if not content_hash or vectorstore is None:
        return 0
    try:
        existing = vectorstore.get(where={"content_hash": content_hash})
    except Exception as exc:
        print(f"Failed looking up chunks for {content_hash[:12]}...: {exc}")
        return 0
    ids = list(_chroma_field(existing, "ids", []) or [])
    if not ids:
        return 0
    try:
        vectorstore.delete(ids=ids)
    except Exception:
        try:
            vectorstore._collection.delete(ids=ids)
        except Exception as exc:
            print(f"Failed deleting chunks for {content_hash[:12]}...: {exc}")
            return 0
    return len(ids)


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
