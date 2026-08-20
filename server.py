import hashlib
import json
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cloak_config import describe
from history_aware_generation import ask_question, ask_question_events
from ingestion_pipeline import (
    add_chunks_to_vectorstore,
    collection_has_hash,
    embedding_pack_is_valid,
    import_embedding_pack,
    load_chunks_from_vectorstore,
    load_documents,
    split_documents,
)
from session_manager import session_manager


ALLOWED_SUFFIXES = {".pdf"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(describe())
    session_manager.start_sweeper()
    yield
    session_manager.shutdown()


app = FastAPI(title="Local RAG Backend API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


def _require_session_id(x_session_id: Optional[str]) -> str:
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="X-Session-Id header is required.")
    return x_session_id.strip()


async def _read_embedding_pack(embeddings: Optional[UploadFile]):
    if embeddings is None:
        return None
    raw = await embeddings.read()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid embeddings JSON: {exc}") from exc


async def _save_pdf(file: UploadFile, dest_dir: str) -> tuple:
    os.makedirs(dest_dir, exist_ok=True)
    incoming_path = os.path.join(dest_dir, f"incoming-{uuid.uuid4().hex}.pdf")
    hasher = hashlib.sha256()
    with open(incoming_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            buffer.write(chunk)
    content_hash = hasher.hexdigest()
    final_path = os.path.join(dest_dir, f"{content_hash}.pdf")
    os.replace(incoming_path, final_path)
    return final_path, content_hash


@app.post("/session/close")
async def close_session(x_session_id: Optional[str] = Header(default=None)):
    session_id = _require_session_id(x_session_id)
    closed = session_manager.close(session_id)
    return {"status": "closed" if closed else "already_closed", "session_id": session_id}


@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    embeddings: Optional[UploadFile] = File(default=None),
    x_session_id: Optional[str] = Header(default=None),
):
    session_id = _require_session_id(x_session_id)
    original_name = file.filename or "document.pdf"
    suffix = os.path.splitext(original_name)[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Currently only PDF files are supported. Other file types will be added later.",
        )

    pack = await _read_embedding_pack(embeddings)
    session = session_manager.get_or_create(session_id)
    file_path, content_hash = await _save_pdf(file, os.path.join(session.temp_dir, "docs"))

    with session_manager.session_scope(session_id) as session:
        already_loaded = any(
            (chunk.metadata or {}).get("content_hash") == content_hash
            for chunk in session.chunks
        )
        if already_loaded:
            print(f"Skipping {original_name}: already in session chunks.")
            return {
                "status": "skipped",
                "message": f"'{original_name}' is already loaded in this session.",
                "content_hash": content_hash,
            }

        if collection_has_hash(session.vectorstore, content_hash):
            loaded = load_chunks_from_vectorstore(session.vectorstore, content_hash)
            if loaded:
                session.chunks.extend(loaded)
                session.rebuild_retrievers()
                print(f"Hydrated {len(loaded)} chunks for {original_name} from session store.")
                return {
                    "status": "restored",
                    "message": f"'{original_name}' restored from session store.",
                    "chunks_created": len(loaded),
                    "content_hash": content_hash,
                }

        print(f"\n--- NEW FILE UPLOADED: {original_name} ({content_hash[:12]}...) ---")

        try:
            if embedding_pack_is_valid(pack, content_hash):
                chunks = import_embedding_pack(session.vectorstore, pack)
                session.chunks.extend(chunks)
                session.rebuild_retrievers()
                return {
                    "status": "restored",
                    "message": f"'{original_name}' restored from client embeddings.",
                    "chunks_created": len(chunks),
                    "content_hash": content_hash,
                }

            if pack is not None:
                print(
                    f"Embedding pack mismatch for {original_name}; recomputing "
                    f"(hash/model/chunks did not match)."
                )

            image_dir = os.path.join(session.temp_dir, "extracted_images", content_hash)
            documents = load_documents(
                file_path=file_path,
                output_image_dir=image_dir,
                source_filename=original_name,
                content_hash=content_hash,
            )
            chunks = split_documents(documents)
            _, embedding_pack = add_chunks_to_vectorstore(
                session.vectorstore, chunks, content_hash, filename=original_name
            )
            session.chunks.extend(chunks)
            session.rebuild_retrievers()
            return {
                "status": "success",
                "message": f"{original_name} ingested successfully.",
                "chunks_created": len(chunks),
                "content_hash": content_hash,
                "embedding_pack": embedding_pack,
            }
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/query")
async def query_rag(
    request: QueryRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    session_id = _require_session_id(x_session_id)
    print(f"\n--- NEW RAG QUERY RECEIVED ---")
    print(f"Query: {request.query}")

    with session_manager.session_scope(session_id) as session:
        return ask_question(request.query, session)


@app.post("/query/stream")
async def query_rag_stream(
    request: QueryRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    session_id = _require_session_id(x_session_id)
    print(f"\n--- NEW RAG STREAM QUERY RECEIVED ---")
    print(f"Query: {request.query}")

    def event_stream():
        with session_manager.session_scope(session_id) as session:
            for event in ask_question_events(request.query, session):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
