import hashlib
import json
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import accounts
from cloak_config import LLM_MODEL, describe, list_ollama_models
from history_aware_generation import ask_question, ask_question_events, load_thread_history
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
WEB_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(describe())
    session_manager.start_sweeper()
    yield
    session_manager.shutdown()


app = FastAPI(title="Cloak", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    content_hashes: Optional[list[str]] = None
    model: Optional[str] = None
    route_mode: Optional[str] = "auto"
    chat_id: Optional[str] = None


class UnloadDocumentRequest(BaseModel):
    content_hash: str


class AuthRequest(BaseModel):
    username: str
    password: str
    confirm_password: Optional[str] = None


class ChatPatch(BaseModel):
    title: Optional[str] = None
    content_hashes: Optional[list[str]] = None


def _require_session_id(x_session_id: Optional[str]) -> str:
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="X-Session-Id header is required.")
    return x_session_id.strip()


def _identity(request: Request, x_session_id: Optional[str]):
    user = accounts.resolve_user(request)
    if user:
        return user["id"], user
    if x_session_id and x_session_id.strip():
        return x_session_id.strip(), None
    raise HTTPException(status_code=401, detail="Sign in or send X-Session-Id.")


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


def _ingest_pdf(session, file_path: str, original_name: str, content_hash: str, pack=None):
    already_loaded = any(
        (chunk.metadata or {}).get("content_hash") == content_hash
        for chunk in session.chunks
    )
    if already_loaded:
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
            return {
                "status": "restored",
                "message": f"'{original_name}' restored from session store.",
                "chunks_created": len(loaded),
                "content_hash": content_hash,
            }

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


def _ensure_pdfs_loaded(session, user, hashes):
    if not user:
        return
    wanted = [item for item in (hashes or []) if item]
    present = {
        (chunk.metadata or {}).get("content_hash")
        for chunk in session.chunks
        if (chunk.metadata or {}).get("content_hash")
    }
    library = {item.get("content_hash"): item for item in accounts.list_pdfs(user["id"])}
    for content_hash in wanted:
        if content_hash in present:
            continue
        path = accounts.pdf_file(user["id"], content_hash)
        if not path.exists():
            continue
        name = (library.get(content_hash) or {}).get("filename") or f"{content_hash[:8]}.pdf"
        _ingest_pdf(session, str(path), name, content_hash)


def _maybe_set_cookie(response: Response, request: Request, user: Optional[dict]) -> None:
    if not user:
        return
    token = user.get("session_token") or ""
    if token and request.cookies.get(accounts.COOKIE_NAME) != token:
        accounts.set_session_cookie(response, token)


@app.post("/auth/register")
async def auth_register(body: AuthRequest, request: Request, response: Response):
    if body.confirm_password is None or body.confirm_password == "":
        raise HTTPException(status_code=400, detail="Confirm your password.")
    if body.password != body.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")
    user = accounts.register(body.username, body.password, accounts.client_ip(request))
    accounts.set_session_cookie(response, user["session_token"])
    return {"user": accounts.public_user(user)}


@app.post("/auth/login")
async def auth_login(body: AuthRequest, request: Request, response: Response):
    user = accounts.login(body.username, body.password, accounts.client_ip(request))
    accounts.set_session_cookie(response, user["session_token"])
    return {"user": accounts.public_user(user)}


@app.post("/auth/logout")
async def auth_logout(request: Request, response: Response):
    user = accounts.resolve_user(request)
    if user:
        accounts.logout(user, accounts.client_ip(request))
    accounts.clear_session_cookie(response)
    return {"status": "signed_out"}


@app.get("/auth/me")
async def auth_me(request: Request, response: Response):
    user = accounts.resolve_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in.")
    _maybe_set_cookie(response, request, user)
    return {"user": accounts.public_user(user)}


@app.get("/pdfs")
async def list_pdfs(request: Request, response: Response):
    user = accounts.require_user(request)
    _maybe_set_cookie(response, request, user)
    return {"documents": accounts.list_pdfs(user["id"])}


@app.delete("/pdfs/{content_hash}")
async def delete_pdf(content_hash: str, request: Request, x_session_id: Optional[str] = Header(default=None)):
    user = accounts.require_user(request)
    session_id, _ = _identity(request, x_session_id)
    with session_manager.session_scope(session_id) as session:
        session.remove_document(content_hash)
    removed = accounts.remove_pdf(user["id"], content_hash)
    if not removed:
        raise HTTPException(status_code=404, detail="PDF not found.")
    return {"status": "deleted", "content_hash": content_hash}


@app.get("/chats")
async def list_chats(request: Request, response: Response):
    user = accounts.require_user(request)
    _maybe_set_cookie(response, request, user)
    return {"chats": [accounts.chat_summary(chat) for chat in accounts.list_chats(user["id"])]}


@app.post("/chats")
async def create_chat(request: Request, response: Response):
    user = accounts.require_user(request)
    _maybe_set_cookie(response, request, user)
    chat = accounts.save_chat(user["id"], accounts.blank_chat())
    return {"chat": chat}


@app.get("/chats/{chat_id}")
async def read_chat(chat_id: str, request: Request, response: Response):
    user = accounts.require_user(request)
    _maybe_set_cookie(response, request, user)
    return {"chat": accounts.get_chat(user["id"], chat_id)}


@app.patch("/chats/{chat_id}")
async def patch_chat(chat_id: str, body: ChatPatch, request: Request):
    user = accounts.require_user(request)
    chat = accounts.get_chat(user["id"], chat_id)
    if body.title is not None:
        title = body.title.strip() or "New chat"
        chat["title"] = title[:80]
    if body.content_hashes is not None:
        chat["content_hashes"] = [item for item in body.content_hashes if item]
    return {"chat": accounts.save_chat(user["id"], chat)}


@app.delete("/chats/{chat_id}")
async def remove_chat(chat_id: str, request: Request):
    user = accounts.require_user(request)
    accounts.delete_chat(user["id"], chat_id)
    return {"status": "deleted", "id": chat_id}


@app.post("/session/close")
async def close_session(request: Request, x_session_id: Optional[str] = Header(default=None)):
    session_id, _user = _identity(request, x_session_id)
    closed = session_manager.close(session_id)
    return {"status": "closed" if closed else "already_closed", "session_id": session_id}


@app.post("/documents/unload")
async def unload_document(
    request_body: UnloadDocumentRequest,
    request: Request,
    x_session_id: Optional[str] = Header(default=None),
):
    session_id, _user = _identity(request, x_session_id)
    content_hash = (request_body.content_hash or "").strip()
    if not content_hash:
        raise HTTPException(status_code=400, detail="content_hash is required.")
    with session_manager.session_scope(session_id) as session:
        removed = session.remove_document(content_hash)
    return {
        "status": "unloaded",
        "content_hash": content_hash,
        "chunks_removed": removed,
    }


@app.post("/upload_pdf")
@app.post("/pdfs")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    embeddings: Optional[UploadFile] = File(default=None),
    x_session_id: Optional[str] = Header(default=None),
):
    session_id, user = _identity(request, x_session_id)
    original_name = file.filename or "document.pdf"
    suffix = os.path.splitext(original_name)[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Currently only PDF files are supported. Other file types will be added later.",
        )

    pack = await _read_embedding_pack(embeddings)
    session = session_manager.get_or_create(session_id)
    dest_dir = str(accounts.pdfs_dir(user["id"])) if user else os.path.join(session.temp_dir, "docs")
    file_path, content_hash = await _save_pdf(file, dest_dir)
    size = os.path.getsize(file_path)

    with session_manager.session_scope(session_id) as session:
        print(f"\n--- NEW FILE UPLOADED: {original_name} ({content_hash[:12]}...) ---")
        try:
            result = _ingest_pdf(session, file_path, original_name, content_hash, pack)
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if user:
        accounts.upsert_pdf(user["id"], content_hash, original_name, size)
        result.pop("embedding_pack", None)
    return result


@app.post("/query")
async def query_rag(
    body: QueryRequest,
    request: Request,
    x_session_id: Optional[str] = Header(default=None),
):
    session_id, user = _identity(request, x_session_id)
    print("\n--- NEW RAG QUERY RECEIVED ---")
    print(f"Query: {body.query}")

    with session_manager.session_scope(session_id) as session:
        session.cancel.clear()
        hashes = body.content_hashes
        if user and body.chat_id:
            chat = accounts.get_chat(user["id"], body.chat_id)
            if hashes is None:
                hashes = list(chat.get("content_hashes") or [])
            load_thread_history(session, chat.get("messages") or [])
        _ensure_pdfs_loaded(session, user, hashes)
        session.sync_included_hashes(hashes)
        session.llm_model = (body.model or "").strip() or LLM_MODEL
        force = (body.route_mode or "auto").strip().lower() or "auto"
        result = ask_question(body.query, session, force_mode=force)
    if user and body.chat_id:
        accounts.append_turn(
            user["id"],
            body.chat_id,
            body.query,
            result.get("answer") or "",
            extra={"citations": result.get("citations") or []},
        )
    return result


@app.post("/query/cancel")
async def cancel_query(request: Request, x_session_id: Optional[str] = Header(default=None)):
    session_id, _user = _identity(request, x_session_id)
    cancelled = session_manager.request_cancel(session_id)
    return {"status": "cancelling" if cancelled else "idle", "session_id": session_id}


class EvalCragRequest(BaseModel):
    path: Optional[str] = None
    limit: int = 40
    mode: str = "retrieval"
    split: int = 0
    html: bool = False
    seed: int = 0


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def models():
    ollama = list_ollama_models()
    if not ollama.get("ok"):
        raise HTTPException(
            status_code=503,
            detail=f"Ollama unreachable at {ollama.get('ollama_url')}: {ollama.get('error')}",
        )
    return {
        "models": ollama.get("models") or [],
        "default": ollama.get("default") or LLM_MODEL,
        "ollama_url": ollama.get("ollama_url"),
    }


@app.post("/eval/crag")
async def eval_crag_endpoint(body: EvalCragRequest):
    from eval_crag import DEFAULT_CRAG_PATH, run_eval

    mode = (body.mode or "retrieval").strip().lower()
    if mode not in {"corpus", "retrieval", "generate"}:
        raise HTTPException(status_code=400, detail="mode must be corpus, retrieval, or generate")
    try:
        return run_eval(
            path=body.path or DEFAULT_CRAG_PATH,
            limit=max(1, min(body.limit, 200)),
            mode=mode,
            split=body.split,
            seed=body.seed,
            html=body.html,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query/stream")
async def query_rag_stream(
    body: QueryRequest,
    request: Request,
    x_session_id: Optional[str] = Header(default=None),
):
    session_id, user = _identity(request, x_session_id)
    print("\n--- NEW RAG STREAM QUERY RECEIVED ---")
    print(f"Query: {body.query}")

    def event_stream():
        final = None
        hashes = body.content_hashes
        with session_manager.session_scope(session_id) as session:
            session.cancel.clear()
            if user and body.chat_id:
                chat = accounts.get_chat(user["id"], body.chat_id)
                if hashes is None:
                    hashes = list(chat.get("content_hashes") or [])
                load_thread_history(session, chat.get("messages") or [])
            _ensure_pdfs_loaded(session, user, hashes)
            session.sync_included_hashes(hashes)
            session.llm_model = (body.model or "").strip() or LLM_MODEL
            force = (body.route_mode or "auto").strip().lower() or "auto"
            for event in ask_question_events(body.query, session, force_mode=force):
                if event.get("type") == "complete":
                    final = event
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        if user and body.chat_id and final:
            accounts.append_turn(
                user["id"],
                body.chat_id,
                body.query,
                final.get("answer") or "",
                extra={"citations": final.get("citations") or []},
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
async def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web UI is missing.")
    return FileResponse(index_path)


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
