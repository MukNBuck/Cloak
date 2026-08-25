"""Account store: users, cookies, IP remember, server-side PDFs and chats."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, Response

DATA_DIR = Path(os.getenv("CLOAK_DATA_DIR", str(Path.home() / ".cloak" / "server")))
USERS_PATH = DATA_DIR / "users.json"
COOKIE_NAME = "cloak_token"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30
PBKDF2_ROUNDS = 180_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{2,40}$")

_lock = threading.Lock()


def _now() -> float:
    return time.time()


def _users_dir(user_id: str) -> Path:
    path = DATA_DIR / "accounts" / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def pdfs_dir(user_id: str) -> Path:
    path = _users_dir(user_id) / "pdfs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chats_dir(user_id: str) -> Path:
    path = _users_dir(user_id) / "chats"
    path.mkdir(parents=True, exist_ok=True)
    return path


def library_path(user_id: str) -> Path:
    return _users_dir(user_id) / "library.json"


def _read_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return fallback


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_users() -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = _read_json(USERS_PATH, {"users": []})
    return list(payload.get("users") or [])


def _save_users(users: list[dict]) -> None:
    _write_json(USERS_PATH, {"users": users})


def _hash_password(password: str, salt: str) -> str:
    raw = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PBKDF2_ROUNDS,
    )
    return raw.hex()


def _public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "username": user["username"],
        "created_at": user.get("created_at"),
    }


def client_ip(request: Request) -> str:
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if forwarded:
        return forwarded
    if request.client and request.client.host:
        return request.client.host
    return ""


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


def register(username: str, password: str, ip: str) -> dict:
    name = (username or "").strip()
    if not USERNAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Username must be 2–40 letters, numbers, dots, underscores, or hyphens.",
        )
    if not password or len(password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters.")
    with _lock:
        users = _load_users()
        if any(item["username"].lower() == name.lower() for item in users):
            raise HTTPException(status_code=409, detail="That username is already taken.")
        salt = secrets.token_hex(16)
        token = secrets.token_urlsafe(32)
        user = {
            "id": str(uuid.uuid4()),
            "username": name,
            "password_hash": _hash_password(password, salt),
            "salt": salt,
            "created_at": _now(),
            "session_token": token,
            "remember_ip": True,
            "bound_ip": ip or "",
        }
        users.append(user)
        _save_users(users)
        _users_dir(user["id"])
        _write_json(library_path(user["id"]), {"documents": []})
        return user


def login(username: str, password: str, ip: str) -> dict:
    name = (username or "").strip()
    with _lock:
        users = _load_users()
        user = next((item for item in users if item["username"].lower() == name.lower()), None)
        if user is None:
            raise HTTPException(status_code=401, detail="Wrong username or password.")
        expected = user.get("password_hash") or ""
        actual = _hash_password(password, user.get("salt") or "")
        if not hmac.compare_digest(expected, actual):
            raise HTTPException(status_code=401, detail="Wrong username or password.")
        token = secrets.token_urlsafe(32)
        for item in users:
            if item["id"] != user["id"] and item.get("bound_ip") == ip:
                item["bound_ip"] = ""
                item["remember_ip"] = False
        user["session_token"] = token
        user["remember_ip"] = True
        user["bound_ip"] = ip or ""
        _save_users(users)
        return user


def logout(user: dict, ip: str) -> None:
    with _lock:
        users = _load_users()
        for item in users:
            if item["id"] == user["id"]:
                item["session_token"] = ""
                item["remember_ip"] = False
                if item.get("bound_ip") == ip:
                    item["bound_ip"] = ""
                break
        _save_users(users)


def user_by_token(token: str) -> Optional[dict]:
    if not token:
        return None
    with _lock:
        users = _load_users()
    return next((item for item in users if item.get("session_token") == token), None)


def user_by_ip(ip: str) -> Optional[dict]:
    if not ip:
        return None
    with _lock:
        users = _load_users()
    return next(
        (
            item
            for item in users
            if item.get("remember_ip") and item.get("bound_ip") == ip and item.get("session_token")
        ),
        None,
    )


def resolve_user(request: Request) -> Optional[dict]:
    token = request.cookies.get(COOKIE_NAME) or ""
    user = user_by_token(token)
    if user:
        return user
    return user_by_ip(client_ip(request))


def require_user(request: Request) -> dict:
    user = resolve_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user


def list_pdfs(user_id: str) -> list[dict]:
    payload = _read_json(library_path(user_id), {"documents": []})
    return list(payload.get("documents") or [])


def upsert_pdf(user_id: str, content_hash: str, filename: str, size: int) -> dict:
    docs = list_pdfs(user_id)
    entry = next((item for item in docs if item.get("content_hash") == content_hash), None)
    if entry is None:
        entry = {
            "content_hash": content_hash,
            "filename": filename,
            "size": size,
            "added_at": _now(),
        }
        docs.append(entry)
    else:
        entry["filename"] = filename or entry.get("filename")
        entry["size"] = size or entry.get("size") or 0
    _write_json(library_path(user_id), {"documents": docs})
    return entry


def remove_pdf(user_id: str, content_hash: str) -> bool:
    docs = list_pdfs(user_id)
    next_docs = [item for item in docs if item.get("content_hash") != content_hash]
    if len(next_docs) == len(docs):
        return False
    _write_json(library_path(user_id), {"documents": next_docs})
    path = pdfs_dir(user_id) / f"{content_hash}.pdf"
    if path.exists():
        path.unlink()
    for chat in list_chats(user_id):
        hashes = [item for item in (chat.get("content_hashes") or []) if item != content_hash]
        if hashes != (chat.get("content_hashes") or []):
            chat["content_hashes"] = hashes
            save_chat(user_id, chat)
    return True


def pdf_file(user_id: str, content_hash: str) -> Path:
    return pdfs_dir(user_id) / f"{content_hash}.pdf"


def blank_chat() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": "New chat",
        "created_at": _now(),
        "updated_at": _now(),
        "content_hashes": [],
        "messages": [],
    }


def list_chats(user_id: str) -> list[dict]:
    chats = []
    folder = chats_dir(user_id)
    for path in folder.glob("*.json"):
        chat = _read_json(path, None)
        if isinstance(chat, dict) and chat.get("id"):
            chats.append(chat)
    chats.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
    return chats


def get_chat(user_id: str, chat_id: str) -> dict:
    path = chats_dir(user_id) / f"{chat_id}.json"
    chat = _read_json(path, None)
    if not isinstance(chat, dict):
        raise HTTPException(status_code=404, detail="Chat not found.")
    return chat


def save_chat(user_id: str, chat: dict) -> dict:
    chat["updated_at"] = _now()
    _write_json(chats_dir(user_id) / f"{chat['id']}.json", chat)
    return chat


def delete_chat(user_id: str, chat_id: str) -> None:
    path = chats_dir(user_id) / f"{chat_id}.json"
    if path.exists():
        path.unlink()


def append_turn(user_id: str, chat_id: str, query: str, answer: str, extra=None) -> dict:
    chat = get_chat(user_id, chat_id)
    chat.setdefault("messages", [])
    chat["messages"].append({"role": "user", "content": query, "ts": _now()})
    assistant = {"role": "assistant", "content": answer, "ts": _now()}
    if extra:
        assistant.update(extra)
    chat["messages"].append(assistant)
    if (chat.get("title") or "New chat") == "New chat" and query.strip():
        chat["title"] = re.sub(r"\s+", " ", query.strip())[:48]
    return save_chat(user_id, chat)


def chat_summary(chat: dict) -> dict:
    return {
        "id": chat.get("id"),
        "title": chat.get("title") or "New chat",
        "updated_at": chat.get("updated_at"),
        "created_at": chat.get("created_at"),
        "content_hashes": list(chat.get("content_hashes") or []),
        "message_count": len(chat.get("messages") or []),
    }


def public_user(user: dict) -> dict:
    return _public_user(user)
