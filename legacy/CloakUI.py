import atexit
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import customtkinter as ctk
import requests
from tkinter import filedialog
from PIL import Image

SERVER_URL = "http://192.168.1.146:8001"
CLOAK_DIR = Path.home() / ".cloak"
CACHE_DIR = CLOAK_DIR / "embeddings"
DOCS_DIR = CLOAK_DIR / "docs"
CHATS_DIR = CLOAK_DIR / "chats"
LIBRARY_PATH = CLOAK_DIR / "library.json"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LOGO_PATH = ASSETS_DIR / "cloaklogo.png"
ICON_PATH = ASSETS_DIR / "cloaklogo.ico"
STATUS_BY_UPLOAD = {
    "success": "Ready",
    "restored": "Ready",
    "skipped": "Ready",
}

BG = "#12151a"
SIDEBAR_BG = "#161a20"
DOCS_BG = "#181c22"
CARD_BG = "#222831"
USER_BG = "#2a3038"
BUBBLE_BG = "#252b33"
ACCENT = "#4db8ff"
ACCENT_HOVER = "#6bc5ff"
ACCENT_DIM = "#1a3348"
SILVER = "#c8cdd5"
MUTED = "#8b939e"
TEXT = "#eef1f4"
LINE = "#2f3640"
DANGER = "#e57373"
PANEL_LINE = "#242a32"
STAGE_LABELS = {
    "routing": "Routing",
    "understanding": "Understanding",
    "fetching": "Fetching chunks",
    "reranking": "Reranking sources",
    "reasoning": "Reasoning",
    "retry": "Retrying",
    "summarize": "Summarizing",
}
LOCAL_MODELS = ()  # filled from the server at runtime
CONNECTION_POLL_MS = 15000
MODELS_REFRESH_MS = 60000
SETTINGS_PATH = CLOAK_DIR / "settings.json"


def load_logo(size):
    if not LOGO_PATH.exists():
        return None
    image = Image.open(LOGO_PATH).convert("RGBA")
    return ctk.CTkImage(light_image=image, dark_image=image, size=size)


def _circle_icon(size, fill, glyph="", glyph_color="#ffffff", glyph_size=None):
    """Soft circular glyph used for avatars / badges (no emoji)."""
    from PIL import ImageDraw, ImageFont

    image = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((1, 1, size * 2 - 2, size * 2 - 2), fill=fill)
    if glyph:
        try:
            font = ImageFont.truetype("segoeui.ttf", (glyph_size or size) * 2)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), glyph, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((size * 2 - tw) / 2 - bbox[0], (size * 2 - th) / 2 - bbox[1] - 1),
            glyph,
            fill=glyph_color,
            font=font,
        )
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return ctk.CTkImage(light_image=image, dark_image=image, size=(size, size))


def _pdf_badge(size=28):
    from PIL import ImageDraw, ImageFont

    image = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, size * 2 - 2, size * 2 - 2), radius=8, fill="#c62828")
    try:
        font = ImageFont.truetype("segoeuib.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    text = "PDF"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size * 2 - tw) / 2 - bbox[0], (size * 2 - th) / 2 - bbox[1]),
        text,
        fill="#ffffff",
        font=font,
    )
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return ctk.CTkImage(light_image=image, dark_image=image, size=(size, size))


def ensure_window_icon():
    """Build a multi-size .ico so Windows can show Cloak in the title bar and taskbar."""
    if not LOGO_PATH.exists():
        return None
    try:
        source = Image.open(LOGO_PATH).convert("RGBA")
        stale = (not ICON_PATH.exists()) or ICON_PATH.stat().st_mtime < LOGO_PATH.stat().st_mtime
        if stale:
            source.save(
                ICON_PATH,
                format="ICO",
                sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
            )
        return ICON_PATH
    except Exception as exc:
        print(f"Could not build window icon: {exc}")
        return None


def _mix_hex(start, end, amount):
    amount = max(0.0, min(1.0, amount))

    def channels(value):
        value = value.lstrip("#")
        return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))

    first = channels(start)
    second = channels(end)
    mixed = tuple(int(first[i] + (second[i] - first[i]) * amount) for i in range(3))
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


CHAT_TEXT_WIDTH = 720
BUBBLE_MAX_WIDTH = 520


def _textbox_inner(box):
    return getattr(box, "_textbox", box)


def _wrap_paragraph(paragraph, font, max_width):
    if paragraph == "":
        return [""]
    words = paragraph.split(" ")
    lines = []
    current = ""
    for word in words:
        trial = word if current == "" else f"{current} {word}"
        if font.measure(trial) <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
        if font.measure(word) <= max_width:
            current = word
            continue
        chunk = ""
        for character in word:
            if font.measure(chunk + character) <= max_width:
                chunk += character
            else:
                if chunk:
                    lines.append(chunk)
                chunk = character
        current = chunk
    lines.append(current)
    return lines


def _wrapped_text_size(text, font, max_width):
    text = text if text else " "
    spacing = font.metrics("linespace") or 20
    lines = []
    for paragraph in text.replace("\r", "").split("\n"):
        lines.extend(_wrap_paragraph(paragraph, font, max_width))
    widths = [font.measure(line) for line in lines] or [0]
    return max(widths), max(1, len(lines)) * spacing


def _fit_textbox(box, min_height=36, wrap_width=CHAT_TEXT_WIDTH, hug=False):
    inner = _textbox_inner(box)
    try:
        text = inner.get("1.0", "end-1c") or " "
    except Exception:
        text = " "
    try:
        from tkinter import font as tkfont

        typeface = tkfont.Font(font=inner.cget("font"))
        spacing = typeface.metrics("linespace") or 20
    except Exception:
        typeface = None
        spacing = 20
    if hug and typeface is not None:
        inner_max = max(40, int(wrap_width) - 10)
        text_w, text_h = _wrapped_text_size(text, typeface, inner_max)
        box.configure(
            width=max(36, min(int(wrap_width), text_w + 10)),
            height=max(min_height, text_h + 8),
        )
        return
    char_w = 8
    if typeface is not None:
        char_w = max(typeface.measure("n"), 7)
    chars_per_line = max(20, int(wrap_width) // char_w)
    lines = 0
    for paragraph in text.split("\n"):
        lines += max(1, (len(paragraph) + chars_per_line - 1) // chars_per_line)
    box.configure(height=max(min_height, int(lines * spacing + 10)))


def _lock_textbox(box):
    inner = _textbox_inner(box)
    try:
        inner.configure(yscrollcommand=lambda *args: None)
    except Exception:
        pass

    def copy_selection(event=None):
        try:
            selected = inner.get("sel.first", "sel.last")
        except Exception:
            selected = inner.get("1.0", "end-1c")
        try:
            box.clipboard_clear()
            box.clipboard_append(selected)
        except Exception:
            pass
        return "break"

    def select_all(event=None):
        inner.tag_add("sel", "1.0", "end-1c")
        inner.mark_set("insert", "1.0")
        return "break"

    def on_key(event):
        key = event.keysym.lower()
        ctrl = bool(event.state & 0x4)
        if ctrl and key == "c":
            return copy_selection(event)
        if ctrl and key == "a":
            return select_all(event)
        if key in ("left", "right", "up", "down", "home", "end", "next", "prior", "shift_l", "shift_r"):
            return None
        return "break"

    for widget in (box, inner):
        widget.bind("<Key>", on_key)
        widget.bind("<Control-c>", copy_selection)
        widget.bind("<Control-a>", select_all)
        widget.bind("<Command-c>", copy_selection)
        widget.bind("<Command-a>", select_all)


def _set_textbox_text(box, text, refit=True):
    inner = _textbox_inner(box)
    inner.configure(state="normal")
    inner.delete("1.0", "end")
    inner.insert("1.0", text or "")
    if refit:
        _fit_textbox(box)


def _write_assistant_text(box, answer="", reasoning=""):
    inner = _textbox_inner(box)
    inner.configure(state="normal")
    inner.delete("1.0", "end")
    answer = (answer or "").strip()
    reasoning = (reasoning or "").strip()
    inner.tag_configure("assistant", foreground=TEXT)
    inner.tag_configure("reason", foreground=MUTED)
    if answer:
        inner.insert("end", answer, "assistant")
    if reasoning:
        if answer:
            inner.insert("end", "\n\n")
        inner.insert("end", "Reasoning\n", "reason")
        inner.insert("end", reasoning, "reason")
    try:
        wrap_width = int(box.cget("width") or CHAT_TEXT_WIDTH)
    except Exception:
        wrap_width = CHAT_TEXT_WIDTH
    _fit_textbox(box, wrap_width=wrap_width)


def make_copyable_text(
    parent,
    text="",
    *,
    text_color=TEXT,
    font_size=15,
    min_height=36,
    width=CHAT_TEXT_WIDTH,
    fg_color="transparent",
    hug=False,
):
    kwargs = dict(
        fg_color=fg_color,
        text_color=text_color,
        font=ctk.CTkFont(size=font_size),
        wrap="word",
        activate_scrollbars=False,
        border_width=0,
        width=width,
        height=min_height,
        corner_radius=0,
    )
    try:
        box = ctk.CTkTextbox(parent, border_spacing=3, **kwargs)
    except TypeError:
        box = ctk.CTkTextbox(parent, **kwargs)
    if text:
        box.insert("1.0", text)
    _lock_textbox(box)
    _fit_textbox(box, min_height=min_height, wrap_width=width, hug=hug)
    return box


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def looks_like_hash_name(name: str) -> bool:
    stem = Path(name or "").stem
    return len(stem) == 64 and all(character in "0123456789abcdef" for character in stem.lower())


def human_filename(content_hash: str, fallback: str = "", library: Optional[list] = None) -> str:
    candidates = [fallback]
    if library:
        for item in library:
            if item.get("content_hash") == content_hash:
                candidates.append(item.get("filename") or "")
    pack = load_json(cache_path(content_hash)) if content_hash else None
    if pack:
        candidates.append(pack.get("filename") or "")
    for name in candidates:
        if name and not looks_like_hash_name(name):
            return Path(name).name
    return Path(fallback).name if fallback else "document.pdf"


def local_pdf_path(content_hash: str) -> Path:
    return DOCS_DIR / f"{content_hash}.pdf"


def remember_pdf(content_hash: str, source_path: str) -> str:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    destination = local_pdf_path(content_hash)
    source = Path(source_path)
    if source.exists() and source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return str(destination)


def _candidate_pdf_paths(entry: dict) -> list:
    content_hash = entry.get("content_hash") or ""
    filename = entry.get("filename") or ""
    candidates = []
    if content_hash:
        candidates.append(local_pdf_path(content_hash))
    stored = entry.get("pdf_path") or ""
    if stored:
        stored_path = Path(stored)
        candidates.append(stored_path)
        if stored_path.suffix.lower() == ".cloak.json":
            candidates.append(stored_path.with_suffix(".pdf"))
    if filename:
        search_roots = [
            Path.cwd(),
            Path.home() / "Documents",
            Path.home() / "Downloads",
            Path.home() / "Desktop",
            CLOAK_DIR / "docs",
        ]
        if stored:
            search_roots.insert(0, Path(stored).parent)
        for root in search_roots:
            try:
                if root.is_dir():
                    candidates.append(root / filename)
            except OSError:
                continue
    unique = []
    seen = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_pdf_path(entry: dict) -> Optional[str]:
    content_hash = entry.get("content_hash") or ""
    for candidate in _candidate_pdf_paths(entry):
        if not candidate.exists() or candidate.suffix.lower() != ".pdf":
            continue
        try:
            if content_hash and sha256_file(candidate) != content_hash:
                continue
        except OSError:
            continue
        if content_hash:
            return remember_pdf(content_hash, str(candidate))
        return str(candidate)
    return None


def sidecar_path(pdf_path: str) -> Path:
    return Path(pdf_path).with_suffix(".cloak.json")


def cache_path(content_hash: str) -> Path:
    return CACHE_DIR / f"{content_hash}.json"


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def find_embedding_pack(pdf_path: str, content_hash: str) -> Optional[Path]:
    cached = cache_path(content_hash)
    if cached.exists():
        pack = load_json(cached)
        if pack and pack.get("content_hash") == content_hash:
            return cached

    sidecar = sidecar_path(pdf_path)
    if sidecar.exists():
        pack = load_json(sidecar)
        if pack and pack.get("content_hash") == content_hash:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sidecar, cached)
            except OSError:
                pass
            return sidecar
    return None


def save_embedding_pack(pdf_path: str, content_hash: str, pack: dict) -> None:
    payload = json.dumps(pack, ensure_ascii=False, separators=(",", ":"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(content_hash).write_text(payload, encoding="utf-8")
    try:
        sidecar_path(pdf_path).write_text(payload, encoding="utf-8")
    except OSError:
        pass


def load_library() -> list:
    CLOAK_DIR.mkdir(parents=True, exist_ok=True)
    data = load_json(LIBRARY_PATH)
    library = data if isinstance(data, list) else []
    known = {item.get("content_hash") for item in library if item.get("content_hash")}
    if CACHE_DIR.exists():
        for pack_file in CACHE_DIR.glob("*.json"):
            pack = load_json(pack_file)
            if not pack or not pack.get("content_hash") or pack["content_hash"] in known:
                continue
            library.append(
                {
                    "content_hash": pack["content_hash"],
                    "filename": human_filename(pack["content_hash"], pack.get("filename") or pack_file.stem),
                    "pdf_path": str(local_pdf_path(pack["content_hash"]))
                    if local_pdf_path(pack["content_hash"]).exists()
                    else "",
                    "last_used": datetime.now(timezone.utc).isoformat(),
                }
            )
            known.add(pack["content_hash"])
    for item in library:
        item["filename"] = human_filename(
            item.get("content_hash") or "",
            item.get("filename") or "",
            library,
        )
    save_library(library)
    return library


def save_library(library: list) -> None:
    CLOAK_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_PATH.write_text(json.dumps(library, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def blank_chat(chat_id: Optional[str] = None) -> dict:
    now = _utc_now()
    return {
        "id": chat_id or str(uuid.uuid4()),
        "title": "New chat",
        "created_at": now,
        "updated_at": now,
        "doc_hashes": [],
        "messages": [],
    }


def load_chat(chat_id: str) -> Optional[dict]:
    data = load_json(chat_path(chat_id))
    if not isinstance(data, dict) or not data.get("id"):
        return None
    data.setdefault("title", "New chat")
    data.setdefault("messages", [])
    data.setdefault("doc_hashes", [])
    return data


def save_chat(chat: dict) -> None:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    chat["updated_at"] = _utc_now()
    chat_path(chat["id"]).write_text(json.dumps(chat, indent=2, ensure_ascii=False), encoding="utf-8")


def list_chats() -> list:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    chats = []
    for path in CHATS_DIR.glob("*.json"):
        data = load_json(path)
        if not isinstance(data, dict) or not data.get("id"):
            continue
        if not (data.get("messages") or data.get("doc_hashes")):
            continue
        chats.append(
            {
                "id": data["id"],
                "title": data.get("title") or "New chat",
                "updated_at": data.get("updated_at") or data.get("created_at") or "",
                "preview": _chat_preview(data),
            }
        )
    chats.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return chats


def delete_chat_file(chat_id: str) -> None:
    path = chat_path(chat_id)
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _chat_preview(chat: dict) -> str:
    for message in chat.get("messages") or []:
        if message.get("role") == "user" and (message.get("content") or "").strip():
            return re.sub(r"\s+", " ", message["content"]).strip()[:80]
    return ""


def title_from_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return "New chat"
    return cleaned[:48] + ("…" if len(cleaned) > 48 else "")


def load_settings() -> dict:
    data = load_json(SETTINGS_PATH)
    return data if isinstance(data, dict) else {}


def save_settings(settings: dict) -> None:
    CLOAK_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def chat_date_label(iso_text: str) -> str:
    if not iso_text:
        return "Earlier"
    try:
        stamp = datetime.fromisoformat(iso_text.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        local = stamp.astimezone()
        today = datetime.now().astimezone().date()
        day = local.date()
        if day == today:
            return "Today"
        if (today - day).days == 1:
            return "Yesterday"
        return local.strftime("%b %d, %Y")
    except Exception:
        return "Earlier"


def upsert_library_entry(library: list, content_hash: str, filename: str, pdf_path: str) -> list:
    filename = human_filename(content_hash, filename, library)
    updated = False
    for item in library:
        if item.get("content_hash") == content_hash:
            existing = item.get("filename") or ""
            if existing and not looks_like_hash_name(existing):
                item["filename"] = existing
            else:
                item["filename"] = filename or existing
            if pdf_path:
                item["pdf_path"] = pdf_path
            item["last_used"] = datetime.now(timezone.utc).isoformat()
            updated = True
            break
    if not updated:
        library.append(
            {
                "content_hash": content_hash,
                "filename": filename,
                "pdf_path": pdf_path,
                "last_used": datetime.now(timezone.utc).isoformat(),
            }
        )
    save_library(library)
    return library


def open_pdf(pdf_path: str, page: Optional[int] = None) -> None:
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(pdf_path)
    uri = path.as_uri()
    if page:
        uri = f"{uri}#page={page}"
    webbrowser.open(uri)


def infer_page_from_excerpt(excerpt: str) -> Optional[int]:
    match = re.search(r"page\s+(\d+)", excerpt or "", flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def render_pdf_page(pdf_path: str, page: int, max_width: int = 640, max_height: int = 0):
    try:
        import pymupdf
        from PIL import Image
        import io
    except ImportError:
        return None, page, None
    document = pymupdf.open(pdf_path)
    try:
        page_index = max(int(page or 1) - 1, 0)
        if page_index >= document.page_count:
            page_index = 0
        pdf_page = document.load_page(page_index)
        pixmap = pdf_page.get_pixmap(matrix=pymupdf.Matrix(1.85, 1.85), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        ratio = min(max_width / float(image.width), 1.0)
        if max_height:
            ratio = min(ratio, max_height / float(image.height))
        if ratio < 1.0:
            image = image.resize(
                (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        return image, page_index + 1, document.page_count
    finally:
        document.close()


class Cloak(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Cloak")
        self.geometry("1440x820")
        self.minsize(1100, 680)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=BG)
        self._apply_window_icon()

        self.session_id = str(uuid.uuid4())
        self._session_closed = False
        self._busy = False
        self._generation = 0
        self._stream_response = None
        self.loaded_hashes = set()
        self.library = load_library()
        self.doc_vars = {}
        self.chats = list_chats()
        self.chat = blank_chat()
        self.chat_id = self.chat["id"]
        self._chat_filter = ""
        self._page_image = None
        self._turn = None
        self._thinking_live = False
        self._pulse_job = None
        self._pulse_n = 0
        self._thought_expanded = True
        self.empty_state = None
        self._logo_small = load_logo((32, 32))
        self._logo_large = load_logo((96, 96))
        self._logo_chat = load_logo((28, 28))
        self._chat_icon = _circle_icon(26, "#243040", "◇", ACCENT, 14)
        self._pdf_badge = _pdf_badge(26)
        self._bar_job = None
        self._pending_uploads = []
        self._stick_to_bottom = True
        self._scroll_job = None
        self._drop_overlay = None
        self._restoring_docs = False
        self._server_connected = False
        self._available_models = []
        self._last_models_fetch = 0.0
        self.route_mode = "auto"
        self.settings = load_settings()
        raw_model = (self.settings.get("model") or "qwen2.5:7b").replace(" (local)", "").strip()
        self.selected_model = raw_model or "qwen2.5:7b"
        if self.settings.get("model") != self.selected_model:
            self.settings["model"] = self.selected_model
            save_settings(self.settings)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=0)
        self._preview_pdf_path = None
        self._preview_page = 1
        self._preview_page_count = 1

        self._build_sidebar()
        self._build_docs_panel()
        self._build_chat()
        self._build_citation_panel()
        self.refresh_library()
        self.refresh_chat_list()
        self._update_context_bar()
        self._enable_drag_drop()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        atexit.register(self.close_server_session)
        self.after(200, self._poll_server)

    def _apply_window_icon(self):
        ico = ensure_window_icon()
        photos = []
        if LOGO_PATH.exists():
            try:
                from PIL import ImageTk

                source = Image.open(LOGO_PATH).convert("RGBA")
                for size in (16, 32, 48, 64, 256):
                    resized = source.resize((size, size), Image.Resampling.LANCZOS)
                    photos.append(ImageTk.PhotoImage(resized, master=self))
                if photos:
                    self.iconphoto(True, *photos)
                    self._window_icon_photos = photos
            except Exception as exc:
                print(f"Could not set window photo icon: {exc}")
        if ico is not None:
            try:
                self.iconbitmap(str(ico))
            except Exception:
                try:
                    self.wm_iconbitmap(str(ico))
                except Exception as exc:
                    print(f"Could not set window bitmap icon: {exc}")
            self.after(200, lambda: self._reapply_iconbitmap(ico))

    def _reapply_iconbitmap(self, ico):
        try:
            self.iconbitmap(str(ico))
        except Exception:
            pass

    def _headers(self):
        return {"X-Session-Id": self.session_id}

    def ui(self, callback):
        self.after(0, callback)

    def set_status(self, text):
        return

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=268, corner_radius=0, fg_color=SIDEBAR_BG)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(3, weight=1)
        self.sidebar.grid_columnconfigure(0, weight=1)

        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, padx=20, pady=(24, 18), sticky="w")
        if self._logo_small:
            ctk.CTkLabel(brand, text="", image=self._logo_small).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(
            brand,
            text="Cloak",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT,
        ).pack(side="left")

        ctk.CTkButton(
            self.sidebar,
            text="New Chat",
            height=42,
            corner_radius=10,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#0b1520",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.new_chat,
        ).grid(row=1, column=0, padx=18, pady=(0, 12), sticky="ew")

        self.chat_search = ctk.CTkEntry(
            self.sidebar,
            placeholder_text="Search chats...",
            height=38,
            corner_radius=10,
            border_width=0,
            fg_color=CARD_BG,
            text_color=TEXT,
            placeholder_text_color=MUTED,
            font=ctk.CTkFont(size=13),
        )
        self.chat_search.grid(row=2, column=0, padx=18, pady=(0, 8), sticky="ew")
        self.chat_search.bind("<KeyRelease>", lambda _event: self._on_chat_search())

        self.chat_list = ctk.CTkScrollableFrame(
            self.sidebar,
            fg_color="transparent",
            scrollbar_button_color=LINE,
            scrollbar_button_hover_color=MUTED,
        )
        self.chat_list.grid(row=3, column=0, sticky="nsew", padx=10, pady=(4, 16))
        self.chat_list.grid_columnconfigure(0, weight=1)

    def _build_docs_panel(self):
        self.docs_panel = ctk.CTkFrame(self, width=340, corner_radius=0, fg_color=DOCS_BG)
        self.docs_panel.grid(row=0, column=1, sticky="nsew")
        self.docs_panel.grid_propagate(False)
        self.docs_panel.grid_rowconfigure(3, weight=1)
        self.docs_panel.grid_columnconfigure(0, weight=1)

        divider = ctk.CTkFrame(self.docs_panel, width=1, fg_color=PANEL_LINE)
        divider.place(x=0, y=0, relheight=1)

        self.model_menu = ctk.CTkOptionMenu(
            self.docs_panel,
            values=["Connecting…"],
            command=self._on_model_change,
            height=40,
            corner_radius=10,
            fg_color=CARD_BG,
            button_color=CARD_BG,
            button_hover_color="#2c333c",
            dropdown_fg_color=CARD_BG,
            dropdown_hover_color="#2c333c",
            text_color=TEXT,
            font=ctk.CTkFont(size=13),
            dropdown_font=ctk.CTkFont(size=13),
        )
        self.model_menu.set(self.selected_model or "Connecting…")
        self.model_menu.grid(row=0, column=0, padx=20, pady=(24, 8), sticky="ew")

        self.route_menu = ctk.CTkOptionMenu(
            self.docs_panel,
            values=["Auto", "Chat", "Docs"],
            command=self._on_route_mode_change,
            height=34,
            corner_radius=10,
            fg_color=CARD_BG,
            button_color=CARD_BG,
            button_hover_color="#2c333c",
            dropdown_fg_color=CARD_BG,
            dropdown_hover_color="#2c333c",
            text_color=SILVER,
            font=ctk.CTkFont(size=12),
            dropdown_font=ctk.CTkFont(size=12),
        )
        saved_route = (self.settings.get("route_mode") or "auto").lower()
        self.route_mode = saved_route if saved_route in {"auto", "chat", "rag"} else "auto"
        label = {"auto": "Auto", "chat": "Chat", "rag": "Docs"}.get(self.route_mode, "Auto")
        self.route_menu.set(label)
        self.route_menu.grid(row=1, column=0, padx=20, pady=(0, 14), sticky="ew")

        ctk.CTkLabel(
            self.docs_panel,
            text="Your documents",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=SILVER,
        ).grid(row=2, column=0, padx=22, pady=(0, 6), sticky="w")

        self.library_list = ctk.CTkScrollableFrame(
            self.docs_panel,
            fg_color="transparent",
            scrollbar_button_color=LINE,
            scrollbar_button_hover_color=MUTED,
        )
        self.library_list.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.library_list.grid_columnconfigure(0, weight=1)

        foot = ctk.CTkFrame(self.docs_panel, fg_color="transparent")
        foot.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 16))
        connected = ctk.CTkFrame(foot, fg_color="transparent")
        connected.pack(side="left")
        self.connection_dot = ctk.CTkLabel(
            connected, text="●", font=ctk.CTkFont(size=10), text_color=MUTED
        )
        self.connection_dot.pack(side="left", padx=(0, 4))
        self.connection_label = ctk.CTkLabel(
            connected,
            text="Connecting…",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
        )
        self.connection_label.pack(side="left")

    def _set_connection_ui(self, connected: bool):
        self._server_connected = connected
        if not hasattr(self, "connection_label"):
            return
        if connected:
            self.connection_dot.configure(text_color="#4caf7a")
            self.connection_label.configure(text="Connected", text_color=MUTED)
        else:
            self.connection_dot.configure(text_color="#c62828")
            self.connection_label.configure(text="Disconnected", text_color=MUTED)

    def _apply_server_models(self, models, default=None):
        models = [name for name in (models or []) if name]
        self._available_models = models
        if not hasattr(self, "model_menu"):
            return
        if not models:
            self.model_menu.configure(values=["No models"])
            self.model_menu.set("No models")
            return
        self.model_menu.configure(values=models)
        if self.selected_model in models:
            self.model_menu.set(self.selected_model)
            return
        pick = default if default in models else models[0]
        self.selected_model = pick
        self.model_menu.set(pick)
        self.settings["model"] = pick
        save_settings(self.settings)

    def _poll_server(self):
        threading.Thread(target=self._check_server, daemon=True).start()
        self.after(CONNECTION_POLL_MS, self._poll_server)

    def _check_server(self):
        connected = False
        models = None
        default = None
        try:
            health = requests.get(f"{SERVER_URL}/health", timeout=1.5)
            connected = health.status_code == 200
            now = time.time()
            need_models = connected and (
                not self._available_models
                or (now - float(getattr(self, "_last_models_fetch", 0) or 0)) * 1000
                >= MODELS_REFRESH_MS
            )
            if need_models:
                models_resp = requests.get(f"{SERVER_URL}/models", timeout=3)
                if models_resp.status_code == 200:
                    data = models_resp.json()
                    models = data.get("models") or []
                    default = data.get("default")
                    self._last_models_fetch = now
        except Exception:
            connected = False
        self.ui(lambda: self._on_server_poll(connected, models, default))

    def _on_server_poll(self, connected, models, default):
        self._set_connection_ui(connected)
        if connected and models is not None:
            self._apply_server_models(models, default)

    def _mark_connected(self):
        if not self._server_connected:
            self._set_connection_ui(True)

    def _on_route_mode_change(self, value):
        mapping = {"Auto": "auto", "Chat": "chat", "Docs": "rag"}
        self.route_mode = mapping.get(value, "auto")
        self.settings["route_mode"] = self.route_mode
        save_settings(self.settings)

    def _on_model_change(self, value):
        if value in ("Connecting…", "No models", "Disconnected"):
            return
        self.selected_model = value
        self.settings["model"] = value
        save_settings(self.settings)

    def _build_chat(self):
        self.chat_wrap = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.chat_wrap.grid(row=0, column=2, sticky="nsew")
        self.chat_wrap.grid_rowconfigure(1, weight=1)
        self.chat_wrap.grid_columnconfigure(0, weight=1)

        left_line = ctk.CTkFrame(self.chat_wrap, width=1, fg_color=PANEL_LINE)
        left_line.place(x=0, y=0, relheight=1)

        self.context_bar = ctk.CTkFrame(self.chat_wrap, fg_color=CARD_BG, corner_radius=0, height=44)
        self.context_bar.grid(row=0, column=0, sticky="ew")
        self.context_bar.grid_propagate(False)
        self.context_label = ctk.CTkLabel(
            self.context_bar,
            text="No PDF in this chat yet.",
            font=ctk.CTkFont(size=13),
            text_color=MUTED,
            anchor="w",
        )
        self.context_label.pack(fill="x", padx=28, pady=12)

        self.chat_scroll = ctk.CTkScrollableFrame(
            self.chat_wrap,
            fg_color="transparent",
            scrollbar_button_color=LINE,
            scrollbar_button_hover_color=MUTED,
        )
        self.chat_scroll.grid(row=1, column=0, sticky="nsew", padx=36, pady=(18, 8))
        self.chat_scroll.grid_columnconfigure(0, weight=1)
        self._hook_chat_scroll()
        self._show_empty_state()

        self.composer = ctk.CTkFrame(self.chat_wrap, fg_color=CARD_BG, corner_radius=18, border_width=0)
        self.composer.grid(row=2, column=0, sticky="ew", padx=28, pady=(8, 22))
        self.composer.grid_columnconfigure(1, weight=1)

        self.attach_btn = ctk.CTkButton(
            self.composer,
            text="+",
            width=42,
            height=42,
            corner_radius=12,
            fg_color="#2c333c",
            hover_color="#353d48",
            text_color=ACCENT,
            font=ctk.CTkFont(size=22, weight="bold"),
            command=self.upload_pdf_thread,
        )
        self.attach_btn.grid(row=0, column=0, padx=(10, 6), pady=10)

        self.msg_entry = ctk.CTkEntry(
            self.composer,
            placeholder_text="Ask a question...",
            border_width=0,
            fg_color=CARD_BG,
            height=44,
            font=ctk.CTkFont(size=15),
            text_color=TEXT,
            placeholder_text_color=MUTED,
        )
        self.msg_entry.grid(row=0, column=1, sticky="ew", padx=(2, 6), pady=10)
        self.msg_entry.bind("<Return>", lambda event: self.send_query_thread())

        self.send_btn = ctk.CTkButton(
            self.composer,
            text="➤",
            width=42,
            height=42,
            corner_radius=12,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#0b1520",
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._composer_button,
        )
        self.send_btn.grid(row=0, column=2, padx=(0, 10), pady=10)

    def _update_context_bar(self):
        if not hasattr(self, "context_label"):
            return
        names = []
        for content_hash in sorted(self.loaded_hashes):
            entry = self.library_entry(content_hash) or {}
            names.append(human_filename(content_hash, entry.get("filename") or "", self.library))
        if not names:
            self.context_label.configure(text="No PDF in this chat yet.")
        elif len(names) == 1:
            self.context_label.configure(
                text=f"{names[0]} is actively being used as context in this conversation."
            )
        else:
            self.context_label.configure(
                text=f"{len(names)} PDFs are actively being used as context in this conversation."
            )

    def _hook_chat_scroll(self):
        scrollbar = getattr(self.chat_scroll, "_scrollbar", None)
        if scrollbar is None:
            return
        original_set = scrollbar.set

        def set_and_track(first, last):
            original_set(first, last)
            try:
                self._stick_to_bottom = float(last) >= 0.985
            except Exception:
                pass

        scrollbar.set = set_and_track

    def _on_chat_wheel(self, event):
        canvas = self.chat_scroll._parent_canvas
        steps = int(-1 * (event.delta / 120))
        if steps == 0:
            steps = -1 if event.delta > 0 else 1
        canvas.yview_scroll(steps, "units")
        return "break"

    def _on_chat_wheel_linux(self, steps):
        self.chat_scroll._parent_canvas.yview_scroll(steps, "units")
        return "break"

    def _redirect_wheel(self, widget):
        inner = _textbox_inner(widget)
        for target in (widget, inner):
            target.bind("<MouseWheel>", self._on_chat_wheel)
            target.bind("<Button-4>", lambda event: self._on_chat_wheel_linux(-1))
            target.bind("<Button-5>", lambda event: self._on_chat_wheel_linux(1))

    def _show_empty_state(self):
        if getattr(self, "empty_state", None) is not None:
            try:
                if self.empty_state.winfo_exists():
                    self.empty_state.grid()
                    return
            except Exception:
                pass
        self.empty_state = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        self.empty_state.grid(sticky="ew", pady=(72, 12))
        self.empty_state.grid_columnconfigure(0, weight=1)
        if self._logo_large:
            ctk.CTkLabel(self.empty_state, text="", image=self._logo_large).grid(row=0, column=0, pady=(0, 16))
        ctk.CTkLabel(
            self.empty_state,
            text="Cloak",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=TEXT,
        ).grid(row=1, column=0, pady=(0, 6))
        ctk.CTkLabel(
            self.empty_state,
            text="What's in the documents?",
            font=ctk.CTkFont(size=16),
            text_color=SILVER,
        ).grid(row=2, column=0, pady=(0, 8))
        ctk.CTkLabel(
            self.empty_state,
            text="Upload a PDF, then ask. Sources stay on this machine.",
            font=ctk.CTkFont(size=14),
            text_color=MUTED,
            wraplength=480,
        ).grid(row=3, column=0)

    def _hide_empty_state(self):
        if getattr(self, "empty_state", None) is not None:
            try:
                self.empty_state.grid_remove()
            except Exception:
                pass

    def _build_citation_panel(self):
        self.cite_panel = ctk.CTkFrame(self, width=520, fg_color=SIDEBAR_BG, corner_radius=0)
        self.cite_panel.grid(row=0, column=3, sticky="nsew")
        self.cite_panel.grid_propagate(False)
        self.cite_panel.grid_rowconfigure(3, weight=1)
        self.cite_panel.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self.cite_panel, fg_color=SIDEBAR_BG)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(16, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Source",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            header,
            text="Close",
            width=70,
            fg_color=CARD_BG,
            hover_color="#1b2228",
            border_width=1,
            border_color=LINE,
            command=self.hide_citation_panel,
        ).grid(row=0, column=1, sticky="e")

        self.cite_title = ctk.CTkLabel(
            self.cite_panel,
            text="",
            font=ctk.CTkFont(size=14, weight="bold"),
            wraplength=480,
            justify="left",
        )
        self.cite_title.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="w")

        nav = ctk.CTkFrame(self.cite_panel, fg_color=SIDEBAR_BG)
        nav.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 10))
        nav.grid_columnconfigure(4, weight=1)
        self.prev_page_btn = ctk.CTkButton(
            nav, text="Prev", width=64, fg_color=CARD_BG, hover_color="#1b2228",
            border_width=1, border_color=LINE, command=lambda: self.shift_preview_page(-1)
        )
        self.prev_page_btn.grid(row=0, column=0, padx=(0, 6))
        self.page_entry = ctk.CTkEntry(nav, width=52, height=32, justify="center")
        self.page_entry.grid(row=0, column=1, padx=(0, 4))
        self.page_entry.bind("<Return>", self.goto_preview_page)
        self.page_count_label = ctk.CTkLabel(nav, text="/ 1", text_color=MUTED)
        self.page_count_label.grid(row=0, column=2, padx=(0, 6))
        self.next_page_btn = ctk.CTkButton(
            nav, text="Next", width=64, fg_color=CARD_BG, hover_color="#1b2228",
            border_width=1, border_color=LINE, command=lambda: self.shift_preview_page(1)
        )
        self.next_page_btn.grid(row=0, column=3, padx=(0, 8))
        self.open_pdf_btn = ctk.CTkButton(
            nav,
            text="Open PDF",
            width=90,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#041114",
            command=self.open_current_citation,
        )
        self.open_pdf_btn.grid(row=0, column=5, sticky="e")

        self.cite_preview_host = ctk.CTkFrame(self.cite_panel, fg_color=CARD_BG, corner_radius=12)
        self.cite_preview_host.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 16))
        self.cite_image_label = None
        self._set_preview_widget(text="PDF preview")

        self._current_citation = None
        self.hide_citation_panel()

    def hide_citation_panel(self):
        self.cite_panel.grid_remove()
        self.grid_columnconfigure(3, weight=0)

    def show_citation_panel(self):
        self.grid_columnconfigure(3, weight=1)
        self.cite_panel.grid()

    def refresh_library(self):
        for child in self.library_list.winfo_children():
            child.destroy()
        self.doc_vars = {}
        pending = list(getattr(self, "_pending_uploads", []) or [])
        items = sorted(self.library, key=lambda item: item.get("last_used") or "", reverse=True)
        if not pending and not items:
            ctk.CTkLabel(
                self.library_list,
                text="No PDFs yet",
                text_color=MUTED,
                font=ctk.CTkFont(size=13),
            ).grid(row=0, column=0, padx=10, pady=12, sticky="w")
            self._update_context_bar()
            return
        row = 0
        for pending_item in pending:
            card = ctk.CTkFrame(self.library_list, fg_color=CARD_BG, corner_radius=12)
            card.grid(row=row, column=0, sticky="ew", pady=4, padx=6)
            card.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(card, text="", image=self._pdf_badge).grid(
                row=0, column=0, rowspan=2, padx=(12, 10), pady=12, sticky="nw"
            )
            ctk.CTkLabel(
                card,
                text=pending_item.get("filename") or "document.pdf",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=TEXT,
                anchor="w",
            ).grid(row=0, column=1, sticky="ew", pady=(12, 0))
            ctk.CTkLabel(
                card,
                text="Uploading…",
                font=ctk.CTkFont(size=12),
                text_color=ACCENT,
                anchor="w",
            ).grid(row=1, column=1, sticky="w", pady=(2, 12))
            row += 1
        for item in items:
            content_hash = item.get("content_hash") or ""
            filename = item.get("filename") or "document.pdf"
            if any((pending_item.get("filename") or "") == filename for pending_item in pending):
                continue
            loaded = content_hash in self.loaded_hashes
            var = ctk.BooleanVar(value=loaded)
            self.doc_vars[content_hash] = var
            card = ctk.CTkFrame(self.library_list, fg_color=CARD_BG, corner_radius=12)
            card.grid(row=row, column=0, sticky="ew", pady=4, padx=6)
            card.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(card, text="", image=self._pdf_badge).grid(
                row=0, column=0, rowspan=2, padx=(12, 10), pady=12, sticky="nw"
            )
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.grid(row=0, column=1, sticky="ew", pady=(10, 0), padx=(0, 8))
            top.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                top,
                text=filename,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=TEXT,
                anchor="w",
            ).grid(row=0, column=0, sticky="ew")
            ctk.CTkButton(
                top,
                text="×",
                width=28,
                height=26,
                corner_radius=8,
                fg_color="transparent",
                hover_color="#3a2428",
                text_color=DANGER,
                font=ctk.CTkFont(size=16),
                command=lambda h=content_hash: self.delete_document(h),
            ).grid(row=0, column=1, sticky="e")
            bottom = ctk.CTkFrame(card, fg_color="transparent")
            bottom.grid(row=1, column=1, sticky="ew", pady=(4, 12), padx=(0, 8))
            bottom.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                bottom,
                text="Include in current chat",
                font=ctk.CTkFont(size=12),
                text_color=MUTED,
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkSwitch(
                bottom,
                text="",
                width=42,
                variable=var,
                command=lambda h=content_hash: self.toggle_document(h),
                progress_color=ACCENT,
                button_color="#f0f3f6",
                button_hover_color="#ffffff",
                fg_color="#3a424c",
            ).grid(row=0, column=1, sticky="e")
            row += 1
        self._update_context_bar()

    def library_entry(self, content_hash: str):
        for item in self.library:
            if item.get("content_hash") == content_hash:
                return item
        return None

    def delete_document(self, content_hash: str):
        entry = self.library_entry(content_hash)
        local = local_pdf_path(content_hash)
        try:
            if local.exists():
                local.unlink()
        except OSError:
            pass
        cached = cache_path(content_hash)
        try:
            if cached.exists():
                cached.unlink()
        except OSError:
            pass
        stored = Path((entry or {}).get("pdf_path") or "")
        try:
            if stored.exists() and CLOAK_DIR in stored.resolve().parents:
                stored.unlink()
        except OSError:
            pass
        self.library = [item for item in self.library if item.get("content_hash") != content_hash]
        save_library(self.library)
        self.loaded_hashes.discard(content_hash)
        self.refresh_library()
        threading.Thread(
            target=self._unload_document,
            args=(content_hash,),
            daemon=True,
        ).start()
        self._persist_chat()

    def refresh_chat_list(self):
        if not hasattr(self, "chat_list"):
            return
        for child in self.chat_list.winfo_children():
            child.destroy()
        query = (self._chat_filter or "").strip().lower()
        self.chats = list_chats()
        items = self.chats
        if query:
            items = [item for item in items if query in (item.get("title") or "").lower()]
        if not items:
            ctk.CTkLabel(
                self.chat_list,
                text="No chats yet" if not query else "No matching chats",
                text_color=MUTED,
                font=ctk.CTkFont(size=13),
            ).grid(row=0, column=0, padx=10, pady=12, sticky="w")
            return
        groups = {}
        for item in items:
            label = chat_date_label(item.get("updated_at") or "")
            groups.setdefault(label, []).append(item)
        row = 0
        for label, bucket in groups.items():
            ctk.CTkLabel(
                self.chat_list,
                text=label,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=MUTED,
            ).grid(row=row, column=0, sticky="w", padx=10, pady=(14, 4))
            row += 1
            for item in bucket:
                active = item.get("id") == self.chat_id
                card = ctk.CTkFrame(
                    self.chat_list,
                    fg_color=ACCENT_DIM if active else "transparent",
                    corner_radius=10,
                )
                card.grid(row=row, column=0, sticky="ew", pady=1)
                card.grid_columnconfigure(1, weight=1)
                ctk.CTkLabel(card, text="", image=self._chat_icon).grid(
                    row=0, column=0, padx=(10, 8), pady=10, sticky="n"
                )
                title = item.get("title") or "New chat"
                text_wrap = ctk.CTkFrame(card, fg_color="transparent")
                text_wrap.grid(row=0, column=1, sticky="ew", pady=8)
                text_wrap.grid_columnconfigure(0, weight=1)
                open_btn = ctk.CTkButton(
                    text_wrap,
                    text=title,
                    anchor="w",
                    height=24,
                    fg_color="transparent",
                    hover_color="#2a313a",
                    text_color=TEXT if active else SILVER,
                    font=ctk.CTkFont(size=13, weight="bold"),
                    command=lambda chat_id=item["id"]: self.open_chat(chat_id),
                )
                open_btn.grid(row=0, column=0, sticky="ew")
                ctk.CTkLabel(
                    text_wrap,
                    text=chat_date_label(item.get("updated_at") or ""),
                    anchor="w",
                    font=ctk.CTkFont(size=11),
                    text_color=MUTED,
                ).grid(row=1, column=0, sticky="w", padx=(8, 0))
                ctk.CTkButton(
                    card,
                    text="×",
                    width=26,
                    height=26,
                    fg_color="transparent",
                    hover_color="#3a2428",
                    text_color=MUTED,
                    font=ctk.CTkFont(size=15),
                    command=lambda chat_id=item["id"]: self.delete_chat(chat_id),
                ).grid(row=0, column=2, padx=(0, 6), pady=8, sticky="ne")
                row += 1

    def _on_chat_search(self):
        self._chat_filter = self.chat_search.get()
        self.refresh_chat_list()

    def _persist_chat(self):
        if not getattr(self, "chat", None):
            return
        self.chat["doc_hashes"] = sorted(self.loaded_hashes)
        if self.chat.get("title") in (None, "", "New chat"):
            preview = _chat_preview(self.chat)
            if preview:
                self.chat["title"] = title_from_text(preview)
        if not (self.chat.get("messages") or self.chat.get("doc_hashes")):
            delete_chat_file(self.chat["id"])
            self.refresh_chat_list()
            return
        save_chat(self.chat)
        self.refresh_chat_list()

    def _append_message(self, role, content, **extra):
        message = {
            "role": role,
            "content": content or "",
            "ts": _utc_now(),
        }
        message.update(extra)
        self.chat.setdefault("messages", []).append(message)
        if role == "user" and (self.chat.get("title") in (None, "", "New chat")):
            self.chat["title"] = title_from_text(content)
        self._persist_chat()

    def _clear_chat_view(self):
        self._stop_thinking_pulse()
        self._turn = None
        for child in self.chat_scroll.winfo_children():
            child.destroy()
        self.empty_state = None
        self.hide_citation_panel()

    def _replay_messages(self):
        messages = self.chat.get("messages") or []
        if not messages:
            self._show_empty_state()
            return
        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                # Skip upload chatter; the top context bar covers that.
                if re.search(r"\bis in this chat\.?\s*$", content, re.I):
                    continue
                self.add_system_message(content, persist=False)
            elif role == "user":
                self.add_user_bubble(content)
            elif role == "assistant":
                self._create_assistant_turn("")
                self._finish_thinking()
                self._render_turn(content, message.get("reasoning") or "")
                self._show_turn_citations(message.get("citations") or [])
                self._turn = None
        self._scroll_chat(force=True)

    def open_chat(self, chat_id: str):
        if chat_id == self.chat_id and not self._busy:
            return
        if self._busy:
            self.cancel_generation()
        self._persist_chat()
        loaded = load_chat(chat_id)
        if loaded is None:
            self.set_status("Chat not found")
            self.refresh_chat_list()
            return
        self.close_server_session()
        self._session_closed = False
        self.session_id = str(uuid.uuid4())
        self.chat = loaded
        self.chat_id = loaded["id"]
        self.loaded_hashes.clear()
        self._generation += 1
        self._busy = False
        self._set_composer_enabled(True)
        self._set_send_generating(False)
        self._clear_chat_view()
        self._replay_messages()
        self.refresh_library()
        self.refresh_chat_list()
        self.set_status("Ready")
        hashes = list(loaded.get("doc_hashes") or [])
        if hashes:
            self.set_status("Restoring documents for this chat...")
            self._restoring_docs = True
            threading.Thread(target=self._restore_chat_docs, args=(hashes,), daemon=True).start()

    def _restore_chat_docs(self, hashes):
        try:
            for content_hash in hashes:
                entry = self.library_entry(content_hash)
                if not entry:
                    continue
                pdf_path = resolve_pdf_path(entry)
                if not pdf_path:
                    continue
                display_name = human_filename(content_hash, entry.get("filename") or "", self.library)

                def mark_pending(name=display_name):
                    self._pending_uploads = [
                        item for item in getattr(self, "_pending_uploads", []) if item.get("filename") != name
                    ]
                    self._pending_uploads.insert(0, {"filename": name, "path": ""})
                    self.refresh_library()

                self.ui(mark_pending)
                self.upload_pdf(pdf_path, display_name)
        finally:
            self.ui(lambda: setattr(self, "_restoring_docs", False))
            self.ui(lambda: self.set_status("Ready"))
            self.ui(self._persist_chat)

    def delete_chat(self, chat_id: str):
        if chat_id == self.chat_id and self._busy:
            self.cancel_generation()
        delete_chat_file(chat_id)
        if chat_id == self.chat_id:
            self.new_chat(skip_save_old=True)
        else:
            self.refresh_chat_list()

    def new_chat(self, skip_save_old=False):
        if self._busy:
            self.cancel_generation()
        if not skip_save_old:
            self._persist_chat()
        self.close_server_session()
        self._session_closed = False
        self.session_id = str(uuid.uuid4())
        self.loaded_hashes.clear()
        self.chat = blank_chat()
        self.chat_id = self.chat["id"]
        self._clear_chat_view()
        self._show_empty_state()
        self.set_status("Ready")
        self.refresh_library()
        self.refresh_chat_list()
        if hasattr(self, "send_btn"):
            self._set_composer_enabled(True)
        self._set_send_generating(False)
        self._busy = False
        self._generation += 1
        threading.Thread(target=self._post_cancel, daemon=True).start()

    def _enable_drag_drop(self):
        try:
            import windnd

            def on_drop(files):
                paths = []
                for item in files or []:
                    if isinstance(item, bytes):
                        try:
                            item = item.decode("utf-8")
                        except UnicodeDecodeError:
                            item = item.decode("mbcs", errors="ignore")
                    paths.append(str(item).strip().strip('"'))
                self.after(0, lambda: self._handle_dropped_files(paths))

            windnd.hook_dropfiles(self, func=on_drop)
        except Exception as exc:
            print(f"Drag-and-drop unavailable ({exc}). Use the + button to upload PDFs.")

    def _handle_dropped_files(self, paths):
        pdfs = [path for path in paths if path.lower().endswith(".pdf") and Path(path).exists()]
        if not pdfs:
            self.set_status("Drop PDF files only")
            return
        for path in pdfs:
            display_name = os.path.basename(path)
            self._pending_uploads = [
                item for item in getattr(self, "_pending_uploads", []) if item.get("filename") != display_name
            ]
            self._pending_uploads.insert(0, {"filename": display_name, "path": path})
            self.refresh_library()
            self.set_status(f"Loading {display_name}...")
            threading.Thread(target=self.upload_pdf, args=(path, display_name), daemon=True).start()

    def close_server_session(self):
        if self._session_closed:
            return
        self._session_closed = True
        try:
            requests.post(f"{SERVER_URL}/session/close", headers=self._headers(), timeout=1)
        except Exception:
            pass

    def on_close(self):
        self._persist_chat()
        self.close_server_session()
        self.destroy()

    def add_system_message(self, text, persist=True):
        self._hide_empty_state()
        label = ctk.CTkLabel(
            self.chat_scroll,
            text=text,
            text_color=MUTED,
            font=ctk.CTkFont(size=13),
            wraplength=720,
            justify="left",
        )
        label.grid(sticky="w", pady=(4, 12))
        self._scroll_chat()
        if persist:
            self._append_message("system", text)
    def add_user_bubble(self, text):
        self._hide_empty_state()
        row = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        row.grid(sticky="ew", pady=(8, 6))
        row.grid_columnconfigure(0, weight=1)
        bubble = ctk.CTkFrame(row, fg_color=BUBBLE_BG, corner_radius=16)
        bubble.grid(row=0, column=0, sticky="w", padx=(0, 64))
        box = make_copyable_text(
            bubble,
            text,
            text_color=TEXT,
            font_size=15,
            min_height=28,
            width=BUBBLE_MAX_WIDTH,
            fg_color=BUBBLE_BG,
            hug=True,
        )
        box.pack(padx=14, pady=10)
        self._redirect_wheel(box)
        self._scroll_chat(force=True)
        return box

    def add_assistant_bubble(self, text, citations=None):
        self._create_assistant_turn("")
        self._complete_turn(
            {
                "answer": text,
                "citations": citations or [],
                "reasoning": "",
            }
        )

    def _create_assistant_turn(self, query=""):
        self._hide_empty_state()
        wrap = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        wrap.grid(sticky="ew", pady=(8, 14))
        wrap.grid_columnconfigure(1, weight=1)

        if query:
            bubble_row = ctk.CTkFrame(wrap, fg_color="transparent")
            bubble_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
            bubble_row.grid_columnconfigure(0, weight=1)
            bubble = ctk.CTkFrame(bubble_row, fg_color=BUBBLE_BG, corner_radius=16)
            bubble.grid(row=0, column=0, sticky="w", padx=(0, 64))
            query_box = make_copyable_text(
                bubble,
                query,
                text_color=TEXT,
                font_size=15,
                min_height=28,
                width=BUBBLE_MAX_WIDTH,
                fg_color=BUBBLE_BG,
                hug=True,
            )
            query_box.pack(padx=14, pady=10)
            self._redirect_wheel(query_box)

        if self._logo_chat:
            ctk.CTkLabel(wrap, text="", image=self._logo_chat).grid(
                row=1, column=0, sticky="n", padx=(0, 12), pady=(4, 0)
            )
        else:
            ctk.CTkLabel(wrap, text="", image=self._chat_icon).grid(
                row=1, column=0, sticky="n", padx=(0, 12), pady=(4, 0)
            )

        body = ctk.CTkFrame(wrap, fg_color="transparent")
        body.grid(row=1, column=1, sticky="ew")
        body.grid_columnconfigure(0, weight=1)

        status = ctk.CTkLabel(
            body,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=14, slant="italic"),
            text_color=MUTED,
        )
        status.grid(row=0, column=0, sticky="w", pady=(0, 6))

        answer = make_copyable_text(body, "", text_color=TEXT, font_size=15, min_height=28, width=560)
        answer.grid(row=1, column=0, sticky="w", pady=(2, 4))
        self._redirect_wheel(answer)

        reasoning_wrap = ctk.CTkFrame(body, fg_color="transparent")
        reasoning_wrap.grid(row=2, column=0, sticky="w", pady=(10, 4))
        reasoning_wrap.grid_columnconfigure(0, weight=1)
        reasoning_title = ctk.CTkLabel(
            reasoning_wrap,
            text="Reasoning",
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=SILVER,
        )
        reasoning_title.grid(row=0, column=0, sticky="w")
        reasoning_body = make_copyable_text(
            reasoning_wrap,
            "",
            text_color=MUTED,
            font_size=13,
            min_height=24,
            width=560,
        )
        reasoning_body.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self._redirect_wheel(reasoning_body)
        reasoning_wrap.grid_remove()

        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        chips = ctk.CTkFrame(actions, fg_color="transparent")
        chips.pack(side="left", fill="x", expand=True)

        turn = {
            "wrap": wrap,
            "generation": self._generation,
            "query": query or "",
            "status": status,
            "status_target": "Thinking",
            "status_shown": "",
            "answer": answer,
            "answer_text": "",
            "reasoning": reasoning_body,
            "reasoning_title": reasoning_title,
            "reasoning_wrap": reasoning_wrap,
            "reasoning_text": "",
            "chips": chips,
            "copy_btn": None,
        }
        copy_btn = ctk.CTkButton(
            actions,
            text="Copy",
            width=58,
            height=28,
            fg_color="transparent",
            hover_color=CARD_BG,
            text_color=MUTED,
            font=ctk.CTkFont(size=12),
            command=lambda item=turn: self._copy_turn(item),
        )
        copy_btn.pack(side="right")
        turn["copy_btn"] = copy_btn
        self._turn = turn
        self._start_thinking_pulse()
        self._apply_stage({"id": "understanding", "detail": ""})
        self._scroll_chat(force=True)

    def _copy_turn(self, turn):
        parts = []
        query = (turn.get("query") or "").strip()
        answer = (turn.get("answer_text") or "").strip()
        reasoning = (turn.get("reasoning_text") or "").strip()
        if query:
            parts.append(query)
        if answer:
            parts.append(answer)
        if reasoning:
            parts.append(f"Reasoning\n{reasoning}")
        text = "\n\n".join(parts).strip()
        if not text:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
        except Exception:
            return
        button = turn.get("copy_btn")
        if button is None:
            return
        button.configure(text="Copied")
        self.after(1200, lambda: self._reset_copy_label(button))

    def _reset_copy_label(self, button):
        try:
            button.configure(text="Copy")
        except Exception:
            pass

    def _start_thinking_pulse(self):
        self._thinking_live = True
        self._pulse_n = 0
        if self._bar_job:
            try:
                self.after_cancel(self._bar_job)
            except Exception:
                pass
        self._tick_status_text()

    def _stop_thinking_pulse(self):
        self._thinking_live = False
        if self._bar_job:
            try:
                self.after_cancel(self._bar_job)
            except Exception:
                pass
            self._bar_job = None
        if self._pulse_job:
            try:
                self.after_cancel(self._pulse_job)
            except Exception:
                pass
            self._pulse_job = None

    def _tick_status_text(self):
        if not self._thinking_live:
            return
        turn = self._turn or {}
        label = turn.get("status")
        if label is None:
            return
        target = turn.get("status_target") or "Thinking"
        shown = turn.get("status_shown") or ""
        if shown != target:
            if not target.startswith(shown):
                shown = ""
            shown = target[: len(shown) + 1]
            turn["status_shown"] = shown
            cursor = "▍" if len(shown) < len(target) else ""
            try:
                label.configure(text=shown + cursor, text_color=ACCENT)
            except Exception:
                return
            self._bar_job = self.after(18, self._tick_status_text)
            return

        self._pulse_n = (self._pulse_n + 1) % 64
        wave = (self._pulse_n / 63.0)
        amount = 0.5 - 0.5 * abs(1.0 - 2.0 * wave)
        color = _mix_hex(MUTED, ACCENT, 0.25 + 0.75 * amount)
        blink = " ▍" if (self._pulse_n // 8) % 2 == 0 else ""
        try:
            label.configure(text=target + blink, text_color=color)
        except Exception:
            return
        self._bar_job = self.after(40, self._tick_status_text)

    def _apply_stage(self, event):
        if (self._turn or {}).get("generation") not in (None, self._generation):
            return
        turn = self._turn
        if not turn:
            return
        stage_id = event.get("id") or ""
        label = STAGE_LABELS.get(stage_id, event.get("label") or "Thinking")
        detail = re.sub(r"\s+", " ", event.get("detail") or "").strip()
        if detail and detail.lower() not in label.lower() and len(detail) < 70:
            status = f"{label} · {detail}"
        else:
            status = label
        if turn.get("status_target") != status:
            turn["status_target"] = status
            turn["status_shown"] = ""

    def _finish_thinking(self):
        turn = self._turn or {}
        self._stop_thinking_pulse()
        status = turn.get("status")
        if status is not None:
            status.grid_remove()

    def _render_turn(self, answer="", reasoning=""):
        turn = self._turn or {}
        box = turn.get("answer")
        if box is None:
            return
        turn["answer_text"] = answer or ""
        turn["reasoning_text"] = reasoning or ""
        _set_textbox_text(box, answer or "")
        self._show_reasoning(reasoning)

    def _show_reasoning(self, reasoning):
        turn = self._turn or {}
        wrap = turn.get("reasoning_wrap")
        title = turn.get("reasoning_title")
        body = turn.get("reasoning")
        text = (reasoning or "").strip()
        if wrap is None:
            return
        if not text:
            wrap.grid_remove()
            return
        wrap.grid()
        if title is not None:
            title.configure(text="Reasoning")
        if body is not None:
            _set_textbox_text(body, text)

    def _show_turn_citations(self, citations):
        turn = self._turn or {}
        chips = turn.get("chips")
        if not chips or not citations:
            return
        for child in chips.winfo_children():
            child.destroy()
        for citation in citations:
            label = citation.get("source") or "Source"
            page = citation.get("page")
            chip = f"{label}"
            if page:
                chip += f"  p.{page}"
            ctk.CTkButton(
                chips,
                text=chip,
                width=1,
                height=30,
                corner_radius=14,
                fg_color=CARD_BG,
                hover_color="#1b2228",
                border_width=1,
                border_color="#1c3a40",
                text_color=SILVER,
                font=ctk.CTkFont(size=12),
                command=lambda item=citation: self.show_citation(item),
            ).pack(side="left", padx=(0, 8), pady=(0, 4))
        self._scroll_chat()

    def _type_answer(self, text, citations, reasoning, index=0):
        if (self._turn or {}).get("generation") not in (None, self._generation):
            return
        turn = self._turn or {}
        if not turn.get("answer"):
            return
        self._finish_thinking()
        self._render_turn(text or "No answer returned.", reasoning)
        self._show_turn_citations(citations)
        self._scroll_chat()
        self._query_finished()

    def _complete_turn(self, event):
        if (self._turn or {}).get("generation") not in (None, self._generation):
            return
        answer = (event.get("answer") or "").strip() or "No answer returned."
        citations = event.get("citations") or []
        reasoning = event.get("reasoning") or ""
        self._append_message(
            "assistant",
            answer,
            reasoning=reasoning,
            citations=citations,
        )
        self._type_answer(answer, citations, reasoning)

    def _fail_turn(self, message):
        if (self._turn or {}).get("generation") not in (None, self._generation):
            return
        turn = self._turn or {}
        self._finish_thinking()
        if turn.get("answer"):
            self._render_turn(message)
        else:
            self.add_system_message(message)
        self._query_finished()

    def _set_send_generating(self, generating):
        if not hasattr(self, "send_btn"):
            return
        if generating:
            self.send_btn.configure(
                text="■",
                font=ctk.CTkFont(size=22, weight="bold"),
            )
        else:
            self.send_btn.configure(
                text="↑",
                font=ctk.CTkFont(size=18, weight="bold"),
            )

    def _set_composer_enabled(self, enabled):
        if hasattr(self, "msg_entry"):
            self.msg_entry.configure(state="normal")
        if hasattr(self, "send_btn"):
            self.send_btn.configure(state="normal")

    def _query_finished(self, generation=None):
        if generation is not None and generation != self._generation:
            return
        self._busy = False
        self._stream_response = None
        self._set_send_generating(False)
        self._set_composer_enabled(True)
        self.set_status("Ready")

    def _scroll_chat(self, force=False):
        if not force and not getattr(self, "_stick_to_bottom", True):
            return
        if self._scroll_job:
            try:
                self.after_cancel(self._scroll_job)
            except Exception:
                pass
        self._scroll_job = self.after_idle(self._flush_chat_scroll)

    def _flush_chat_scroll(self):
        self._scroll_job = None
        try:
            self.chat_scroll._parent_canvas.yview_moveto(1.0)
            self._stick_to_bottom = True
        except Exception:
            pass

    def upload_pdf_thread(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not file_path:
            return
        display_name = os.path.basename(file_path)
        self._pending_uploads = [
            item for item in getattr(self, "_pending_uploads", []) if item.get("filename") != display_name
        ]
        self._pending_uploads.insert(0, {"filename": display_name, "path": file_path})
        self.refresh_library()
        self.set_status(f"Loading {display_name}...")
        threading.Thread(target=self.upload_pdf, args=(file_path, display_name), daemon=True).start()

    def toggle_document(self, content_hash: str):
        var = self.doc_vars.get(content_hash)
        if var is None:
            return
        if not var.get():
            self.loaded_hashes.discard(content_hash)
            self._persist_chat()
            self._update_context_bar()
            threading.Thread(
                target=self._unload_document,
                args=(content_hash,),
                daemon=True,
            ).start()
            return
        if content_hash in self.loaded_hashes:
            self._update_context_bar()
            return
        entry = self.library_entry(content_hash)
        if not entry:
            var.set(False)
            return
        pdf_path = resolve_pdf_path(entry)
        if not pdf_path:
            self.set_status(f"Could not find {entry.get('filename') or 'that PDF'} automatically.")
            var.set(False)
            return
        entry["pdf_path"] = pdf_path
        save_library(self.library)
        display_name = human_filename(content_hash, entry.get("filename") or "", self.library)
        self._pending_uploads = [
            item for item in getattr(self, "_pending_uploads", []) if item.get("filename") != display_name
        ]
        self._pending_uploads.insert(0, {"filename": display_name, "path": pdf_path})
        self.refresh_library()
        self.set_status(f"Loading {display_name}...")
        threading.Thread(
            target=self.upload_pdf,
            args=(pdf_path, display_name),
            daemon=True,
        ).start()

    def _unload_document(self, content_hash: str):
        try:
            response = requests.post(
                f"{SERVER_URL}/documents/unload",
                json={"content_hash": content_hash},
                headers=self._headers(),
                timeout=30,
            )
            if response.status_code != 200:
                print(f"Unload failed {response.status_code}: {response.text[:300]}")
        except Exception as exc:
            print(f"Unload connection error: {exc}")

    def _query_payload(self, query: str):
        return {
            "query": query,
            "content_hashes": sorted(self.loaded_hashes),
            "model": self.selected_model,
            "route_mode": getattr(self, "route_mode", None) or "auto",
        }

    def upload_pdf(self, file_path, display_name=None):
        self.ui(lambda: self.set_status("Loading document..."))
        try:
            with open(file_path, "rb") as handle:
                pdf_bytes = handle.read()
            content_hash = sha256_bytes(pdf_bytes)
            stored_path = remember_pdf(content_hash, file_path)
            display_name = human_filename(
                content_hash,
                display_name or os.path.basename(file_path),
                self.library,
            )
            pack_path = find_embedding_pack(file_path, content_hash) or find_embedding_pack(
                stored_path, content_hash
            )
            files = {"file": (display_name, pdf_bytes, "application/pdf")}
            if pack_path is not None:
                files["embeddings"] = (
                    "embeddings.json",
                    pack_path.read_bytes(),
                    "application/json",
                )

            response = requests.post(
                f"{SERVER_URL}/upload_pdf",
                files=files,
                headers=self._headers(),
            )
            if response.status_code == 200:
                data = response.json()
                pack = data.get("embedding_pack")
                content_hash = data.get("content_hash") or content_hash
                if pack:
                    save_embedding_pack(file_path, content_hash, pack)
                stored_path = remember_pdf(content_hash, file_path)
                self.library = upsert_library_entry(
                    self.library,
                    content_hash,
                    display_name,
                    stored_path,
                )
                self.loaded_hashes.add(content_hash)
                status = STATUS_BY_UPLOAD.get(data.get("status", "success"), "Ready")
                self.ui(lambda: self._mark_connected())
                self.ui(lambda name=display_name, st=status: self._after_upload(name, st))
            else:
                detail = response.text.strip().replace("\n", " ")
                self.ui(lambda name=display_name: self._after_upload_failed(name, f"Upload failed ({response.status_code})"))
                print(f"Upload failed {response.status_code}: {detail[:500]}")
        except Exception as exc:
            self.ui(lambda name=display_name or os.path.basename(file_path): self._after_upload_failed(name, "Connection error"))
            print(f"Upload connection error: {exc}")

    def _clear_pending_upload(self, filename):
        self._pending_uploads = [
            item for item in getattr(self, "_pending_uploads", []) if item.get("filename") != filename
        ]

    def _after_upload(self, filename, status):
        self._clear_pending_upload(filename)
        self.refresh_library()
        self._update_context_bar()
        self._persist_chat()

    def _after_upload_failed(self, filename, message):
        self._clear_pending_upload(filename)
        self.refresh_library()
        print(f"Upload failed: {message}")

    def _composer_button(self):
        if self._busy:
            self.cancel_generation()
            return
        self.send_query_thread()

    def send_query_thread(self):
        query = self.msg_entry.get().strip()
        if self._busy:
            if query:
                self.cancel_generation(start_next=query)
            else:
                self.cancel_generation()
            return
        if not query:
            return
        self._start_query(query)

    def _start_query(self, query):
        self._busy = True
        self._set_send_generating(True)
        self.msg_entry.delete(0, "end")
        self._append_message("user", query)
        self._create_assistant_turn(query)
        self.set_status("Working")
        generation = self._generation
        threading.Thread(target=self.send_query, args=(query, generation), daemon=True).start()

    def cancel_generation(self, start_next=None):
        if not self._busy and not start_next:
            return
        self._generation += 1
        threading.Thread(target=self._post_cancel, daemon=True).start()
        old_turn = self._turn
        self._finish_thinking()
        if old_turn and old_turn.get("answer"):
            old_turn["answer_text"] = "Stopped."
            old_turn["reasoning_text"] = ""
            _set_textbox_text(old_turn["answer"], "Stopped.")
            wrap = old_turn.get("reasoning_wrap")
            if wrap is not None:
                wrap.grid_remove()
        if start_next:
            self._start_query(start_next)
            return
        self._busy = False
        self._set_send_generating(False)
        self.set_status("Ready")

    def _close_stream(self):
        response = self._stream_response
        self._stream_response = None
        if response is None:
            return
        try:
            response.close()
        except Exception:
            pass

    def _post_cancel(self):
        try:
            requests.post(
                f"{SERVER_URL}/query/cancel",
                headers=self._headers(),
                timeout=2,
            )
        except Exception:
            pass

    def send_query(self, query, generation=None):
        if generation is None:
            generation = self._generation
        try:
            if self._stream_query(query, generation):
                return
            if generation != self._generation:
                return
            self._send_query_fallback(query, generation)
        except Exception as exc:
            print(f"Query failed: {exc}")
            if generation == self._generation:
                self.ui(lambda: self._fail_turn("Could not connect to the backend."))

    def _stream_query(self, query, generation):
        response = requests.post(
            f"{SERVER_URL}/query/stream",
            json=self._query_payload(query),
            headers=self._headers(),
            timeout=300,
            stream=True,
        )
        self._stream_response = response
        if response.status_code in (404, 405):
            return False
        if response.status_code != 200:
            if generation == self._generation:
                self.ui(lambda code=response.status_code: self._fail_turn(f"Backend error {code}"))
            return True
        self.ui(lambda: self._mark_connected())
        got_complete = False
        try:
            for line in response.iter_lines(decode_unicode=True):
                if generation != self._generation:
                    break
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line.split(":", 1)[1].strip()
                if not payload:
                    continue
                event = json.loads(payload)
                if event.get("type") == "complete":
                    got_complete = True
                self.ui(lambda e=event, g=generation: self._on_stream_event(e, g))
        except Exception as exc:
            if generation != self._generation:
                return True
            print(f"Stream ended: {exc}")
        finally:
            try:
                response.close()
            except Exception:
                pass
            if self._stream_response is response:
                self._stream_response = None
        if generation != self._generation:
            return True
        if not got_complete:
            self.ui(lambda: self._fail_turn("The backend stopped before finishing."))
        return True

    def _on_stream_event(self, event, generation=None):
        if generation is not None and generation != self._generation:
            return
        kind = event.get("type")
        if kind == "stage":
            self._apply_stage(event)
        elif kind == "complete":
            self._complete_turn(event)

    def _send_query_fallback(self, query, generation=None):
        if generation is not None and generation != self._generation:
            return
        self.ui(lambda: self._apply_stage({"id": "fetching", "detail": "Searching the documents"}))
        response = requests.post(
            f"{SERVER_URL}/query",
            json=self._query_payload(query),
            headers=self._headers(),
            timeout=300,
        )
        if generation is not None and generation != self._generation:
            return
        if response.status_code == 200:
            data = response.json()
            self.ui(lambda d=data: self._complete_turn(d))
        else:
            self.ui(lambda code=response.status_code: self._fail_turn(f"Backend error {code}"))

    def show_citation(self, citation: dict):
        self._current_citation = citation
        source = citation.get("source") or "document.pdf"
        excerpt = citation.get("excerpt") or ""
        self._set_preview_widget(text="Loading PDF page...")

        entry = self.library_entry(citation.get("content_hash") or "")
        pdf_path = resolve_pdf_path(entry) if entry else None
        page = citation.get("page") or infer_page_from_excerpt(excerpt) or 1
        self._preview_pdf_path = pdf_path
        self._preview_page = int(page)
        self.cite_title.configure(text=source)
        self.show_citation_panel()
        self.render_current_preview()

    def _sync_page_controls(self):
        if getattr(self, "page_entry", None) is None:
            return
        self.page_entry.delete(0, "end")
        self.page_entry.insert(0, str(self._preview_page or 1))
        if getattr(self, "page_count_label", None) is not None:
            self.page_count_label.configure(text=f"/ {self._preview_page_count or 1}")

    def goto_preview_page(self, event=None):
        if not self._preview_pdf_path:
            return "break"
        raw = self.page_entry.get().strip()
        try:
            page = int(raw)
        except ValueError:
            self._sync_page_controls()
            return "break"
        page = max(1, min(self._preview_page_count or 1, page))
        if page != self._preview_page:
            self._preview_page = page
            self.render_current_preview()
        else:
            self._sync_page_controls()
        return "break"

    def _preview_fit(self):
        width = max(self.cite_preview_host.winfo_width() - 16, 0)
        height = max(self.cite_preview_host.winfo_height() - 16, 0)
        if width < 80:
            width = 480
        if height < 80:
            height = 640
        return width, height

    def _set_preview_widget(self, image=None, text=""):
        import tkinter as tk
        from PIL import ImageTk

        for child in list(self.cite_preview_host.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass
        self.cite_image_label = None
        self._page_image = None
        if image is not None:
            photo = ImageTk.PhotoImage(image, master=self)
            self._page_image = photo
            label = tk.Label(
                self.cite_preview_host,
                image=photo,
                bg=CARD_BG,
                borderwidth=0,
                highlightthickness=0,
            )
            label.image = photo
            label.pack(expand=True, padx=8, pady=8)
            self.cite_image_label = label
            return
        self.cite_image_label = ctk.CTkLabel(
            self.cite_preview_host,
            text=text,
            fg_color=CARD_BG,
            wraplength=480,
            justify="center",
            text_color=MUTED,
        )
        self.cite_image_label.pack(expand=True, padx=8, pady=8)

    def render_current_preview(self):
        pdf_path = self._preview_pdf_path
        citation = self._current_citation or {}
        if not pdf_path or not Path(pdf_path).exists():
            self._set_preview_widget(text="Original PDF is not on this computer.")
            self._sync_page_controls()
            return
        max_width, max_height = self._preview_fit()
        try:
            image, page, count = render_pdf_page(
                pdf_path,
                self._preview_page,
                max_width=max_width,
                max_height=max_height,
            )
        except Exception as exc:
            self._set_preview_widget(text=f"Could not render PDF page: {exc}")
            return
        self._preview_page = page or self._preview_page
        self._preview_page_count = count or 1
        self._sync_page_controls()
        source = citation.get("source") or Path(pdf_path).name
        self.cite_title.configure(text=source)
        if image is None:
            self._set_preview_widget(text="Install pymupdf to preview PDF pages.")
            return
        self._set_preview_widget(image=image, text="")

    def shift_preview_page(self, delta: int):
        if not self._preview_pdf_path:
            return
        next_page = max(1, min(self._preview_page_count, self._preview_page + delta))
        if next_page == self._preview_page:
            return
        self._preview_page = next_page
        self.render_current_preview()

    def open_current_citation(self):
        citation = self._current_citation or {}
        pdf_path = self._preview_pdf_path
        if not pdf_path:
            entry = self.library_entry(citation.get("content_hash") or "")
            pdf_path = resolve_pdf_path(entry) if entry else None
        if not pdf_path:
            self.set_status("Could not find that PDF on this computer.")
            return
        try:
            open_pdf(pdf_path, self._preview_page or citation.get("page"))
        except Exception as exc:
            self.set_status(f"Could not open PDF: {exc}")


if __name__ == "__main__":
    app = Cloak()
    app.mainloop()
