"""Local tools for the chat path. 7B quality comes from tools, not a longer prompt.

Kept small on purpose: Qwen2.5-7B follows a handful of tools; a kitchen-sink
catalog makes it call the wrong one. Intentionally omitted: web search (this
app is local/NDA), a Python interpreter (it dumps code instead of answering),
and filesystem access.
"""

from __future__ import annotations

import ast
import datetime as _dt
import hashlib
import json
import math
import operator
import os
import re
from uuid import uuid4

MAX_TOOL_ROUNDS = 4
MAX_TOOL_CHARS = 8000
TOOL_NUM_PREDICT = 640

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_FUNCS = {
    "abs": abs,
    "ceil": math.ceil,
    "cos": math.cos,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "round": round,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

STAGE_LABELS = {
    "calculator": "Using calculator",
    "clock": "Checking the clock",
    "date_math": "Calculating dates",
    "convert_units": "Converting units",
    "list_pdfs": "Listing PDFs",
    "search_pdfs": "Searching PDFs",
    "read_pdf_page": "Reading PDF page",
    "hash_text": "Hashing text",
}

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate arithmetic. Use for any math or word problem. "
                "Pass an expression such as 0.15*240, (12+8)/5, or 15% of 847."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python-style math expression",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clock",
            "description": (
                "Current date and time. Optional IANA timezone like America/New_York."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone, or empty for local",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_math",
            "description": (
                "Calendar math. start/end as YYYY-MM-DD or today/tomorrow/yesterday. "
                "Set days (or weeks) to shift start. Set end to get days between."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "YYYY-MM-DD or today"},
                    "days": {"type": "integer", "description": "Days to add; negative subtracts"},
                    "weeks": {"type": "integer", "description": "Weeks to add"},
                    "end": {"type": "string", "description": "Other date for a day count"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": (
                "Convert length, mass, temperature, volume, area, time, speed, or data size. "
                "No live currency rates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_pdfs",
            "description": "List PDFs attached to this chat, with page ranges.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pdfs",
            "description": (
                "Search attached PDF excerpts. Pass keywords, a name, or a clause — not pronouns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pdf_page",
            "description": "Read text from one page of an attached PDF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "source": {
                        "type": "string",
                        "description": "Filename; optional if only one PDF is attached",
                    },
                },
                "required": ["page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hash_text",
            "description": "SHA-256 (default), SHA-1, or MD5 of a string. Local only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "algorithm": {
                        "type": "string",
                        "description": "sha256, sha1, or md5",
                    },
                },
                "required": ["text"],
            },
        },
    },
]


def stage_for_calls(calls) -> str:
    labels = []
    for call in calls or []:
        name = _call_field(call, "name") or ""
        label = STAGE_LABELS.get(name, "Using tools")
        if label not in labels:
            labels.append(label)
    return " · ".join(labels) or "Using tools"


def normalize_tool_calls(message) -> list[dict]:
    raw = list(getattr(message, "tool_calls", None) or [])
    if not raw:
        extra = getattr(message, "additional_kwargs", None) or {}
        raw = list(extra.get("tool_calls") or [])
    out = []
    for item in raw:
        name = _call_field(item, "name")
        args = _call_field(item, "args")
        call_id = _call_field(item, "id") or ""
        if not name:
            fn = _call_field(item, "function") or {}
            if isinstance(fn, dict):
                name = fn.get("name")
                args = args if args is not None else fn.get("arguments")
            elif fn is not None:
                name = getattr(fn, "name", None)
                args = args if args is not None else getattr(fn, "arguments", None)
        args = _as_args(args)
        if not name:
            continue
        out.append(
            {
                "name": str(name),
                "args": args,
                "id": str(call_id) or f"call_{uuid4().hex[:12]}",
            }
        )
    return out


def execute_tool(name, args, session=None):
    """Run one tool. Returns (text, docs_for_citations). Never raises."""
    args = _as_args(args)
    try:
        if name == "calculator":
            expr = _arg(args, "expression", "expr", "input", "query", "_raw")
            _ok, text = eval_expression(expr)
            return text, []
        if name == "clock":
            return format_clock(_arg(args, "timezone", "tz")), []
        if name == "date_math":
            return date_math(args), []
        if name == "convert_units":
            return convert_units(args), []
        if name == "list_pdfs":
            return list_pdfs(session), []
        if name == "search_pdfs":
            return search_pdfs(session, _arg(args, "query", "q", "text", "input", "keywords"))
        if name == "read_pdf_page":
            return read_pdf_page(
                session,
                _arg(args, "source", "filename", "file", "pdf"),
                args.get("page", args.get("p", args.get("page_number"))),
            )
        if name == "hash_text":
            return hash_text(_arg(args, "text", "input", "value"), _arg(args, "algorithm", "algo", default="sha256")), []
        return f"Unknown tool {name!r}.", []
    except Exception as exc:
        return f"Tool {name} failed: {exc}", []


def eval_expression(expr: str):
    """Safe arithmetic. Returns (ok, display_or_error)."""
    raw = (expr or "").strip()
    if not raw:
        return False, "Empty expression."
    text = raw.replace("×", "*").replace("÷", "/").replace("^", "**")
    text = re.sub(
        r"^(?:what(?:'s|\s+is)|calculate|compute|eval(?:uate)?)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"(\d+(?:\.\d+)?)\s*percent\b", r"(\1/100)", text, flags=re.I)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", text)
    text = re.sub(r"(?<=[\d)])\s*of\s*(?=[\d(])", "*", text, flags=re.I)
    text = re.sub(r"\s+", "", text)
    if not text or len(text) > 400:
        return False, f"Could not evaluate {raw!r}."
    try:
        tree = ast.parse(text, mode="eval")
        value = _eval_ast(tree)
        if not isinstance(value, bool):
            value = _check_num(value)
    except Exception as exc:
        return False, f"Could not evaluate {raw!r}: {exc}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, float):
        value = _fmt_num(value)
    return True, f"{raw} = {value}"


def format_clock(timezone: str = "") -> str:
    tz = (timezone or "").strip()
    now = _dt.datetime.now().astimezone()
    if tz:
        try:
            from zoneinfo import ZoneInfo

            now = _dt.datetime.now(ZoneInfo(tz))
        except Exception:
            return (
                f"Unknown timezone {tz!r}. Use an IANA name like America/New_York "
                f"or Europe/London. Local time is {now.strftime('%A, %B %d, %Y at %H:%M:%S %Z')}."
            )
    return now.strftime("%A, %B %d, %Y at %H:%M:%S %Z (UTC%z)")


def parse_date(value: str):
    text = (value or "").strip()
    key = text.lower()
    today = _dt.datetime.now().astimezone().date()
    if not text or key in ("today", "now"):
        return today
    if key == "tomorrow":
        return today + _dt.timedelta(days=1)
    if key == "yesterday":
        return today - _dt.timedelta(days=1)
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%d %B %Y"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Could not parse date {value!r}. Use YYYY-MM-DD.")


def date_math(args) -> str:
    args = _as_args(args)
    start = parse_date(_arg(args, "start", "date", "from", default="today"))
    end_raw = _arg(args, "end", "until", "to")
    days = _as_int(args.get("days"), None)
    weeks = _as_int(args.get("weeks"), None)
    shift = 0
    if days is not None:
        shift += days
    if weeks is not None:
        shift += 7 * weeks
    if end_raw:
        end = parse_date(end_raw)
        delta = (end - start).days
        return (
            f"{_fmt_date(start)} → {_fmt_date(end)}: {delta} day"
            f"{'' if abs(delta) == 1 else 's'}"
        )
    if shift:
        result = start + _dt.timedelta(days=shift)
        sign = "+" if shift > 0 else ""
        return f"{_fmt_date(start)} {sign}{shift} days → {_fmt_date(result)}"
    return _fmt_date(start)


def convert_units(args) -> str:
    args = _as_args(args)
    value = _as_float(_arg(args, "value", "amount", "n", "quantity"))
    src = _arg(args, "from_unit", "from", "unit_from", "source")
    dst = _arg(args, "to_unit", "to", "unit_to", "dest", "destination")
    if value is None:
        return "convert_units needs a numeric value."
    left = _lookup_unit(src)
    right = _lookup_unit(dst)
    if left is None:
        return f"Unknown unit {src!r}."
    if right is None:
        return f"Unknown unit {dst!r}."
    cat_a, key_a = left
    cat_b, key_b = right
    if cat_a != cat_b:
        return f"Cannot convert {src} ({cat_a}) to {dst} ({cat_b})."
    if cat_a == "temperature":
        kelvin = _to_kelvin(value, key_a)
        out = _from_kelvin(kelvin, key_b)
    else:
        base = value * _UNITS[cat_a][key_a]
        out = base / _UNITS[cat_b][key_b]
    return f"{_fmt_num(value)} {src} = {_fmt_num(out)} {dst}"


def attached_pdfs(session):
    files = []
    index = {}
    for chunk in getattr(session, "chunks", None) or []:
        meta = chunk.metadata or {}
        digest = meta.get("content_hash") or ""
        name = meta.get("source") or "document.pdf"
        key = digest or name
        item = index.get(key)
        if item is None:
            item = {"name": name, "hash": digest, "pages": set()}
            index[key] = item
            files.append(item)
        elif name and item["name"] == "document.pdf":
            item["name"] = name
        page = meta.get("page")
        try:
            if page is not None:
                item["pages"].add(int(page))
        except (TypeError, ValueError):
            pass
    return files


def attached_pdf_names(session) -> list[str]:
    names = []
    seen = set()
    for item in attached_pdfs(session):
        name = item["name"]
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def list_pdfs(session) -> str:
    files = attached_pdfs(session)
    if not files:
        return "No PDFs are attached to this chat. Ask the user to check a file in PDFs."
    lines = []
    for item in files:
        pages = sorted(item["pages"])
        extra = ""
        if pages:
            extra = f" (pages {pages[0]}–{pages[-1]})" if pages[0] != pages[-1] else f" (page {pages[0]})"
        lines.append(f"- {item['name']}{extra}")
    return "Attached PDFs:\n" + "\n".join(lines)


def search_pdfs(session, query: str):
    query = (query or "").strip()
    if not query:
        return "search_pdfs needs a query string.", []
    if not getattr(session, "chunks", None):
        return "No PDFs are attached to this chat. Ask the user to check a file in PDFs.", []
    docs = _retrieve_docs(session, query, k=6)
    if not docs:
        return "No matching passages in the attached PDFs.", []
    parts = []
    for index, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        src = meta.get("source") or "PDF"
        page = meta.get("page")
        loc = src if page in (None, "") else f"{src} p.{page}"
        text = re.sub(r"\s+", " ", (doc.page_content or "").strip())
        if len(text) > 700:
            text = text[:700] + "…"
        parts.append(f"[{index}] {loc}\n{text}")
    return "\n\n".join(parts), docs


def read_pdf_page(session, source, page):
    if not getattr(session, "chunks", None):
        return "No PDFs are attached to this chat. Ask the user to check a file in PDFs.", []
    try:
        page_i = int(float(page))
    except (TypeError, ValueError):
        return "read_pdf_page needs a page number.", []
    source = (source or "").strip()
    hits = []
    sources = set()
    for chunk in session.chunks:
        meta = chunk.metadata or {}
        try:
            chunk_page = int(float(meta.get("page")))
        except (TypeError, ValueError):
            continue
        if chunk_page != page_i:
            continue
        src = meta.get("source") or "document.pdf"
        sources.add(src)
        if source and not _source_match(src, source):
            continue
        hits.append(chunk)
    if not hits and source and sources:
        names = ", ".join(sorted(sources))
        return f"No text for {source!r} p.{page_i}. Pages with that number exist in: {names}.", []
    if not hits:
        available = sorted(
            {
                int(float((chunk.metadata or {}).get("page")))
                for chunk in session.chunks
                if (chunk.metadata or {}).get("page") is not None
            },
            key=lambda item: item,
        )
        hint = f" Available pages include {available[:12]}." if available else ""
        return f"No text for page {page_i}.{hint}", []
    files = sorted({(doc.metadata or {}).get("source") or "document.pdf" for doc in hits})
    if not source and len(files) > 1:
        return (
            f"Page {page_i} exists in: {', '.join(files)}. Pass source as the filename.",
            [],
        )
    parts = []
    for doc in hits:
        text = (doc.page_content or "").strip()
        if text:
            parts.append(text)
    body = "\n\n".join(parts).strip()
    if len(body) > MAX_TOOL_CHARS:
        body = body[:MAX_TOOL_CHARS] + "…"
    header = f"{files[0]} p.{page_i}"
    return f"{header}\n{body}" if body else f"{header} is empty.", hits[:4]


def hash_text(text: str, algorithm: str = "sha256") -> str:
    algo = (algorithm or "sha256").strip().lower().replace("-", "")
    data = (text or "").encode("utf-8")
    if algo in ("sha256", "sha2"):
        digest = hashlib.sha256(data).hexdigest()
        label = "sha256"
    elif algo in ("sha1",):
        digest = hashlib.sha1(data).hexdigest()
        label = "sha1"
    elif algo in ("md5",):
        digest = hashlib.md5(data).hexdigest()
        label = "md5"
    else:
        return "algorithm must be sha256, sha1, or md5."
    return f"{label}: {digest}"


def _retrieve_docs(session, query, k=6):
    allowed = {
        (chunk.metadata or {}).get("content_hash")
        for chunk in (session.chunks or [])
        if (chunk.metadata or {}).get("content_hash")
    }
    docs = []
    retriever = getattr(session, "hybrid_retriever", None) or getattr(session, "bm25_retriever", None)
    if retriever is not None:
        try:
            docs = list(retriever.invoke(query) or [])
        except Exception as exc:
            print(f"PDF tool search failed ({exc})")
            docs = []
    if allowed:
        docs = [doc for doc in docs if (doc.metadata or {}).get("content_hash") in allowed]
    if not docs:
        docs = _keyword_scan(session, query, k=max(k, 8))
    seen = set()
    unique = []
    for doc in docs:
        meta = doc.metadata or {}
        key = (meta.get("content_hash"), meta.get("page"), (doc.page_content or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
        if len(unique) >= k:
            break
    return unique


def _keyword_scan(session, query, k=6):
    terms = [term for term in re.findall(r"[a-z0-9][a-z0-9_./+-]*", (query or "").lower()) if len(term) > 1]
    if not terms:
        return []
    scored = []
    for chunk in session.chunks or []:
        text = (chunk.page_content or "").lower()
        if not text:
            continue
        score = sum(text.count(term) for term in terms)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: -item[0])
    return [chunk for _, chunk in scored[:k]]


def _source_match(actual, asked) -> bool:
    a = (actual or "").lower()
    b = (asked or "").lower().strip()
    if not b:
        return True
    return b in a or os.path.basename(a) == os.path.basename(b) or os.path.basename(b) in a


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        if isinstance(node.value, bool):
            raise ValueError("unsupported")
        return node.value
    num_type = getattr(ast, "Num", None)
    if num_type is not None and isinstance(node, num_type):
        return node.n
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _check_num(_eval_ast(node.left))
        right = _check_num(_eval_ast(node.right))
        if isinstance(node.op, ast.Pow) and (abs(right) > 12 or (abs(left) > 1e6 and abs(right) > 8)):
            raise ValueError("exponent too large")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_check_num(_eval_ast(node.operand)))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        op = _CMP.get(type(node.ops[0]))
        if op is None:
            raise ValueError("unsupported")
        return op(_check_num(_eval_ast(node.left)), _check_num(_eval_ast(node.comparators[0])))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        if node.keywords:
            raise ValueError("unsupported")
        fn = _FUNCS[node.func.id]
        args = [_check_num(_eval_ast(arg)) for arg in node.args]
        if node.func.id == "round" and not 1 <= len(args) <= 2:
            raise ValueError("unsupported")
        if node.func.id != "round" and not 1 <= len(args) <= 2:
            raise ValueError("unsupported")
        if node.func.id == "log" and not 1 <= len(args) <= 2:
            raise ValueError("unsupported")
        return fn(*args)
    raise ValueError("unsupported")


def _check_num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("unsupported")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError("overflow")
    if abs(value) > 1e15:
        raise ValueError("value too large")
    return value


def _fmt_num(value):
    if isinstance(value, float):
        text = f"{value:.10g}"
        return text
    return value


def _fmt_date(value: _dt.date) -> str:
    return value.strftime("%A, %B %d, %Y")


def _as_args(raw):
    if raw is None:
        return {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": text}
        return loaded if isinstance(loaded, dict) else {"_raw": text}
    if isinstance(raw, dict):
        return raw
    return {}


def _call_field(call, key, default=None):
    if isinstance(call, dict):
        return call.get(key, default)
    return getattr(call, key, default)


def _arg(args, *keys, default=""):
    for key in keys:
        if key in args and args[key] not in (None, ""):
            return args[key]
    return default


def _as_int(value, default=0):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Canonical factor is "how many bases in one of this unit". Temperature is special.
_UNITS = {
    "length": {
        "m": 1.0,
        "km": 1000.0,
        "cm": 0.01,
        "mm": 0.001,
        "um": 1e-6,
        "nm": 1e-9,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.344,
        "nmi": 1852.0,
    },
    "mass": {
        "g": 0.001,
        "kg": 1.0,
        "mg": 1e-6,
        "lb": 0.45359237,
        "oz": 0.028349523125,
        "ton": 907.18474,
        "tonne": 1000.0,
        "st": 6.35029318,
    },
    "volume": {
        "l": 1.0,
        "ml": 0.001,
        "m3": 1000.0,
        "gal": 3.785411784,
        "qt": 0.946352946,
        "pt": 0.473176473,
        "cup": 0.2365882365,
        "floz": 0.0295735295625,
        "tbsp": 0.01478676478125,
        "tsp": 0.00492892159375,
    },
    "area": {
        "m2": 1.0,
        "km2": 1e6,
        "cm2": 1e-4,
        "ft2": 0.09290304,
        "in2": 0.00064516,
        "acre": 4046.8564224,
        "ha": 10000.0,
        "mi2": 2589988.110336,
    },
    "time": {
        "s": 1.0,
        "ms": 0.001,
        "min": 60.0,
        "hr": 3600.0,
        "day": 86400.0,
        "week": 604800.0,
        "year": 31557600.0,
    },
    "speed": {
        "mps": 1.0,
        "kph": 1000.0 / 3600.0,
        "mph": 1609.344 / 3600.0,
        "knot": 1852.0 / 3600.0,
        "fps": 0.3048,
    },
    "data": {
        "b": 1.0,
        "kb": 1000.0,
        "mb": 1e6,
        "gb": 1e9,
        "tb": 1e12,
        "kib": 1024.0,
        "mib": 1024.0 ** 2,
        "gib": 1024.0 ** 3,
        "tib": 1024.0 ** 4,
    },
}

_UNIT_ALIAS = {
    "meter": ("length", "m"),
    "meters": ("length", "m"),
    "metre": ("length", "m"),
    "metres": ("length", "m"),
    "kilometer": ("length", "km"),
    "kilometers": ("length", "km"),
    "kilometre": ("length", "km"),
    "kilometres": ("length", "km"),
    "centimeter": ("length", "cm"),
    "centimeters": ("length", "cm"),
    "millimeter": ("length", "mm"),
    "millimeters": ("length", "mm"),
    "inch": ("length", "in"),
    "inches": ("length", "in"),
    "foot": ("length", "ft"),
    "feet": ("length", "ft"),
    "yard": ("length", "yd"),
    "yards": ("length", "yd"),
    "mile": ("length", "mi"),
    "miles": ("length", "mi"),
    "nauticalmile": ("length", "nmi"),
    "nauticalmiles": ("length", "nmi"),
    "gram": ("mass", "g"),
    "grams": ("mass", "g"),
    "kilogram": ("mass", "kg"),
    "kilograms": ("mass", "kg"),
    "milligram": ("mass", "mg"),
    "pound": ("mass", "lb"),
    "pounds": ("mass", "lb"),
    "lbs": ("mass", "lb"),
    "ounce": ("mass", "oz"),
    "ounces": ("mass", "oz"),
    "stone": ("mass", "st"),
    "tonnes": ("mass", "tonne"),
    "metricton": ("mass", "tonne"),
    "uston": ("mass", "ton"),
    "celsius": ("temperature", "c"),
    "centigrade": ("temperature", "c"),
    "fahrenheit": ("temperature", "f"),
    "kelvin": ("temperature", "k"),
    "degc": ("temperature", "c"),
    "degf": ("temperature", "f"),
    "degk": ("temperature", "k"),
    "liter": ("volume", "l"),
    "liters": ("volume", "l"),
    "litre": ("volume", "l"),
    "litres": ("volume", "l"),
    "milliliter": ("volume", "ml"),
    "milliliters": ("volume", "ml"),
    "gallon": ("volume", "gal"),
    "gallons": ("volume", "gal"),
    "usgallon": ("volume", "gal"),
    "quart": ("volume", "qt"),
    "quarts": ("volume", "qt"),
    "pint": ("volume", "pt"),
    "pints": ("volume", "pt"),
    "cups": ("volume", "cup"),
    "fluidounce": ("volume", "floz"),
    "fluidounces": ("volume", "floz"),
    "floz": ("volume", "floz"),
    "tablespoon": ("volume", "tbsp"),
    "teaspoon": ("volume", "tsp"),
    "cubicmeter": ("volume", "m3"),
    "cubicmeters": ("volume", "m3"),
    "sqm": ("area", "m2"),
    "squaremeter": ("area", "m2"),
    "squaremeters": ("area", "m2"),
    "sqft": ("area", "ft2"),
    "squarefoot": ("area", "ft2"),
    "squarefeet": ("area", "ft2"),
    "sqin": ("area", "in2"),
    "acre": ("area", "acre"),
    "acres": ("area", "acre"),
    "hectare": ("area", "ha"),
    "hectares": ("area", "ha"),
    "second": ("time", "s"),
    "seconds": ("time", "s"),
    "sec": ("time", "s"),
    "millisecond": ("time", "ms"),
    "minute": ("time", "min"),
    "minutes": ("time", "min"),
    "mins": ("time", "min"),
    "hour": ("time", "hr"),
    "hours": ("time", "hr"),
    "hrs": ("time", "hr"),
    "h": ("time", "hr"),
    "day": ("time", "day"),
    "days": ("time", "day"),
    "week": ("time", "week"),
    "weeks": ("time", "week"),
    "year": ("time", "year"),
    "years": ("time", "year"),
    "mps": ("speed", "mps"),
    "m/s": ("speed", "mps"),
    "kph": ("speed", "kph"),
    "km/h": ("speed", "kph"),
    "kmh": ("speed", "kph"),
    "mph": ("speed", "mph"),
    "knot": ("speed", "knot"),
    "knots": ("speed", "knot"),
    "ft/s": ("speed", "fps"),
    "fps": ("speed", "fps"),
    "byte": ("data", "b"),
    "bytes": ("data", "b"),
    "kilobyte": ("data", "kb"),
    "megabyte": ("data", "mb"),
    "gigabyte": ("data", "gb"),
    "terabyte": ("data", "tb"),
    "kibibyte": ("data", "kib"),
    "mebibyte": ("data", "mib"),
    "gibibyte": ("data", "gib"),
}


def _norm_unit_token(text: str) -> str:
    token = (text or "").strip().lower()
    token = token.replace("°", "").replace("º", "")
    token = token.replace(" ", "").replace(".", "")
    token = token.replace("²", "2").replace("³", "3")
    token = token.replace("square", "sq").replace("cubic", "cu")
    token = token.replace("degrees", "deg").replace("degree", "deg")
    return token


def _lookup_unit(text: str):
    token = _norm_unit_token(text)
    if not token:
        return None
    if token in _UNIT_ALIAS:
        return _UNIT_ALIAS[token]
    if token in ("c", "f", "k"):
        return ("temperature", token)
    hits = []
    for category, table in _UNITS.items():
        if token in table:
            hits.append((category, token))
    if len(hits) == 1:
        return hits[0]
    if token in ("in",):
        return ("length", "in")
    return None


def _to_kelvin(value: float, unit: str) -> float:
    if unit == "c":
        return value + 273.15
    if unit == "f":
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    if unit == "k":
        return value
    raise ValueError("unknown temperature")


def _from_kelvin(value: float, unit: str) -> float:
    if unit == "c":
        return value - 273.15
    if unit == "f":
        return (value - 273.15) * 9.0 / 5.0 + 32.0
    if unit == "k":
        return value
    raise ValueError("unknown temperature")
