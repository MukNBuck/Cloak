"""ChatGPT-class local harness for Cloak's Ollama models.

Uses the model's native chat template (System + turns) instead of stuffing
instructions into a raw user string. That is the main quality gap on 7B/8B.
"""

from __future__ import annotations

import datetime as _dt
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from chat_tools import eval_expression, format_clock

MATH_QUERY_RE = re.compile(
    r"^\s*(?:(?:what(?:'s| is)|calculate|compute|eval(?:uate)?)\s+)?("
    r"[\d\s.+\-*/()x×÷]+"
    r")\s*\??\s*$",
    re.I,
)
CLOCK_QUERY_RE = re.compile(
    r"^\s*("
    r"what(?:'s|\s+is)(?:\s+the)?(?:\s+current)?\s+(?:date|time|day)(?:\s+today|\s+is it)?"
    r"|what\s+(?:date|time|day)\s+is it"
    r"|(?:today'?s|current)\s+(?:date|time)"
    r")\s*\??\s*$",
    re.I,
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


def system_prompt(pdf_names=None) -> str:
    now = _dt.datetime.now().astimezone()
    names = [str(name).strip() for name in (pdf_names or []) if str(name).strip()]
    pdf_line = ""
    if names:
        shown = names[:8]
        extra = len(names) - len(shown)
        pdf_line = " Attached PDFs: " + ", ".join(shown)
        if extra:
            pdf_line += f" (+{extra} more)"
        pdf_line += "."
    return (
        "You are Cloak, a private general assistant running locally on this machine. "
        "You work like ChatGPT or Gemini: be direct, capable, and conversational. "
        "Nothing you say leaves this computer.\n"
        "Answer in plain language by default: explain, plan, tutor, and brainstorm in prose. "
        "Match the user's language. Use Markdown when it helps (lists, headings, short quotes).\n"
        "Do not write code unless the user asked you to code, debug, implement, or show syntax. "
        "Do not answer ordinary questions with a Python script, a calculator program, or a demo. "
        "If they did not name a language, do not default to Python; pick what they implied or ask. "
        "When they do want code, use a fenced block with a language tag and keep it as short as the request allows.\n"
        "For math, use LaTeX with delimiters: inline \\(like this\\) and display \\[like this\\]. "
        "Show short working, then the answer. Do not leave raw TeX without those delimiters. "
        "If you are unsure, say so and still give the best answer you can. "
        "Do not refuse ordinary requests. Do not claim you are ChatGPT, Gemini, or Google.\n"
        "Call tools instead of guessing: calculator for math, clock for the time, "
        "date_math for calendars, convert_units for conversions, hash_text for checksums, "
        "and list_pdfs / search_pdfs / read_pdf_page for attached files. "
        "Do not write Python to calculate."
        f"{pdf_line}\n"
        f"Local date and time: {now.strftime('%A, %B %d, %Y %H:%M %Z')}."
    )


def history_char_budget() -> int:
    try:
        from cloak_config import CHARS_PER_TOKEN, NUM_CTX, NUM_PREDICT, PROMPT_RESERVE_TOKENS
    except Exception:
        return 24000
    tokens = max(2048, int(NUM_CTX) - int(NUM_PREDICT) - int(PROMPT_RESERVE_TOKENS) - 256)
    return int(tokens * float(CHARS_PER_TOKEN))


def strip_chat(raw: str) -> str:
    """Drop model-internal think tags. Keep newlines and Markdown intact."""
    text = raw or ""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


def thread_messages(chat_history, query, limit=24, clip=None, pdf_names=None):
    """Build a native chat-template message list, packing newest turns first."""
    system = SystemMessage(content=system_prompt(pdf_names=pdf_names))
    query = (query or "").strip()
    if len(query) > 32000:
        query = query[:32000] + "…"

    budget = max(2000, history_char_budget() - len(system.content) - len(query))
    prior = list(chat_history or [])[-limit:]
    kept = []
    used = 0
    for item in reversed(prior):
        text = (getattr(item, "content", None) or "").strip()
        if not text:
            continue
        remaining = budget - used
        if remaining <= 240:
            break
        if len(text) > remaining:
            cut = text[:remaining].rsplit("\n", 1)[0]
            if len(cut) < 80:
                cut = text[:remaining]
            text = cut.rstrip() + "\n…"
        used += len(text)
        if isinstance(item, HumanMessage) or getattr(item, "type", "") == "human":
            kept.append(HumanMessage(content=text))
        else:
            kept.append(AIMessage(content=text))
    kept.reverse()
    return [system, *kept, HumanMessage(content=query)]


def try_direct_math(query: str):
    """Exact arithmetic without spending a generation when the query is just math."""
    match = MATH_QUERY_RE.match(query or "")
    if not match:
        return None
    expr = match.group(1).replace("×", "*").replace("÷", "/").replace("x", "*")
    if not re.search(r"[+\-*/]", expr):
        return None
    ok, text = eval_expression(expr)
    return text if ok else None


def try_direct_clock(query: str):
    if not CLOCK_QUERY_RE.match((query or "").strip()):
        return None
    return format_clock()
