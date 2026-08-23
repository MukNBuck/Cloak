"""Turn router: chat vs RAG before retrieval.

Keeps Cloak feeling like a general local assistant when no evidence is needed,
and document-grounded when the user is asking about included PDFs.
"""

from __future__ import annotations

import re

CHAT_RE = re.compile(
    r"^\s*("
    r"thanks|thank you|thx|ty|"
    r"ok|okay|got it|cool|nice|great|"
    r"hi|hello|hey|"
    r"who are you|what (are|can) you|"
    r"explain (that|this|your answer)( (again|simpler|simply|like i.{0,12}))?|"
    r"say that (again|differently)|rephrase|"
    r"continue|go on|"
    r"never ?mind|nm"
    r")\b",
    re.I,
)

NEEDS_EVIDENCE_RE = re.compile(
    r"\b("
    r"according to|in the (pdf|document|doc|contract|file|pages?)|"
    r"from the (pdf|document|doc|contract|file)|"
    r"cite|citation|source|quote|"
    r"what does .{1,40} say|"
    r"find (the |where )|"
    r"where (is|does|in)|"
    r"page \d+"
    r")\b",
    re.I,
)

FACTUAL_RE = re.compile(
    r"\b("
    r"who|what|when|where|why|how|which|"
    r"list|name|define|definition|"
    r"how many|how much|difference|compare|versus|\bvs\b|"
    r"summarize|summary|overview|tl;?dr|"
    r"extract|enumerate"
    r")\b",
    re.I,
)

INTENT_SUMMARIZE_RE = re.compile(
    r"\b(summarize|summary|overview|tl;?dr|recap|walk me through)\b", re.I
)
INTENT_COMPARE_RE = re.compile(
    r"\b(compare|versus|\bvs\.?\b|difference between|differ from|contrast)\b", re.I
)
INTENT_EXTRACT_RE = re.compile(
    r"\b("
    r"list all|list the|extract|enumerate|tabulate|all of the|"
    r"name the|which (clubs?|teams?|songs?|movies?|films?|awards?|members?|cities)|"
    r"who (are|were) (the |in )|"
    r"what (songs?|teams?|clubs?|movies?|awards?|members?)"
    r")\b",
    re.I,
)
INTENT_COMPUTE_RE = re.compile(
    r"\b("
    r"how many years|how much (older|younger|more|less)|age difference|"
    r"percentage|percent|what percent|how long would|area of|"
    r"times (as )?(many|much)|compared to|how many more|how many (of|times)"
    r")\b",
    re.I,
)

ROUTE_LABELS = {
    "chat": "Chat",
    "rag_lookup": "Document lookup",
    "rag_summarize": "Summarize documents",
    "rag_compare": "Compare documents",
    "rag_compute": "Compute from documents",
    "rag_extract": "Extract from documents",
}


def hop_subqueries(query: str) -> list[str]:
    """Split nested fact questions into an inner search and a shorter outer search.

    Example: "who was the goalkeeper of the team that won the final"
    -> "team that won the final", "goalkeeper team"
    """
    text = (query or "").strip().rstrip("?").strip()
    if not text:
        return []
    extras: list[str] = []

    match = re.search(
        r"\b(?:who|what|which)\s+(?:was|is|were|are|did)?\s*(?:the\s+)?"
        r"(.+?)\s+of\s+(?:the\s+)?(.+?)\s+that\s+(.+)$",
        text,
        re.I,
    )
    if match:
        outer, entity, clause = (group.strip() for group in match.groups())
        if entity and clause:
            extras.append(f"{entity} that {clause}")
        if outer and entity:
            extras.append(f"{outer} {entity}")

    match = re.search(
        r"\b(?:what is the name of|name of|who is|who was)\s+(?:the\s+)?"
        r"(.+?)\s+who\s+(.+)$",
        text,
        re.I,
    )
    if match:
        role, clause = (group.strip() for group in match.groups())
        if role and clause:
            extras.append(f"{role} who {clause}")
        if role:
            extras.append(role)

    seen = {text.lower()}
    unique = []
    for item in extras:
        key = item.lower()
        if key in seen or len(item) < 6:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:2]


def classify_intent(query: str) -> str:
    text = query or ""
    if INTENT_SUMMARIZE_RE.search(text):
        return "summarize"
    if INTENT_COMPUTE_RE.search(text):
        return "compute"
    if INTENT_COMPARE_RE.search(text):
        return "compare"
    if INTENT_EXTRACT_RE.search(text):
        return "extract"
    return "lookup"


def is_chitchat(query: str) -> bool:
    text = (query or "").strip()
    if not text:
        return True
    if len(text) > 160:
        return False
    return bool(CHAT_RE.search(text))


def needs_evidence(query: str) -> bool:
    return bool(NEEDS_EVIDENCE_RE.search(query or ""))


def looks_factual(query: str) -> bool:
    return bool(FACTUAL_RE.search(query or ""))


def decide_route(
    query: str,
    *,
    doc_count: int = 0,
    force_mode: str | None = None,
) -> tuple[str, str]:
    """Return (route, reason).

    force_mode: auto | chat | rag (None treated as auto)
    """
    force = (force_mode or "auto").strip().lower() or "auto"
    docs = max(0, int(doc_count or 0))
    text = (query or "").strip()

    if force == "chat":
        return "chat", "user_forced_chat"
    if force == "rag":
        intent = classify_intent(text)
        return _rag_route_for_intent(intent), "user_forced_rag"

    if docs <= 0:
        return "chat", "no_docs"

    if is_chitchat(text):
        return "chat", "chitchat"

    intent = classify_intent(text)
    if intent in {"summarize", "compare", "compute", "extract"}:
        return _rag_route_for_intent(intent), f"intent:{intent}"

    if needs_evidence(text) or looks_factual(text):
        return "rag_lookup", "factual_with_docs"

    # Short ambiguous messages with docs loaded: prefer chat so RAG isn't forced.
    if len(text) < 48 and not needs_evidence(text):
        return "chat", "short_ambiguous"

    return "rag_lookup", "docs_default"


def _rag_route_for_intent(intent: str) -> str:
    return {
        "summarize": "rag_summarize",
        "compare": "rag_compare",
        "compute": "rag_compute",
        "extract": "rag_extract",
        "lookup": "rag_lookup",
    }.get(intent, "rag_lookup")


def route_label(route: str) -> str:
    return ROUTE_LABELS.get(route, route)
