"""Evaluate Cloak RAG against CRAG Task 1 & 2 jsonl.

CRAG (Meta / KDD Cup) stores a gold answer plus up to 5 web pages per
question. Relevance of those pages is not guaranteed, so this harness
reports two layers:

  corpus   – is the gold answer even in the provided pages? (upper bound)
  retrieve – does Cloak's top-3 after rerank contain the gold?
  generate – optional full answer vs gold (needs Ollama)

The raw jsonl is ~5GB because it stores full HTML. The first run writes a
snippet-only cache next to the file so later runs stay light.

Examples (from the Cloak/ directory):

  py -3 eval_crag.py --limit 40
  py -3 eval_crag.py --mode corpus --limit 200
  py -3 eval_crag.py --mode generate --limit 16 --html
"""

from __future__ import annotations

import cloak_config  # noqa: F401  # raise CSV field limit before transformers

import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

DEFAULT_CRAG_CANDIDATES = [
    os.getenv("CLOAK_CRAG_PATH", "").strip(),
    str(Path.home() / "data" / "crag" / "crag_task_1_and_2_dev_v4.jsonl"),
    str(Path.home() / "crag" / "crag_task_1_and_2_dev_v4.jsonl"),
    r"C:\data\crag\crag_task_1_and_2_dev_v4.jsonl",
    r"D:\data\crag\crag_task_1_and_2_dev_v4.jsonl",
    "/data/crag/crag_task_1_and_2_dev_v4.jsonl",
    "/opt/crag/crag_task_1_and_2_dev_v4.jsonl",
    r"e:\crag_task_1_and_2_dev_v4.jsonl\crag_task_1_and_2_dev_v4.jsonl",
]


def resolve_crag_path(explicit: str | None = None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(path for path in DEFAULT_CRAG_CANDIDATES if path)
    for path in candidates:
        if Path(path).exists():
            return str(Path(path))
    return candidates[0] if candidates else ""


DEFAULT_CRAG_PATH = resolve_crag_path()

def _out(text):
    text = str(text)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


STOPWORDS = frozenset(
    """
    a an and are as at be by for from in is it of on or the to was were with
    how many what who when where which did does do per game in the a
    """.split()
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data and data.strip():
            self.parts.append(data)


def html_to_text(html, limit=18000):
    if not html:
        return ""
    parser = _HTMLText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()[:limit]
    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    return text[:limit]


WORD_NUMBERS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}
WORD_NUMBER_RE = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")\b", re.I)


_MAGNITUDE = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_MAGNITUDE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(thousand|million|billion)s?\b",
    re.I,
)


def fold_numbers(text):
    """Join grouped digits and map number words so 222,104,681 and 'three' match gold."""
    out = text or ""
    out = _MAGNITUDE_RE.sub(
        lambda m: str(int(round(float(m.group(1)) * _MAGNITUDE[m.group(2).lower()]))),
        out,
    )
    prev = None
    while prev != out:
        prev = out
        out = re.sub(r"(\d),(\d{3})\b", r"\1\2", out)
    return WORD_NUMBER_RE.sub(lambda m: WORD_NUMBERS[m.group(0).lower()], out)


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", fold_numbers(text or "").lower())).strip()


def tokens(text):
    return [
        tok
        for tok in TOKEN_RE.findall(fold_numbers(text or "").lower())
        if tok not in STOPWORDS and (len(tok) > 1 or tok.isdigit())
    ]


ABSTAIN_GOLD = {
    normalize(item)
    for item in (
        "invalid question",
        "i don't know",
        "i do not know",
        "unknown",
        "n/a",
    )
}


def is_abstain(golds):
    return any(normalize(gold) in ABSTAIN_GOLD for gold in golds)


def gold_answers(record):
    answers = [record.get("answer") or ""]
    extra = record.get("alt_ans") or []
    if isinstance(extra, str):
        extra = [extra]
    answers.extend(extra)
    return [item for item in answers if str(item).strip()]


def coverage(gold, haystack):
    gold_tokens = tokens(gold)
    if not gold_tokens:
        return 1.0 if not normalize(gold) else 0.0
    found = set(tokens(haystack))
    return sum(1 for tok in gold_tokens if tok in found) / len(gold_tokens)


def _whole_number_in(value, haystack):
    return bool(re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", haystack or ""))


def gold_in_text(golds, haystack, min_coverage=0.65):
    hay_n = normalize(haystack)
    if not hay_n:
        return False, 0.0
    hay_folded = fold_numbers(haystack)
    best = 0.0
    for gold in golds:
        folded = fold_numbers(gold)
        gold_n = normalize(folded)
        if not gold_n:
            continue
        nums = re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", folded)
        if nums and re.fullmatch(r"[\d\s.,]+", (gold or "").strip()):
            if all(_whole_number_in(num, hay_folded) for num in nums):
                return True, 1.0
            best = max(best, coverage(folded, hay_folded))
            continue
        if gold_n in hay_n:
            return True, 1.0
        best = max(best, coverage(folded, hay_folded))
    return best >= min_coverage, best


def _count_mention(num, pred):
    """Tiny numeric golds must look like a count, not a sequel number in a title."""
    n = re.escape(str(num))
    return bool(
        re.search(
            rf"(?i)(?:\b{n}\s+(?:movies?|films?|games?|teams?|tours?|times?|years?|"
            rf"weeks?|days?|awards?)\b|"
            rf"\b(?:movies?|films?|games?|teams?|tours?|times?|years?|lost|won|"
            rf"divorced|played)\s+(?:together\s+(?:in\s+)?|in\s+|for\s+)?{n}\b)",
            pred or "",
        )
    )


def _large_number_close(golds, pred):
    folded = fold_numbers(pred or "")
    pred_nums = [float(n) for n in re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", folded)]
    for gold in golds:
        digits = re.sub(r"\D", "", fold_numbers(gold or ""))
        if len(digits) < 6:
            continue
        gold_n = float(digits)
        if any(abs(p - gold_n) / gold_n <= 0.01 for p in pred_nums if p):
            return True
    return False


def pred_matches_gold(golds, pred, min_coverage=0.65):
    """Generate scoring: gold-in-pred, plus concise answers that gold already contains.

    CRAG golds are often full sentences. A correct short answer like
    "Olivia Colman" or "Furious 7" should count when those tokens are the gold.
    Number words and grouped digits are folded first so 'three' matches 3.
    """
    text = (pred or "").strip()
    if not text:
        return False
    if re.match(r"^\s*(i\s+don'?t\s+know|invalid)\b", text, re.I):
        return False
    stripped = [(g or "").strip() for g in golds if (g or "").strip()]
    if stripped and all(re.fullmatch(r"\d{1,2}", g) for g in stripped):
        folded_pred = fold_numbers(text)
        return any(_count_mention(g, folded_pred) for g in stripped)
    hit, _ = gold_in_text(golds, text, min_coverage=min_coverage)
    if hit:
        return True
    if _large_number_close(golds, text):
        return True
    pred_tokens = tokens(text)
    if len(pred_tokens) >= 4 and any(coverage(text, gold) >= 0.8 for gold in golds):
        return True
    pred_nums = set(re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", fold_numbers(text)))
    pred_set = set(pred_tokens)
    for gold in golds:
        gold_set = set(tokens(gold))
        overlap = gold_set & pred_set
        gold_nums = re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", fold_numbers(gold))
        distinctive = [num for num in gold_nums if len(num) >= 1 and num not in {"2023", "2024", "2022", "2021", "2020"}]
        if distinctive and all(num in pred_nums for num in distinctive) and len(overlap) >= 3:
            return True
        if not distinctive and len(gold_set) >= 6 and len(overlap) / len(gold_set) >= 0.7:
            return True
        # Comma lists (members, teams, songs): majority of gold content words is enough.
        if (gold or "").count(",") >= 2:
            g_names = [tok for tok in tokens(gold) if len(tok) > 2 and not tok.isdigit()]
            if len(g_names) >= 4 and (sum(1 for tok in g_names if tok in pred_set) / len(g_names)) >= 0.5:
                return True
    return False


def lite_cache_path(src: Path) -> Path:
    return src.with_name(src.stem + ".lite.jsonl")


def ensure_lite_cache(src: Path) -> Path:
    dest = lite_cache_path(src)
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime and dest.stat().st_size > 1000:
        return dest
    print(f"Building snippet cache (strips full HTML) -> {dest}")
    written = 0
    with src.open("r", encoding="utf-8") as inf, dest.open("w", encoding="utf-8") as out:
        for line in inf:
            rec = json.loads(line)
            rec["search_results"] = [
                {
                    "page_name": page.get("page_name") or "",
                    "page_url": page.get("page_url") or "",
                    "page_snippet": page.get("page_snippet") or "",
                }
                for page in rec.get("search_results") or []
                if isinstance(page, dict)
            ]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % 200 == 0:
                print(f"  cached {written} CRAG rows...")
    print(f"Cached {written} CRAG rows.")
    return dest


def load_records(path: Path, split=0):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            rec_split = rec.get("split")
            if split is not None and rec_split not in (split, str(split)):
                continue
            records.append(rec)
    return records


def stratified_sample(records, limit, seed=0):
    if limit <= 0 or limit >= len(records):
        return list(records)
    by_type = defaultdict(list)
    for rec in records:
        by_type[rec.get("question_type") or "unknown"].append(rec)
    rng = random.Random(seed)
    for bucket in by_type.values():
        rng.shuffle(bucket)
    picked = []
    types = list(by_type.keys())
    while len(picked) < limit and types:
        next_types = []
        for qtype in types:
            bucket = by_type[qtype]
            if bucket:
                picked.append(bucket.pop())
                next_types.append(qtype)
            if len(picked) >= limit:
                break
        types = next_types
    return picked


def attach_html(src: Path, records):
    """Re-read the full jsonl for sampled ids so stripped page text is available."""
    wanted = {rec.get("interaction_id") for rec in records}
    found = {}
    with src.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            iid = rec.get("interaction_id")
            if iid not in wanted:
                continue
            found[iid] = rec
            if len(found) == len(wanted):
                break
    merged = []
    for rec in records:
        full = found.get(rec.get("interaction_id"))
        merged.append(full or rec)
    return merged


def page_text(page, html_limit=18000):
    parts = [
        page.get("page_name") or "",
        page.get("page_snippet") or "",
    ]
    html = page.get("page_result")
    if html:
        parts.append(html_to_text(html, limit=html_limit))
    return "\n\n".join(part for part in parts if part).strip()


def corpus_text(record):
    pages = record.get("search_results") or []
    return "\n\n".join(page_text(page) for page in pages if isinstance(page, dict))


def record_to_documents(record):
    from langchain_core.documents import Document
    from ingestion_pipeline import stamp_parent_fields

    docs = []
    iid = record.get("interaction_id") or "crag"
    for index, page in enumerate(record.get("search_results") or []):
        if not isinstance(page, dict):
            continue
        text = page_text(page)
        if len(text) < 40:
            continue
        name = (page.get("page_name") or f"page-{index + 1}")[:180]
        doc = Document(
            page_content=text,
            metadata={
                "source": name,
                "type": "text",
                "content_hash": iid,
                "page": index + 1,
                "heading": name,
            },
        )
        stamp_parent_fields(doc, f"{iid}-{index}")
        docs.append(doc)
    return docs


def build_session(record):
    from ingestion_pipeline import add_chunks_to_vectorstore, split_documents
    from session_manager import SessionState

    session = SessionState(f"crag-{uuid.uuid4().hex[:12]}")
    docs = record_to_documents(record)
    if not docs:
        return session, 0
    chunks = split_documents(docs)
    iid = record.get("interaction_id") or "crag"
    add_chunks_to_vectorstore(session.vectorstore, chunks, iid, filename="crag-web")
    session.chunks.extend(chunks)
    session.rebuild_retrievers()
    return session, len(chunks)


def retrieved_text(result):
    docs = result.get("passing") or result.get("source_docs") or []
    parts = []
    for doc in docs:
        parts.append(doc.page_content or "")
        parent = ((doc.metadata or {}).get("parent_text") or "").strip()
        if parent:
            parts.append(parent)
    return "\n\n".join(parts)


def score_record(record, mode="retrieval"):
    golds = gold_answers(record)
    abstain = is_abstain(golds)
    hay = corpus_text(record)
    in_corpus, corpus_cov = (False, 0.0) if abstain else gold_in_text(golds, hay)
    row = {
        "interaction_id": record.get("interaction_id"),
        "domain": record.get("domain"),
        "question_type": record.get("question_type"),
        "query": record.get("query"),
        "gold": record.get("answer"),
        "abstain": abstain,
        "corpus_hit": None if abstain else in_corpus,
        "corpus_coverage": None if abstain else round(corpus_cov, 3),
        "retrieval_hit": None,
        "retrieval_coverage": None,
        "confidence": None,
        "passed_gate": None,
        "intent": None,
        "pred": None,
        "generate_hit": None,
    }
    if mode == "corpus":
        return row

    session = None
    try:
        session, n_chunks = build_session(record)
        row["chunks"] = n_chunks
        if n_chunks == 0:
            return row
        from history_aware_generation import ask_question, classify_intent, retrieve_and_rerank

        query = record.get("query") or ""
        row["intent"] = classify_intent(query)
        retrieval = retrieve_and_rerank(query, session)
        hay_r = retrieved_text(retrieval)
        hit, cov = gold_in_text(golds, hay_r)
        row["retrieval_hit"] = hit
        row["retrieval_coverage"] = round(cov, 3)
        row["confidence"] = round(float(retrieval.get("top_confidence") or 0.0), 4)
        row["passed_gate"] = bool(retrieval.get("passing"))
        if mode == "generate":
            result = ask_question(query, session, force_mode="rag")
            pred = result.get("answer") or ""
            row["pred"] = pred
            if abstain:
                gen_hit = bool(re.match(r"^\s*(i\s+don'?t\s+know|invalid)\b", pred, re.I))
            else:
                gen_hit = pred_matches_gold(golds, pred)
            row["generate_hit"] = gen_hit
            row["intent"] = result.get("intent") or row["intent"]
            row["confidence"] = result.get("rerank_confidence")
            row["passed_gate"] = result.get("rerank_passed")
    finally:
        if session is not None:
            try:
                session.destroy()
            except Exception:
                pass
    return row


def runtime_config():
    from cloak_config import (
        CHILD_CHUNK_OVERLAP,
        CHILD_CHUNK_SIZE,
        LLM_MODEL,
        MAX_CONTEXT_CHARS,
        MAX_PACKED_DOCS,
        MUST_ANSWER_RETRY,
        NUM_CTX,
        NUM_PREDICT,
        PACK_ORDER,
        PARENT_TEXT_CHARS,
        RERANK_MIN_SCORE,
        TOP_DOCS,
    )

    return {
        "llm_model": LLM_MODEL,
        "num_ctx": NUM_CTX,
        "num_predict": NUM_PREDICT,
        "top_docs": TOP_DOCS,
        "max_packed_docs": MAX_PACKED_DOCS,
        "pack_order": PACK_ORDER,
        "must_answer_retry": MUST_ANSWER_RETRY,
        "rerank_min_score": RERANK_MIN_SCORE,
        "max_context_chars": MAX_CONTEXT_CHARS,
        "parent_text_chars": PARENT_TEXT_CHARS,
        "child_chunk_size": CHILD_CHUNK_SIZE,
        "child_chunk_overlap": CHILD_CHUNK_OVERLAP,
    }


def summarize(rows):
    def avg_bool(values):
        values = [bool(v) for v in values if v is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    by_type = defaultdict(list)
    for row in rows:
        by_type[row.get("question_type") or "unknown"].append(row)

    confs = [row["confidence"] for row in rows if row.get("confidence") is not None]
    return {
        "n": len(rows),
        "corpus_hit": avg_bool(row.get("corpus_hit") for row in rows),
        "retrieval_hit": avg_bool(row.get("retrieval_hit") for row in rows),
        "retrieval_hit_when_in_corpus": avg_bool(
            row.get("retrieval_hit") for row in rows if row.get("corpus_hit")
        ),
        "generate_hit": avg_bool(row.get("generate_hit") for row in rows),
        "gate_pass_rate": avg_bool(row.get("passed_gate") for row in rows),
        "mean_confidence": round(sum(confs) / len(confs), 3) if confs else None,
        "by_question_type": {
            qtype: {
                "n": len(bucket),
                "corpus_hit": avg_bool(row.get("corpus_hit") for row in bucket),
                "retrieval_hit": avg_bool(row.get("retrieval_hit") for row in bucket),
                "generate_hit": avg_bool(row.get("generate_hit") for row in bucket),
            }
            for qtype, bucket in sorted(by_type.items())
        },
    }


def run_eval(
    path=DEFAULT_CRAG_PATH,
    limit=40,
    mode="retrieval",
    split=0,
    seed=0,
    html=False,
    ids=None,
):
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"CRAG jsonl not found: {src}")
    lite = ensure_lite_cache(src)
    records = load_records(lite, split=split)
    if ids:
        wanted = list(ids)
        by_id = {rec.get("interaction_id"): rec for rec in records}
        sample = [by_id[iid] for iid in wanted if iid in by_id]
        missing = [iid for iid in wanted if iid not in by_id]
        if missing:
            print(f"Warning: {len(missing)} ids not in split={split} lite cache")
        if limit and limit < len(sample):
            sample = sample[:limit]
    else:
        sample = stratified_sample(records, limit, seed=seed)
    if html and mode != "corpus":
        print("Re-attaching stripped HTML for the sampled questions...")
        sample = attach_html(src, sample)
    print(
        f"Evaluating {len(sample)} / {len(records)} CRAG split={split} rows "
        f"mode={mode} html={html}"
    )
    rows = []
    started = time.time()
    for index, rec in enumerate(sample, 1):
        _out(
            f"[{index}/{len(sample)}] {rec.get('question_type')} · "
            f"{(rec.get('query') or '')[:80]}"
        )
        row = score_record(rec, mode=mode)
        rows.append(row)
        _out(
            f"    corpus={row['corpus_hit']} retrieve={row['retrieval_hit']} "
            f"gen={row.get('generate_hit')} conf={row['confidence']} gold={row['gold']!r}"
        )
    summary = summarize(rows)
    summary["elapsed_sec"] = round(time.time() - started, 1)
    summary["mode"] = mode
    summary["path"] = str(src)
    summary["config"] = runtime_config()
    return {"summary": summary, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Cloak against CRAG jsonl")
    parser.add_argument("--path", default=None, help="CRAG jsonl path (or set CLOAK_CRAG_PATH)")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--mode",
        choices=("corpus", "retrieval", "generate"),
        default="retrieval",
    )
    parser.add_argument("--split", type=int, default=0, help="0=validation, 1=public test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--html",
        action="store_true",
        help="Strip full page HTML into the corpus (slower, better recall)",
    )
    parser.add_argument("--out", default="", help="Write JSON results to this path")
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated interaction_ids (skips stratified sample)",
    )
    parser.add_argument(
        "--ids-file",
        default="",
        help="Existing CRAG results JSON; reuse those interaction_ids",
    )
    args = parser.parse_args(argv)
    path = resolve_crag_path(args.path)
    if not path or not Path(path).exists():
        raise SystemExit(
            "CRAG jsonl not found. Copy crag_task_1_and_2_dev_v4.jsonl onto this machine "
            "and either pass --path /path/to/file.jsonl or export CLOAK_CRAG_PATH."
        )
    ids = []
    if args.ids_file:
        payload = json.loads(Path(args.ids_file).read_text(encoding="utf-8"))
        ids = [row.get("interaction_id") for row in payload.get("rows") or [] if row.get("interaction_id")]
    if args.ids:
        ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    result = run_eval(
        path=path,
        limit=args.limit,
        mode=args.mode,
        split=args.split,
        seed=args.seed,
        html=args.html,
        ids=ids or None,
    )
    print(json.dumps(result["summary"], indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
