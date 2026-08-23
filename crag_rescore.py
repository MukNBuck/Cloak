"""Re-score a frozen generate JSON with number-aware metrics.

Official generate_hit is token overlap. CRAG golds are often a bare digit or a
long sentence, so that overlap both over-credits (name appears in a wrong
answer) and under-credits (222,104,681 vs 222104681; "three" vs 3).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_crag import coverage, gold_in_text, is_abstain, normalize, pred_matches_gold, tokens

IDK_RE = re.compile(r"^\s*(i\s+don'?t\s+know|invalid)\b", re.I)
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
WORD_RE = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")\b", re.I)


def collapse_grouped_digits(text):
    prev = None
    out = text or ""
    while prev != out:
        prev = out
        out = re.sub(r"(\d),(\d{3})\b", r"\1\2", out)
    return out


def fold_number_words(text):
    return WORD_RE.sub(lambda m: WORD_NUMBERS[m.group(0).lower()], text or "")


def fold_text(text):
    return fold_number_words(collapse_grouped_digits(text or ""))


def extract_numbers(text):
    folded = fold_text(text)
    found = []
    for raw in re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", folded):
        try:
            found.append(float(raw) if "." in raw else int(raw))
        except ValueError:
            continue
    return found


def _is_year(n):
    try:
        v = float(n)
    except (TypeError, ValueError):
        return False
    return v.is_integer() and 1900 <= int(v) <= 2100


def distinctive_numbers(text, query=""):
    """Drop years and numbers already in the question so 2022/2049 don't 'match'."""
    skip = {float(n) for n in extract_numbers(query)}
    out = []
    for n in extract_numbers(text):
        if float(n) in skip:
            continue
        if _is_year(n):
            continue
        out.append(n)
    return out


def numbers_agree(gold, pred, query="", tol=0.51):
    golds = distinctive_numbers(gold, query)
    if not golds:
        return None
    preds = extract_numbers(pred)
    if not preds:
        return False
    return any(any(abs(float(g) - float(p)) <= tol for p in preds) for g in golds)


def numeric_fold_hit(golds, pred):
    folded_golds = [fold_text(g) for g in golds]
    folded_pred = fold_text(pred)
    return pred_matches_gold(folded_golds, folded_pred) or any(
        gold_in_text([fold_text(g)], folded_pred)[0] for g in golds
    )


def short_entity_hit(gold, pred, query=""):
    """Bare golds like 'the weeknd' / '3' — substring of a long gold is not required."""
    gtoks = tokens(fold_text(gold))
    if not gtoks:
        nums = distinctive_numbers(gold, query) or extract_numbers(gold)
        if len(nums) == 1:
            return bool(numbers_agree(gold, pred, query)) and not IDK_RE.match(pred or "")
        return None
    if len(gtoks) > 2:
        return None
    pred_n = normalize(fold_text(pred))
    gold_n = normalize(fold_text(gold))
    if not gold_n or not pred_n:
        return False
    if gold_n not in pred_n and gtoks[0] not in tokens(fold_text(pred)):
        return False
    agree = numbers_agree(gold, pred, query)
    if agree is False:
        return False
    return True


def calibrated_hit(row):
    gold = row.get("gold") or ""
    pred = row.get("pred") or ""
    query = row.get("query") or ""
    if row.get("abstain"):
        return bool(IDK_RE.match(pred))
    if not pred or IDK_RE.match(pred):
        return False
    agree = numbers_agree(gold, pred, query)
    short = short_entity_hit(gold, pred, query)
    official = bool(row.get("generate_hit"))
    fold = numeric_fold_hit([gold], pred)
    if agree is False:
        return False
    if short is False:
        return False
    if agree is True:
        return True
    if fold:
        return True
    if official and agree is not False and short is not False:
        return True
    return False


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "crag_results/local_generate40.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    buckets = Counter()
    by_type = defaultdict(Counter)
    disagreements = []
    details = []
    for row in rows:
        gold = row.get("gold") or ""
        pred = row.get("pred") or ""
        official = bool(row.get("generate_hit"))
        fold = False if row.get("abstain") else numeric_fold_hit([gold], pred)
        if row.get("abstain"):
            fold = bool(IDK_RE.match(pred))
        query = row.get("query") or ""
        agree = numbers_agree(gold, pred, query)
        calib = calibrated_hit(row)
        if official and not calib:
            tag = "metric_false_positive"
        elif calib and not official:
            tag = "metric_false_negative"
        elif official:
            tag = "true_positive"
        else:
            tag = "true_negative"
        buckets[tag] += 1
        by_type[row.get("question_type") or "?"][tag] += 1
        item = {
            "type": row.get("question_type"),
            "intent": row.get("intent"),
            "retr": row.get("retrieval_hit"),
            "corpus": row.get("corpus_hit"),
            "official": official,
            "numeric_fold": fold,
            "numbers_agree": agree,
            "calibrated": calib,
            "tag": tag,
            "q": (row.get("query") or "")[:140],
            "gold": gold[:160],
            "pred": (pred or "")[:200],
        }
        details.append(item)
        if tag.startswith("metric_"):
            disagreements.append(item)

    n = len(rows)
    official_n = sum(1 for r in rows if r.get("generate_hit"))
    calib_n = sum(1 for d in details if d["calibrated"])
    fold_n = sum(1 for d in details if d["numeric_fold"])
    out = {
        "file": str(path),
        "n": n,
        "official_generate_hit": round(official_n / n, 3),
        "numeric_fold_hit": round(fold_n / n, 3),
        "calibrated_hit": round(calib_n / n, 3),
        "buckets": dict(buckets),
        "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "disagreements": disagreements,
        "details": details,
    }
    dest = path.with_name(path.stem + "_rescore.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k not in ("details", "disagreements")}, indent=2))
    print("\nDisagreements:")
    for item in disagreements:
        print(f"  [{item['tag']}] {item['type']} | {item['q']}")
        print(f"    gold: {item['gold']}")
        print(f"    pred: {item['pred']}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
