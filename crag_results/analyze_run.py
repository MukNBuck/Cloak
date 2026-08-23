"""Offline diagnosis of a generate JSON: metric vs model vs retrieval."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_crag import coverage, gold_in_text, normalize, pred_matches_gold, tokens

IDK_RE = re.compile(r"^\s*(i\s+don'?t\s+know|invalid)\b", re.I)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def gold_nums(text):
    return [float(x) for x in NUM_RE.findall(text or "")]


def classify(row):
    gold = row.get("gold") or ""
    pred = row.get("pred") or ""
    abstain = bool(row.get("abstain"))
    retr = row.get("retrieval_hit")
    gen = bool(row.get("generate_hit"))
    pred_n = normalize(pred)
    gold_n = normalize(gold)
    cov_g_in_p = coverage(gold, pred) if pred else 0.0
    cov_p_in_g = coverage(pred, gold) if pred else 0.0
    idk = bool(IDK_RE.match(pred))
    loose = pred_matches_gold([gold], pred) if pred and not abstain else False
    gnums = gold_nums(gold)
    pnums = gold_nums(pred)
    num_ok = bool(gnums) and any(any(abs(a - b) <= 0.51 for b in pnums) for a in gnums)

    if abstain:
        bucket = "false_premise_should_abstain"
        if gen:
            bucket = "false_premise_ok"
        elif not idk:
            bucket = "false_premise_answered"
        return bucket, cov_g_in_p, cov_p_in_g, loose, num_ok

    if retr is False:
        if gen:
            return "gen_ok_without_retr", cov_g_in_p, cov_p_in_g, loose, num_ok
        return "blocked_by_retrieval", cov_g_in_p, cov_p_in_g, loose, num_ok

    if gen:
        return "retr_and_gen_ok", cov_g_in_p, cov_p_in_g, loose, num_ok

    if idk:
        return "retrieved_then_refused", cov_g_in_p, cov_p_in_g, loose, num_ok
    if gold_n and gold_n in pred_n:
        return "metric_false_negative", cov_g_in_p, cov_p_in_g, loose, num_ok
    if loose:
        return "strict_metric_miss_loose_hit", cov_g_in_p, cov_p_in_g, loose, num_ok
    if num_ok and gnums:
        return "number_ok_wording_miss", cov_g_in_p, cov_p_in_g, loose, num_ok
    if cov_g_in_p >= 0.5:
        return "partial_overlap", cov_g_in_p, cov_p_in_g, loose, num_ok
    if cov_p_in_g >= 0.8:
        return "short_correctish", cov_g_in_p, cov_p_in_g, loose, num_ok
    return "wrong_or_incomplete", cov_g_in_p, cov_p_in_g, loose, num_ok


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "local_generate40.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    buckets = Counter()
    by_type = defaultdict(Counter)
    details = []
    for row in rows:
        bucket, c1, c2, loose, num_ok = classify(row)
        buckets[bucket] += 1
        by_type[row.get("question_type") or "?"][bucket] += 1
        details.append(
            {
                "type": row.get("question_type"),
                "intent": row.get("intent"),
                "bucket": bucket,
                "retr": row.get("retrieval_hit"),
                "gen": row.get("generate_hit"),
                "gate": row.get("passed_gate"),
                "conf": row.get("confidence"),
                "chunks": row.get("chunks"),
                "cov_gold_in_pred": round(c1, 3),
                "cov_pred_in_gold": round(c2, 3),
                "loose": loose,
                "num_ok": num_ok,
                "q": (row.get("query") or "")[:160],
                "gold": (row.get("gold") or "")[:180],
                "pred": (row.get("pred") or "")[:220],
            }
        )

    n = len(rows)
    official = sum(1 for r in rows if r.get("generate_hit")) / n
    loose_n = 0
    for row in rows:
        if row.get("abstain"):
            loose_n += int(bool(IDK_RE.match(row.get("pred") or "")))
        else:
            loose_n += int(pred_matches_gold([row.get("gold") or ""], row.get("pred") or ""))
    ceiling = sum(
        1
        for r in rows
        if r.get("abstain")
        or r.get("retrieval_hit")
        or r.get("generate_hit")
    ) / n

    out = {
        "file": str(path),
        "n": n,
        "official_generate_hit": round(official, 3),
        "loose_generate_hit": round(loose_n / n, 3),
        "retrieval_ceiling_if_perfect_gen": round(ceiling, 3),
        "buckets": dict(buckets),
        "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "details": details,
    }
    dest = path.with_name(path.stem + "_diagnosis.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "details"}, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
