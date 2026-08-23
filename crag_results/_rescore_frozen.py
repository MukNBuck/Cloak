import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_crag import is_abstain, pred_matches_gold

data = json.loads(Path("crag_results/local_generate40_v5.json").read_text(encoding="utf-8"))
hits = 0
changed = []
for row in data["rows"]:
    gold = row.get("gold") or ""
    pred = row.get("pred") or ""
    old = bool(row.get("generate_hit"))
    if is_abstain([gold]):
        new = bool(re.match(r"^\s*(i\s+don'?t\s+know|invalid)\b", pred, re.I))
    else:
        new = pred_matches_gold([gold], pred)
    hits += int(new)
    if new != old:
        changed.append(
            {
                "old": old,
                "new": new,
                "type": row.get("question_type"),
                "q": row.get("query"),
                "gold": gold,
                "pred": pred,
            }
        )
print("frozen new generate_hit", hits, round(hits / 40, 3))
print("changed", len(changed))
for item in changed:
    print(" old", item["old"], "new", item["new"], item["type"])
    print("  Q", (item["q"] or "")[:100])
    print("  G", (item["gold"] or "")[:100])
    print("  P", (item["pred"] or "")[:120])
    print()
