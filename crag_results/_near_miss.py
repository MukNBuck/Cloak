import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_crag import coverage, is_abstain, pred_matches_gold, tokens

data = json.loads(Path("crag_results/local_generate40_v4.json").read_text(encoding="utf-8"))
print("official", data["summary"]["generate_hit"])
for row in data["rows"]:
    gold = row.get("gold") or ""
    pred = row.get("pred") or ""
    if row.get("generate_hit") or is_abstain([gold]):
        continue
    cov = coverage(gold, pred)
    rev = coverage(pred, gold)
    print(round(cov, 3), round(rev, 3), "names", len(set(tokens(gold)) & set(tokens(pred))), "/", len(set(tokens(gold))), (row.get("query") or "")[:70])
    print("  P", (pred or "")[:120])
