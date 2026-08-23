import json
from pathlib import Path

prev = json.loads(Path("crag_results/local_generate40_v7.json").read_text(encoding="utf-8"))
cur = json.loads(Path("crag_results/local_generate40_v8.json").read_text(encoding="utf-8"))
rows_prev = {r["interaction_id"]: r for r in prev["rows"]}
print("v8", cur["summary"]["generate_hit"], "v7", prev["summary"]["generate_hit"])
print("--- changed ---")
for r in cur["rows"]:
    old = bool(rows_prev[r["interaction_id"]].get("generate_hit"))
    new = bool(r.get("generate_hit"))
    if old == new:
        continue
    print(" v7", old, "v8", new, r.get("question_type"))
    print(" Q", (r.get("query") or "")[:110])
    print(" G", (r.get("gold") or "")[:110])
    print(" P8", (r.get("pred") or "")[:150])
    print(" P7", (rows_prev[r["interaction_id"]].get("pred") or "")[:150])
    print()
print("--- misses ---")
for r in cur["rows"]:
    if r.get("generate_hit"):
        continue
    print(r.get("question_type"), "retr", r.get("retrieval_hit"), "corp", r.get("corpus_hit"))
    print(" Q", (r.get("query") or "")[:110])
    print(" G", (r.get("gold") or "")[:90])
    print(" P", (r.get("pred") or "")[:150])
    print()
