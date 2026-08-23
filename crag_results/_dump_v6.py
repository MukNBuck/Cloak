import json
from pathlib import Path

v5 = json.loads(Path("crag_results/local_generate40_v5.json").read_text(encoding="utf-8"))
v6 = json.loads(Path("crag_results/local_generate40_v6.json").read_text(encoding="utf-8"))

rows5 = {r["interaction_id"]: r for r in v5["rows"]}
changed = []
misses = []
hits = []
for r in v6["rows"]:
    old = bool(rows5[r["interaction_id"]].get("generate_hit"))
    new = bool(r.get("generate_hit"))
    item = {
        "type": r.get("question_type"),
        "retr": r.get("retrieval_hit"),
        "corpus": r.get("corpus_hit"),
        "abs": r.get("abstain"),
        "intent": r.get("intent"),
        "q": r.get("query"),
        "gold": r.get("gold"),
        "pred": r.get("pred"),
        "pred_v5": rows5[r["interaction_id"]].get("pred"),
        "v5": old,
        "v6": new,
    }
    if old != new:
        changed.append(item)
    if new:
        hits.append(item)
    else:
        misses.append(item)

out = {
    "generate_hit": v6["summary"]["generate_hit"],
    "miss_n": len(misses),
    "hit_n": len(hits),
    "changed": [
        {
            "v5": c["v5"],
            "v6": c["v6"],
            "type": c["type"],
            "q": c["q"],
            "gold": c["gold"],
            "pred": c["pred"],
            "pred_v5": c["pred_v5"],
        }
        for c in changed
    ],
    "misses": [
        {
            "type": m["type"],
            "retr": m["retr"],
            "corpus": m["corpus"],
            "abs": m["abs"],
            "intent": m["intent"],
            "q": m["q"],
            "gold": m["gold"],
            "pred": m["pred"],
        }
        for m in misses
    ],
}
Path("crag_results/_v6_misses.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("v6 hit", out["hit_n"], "miss", out["miss_n"], "rate", out["generate_hit"])
print("changed", len(changed))
for c in changed:
    print(" v5", c["v5"], "v6", c["v6"], c["type"])
    print("  Q", (c["q"] or "")[:110])
    print("  G", (c["gold"] or "")[:110])
    print("  P6", (c["pred"] or "")[:140])
    print("  P5", (c["pred_v5"] or "")[:140])
    print()
print("--- misses ---")
for m in misses:
    print(m["type"], "retr", m["retr"], "corp", m["corpus"], "int", m["intent"])
    print(" Q", (m["q"] or "")[:110])
    print(" G", (m["gold"] or "")[:110])
    print(" P", (m["pred"] or "")[:140])
    print()
