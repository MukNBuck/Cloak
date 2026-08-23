import json
from pathlib import Path

prev = json.loads(Path("crag_results/local_generate40_v6.json").read_text(encoding="utf-8"))
cur = json.loads(Path("crag_results/local_generate40_v7.json").read_text(encoding="utf-8"))
rows_prev = {r["interaction_id"]: r for r in prev["rows"]}
changed = []
misses = []
for r in cur["rows"]:
    old = bool(rows_prev[r["interaction_id"]].get("generate_hit"))
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
        "pred_prev": rows_prev[r["interaction_id"]].get("pred"),
        "prev": old,
        "now": new,
    }
    if old != new:
        changed.append(item)
    if not new:
        misses.append(item)

print("v7 hit", sum(1 for r in cur["rows"] if r.get("generate_hit")), "rate", cur["summary"]["generate_hit"])
print("changed vs v6", len(changed))
for c in changed:
    print(" v6", c["prev"], "v7", c["now"], c["type"])
    print("  Q", (c["q"] or "")[:110])
    print("  G", (c["gold"] or "")[:110])
    print("  P7", (c["pred"] or "")[:160])
    print("  P6", (c["pred_prev"] or "")[:160])
    print()
print("--- misses ---")
for m in misses:
    print(m["type"], "retr", m["retr"], "corp", m["corpus"], "int", m["intent"])
    print(" Q", (m["q"] or "")[:110])
    print(" G", (m["gold"] or "")[:110])
    print(" P", (m["pred"] or "")[:160])
    print()
Path("crag_results/_v7_misses.json").write_text(
    json.dumps(
        {
            "generate_hit": cur["summary"]["generate_hit"],
            "changed": [
                {k: c[k] for k in ("prev", "now", "type", "q", "gold", "pred", "pred_prev")}
                for c in changed
            ],
            "misses": [
                {k: m[k] for k in ("type", "retr", "corpus", "abs", "intent", "q", "gold", "pred")}
                for m in misses
            ],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
