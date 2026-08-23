import json
from pathlib import Path

data = json.loads(Path("crag_results/local_generate40_v5.json").read_text(encoding="utf-8"))
misses = [row for row in data["rows"] if not row.get("generate_hit")]
hits = [row for row in data["rows"] if row.get("generate_hit")]
out = {
    "generate_hit": data["summary"]["generate_hit"],
    "miss_n": len(misses),
    "hit_n": len(hits),
    "misses": [
        {
            "type": r.get("question_type"),
            "retr": r.get("retrieval_hit"),
            "corpus": r.get("corpus_hit"),
            "abs": r.get("abstain"),
            "q": r.get("query"),
            "gold": r.get("gold"),
            "pred": r.get("pred"),
        }
        for r in misses
    ],
    "hits": [
        {
            "type": r.get("question_type"),
            "q": r.get("query"),
            "gold": r.get("gold"),
            "pred": r.get("pred"),
        }
        for r in hits
    ],
}
Path("crag_results/_v5_misses.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("hit", out["hit_n"], "miss", out["miss_n"], "rate", out["generate_hit"])
