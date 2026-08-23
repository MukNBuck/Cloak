import json
from pathlib import Path

rows = json.loads(Path("local_generate40.json").read_text(encoding="utf-8"))["rows"]


def dump(text):
    print(str(text).encode("ascii", "replace").decode("ascii"))


retr_ok = [r for r in rows if r.get("retrieval_hit") and not r.get("generate_hit") and not r.get("abstain")]
gen_ok = [r for r in rows if r.get("generate_hit")]
idk = [r for r in rows if (r.get("pred") or "").lower().startswith("i don't know")]
swc = [r for r in rows if r.get("question_type") == "simple_w_condition"]

dump("generate_hits %s" % len(gen_ok))
dump("idk %s" % len(idk))
dump("retr_ok_gen_fail %s" % len(retr_ok))
dump("--- simple_w_condition ---")
for r in swc:
    dump("retr=%s gen=%s gate=%s intent=%s" % (r.get("retrieval_hit"), r.get("generate_hit"), r.get("passed_gate"), r.get("intent")))
    dump(" Q: %s" % (r.get("query") or "")[:140])
    dump(" G: %s" % (r.get("gold") or "")[:180])
    dump(" P: %s" % ((r.get("pred") or "")[:200].replace("\n", " ")))
dump("--- retr ok gen fail ---")
for r in retr_ok:
    dump("%s intent=%s" % (r.get("question_type"), r.get("intent")))
    dump(" Q: %s" % (r.get("query") or "")[:140])
    dump(" G: %s" % (r.get("gold") or "")[:180])
    dump(" P: %s" % ((r.get("pred") or "")[:200].replace("\n", " ")))
