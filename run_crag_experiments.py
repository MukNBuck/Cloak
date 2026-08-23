"""One-factor-at-a-time CRAG ablations on a frozen id set.

Each run is a subprocess so cloak_config env vars are picked up at import.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "crag_results"
DEFAULT_IDS = RESULTS / "local_generate40.json"
CRAG_PATH = os.getenv(
    "CLOAK_CRAG_PATH",
    r"e:\crag_task_1_and_2_dev_v4.jsonl\crag_task_1_and_2_dev_v4.jsonl",
)

EXPERIMENTS = {
    "pack3": {
        "out": "exp_pack3_generate40.json",
        "mode": "generate",
        "env": {"CLOAK_MAX_PACKED_DOCS": "3"},
        "note": "Fewer distractors in the packed window (Liu et al. 2023 / 7B attention).",
    },
    "ushape": {
        "out": "exp_ushape_generate40.json",
        "mode": "generate",
        "env": {"CLOAK_PACK_ORDER": "u_shape"},
        "note": "Lost-in-the-Middle: rank-1 first, rank-2 last.",
    },
    "nomust": {
        "out": "exp_nomust_generate40.json",
        "mode": "generate",
        "env": {"CLOAK_MUST_ANSWER_RETRY": "0"},
        "note": "Disable must-answer retry that invents answers on false-premise.",
    },
    "chunk320": {
        "out": "exp_chunk320_retrieval40.json",
        "mode": "retrieval",
        "env": {"CLOAK_CHILD_CHUNK_SIZE": "320", "CLOAK_CHILD_CHUNK_OVERLAP": "64"},
        "note": "Smaller child chunks for exact-fact retrieval.",
    },
    "chunk720": {
        "out": "exp_chunk720_retrieval40.json",
        "mode": "retrieval",
        "env": {"CLOAK_CHILD_CHUNK_SIZE": "720", "CLOAK_CHILD_CHUNK_OVERLAP": "120"},
        "note": "Larger child chunks for multi-hop / list pages.",
    },
}


def run_one(name, spec, ids_file: Path, limit: int):
    out = RESULTS / spec["out"]
    env = os.environ.copy()
    env["CLOAK_VISION"] = "0"
    env["CLOAK_CRAG_PATH"] = CRAG_PATH
    env.update(spec["env"])
    cmd = [
        sys.executable,
        str(ROOT / "eval_crag.py"),
        "--mode",
        spec["mode"],
        "--html",
        "--ids-file",
        str(ids_file),
        "--limit",
        str(limit),
        "--out",
        str(out),
    ]
    print(f"\n=== {name} ===")
    print(spec["note"])
    print(" ".join(cmd))
    print("env", spec["env"])
    completed = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with code {completed.returncode}")
    if out.exists():
        summary = json.loads(out.read_text(encoding="utf-8")).get("summary") or {}
        print(f"{name} summary:", json.dumps(summary, indent=2)[:2000])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="Comma-separated experiment names")
    parser.add_argument("--ids-file", default=str(DEFAULT_IDS))
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiments {unknown}. Choose from {list(EXPERIMENTS)}")
    ids_file = Path(args.ids_file)
    if not ids_file.exists():
        raise SystemExit(f"ids file not found: {ids_file}")
    RESULTS.mkdir(exist_ok=True)
    for name in names:
        run_one(name, EXPERIMENTS[name], ids_file, args.limit)


if __name__ == "__main__":
    main()
