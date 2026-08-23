#!/usr/bin/env bash
# Run CRAG eval presets on the 5070 box.
# Usage (from Cloak/):
#   bash run_crag_5070.sh smoke
#   bash run_crag_5070.sh corpus
#   bash run_crag_5070.sh retrieval
#   bash run_crag_5070.sh generate
#   bash run_crag_5070.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

VENV="${CLOAK_VENV:-$ROOT/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

export CLOAK_VISION="${CLOAK_VISION:-0}"
export CLOAK_LLM_MODEL="${CLOAK_LLM_MODEL:-qwen2.5:14b}"

OUT_DIR="${CLOAK_CRAG_OUT:-$ROOT/crag_results}"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MODE_ARG="${1:-smoke}"

run_one() {
  local name="$1"
  shift
  local out="$OUT_DIR/${name}_${STAMP}.json"
  echo
  echo "======== $name ========"
  echo "cmd: python eval_crag.py $*"
  echo "out: $out"
  python eval_crag.py "$@" --out "$out"
}

case "$MODE_ARG" in
  smoke)
    # Fast sanity: corpus upper bound + small retrieval sample
    run_one smoke_corpus --mode corpus --limit 80 --html
    run_one smoke_retrieval --mode retrieval --limit 20 --html
    ;;
  corpus)
    run_one corpus --mode corpus --limit "${LIMIT:-200}" --html
    ;;
  retrieval)
    run_one retrieval --mode retrieval --limit "${LIMIT:-100}" --html
    ;;
  generate)
    run_one generate --mode generate --limit "${LIMIT:-40}" --html
    ;;
  all)
    run_one corpus --mode corpus --limit 200 --html
    run_one retrieval --mode retrieval --limit 100 --html
    run_one generate --mode generate --limit 40 --html
    ;;
  *)
    echo "Unknown mode: $MODE_ARG"
    echo "Use: smoke | corpus | retrieval | generate | all"
    exit 1
    ;;
esac

echo
echo "Results in $OUT_DIR"
