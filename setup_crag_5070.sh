#!/usr/bin/env bash
# One-time setup for CRAG eval on the RTX 5070 box.
# Run from the Cloak/ directory:  bash setup_crag_5070.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
VENV="${CLOAK_VENV:-$ROOT/.venv}"

echo "==> Python"
"$PY" -V

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating venv at $VENV"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel

echo "==> Installing CUDA PyTorch (cu124 wheels; adjust if needed)"
# RTX 50-series needs a recent CUDA build. Override with TORCH_INDEX if you prefer.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
python -m pip install --index-url "$TORCH_INDEX" torch torchvision torchaudio \
  || python -m pip install torch

echo "==> Installing Cloak deps"
python -m pip install -r requirements.txt

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  echo "==> Wrote .env from .env.example — edit CLOAK_CRAG_PATH"
fi

echo "==> GPU check"
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    print("WARNING: CUDA not available — embed/rerank will be CPU-only")
PY

if command -v ollama >/dev/null 2>&1; then
  echo "==> Pulling Ollama chat models (needed for --mode generate)"
  ollama pull "${CLOAK_LLM_MODEL:-qwen2.5:14b}"
  ollama pull qwen2.5:7b || true
else
  echo "WARNING: ollama not on PATH — install it before --mode generate"
fi

echo
echo "Setup done."
echo "  1. Place CRAG jsonl and set CLOAK_CRAG_PATH in .env"
echo "  2. source $VENV/bin/activate"
echo "  3. bash run_crag_5070.sh smoke"
echo "See CRAG_5070.md for the full ladder."
