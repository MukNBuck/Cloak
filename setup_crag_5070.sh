#!/usr/bin/env bash
# One-time setup for CRAG eval on the RTX 5070 box.
# Run from the Cloak/ directory:  bash setup_crag_5070.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
VENV="${CLOAK_VENV:-$ROOT/.venv}"
LLM_MODEL="${CLOAK_LLM_MODEL:-qwen2.5:14b}"

install_ollama() {
  if command -v ollama >/dev/null 2>&1; then
    echo "==> Ollama already installed: $(command -v ollama)"
    return 0
  fi
  echo "==> Installing Ollama"
  if [[ "$(id -u)" -eq 0 ]]; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    curl -fsSL https://ollama.com/install.sh | sudo sh
  fi
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: Ollama install finished but 'ollama' is not on PATH."
    echo "Open a new shell or check /usr/local/bin/ollama"
    exit 1
  fi
}

start_ollama() {
  if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "==> Ollama service already responding on :11434"
    return 0
  fi
  echo "==> Starting Ollama service"
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable --now ollama 2>/dev/null || true
    if systemctl is-active --quiet ollama 2>/dev/null; then
      echo "==> ollama.service is active"
      return 0
    fi
  fi
  # Fallback: background daemon for this session
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  echo $! >/tmp/ollama-serve.pid
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      echo "==> Ollama serve is up (pid $(cat /tmp/ollama-serve.pid))"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: Ollama did not start. Check /tmp/ollama-serve.log"
  exit 1
}

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

install_ollama
start_ollama

echo "==> Pulling Ollama chat models (needed for --mode generate)"
ollama pull "$LLM_MODEL"
ollama pull qwen2.5:7b || true

echo
echo "Setup done."
echo "  1. Place CRAG jsonl and set CLOAK_CRAG_PATH in .env"
echo "  2. source $VENV/bin/activate"
echo "  3. bash run_crag_5070.sh smoke"
echo "See CRAG_5070.md for the full ladder."
