#!/usr/bin/env bash
# Idempotent setup for the Cloak RAG Cloud Agent environment (CPU-only).
# Prepares: system libs, a Python venv with CPU PyTorch + deps, Ollama,
# the on-disk Hugging Face model cache, and a small chat model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> System packages (venv, PDF/OCR libs, zstd for the Ollama installer)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3.12-venv poppler-utils tesseract-ocr libgl1 libglib2.0-0 zstd curl ca-certificates

PY="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  echo "==> Creating venv at .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel

echo "==> CPU PyTorch (no GPU in the Cloud Agent VM)"
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

echo "==> Cloak Python dependencies"
pip install -r requirements.txt
# Some deps (timm/unstructured-inference) can pull a CUDA-ABI torchvision from
# PyPI, which breaks against CPU torch. Pin the matching CPU build.
pip install --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps torchvision

if [ ! -f .env ]; then
  echo "==> Writing CPU .env"
  cat > .env <<'ENV'
# Cloud Agent (CPU-only) local RAG settings. See .env.example for the 5070/GPU box.
CLOAK_LLM_MODEL=qwen2.5:1.5b
CLOAK_VISION_MODEL=qwen2.5vl:7b
CLOAK_VISION=0
CLOAK_EMBEDDING_MODEL=BAAI/bge-m3
CLOAK_RERANK_MODEL=BAAI/bge-reranker-v2-m3
CLOAK_DEVICE=cpu
CLOAK_PDF_STRATEGY=fast
OLLAMA_HOST=http://127.0.0.1:11434
# CRAG dataset is a ~5GB external artifact and is not shipped here.
CLOAK_CRAG_PATH=/data/crag/crag_task_1_and_2_dev_v4.jsonl
ENV
fi

echo "==> Ollama runtime"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sudo sh
fi

echo "==> Caching embedding + rerank models to disk"
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"):
    snapshot_download(
        repo_id=repo,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors",
                        "tokenizer*", "sentencepiece*", "*.py"],
    )
    print("cached", repo)
PY

echo "==> Pulling the CPU chat model into the durable Ollama store"
STARTED_SERVE=0
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  nohup ollama serve >/tmp/ollama-install-serve.log 2>&1 &
  STARTED_SERVE=$!
  for _ in $(seq 1 60); do
    curl -fsS --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi
ollama pull qwen2.5:1.5b
if [ "$STARTED_SERVE" != "0" ]; then
  kill "$STARTED_SERVE" >/dev/null 2>&1 || true
fi

echo "==> Install complete."
