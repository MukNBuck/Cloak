#!/usr/bin/env bash
# Foreground FastAPI backend for the Cloak RAG API (served in a Cursor terminal).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Wait for Ollama so /models works as soon as the server accepts requests.
OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"
for _ in $(seq 1 30); do
  curl -fsS --max-time 1 "$OLLAMA_URL/api/tags" >/dev/null 2>&1 && break
  sleep 1
done

exec uvicorn server:app --host 0.0.0.0 --port 8001
