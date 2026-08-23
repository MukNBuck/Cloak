#!/usr/bin/env bash
# Per-boot reconciliation: bring up the Ollama daemon (no systemd in the VM).
# Idempotent: a no-op if Ollama already answers on :11434.
set -euo pipefail

OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"

if curl -fsS --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "Ollama already responding at $OLLAMA_URL"
  exit 0
fi

mkdir -p /tmp/cloak
echo "Starting Ollama daemon..."
nohup ollama serve >/tmp/cloak/ollama-serve.log 2>&1 &

for _ in $(seq 1 60); do
  if curl -fsS --max-time 1 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    echo "Ollama is up at $OLLAMA_URL"
    exit 0
  fi
  sleep 1
done

echo "ERROR: Ollama did not become ready; see /tmp/cloak/ollama-serve.log" >&2
exit 1
