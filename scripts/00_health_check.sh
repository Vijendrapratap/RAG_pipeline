#!/usr/bin/env bash
# Health-check every service brought up by docker-compose.yml.
# Pass: ✅; fail: ❌. Exit 0 only if all checks pass.
#
# Phase F: Open WebUI, Redis, and the Tantivy `:8765` sidecar are gone.
# The dashboard + retrieval API live in the rag-api container; Tantivy
# runs in-process inside it.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

QDRANT_API_KEY="${QDRANT_API_KEY:-}"

PASS=0
FAIL=0

check() {
  local name="$1"
  shift
  if "$@" > /dev/null 2>&1; then
    echo "✅ $name"
    PASS=$((PASS + 1))
  else
    echo "❌ $name"
    FAIL=$((FAIL + 1))
  fi
}

check "rag-api (8080)"     curl -fsS http://localhost:8080/api/health
check "Ollama (11434)"     curl -fsS http://localhost:11434/api/tags

if [[ -n "$QDRANT_API_KEY" ]]; then
  check "Qdrant (6333)"    curl -fsS -H "api-key: $QDRANT_API_KEY" http://localhost:6333/collections
else
  check "Qdrant (6333)"    curl -fsS http://localhost:6333/collections
fi

check "Infinity (7997)"    curl -fsS http://localhost:7997/health
check "Postgres (5432)"    pg_isready -h localhost -p 5432 -U owui

echo
echo "$PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
