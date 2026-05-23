#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Load .env. PRD spec uses `export $(grep -v '^#' .env | xargs)`; we use the
# safer set -a / source / set +a (handles values with spaces and quoted
# strings, doesn't need xargs). Matches the pattern in scripts/02_init_qdrant.sh.
if [[ ! -f .env ]]; then
  echo "❌ .env not found at $REPO_ROOT/.env" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

psql "postgresql://owui:${POSTGRES_PASSWORD}@localhost:5432/openwebui" \
    -f infra/postgres/analytics_schema.sql
