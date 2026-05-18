#!/usr/bin/env bash
set -euo pipefail

# Phase 13 — per-file content tagging via Qwen 2.5 7B (Ollama).
# Runs AFTER bulk_ingest_hardened.py. Resumable: re-runs skip files
# already tagged (file_meta.tagged_at IS NOT NULL).
#
# Usage:
#   scripts/06_enrich_tags.sh                  # tag all untagged files
#   scripts/06_enrich_tags.sh --limit 10       # tag at most 10 files
#   scripts/06_enrich_tags.sh --dry-run        # show what would be tagged
#   scripts/06_enrich_tags.sh --model qwen2.5:14b  # override model

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f .env ]]; then
  echo "❌ .env not found at $REPO_ROOT/.env" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

exec python -m ingestion.enrich_content_tags "$@"
