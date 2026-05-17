#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Load .env. PRD spec uses `export $(grep -v '^#' .env | xargs)`; we use the
# safer set -a / source / set +a which handles values with spaces and quoted
# strings correctly, and doesn't require xargs.
if [[ ! -f .env ]]; then
  echo "❌ .env not found at $REPO_ROOT/.env" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

# Auto-activate the project venv if present at the standard location.
# PRD bash script invokes `python` directly; on Ubuntu 24.04 there is no
# `python` symlink unless the venv is active, so this is necessary in practice.
VENV_ACTIVATE="${HOME}/.venvs/transcript-rag/bin/activate"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

python infra/qdrant/qdrant_setup.py
