#!/usr/bin/env bash
# scripts/setup_env.sh — idempotent .env validator and fixer.
#
# What it does:
#   1. Creates .env from .env.example if missing.
#   2. Fills any missing required key with a sensible default.
#   3. Generates real secrets for any key whose value is `replace_me`
#      or empty, using `openssl rand -hex 32`.
#   4. Warns (does not fail) if data paths don't exist on the host
#      yet — you may be running this before mounting your data drive.
#
# Safe to re-run any time. Will never overwrite an existing real value.
#
# Usage:
#   bash scripts/setup_env.sh           # validate + fix, print summary
#   bash scripts/setup_env.sh --check   # validate only, exit 1 if broken

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
TEMPLATE="$REPO_ROOT/.env.example"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

if [[ ! -f "$TEMPLATE" ]]; then
  echo "❌ .env.example not found at $TEMPLATE" >&2
  exit 1
fi

# Bootstrap .env from template if absent.
if [[ ! -f "$ENV_FILE" ]]; then
  if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "❌ .env not found (run without --check to create it)" >&2
    exit 1
  fi
  cp "$TEMPLATE" "$ENV_FILE"
  echo "📄 Created $ENV_FILE from .env.example"
fi

# --- helpers --------------------------------------------------------------

get_val() { grep -E "^${1}=" "$ENV_FILE" | head -n1 | cut -d= -f2- ; }

set_val() {
  local key="$1" val="$2"
  if grep -qE "^${key}=" "$ENV_FILE"; then
    # Replace existing line. Use | as sed delimiter to allow / in values.
    sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    printf '\n%s=%s\n' "$key" "$val" >> "$ENV_FILE"
  fi
}

gen_secret() { openssl rand -hex 32; }

# --- required keys --------------------------------------------------------

SECRETS=(WEBUI_SECRET_KEY POSTGRES_PASSWORD QDRANT_API_KEY)

# key=default — defaults applied only when the key is missing entirely.
declare -A DEFAULTS=(
  [QDRANT_URL]="http://localhost:6333"
  [OLLAMA_URL]="http://localhost:11434"
  [TANTIVY_URL]="http://localhost:8765"
  [RERANKER_URL]="http://localhost:7997"
  [POSTGRES_DSN]='postgresql://owui:${POSTGRES_PASSWORD}@localhost:5432/openwebui'
  [QDRANT_COLLECTION]="transcripts"
  [EMBED_MODEL]="bge-m3"
  [RAW_TRANSCRIPTS_DIR]="/mnt/d/GuruAudio/Output"
  [RAW_TRANSCRIPTS_BASE_DIR]="/mnt/d/GuruAudio/Output"
  [PROCESSED_CHUNKS_DIR]="/mnt/d/Vishvas-rag-pipeline/data/processed"
  [TANTIVY_DIR]="/mnt/d/Vishvas-rag-pipeline/data/tantivy"
  [DEAD_LETTER_DIR]="./dead_letter"
  [INGEST_BATCH_SIZE]="32"
  [INGEST_PER_FILE_TIMEOUT_SEC]="1800"
  [INGEST_MAX_FILE_SIZE_MB]="500"
)

PATH_KEYS=(RAW_TRANSCRIPTS_DIR RAW_TRANSCRIPTS_BASE_DIR)

ISSUES=0
FIXED=0

# --- secrets pass ---------------------------------------------------------

for k in "${SECRETS[@]}"; do
  cur="$(get_val "$k")"
  if [[ -z "$cur" || "$cur" == "replace_me" ]]; then
    if [[ $CHECK_ONLY -eq 1 ]]; then
      echo "❌ $k is not set" >&2
      ISSUES=$((ISSUES + 1))
    else
      new_val="$(gen_secret)"
      set_val "$k" "$new_val"
      echo "🔑 Generated $k"
      FIXED=$((FIXED + 1))
    fi
  else
    echo "✅ $k set"
  fi
done

# --- defaults pass --------------------------------------------------------

for k in "${!DEFAULTS[@]}"; do
  cur="$(get_val "$k")"
  if [[ -z "$cur" ]]; then
    if [[ $CHECK_ONLY -eq 1 ]]; then
      echo "❌ $k is missing" >&2
      ISSUES=$((ISSUES + 1))
    else
      set_val "$k" "${DEFAULTS[$k]}"
      echo "➕ Added $k=${DEFAULTS[$k]}"
      FIXED=$((FIXED + 1))
    fi
  fi
done

# --- path sanity (warn only) ---------------------------------------------

echo
for k in "${PATH_KEYS[@]}"; do
  cur="$(get_val "$k")"
  if [[ -n "$cur" && -d "$cur" ]]; then
    n=$(find "$cur" -maxdepth 3 -type f \( -name '*.json' -o -name '*.txt' \) 2>/dev/null | head -n 1)
    if [[ -n "$n" ]]; then
      echo "✅ $k -> $cur (contains transcripts)"
    else
      echo "⚠️  $k -> $cur exists but no .json/.txt found in top 3 levels"
    fi
  else
    echo "⚠️  $k -> $cur does not exist on this host yet (mount or edit before ingesting)"
  fi
done

# --- summary --------------------------------------------------------------

echo
if [[ $CHECK_ONLY -eq 1 ]]; then
  if [[ $ISSUES -eq 0 ]]; then
    echo "✅ .env passes validation."
    exit 0
  fi
  echo "❌ $ISSUES issue(s). Re-run without --check to auto-fix." >&2
  exit 1
fi

echo "Done. Fixed $FIXED key(s)."
echo "Inspect: $ENV_FILE"
