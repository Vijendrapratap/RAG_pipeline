#!/usr/bin/env bash
# scripts/start_pipeline.sh — one-command first-run setup.
#
# Brings the system from a fresh clone to "ready to ingest" by running
# every phase-0 through phase-6 setup step in order, with prerequisite
# checks between each. Safe to re-run: every step is idempotent.
#
# This does NOT chunk, ingest, or run the preflight. Those are
# operator-triggered because they take hours-to-days and you want to
# decide when. See `docs/getting_started.md` for the next steps after
# this script reports "READY".
#
# Usage:
#   bash scripts/start_pipeline.sh           # interactive, prompts on failure
#   bash scripts/start_pipeline.sh --yes     # non-interactive, fail fast

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

YES=0
[[ "${1:-}" == "--yes" ]] && YES=1

# Phase F: the Tantivy `:8765` sidecar is gone — Tantivy now runs in-process
# inside the rag-api container. Open WebUI and Redis are also gone.

step() { echo; echo "▶ $*"; }
ok()   { echo "  ✅ $*"; }
warn() { echo "  ⚠️  $*"; }
die()  { echo "  ❌ $*" >&2; exit 1; }

confirm() {
  [[ $YES -eq 1 ]] && return 0
  read -r -p "  Continue anyway? [y/N] " a
  [[ "$a" == "y" || "$a" == "Y" ]]
}

# Wait for a URL to return HTTP 2xx/3xx, up to N seconds. Fail otherwise.
wait_http() {
  local name="$1" url="$2" timeout="${3:-90}" headers="${4:-}"
  local start=$SECONDS
  while (( SECONDS - start < timeout )); do
    if [[ -n "$headers" ]]; then
      curl -fsS -m 3 -H "$headers" "$url" >/dev/null 2>&1 && { ok "$name reachable"; return 0; }
    else
      curl -fsS -m 3 "$url" >/dev/null 2>&1 && { ok "$name reachable"; return 0; }
    fi
    sleep 2
  done
  return 1
}

# ============================================================
step "1/8  Prerequisite checks"
# ============================================================

command -v docker  >/dev/null 2>&1 || die "docker not on PATH"
command -v curl    >/dev/null 2>&1 || die "curl not on PATH"
command -v openssl >/dev/null 2>&1 || die "openssl not on PATH"
docker compose version >/dev/null 2>&1 || die "docker compose v2 plugin missing"
ok "docker + compose + curl + openssl present"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L | head -n1 | sed 's/^/  /'
  ok "GPU visible to host"
else
  warn "nvidia-smi not found — Ollama and reranker will run CPU-only (very slow)"
  confirm || die "aborting"
fi

# ============================================================
step "2/8  Validate .env"
# ============================================================

bash scripts/setup_env.sh || die "setup_env.sh failed"

# Source it so subsequent steps see the values.
set -a
# shellcheck disable=SC1091
source .env
set +a

# ============================================================
step "3/8  docker compose up -d"
# ============================================================

docker compose up -d || die "docker compose up failed"
ok "containers started"

# ============================================================
step "4/8  Wait for services to become reachable"
# ============================================================

wait_http "rag-api"          "http://localhost:8080/api/health"       90 || warn "rag-api not responding yet (continuing — Postgres / Qdrant may still be cold)"
wait_http "Ollama"           "http://localhost:11434/api/tags"        60 || die  "Ollama did not start"
if [[ -n "${QDRANT_API_KEY:-}" ]]; then
  wait_http "Qdrant"         "http://localhost:6333/collections"      60 "api-key: $QDRANT_API_KEY" || die "Qdrant did not start"
else
  wait_http "Qdrant"         "http://localhost:6333/collections"      60 || die "Qdrant did not start"
fi
wait_http "Infinity reranker" "http://localhost:7997/health"          120 || warn "Reranker not responding (lazy-downloads on first call, may be OK)"

# Postgres has no HTTP health endpoint; check via docker.
docker compose exec -T postgres pg_isready -U owui >/dev/null 2>&1 && ok "Postgres ready" || die "Postgres not ready"

# ============================================================
step "5/8  Pull Ollama models (idempotent — skips if already present)"
# ============================================================

bash scripts/01_pull_models.sh || die "model pull failed"

# ============================================================
step "6/8  Initialize Qdrant collection + Postgres schema"
# ============================================================

bash scripts/02_init_qdrant.sh   || die "Qdrant init failed"
bash scripts/03_init_postgres.sh || die "Postgres init failed"

# ============================================================
step "7/7  Final health check"
# ============================================================

bash scripts/00_health_check.sh || warn "Some health checks failed — see output above"

cat <<'EOF'

================================================================
✅ READY

The system is up. Next steps (operator-triggered, in order):

  1. Chunk your transcripts (production corpus is whisperX/whisper JSON):
       python -m ingestion.chunker_json "$RAW_TRANSCRIPTS_DIR" "$PROCESSED_CHUNKS_DIR" --format whisperx
       # OR if you have plain .txt:
       python -m ingestion.chunker_text "$RAW_TRANSCRIPTS_DIR" "$PROCESSED_CHUNKS_DIR"

  2. Preflight on a 100 GB random slice (MANDATORY before full run, ~8–24 h):
       bash scripts/preflight.sh

  3. Full ingestion (8–14 days on 5 TB; resumable):
       tmux new -s ingest
       python -m ingestion.bulk_ingest_hardened --chunks-dir "$PROCESSED_CHUNKS_DIR"

  4. Build the file-summary index (after Phase 13 enrichment populates
     summary_english / summary_hindi in file_meta):
       python -m ingestion.build_summary_index

  5. Open the dashboard at http://localhost:8080 — log in with
     $DASHBOARD_PASSWORD if you set one.

Dashboard:  http://localhost:8080
API health: http://localhost:8080/api/health
EOF
