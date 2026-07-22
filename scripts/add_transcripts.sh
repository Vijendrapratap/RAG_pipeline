#!/usr/bin/env bash
# Add newly-transcribed tracks to the search system, in one command.
#
# It is incremental and safe to re-run: only NEW tracks are processed; anything
# already loaded is skipped, never duplicated. Run it whenever WhisperX has
# produced new "cleaned" transcripts.
#
# Usage (run from the Ubuntu / WSL shell):
#   bash scripts/add_transcripts.sh
#       -> process everything new under the whole transcript source folder
#   bash scripts/add_transcripts.sh "/mnt/d/Transcription whisperx/Output/Some Event"
#       -> process just one folder (still incremental)
#
# Why a wrapper: the only thing that must stay constant between runs is
# --base-dir. Get it wrong and every track looks "new" and loads duplicates.
# This script pins it to SOURCE_ROOT so it can't be set wrong by hand.
set -euo pipefail

# Where WhisperX drops new "cleaned" transcripts (<stem>.raw.json + .cleaned.txt).
SOURCE_ROOT="/mnt/d/Transcription whisperx/Output"

# Process the whole source root by default; allow a narrower folder as $1.
INPUT_DIR="${1:-$SOURCE_ROOT}"

# Always run from the project root so the relative paths below resolve.
cd "$(dirname "$0")/.."
PY=.venv/bin/python

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: input folder not found: $INPUT_DIR" >&2
  echo "       Check the path, or pass the correct folder as the first argument." >&2
  exit 2
fi

# Load secrets (QDRANT_API_KEY etc.). Required — without it, step 2 fails 401.
set -a; source .env; set +a

# --------------------------------------------------------------------------
# [1/5] Staleness sweep. A transcript edited AFTER it was last chunked (a VAD
# re-transcribe, the energy-gated recall pass, a hallucination/loop scrub) is
# otherwise skipped forever: the chunker skips any track whose .chunks.json
# already exists, and bulk_ingest skips any file marked 'ok'. So the stores keep
# serving the OLD text. This step DELETES such a track from every store and drops
# its .chunks.json, so the steps below rebuild it from the current transcript.
#
# The reindex is DESTRUCTIVE (it deletes points before they are rebuilt), so gate
# it on the embed backend being reachable FIRST — never delete data we then can't
# re-embed. bulk_ingest health-checks too, but only AFTER this would have deleted.
echo "==> [1/5] Staleness sweep (reindex transcripts edited since last ingest)"
# Gate on Qdrant reachable AND the embed model actually embedding (not just
# Ollama answering) — the reindex below deletes vectors before rebuilding them,
# so a false "healthy" would strand tracks deleted-but-not-re-embedded.
if ! bash scripts/backend_health.sh; then
  echo "ERROR: Stage 3 backends not ready to ingest from this (WSL) shell." >&2
  echo "       Qdrant must be reachable AND the embed model must actually embed." >&2
  echo "       Refusing to reindex before re-embed is possible — nothing was changed." >&2
  echo "       Common cause: OLLAMA_URL unreachable from WSL, or EMBED_MODEL not pulled." >&2
  exit 3
fi

STALE_SRC="$(mktemp)"; STALE_CHUNKS="$(mktemp)"
trap 'rm -f "$STALE_SRC" "$STALE_CHUNKS"' EXIT
"$PY" -m ingestion.find_stale_transcripts \
  --source-root "$INPUT_DIR" --base-dir "$SOURCE_ROOT" \
  --chunks-dir data/processed \
  --out-sources "$STALE_SRC" --out-chunks "$STALE_CHUNKS" --limit-preview 12
n_stale=$(grep -c . "$STALE_SRC" 2>/dev/null || echo 0)
if [[ "$n_stale" -gt 0 ]]; then
  echo "    $n_stale changed track(s) -> deleting old store data so they re-embed cleanly"
  "$PY" -m ingestion.reindex_file --source-list "$STALE_SRC"
  while IFS= read -r cj; do
    [[ -n "$cj" && -f "data/processed/$cj" ]] && rm -f "data/processed/$cj"
  done < "$STALE_CHUNKS"
else
  echo "    no changed transcripts — nothing to reindex"
fi

echo
echo "==> [2/5] Chunking new transcripts from: $INPUT_DIR"
"$PY" -m ingestion.chunker_cleaned \
  "$INPUT_DIR" data/processed \
  --base-dir "$SOURCE_ROOT" -r

echo
echo "==> [3/5] Loading new chunks into Qdrant + Tantivy"
"$PY" -m ingestion.bulk_ingest_hardened --chunks-dir data/processed

echo
echo "==> [4/5] Refreshing the search app (so it reopens the BM25 index)"
docker compose restart rag-api

echo
echo "==> [5/5] Verifying (sampled check that chunks are searchable)"
# Non-fatal: a verify hiccup shouldn't undo a successful load. Read the line.
"$PY" -m ingestion.verify_ingestion --chunks-dir data/processed || \
  echo "    (verify reported an issue — see dead_letter/ and the line above)"

echo
echo "==> Done. New transcripts are live."
echo "    Open the Vishvas app, or browse http://localhost:8081 and search."
