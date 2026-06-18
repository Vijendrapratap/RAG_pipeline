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

echo "==> [1/4] Chunking new transcripts from: $INPUT_DIR"
"$PY" -m ingestion.chunker_cleaned \
  "$INPUT_DIR" data/processed \
  --base-dir "$SOURCE_ROOT" -r

echo
echo "==> [2/4] Loading new chunks into Qdrant + Tantivy"
"$PY" -m ingestion.bulk_ingest_hardened --chunks-dir data/processed

echo
echo "==> [3/4] Refreshing the search app (so it reopens the BM25 index)"
docker compose restart rag-api

echo
echo "==> [4/4] Verifying (sampled check that chunks are searchable)"
# Non-fatal: a verify hiccup shouldn't undo a successful load. Read the line.
"$PY" -m ingestion.verify_ingestion --chunks-dir data/processed || \
  echo "    (verify reported an issue — see dead_letter/ and the line above)"

echo
echo "==> Done. New transcripts are live."
echo "    Open the Vishvas app, or browse http://localhost:8081 and search."
