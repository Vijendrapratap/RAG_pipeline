#!/usr/bin/env bash
# Selects a random 100GB slice of raw transcripts, runs the full pipeline,
# reports timing + failure stats.
set -euo pipefail

SLICE_DIR="${SLICE_DIR:-/data/preflight_slice}"
RAW_DIR="${RAW_DIR:-/data/raw-transcripts}"
PROCESSED_DIR="${PROCESSED_DIR:-/data/processed_preflight}"

# 1. Random 100GB sample (assumes du-sortable file listing)
python scripts/select_random_slice.py \
    --src "$RAW_DIR" --dst "$SLICE_DIR" --target-gb 100

# 2. Chunk
python -m ingestion.chunker_text "$SLICE_DIR" "$PROCESSED_DIR"

# 3. Ingest
python -m ingestion.bulk_ingest_hardened \
    --chunks-dir "$PROCESSED_DIR" --batch-size 32

# 4. Verify
python -m ingestion.verify_ingestion

# 5. Eval
python -m eval.run_eval --queries eval/golden_queries.yaml
