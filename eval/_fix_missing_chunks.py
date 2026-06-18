"""TEMP probe: find chunk files whose chunks are missing from Qdrant and
(optionally, with --reset) clear their progress rows so a re-ingest picks them
up. Surgical by design: TantivyWriter.add() is append-only, so we must re-ingest
ONLY files whose current chunks are absent — never already-present files.

Run from the project root with .env sourced:
    .venv/bin/python -m eval._fix_missing_chunks           # detect only
    .venv/bin/python -m eval._fix_missing_chunks --reset   # detect + reset
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from ingestion.bulk_ingest_hardened import (
    PROGRESS_DB_PATH,
    QDRANT_COLLECTION,
    QDRANT_KEY,
    QDRANT_URL,
    chunk_uuid,
)
from ingestion.utils.progress_db import ProgressDB

CHUNKS_DIR = Path("data/processed")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)


def present(ids: list[str]) -> set[str]:
    if not ids:
        return set()
    found = client.retrieve(
        collection_name=QDRANT_COLLECTION, ids=ids,
        with_payload=False, with_vectors=False,
    )
    return {str(p.id) for p in found}


def main() -> int:
    missing_files: list[tuple[str, int, int]] = []
    for f in sorted(CHUNKS_DIR.glob("*.chunks.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        sf = data.get("source_file", f.name)
        ids = [chunk_uuid(sf, i, c["text"]) for i, c in enumerate(data.get("chunks", []))]
        found = present(ids)
        miss = [cid for cid in ids if cid not in found]
        if miss:
            missing_files.append((f.name, len(ids), len(miss)))

    total_missing = sum(m for _, _, m in missing_files)
    print(f"progress db: {PROGRESS_DB_PATH}")
    print(f"Files with missing chunks: {len(missing_files)} ({total_missing} chunks)")
    for name, total, miss in missing_files:
        print(f"  {miss}/{total} missing  ::  {name}")

    if "--reset" in sys.argv and missing_files:
        with ProgressDB(PROGRESS_DB_PATH) as db:
            for name, _, _ in missing_files:
                db.reset_to_pending(name)
        print(f"\nReset {len(missing_files)} files to pending — re-run bulk_ingest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
