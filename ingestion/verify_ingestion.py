"""Post-ingest sampling validation. Per PRD §6 Phase 4.

Samples N chunks from the source .chunks.json files and verifies each one is
present in both Qdrant (by chunk_id) and Tantivy (by chunk_id). Exits 0 iff
both presence rates are >= THRESHOLD (default 99.9%).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from ingestion.bulk_ingest_hardened import (
    QDRANT_COLLECTION, QDRANT_KEY, QDRANT_URL, TANTIVY_DIR, chunk_uuid,
)

try:
    import tantivy  # type: ignore[import-not-found]
    TANTIVY_AVAILABLE = True
except ImportError:
    tantivy = None
    TANTIVY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("verify_ingestion")


def sample_chunk_ids(chunks_dir: Path, n: int) -> list[tuple[str, str]]:
    """Return up to n (chunk_id, source_file) pairs sampled across the dir."""
    pool: list[tuple[str, str]] = []
    for f in sorted(chunks_dir.glob("*.chunks.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("skipping %s during sample: %s", f.name, e)
            continue
        sf = data.get("source_file", f.name)
        for i, c in enumerate(data.get("chunks", [])):
            pool.append((chunk_uuid(sf, i, c["text"]), sf))
    if not pool:
        return []
    if len(pool) <= n:
        return pool
    return random.sample(pool, n)


def qdrant_present(client: QdrantClient, ids: list[str]) -> int:
    if not ids:
        return 0
    found = client.retrieve(
        collection_name=QDRANT_COLLECTION, ids=ids, with_payload=False,
        with_vectors=False,
    )
    return len(found)


def tantivy_present(index_dir: Path, ids: list[str]) -> int:
    if not TANTIVY_AVAILABLE or not index_dir.exists():
        return 0
    index = tantivy.Index.open(str(index_dir))
    index.reload()
    searcher = index.searcher()
    n_found = 0
    for cid in ids:
        q = index.parse_query(f'"{cid}"', ["chunk_id"])
        hits = searcher.search(q, limit=1).hits
        if hits:
            n_found += 1
    return n_found


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify Qdrant/Tantivy ingestion completeness.")
    p.add_argument("--chunks-dir", required=True,
                   help="Directory of *.chunks.json files to sample from")
    p.add_argument("--sample-size", type=int, default=1000,
                   help="Number of chunks to spot-check (PRD default: 1000)")
    p.add_argument("--threshold", type=float, default=99.9,
                   help="Pass threshold percentage (default 99.9)")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional RNG seed for reproducible sampling")
    args = p.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    pairs = sample_chunk_ids(Path(args.chunks_dir), args.sample_size)
    if not pairs:
        log.error("no chunks found under %s", args.chunks_dir)
        return 1
    ids = [cid for cid, _ in pairs]

    qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    q_found = qdrant_present(qclient, ids)
    t_found = tantivy_present(TANTIVY_DIR, ids)

    n = len(ids)
    q_pct = 100.0 * q_found / n
    t_pct = 100.0 * t_found / n
    print(f"Sampled {n} chunks. Qdrant: {q_pct:.1f}% present. Tantivy: {t_pct:.1f}% present.")
    return 0 if q_pct >= args.threshold and t_pct >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
