"""Build the file-level summary vector index. Dashboard track, Phase C.

The Phase 13 enrichment pass writes a Hindi + English summary per file into
Postgres `file_meta` — but those summaries were never embedded, so they could
not be *searched*. This script fixes that: it reads each tagged file's
summary, embeds it with bge-m3, and upserts one vector per file into the
Qdrant `transcript_summaries` collection.

That unlocks file-level retrieval (scope=summaries) and two-stage retrieval
(scope=two_stage) in the dashboard API — cross-corpus / topic questions that
chunk search alone handles poorly.

Resumable: files already in the summary collection are skipped unless
`--rebuild`. Idempotent: the point id is uuid5(source_file), so a re-run
overwrites rather than duplicates.

CLI:
    python -m ingestion.build_summary_index \\
        [--limit N] [--rebuild] [--batch-size 32] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import time
import traceback
import uuid
from typing import Any

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from ingestion.utils.retries import retry_with_backoff

try:
    import psycopg2  # type: ignore[import-not-found]
    PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover
    psycopg2 = None
    PSYCOPG2_AVAILABLE = False

# Same UUIDv5 namespace the chunk ingester uses (PRD §6 Phase 4 #13) so IDs
# are stable across machines and re-runs.
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")
SUMMARY_COLLECTION = os.environ.get(
    "QDRANT_SUMMARY_COLLECTION", "transcript_summaries"
)
LOG_FILE = os.environ.get("SUMMARY_INDEX_LOG", "summary_index.log")
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT_S", "60"))

PG_DSN = os.environ.get(
    "PG_DSN",
    "postgresql://owui:{}@localhost:5432/openwebui".format(
        os.environ.get("POSTGRES_PASSWORD", "")
    ),
)

# file_meta columns pulled into each summary point. Order matters — it must
# match the cursor unpacking in iter_taggable_files().
_COLUMNS: tuple[str, ...] = (
    "source_file", "summary_english", "summary_hindi",
    "topics", "people_named", "places_named", "scriptures_referenced",
    "event_type", "primary_language",
    "season", "location", "event_id", "track_type", "session_date",
    "chunk_count", "duration_sec",
)

_LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("build_summary_index")


def setup_logging() -> None:
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(_LOG_FMT)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=100 * 1024 * 1024, backupCount=10
    )
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.propagate = False


# ---- pure helpers -------------------------------------------------------


def build_summary_text(row: dict[str, Any]) -> str:
    """Compose the text embedded for a file's summary vector.

    English + Hindi summaries together give bge-m3 (multilingual) a strong
    bilingual file signal; topic / scripture / people tags are appended so a
    keyword-ish query can still latch on. Pure function — unit-tested.
    """
    parts: list[str] = []
    if row.get("summary_english"):
        parts.append(str(row["summary_english"]).strip())
    if row.get("summary_hindi"):
        parts.append(str(row["summary_hindi"]).strip())
    if row.get("topics"):
        parts.append("Topics: " + ", ".join(row["topics"]))
    if row.get("scriptures_referenced"):
        parts.append("Scriptures: " + ", ".join(row["scriptures_referenced"]))
    if row.get("people_named"):
        parts.append("People: " + ", ".join(row["people_named"]))
    return "\n".join(p for p in parts if p).strip()


def build_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Qdrant payload for a summary point. Drops None / empty values so the
    payload stays lean and `is None` filter checks behave."""
    payload: dict[str, Any] = {}
    for key in _COLUMNS:
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, list) and not val:
            continue
        # session_date arrives as a date object — store ISO string.
        payload[key] = val.isoformat() if hasattr(val, "isoformat") else val
    return payload


# ---- Ollama embedding ---------------------------------------------------


@retry_with_backoff(max_tries=5, base=1.0, max_delay=60.0)
def embed(texts: list[str]) -> list[list[float]]:
    """Batched bge-m3 embeddings via Ollama /api/embed."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    embs = r.json().get("embeddings")
    if not embs or len(embs) != len(texts):
        raise ValueError(
            f"Ollama returned {len(embs) if embs else 0} embeddings for "
            f"{len(texts)} inputs"
        )
    return embs


# ---- Qdrant -------------------------------------------------------------


def ensure_summary_collection(client: QdrantClient, name: str) -> None:
    """Create the summary collection + payload indexes if absent. Idempotent."""
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8, always_ram=True,
                )
            ),
            hnsw_config=HnswConfigDiff(m=32, ef_construct=256),
            on_disk_payload=True,
        )
        log.info("created collection %r", name)
    else:
        log.info("collection %r already exists", name)

    for field, schema in (
        ("source_file", PayloadSchemaType.KEYWORD),
        ("event_type", PayloadSchemaType.KEYWORD),
        ("primary_language", PayloadSchemaType.KEYWORD),
        ("season", PayloadSchemaType.KEYWORD),
        ("location", PayloadSchemaType.KEYWORD),
        ("event_id", PayloadSchemaType.KEYWORD),
        ("track_type", PayloadSchemaType.KEYWORD),
        ("topics", PayloadSchemaType.KEYWORD),
        ("scriptures_referenced", PayloadSchemaType.KEYWORD),
        ("people_named", PayloadSchemaType.KEYWORD),
        ("session_date", PayloadSchemaType.DATETIME),
    ):
        try:
            client.create_payload_index(name, field, schema)
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise


def fetch_indexed_source_files(client: QdrantClient, name: str) -> set[str]:
    """Scroll the summary collection and collect source_files already
    indexed, so a resumed run can skip them."""
    indexed: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name, with_payload=["source_file"],
            with_vectors=False, limit=1000, offset=offset,
        )
        for p in points:
            sf = (p.payload or {}).get("source_file")
            if sf:
                indexed.add(sf)
        if offset is None:
            break
    return indexed


# ---- Postgres -----------------------------------------------------------


def iter_taggable_files(conn: Any) -> list[dict[str, Any]]:
    """Rows from file_meta that have a summary worth indexing."""
    cols = ", ".join(_COLUMNS)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {cols} FROM file_meta "
            "WHERE tagged_at IS NOT NULL "
            "  AND (summary_english IS NOT NULL OR summary_hindi IS NOT NULL) "
            "ORDER BY source_file"
        )
        return [dict(zip(_COLUMNS, r)) for r in cur.fetchall()]


# ---- main ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build the Qdrant transcript_summaries vector index."
    )
    p.add_argument("--limit", type=int, default=None,
                   help="Index at most N files this run.")
    p.add_argument("--rebuild", action="store_true",
                   help="Re-embed every file, even those already indexed.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip embed/upsert; just report what would be done.")
    args = p.parse_args(argv)

    setup_logging()

    if not PSYCOPG2_AVAILABLE:
        log.error("psycopg2 not importable — install psycopg2-binary")
        return 2

    try:
        conn = psycopg2.connect(PG_DSN, connect_timeout=10)
    except Exception as e:
        log.error("Postgres connection failed: %s", e)
        return 2

    qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    ensure_summary_collection(qclient, SUMMARY_COLLECTION)

    try:
        rows = iter_taggable_files(conn)
    except Exception as e:
        log.error("file_meta query failed: %s", e)
        conn.close()
        return 2
    conn.close()

    indexed: set[str] = set()
    if not args.rebuild:
        indexed = fetch_indexed_source_files(qclient, SUMMARY_COLLECTION)
        log.info("%d files already in summary index — will skip", len(indexed))

    pending = [r for r in rows if r["source_file"] not in indexed]
    if args.limit is not None:
        pending = pending[: args.limit]
    log.info(
        "summary index: %d tagged files, %d to process (rebuild=%s, dry_run=%s)",
        len(rows), len(pending), args.rebuild, args.dry_run,
    )

    started = time.monotonic()
    ok = 0
    failed = 0
    skipped_empty = 0

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prepared: list[tuple[dict[str, Any], str]] = []
        for row in batch:
            text = build_summary_text(row)
            if not text:
                skipped_empty += 1
                log.warning("%s: empty summary text — skipped",
                            row["source_file"])
                continue
            prepared.append((row, text))

        if not prepared or args.dry_run:
            ok += len(prepared)
            continue

        try:
            vectors = embed([t for _, t in prepared])
        except Exception as e:
            failed += len(prepared)
            log.error("batch embed failed (%d files): %s\n%s",
                      len(prepared), e, traceback.format_exc())
            continue

        points: list[PointStruct] = []
        for (row, _text), vec in zip(prepared, vectors):
            pid = str(uuid.uuid5(NAMESPACE, row["source_file"]))
            points.append(PointStruct(
                id=pid, vector=vec, payload=build_payload(row),
            ))
        try:
            qclient.upsert(collection_name=SUMMARY_COLLECTION,
                           points=points, wait=True)
            ok += len(points)
        except Exception as e:
            failed += len(points)
            log.error("qdrant upsert failed (%d points): %s", len(points), e)

        done = start + len(batch)
        log.info("[%d/%d] indexed (ok=%d failed=%d)",
                 done, len(pending), ok, failed)

    elapsed = time.monotonic() - started
    log.info(
        "summary index summary: ok=%d failed=%d skipped_empty=%d elapsed=%.1fs",
        ok, failed, skipped_empty, elapsed,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
