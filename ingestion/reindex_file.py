"""Per-file reindex: delete one transcript's data across every store so it can
be cleanly re-chunked and re-ingested. (Stage 1.3)

Re-chunking is otherwise a no-op-that-orphans. ``bulk_ingest_hardened`` skips
files already marked ``ok``; and even forced, a changed transcript orphans its
old data because Qdrant point IDs hash the chunk text and Tantivy is
append-only. No delete path existed anywhere. This is that path. For each
``source_file`` — the qualified relative path stored in every store (see
``chunker_cleaned.qualified_source``, e.g.
``"Live Masters 2010/01 NOIDA .../04 PRAVACHAN.json"``) — it runs five deletes:

  1. Qdrant   — delete every point whose ``payload.source_file == src``
  2. Tantivy  — ``delete_documents("source_file", src)`` + commit
  3. Postgres — ``DELETE FROM chunk_meta WHERE source_file = src``
  4. Postgres — ``UPDATE file_meta SET tagged_at = NULL WHERE source_file = src``
                (so a re-chunked file re-enriches in Stage 2)
  5. Progress — clear the ``ingest_status`` row so bulk_ingest re-processes it

After this, re-chunk (``chunker_cleaned``) and re-ingest
(``bulk_ingest_hardened --no-skip-existing``) rebuild the file cleanly.

This is a DESTRUCTIVE, single-writer operation: no other process may write the
stores concurrently (CLAUDE.md — never run two writers). ``--dry-run`` reports
the per-store counts it *would* delete without touching anything.

Usage:
    python -m ingestion.reindex_file --source "Live Masters 2010/.../04 PRAVACHAN.json"
    python -m ingestion.reindex_file --source-list files.txt        # one per line
    python -m ingestion.reindex_file --worklist ingestion/chunk_audit_worklist.json \
        --oversize-only [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import psycopg2
from qdrant_client import QdrantClient, models

from ingestion.bulk_ingest_hardened import (
    PG_DSN,
    PROGRESS_DB_PATH,
    QDRANT_COLLECTION,
    QDRANT_KEY,
    QDRANT_URL,
    TANTIVY_DIR,
    TantivyWriter,
)
from ingestion.utils.progress_db import ProgressDB

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("reindex_file")


def progress_key_for_source(source_file: str) -> str:
    """The ``ingest_status`` key (the flat .chunks.json name) for a qualified
    ``source_file``. Mirrors ``chunker_cleaned.output_basename`` exactly so the
    progress row matches what bulk_ingest wrote."""
    key = source_file[:-5] if source_file.endswith(".json") else source_file
    safe = re.sub(r'[\\/:*?"<>|]+', "__", key)
    if len(safe) > 180:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[:170]}__{digest}"
    return f"{safe}.chunks.json"


def _src_filter(src: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(
        key="source_file", match=models.MatchValue(value=src))])


def qdrant_count(client: QdrantClient, src: str) -> int:
    return client.count(QDRANT_COLLECTION, count_filter=_src_filter(src),
                        exact=True).count


def reindex_one(
    src: str,
    qclient: QdrantClient,
    tantivy_writer: TantivyWriter,
    pg_conn: Any,
    progress: ProgressDB,
    dry_run: bool,
) -> dict[str, Any]:
    """Delete `src` from all five stores. Returns per-store counts. Postgres
    and Qdrant errors are surfaced (raised) — a partial delete must not look
    like success."""
    n_qdrant = qdrant_count(qclient, src)
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunk_meta WHERE source_file = %s", (src,))
        n_chunk_meta = cur.fetchone()[0]

    if dry_run:
        return {"source_file": src, "qdrant": n_qdrant, "chunk_meta": n_chunk_meta,
                "dry_run": True}

    # 1. Qdrant filtered delete
    qclient.delete(QDRANT_COLLECTION,
                   points_selector=models.FilterSelector(filter=_src_filter(src)),
                   wait=True)
    # 2. Tantivy delete (commit so it takes effect immediately)
    tantivy_writer.delete_documents(src)
    tantivy_writer.commit()
    # 3 + 4. Postgres chunk_meta delete + file_meta re-enrich flag
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM chunk_meta WHERE source_file = %s", (src,))
        deleted_cm = cur.rowcount
        cur.execute(
            "UPDATE file_meta SET tagged_at = NULL WHERE source_file = %s", (src,))
        updated_fm = cur.rowcount
    pg_conn.commit()
    # 5. Progress reset so bulk_ingest re-processes the file
    pkey = progress_key_for_source(src)
    progress.reset_to_pending(pkey)

    log.info("reindexed %s: qdrant=%d chunk_meta=%d file_meta=%d progress_key=%s",
             src, n_qdrant, deleted_cm, updated_fm, pkey)
    return {"source_file": src, "qdrant": n_qdrant, "chunk_meta": deleted_cm,
            "file_meta": updated_fm, "progress_key": pkey}


def _load_sources(args: argparse.Namespace) -> list[str]:
    srcs: list[str] = list(args.source or [])
    if args.source_list:
        srcs += [ln.strip() for ln in Path(args.source_list).read_text(
            encoding="utf-8").splitlines() if ln.strip()]
    if args.worklist:
        wl = json.loads(Path(args.worklist).read_text(encoding="utf-8"))
        rows = wl.get("worklist", [])
        if args.oversize_only:
            rows = [r for r in rows if r.get("over_char_chunks", 0)
                    or r.get("over_token_chunks", 0)]
        srcs += [r["source_file"] for r in rows]
    # de-dup, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for s in srcs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-file reindex delete across all stores (1.3).")
    p.add_argument("--source", action="append", help="A source_file (repeatable).")
    p.add_argument("--source-list", help="File with one source_file per line.")
    p.add_argument("--worklist", help="chunk_audit_worklist.json to pull source_files from.")
    p.add_argument("--oversize-only", action="store_true",
                   help="With --worklist: only files with an oversize chunk "
                        "(the ~50-file re-chunk set); default is every flagged file.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report per-store counts without deleting anything.")
    p.add_argument("--out", default=None, help="Write the per-file report JSON here.")
    args = p.parse_args(argv)

    sources = _load_sources(args)
    if not sources:
        log.error("no source_files given (use --source / --source-list / --worklist)")
        return 2
    log.info("%s %d file(s)%s", "DRY-RUN over" if args.dry_run else "reindexing",
             len(sources), " (oversize-only)" if args.oversize_only else "")

    qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    tantivy_writer = TantivyWriter(TANTIVY_DIR)
    pg_conn = psycopg2.connect(PG_DSN, connect_timeout=10)
    progress = ProgressDB(PROGRESS_DB_PATH)

    reports: list[dict[str, Any]] = []
    tot_q = tot_cm = 0
    try:
        for src in sources:
            rep = reindex_one(src, qclient, tantivy_writer, pg_conn, progress,
                              args.dry_run)
            reports.append(rep)
            tot_q += rep.get("qdrant", 0)
            tot_cm += rep.get("chunk_meta", 0)
    finally:
        if not args.dry_run:
            tantivy_writer.close()  # final commit + drop the writer lock
        pg_conn.close()
        progress.close()

    verb = "would delete" if args.dry_run else "deleted"
    log.info("done: %d files, %s %d qdrant points / %d chunk_meta rows",
             len(reports), verb, tot_q, tot_cm)
    if args.out:
        Path(args.out).write_text(
            json.dumps({"dry_run": args.dry_run, "reports": reports},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("report -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
