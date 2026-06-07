"""Offline PageIndex tree builder. Companion to the `backend=pageindex`
retrieval option (rag_api.pageindex).

For each selected transcript it reconstructs the full text from chunk_meta,
groups consecutive chunks into sections, asks the local Ollama model for a
title + summary per section (and a one-line doc description), and writes the
resulting tree to a JSON file under PAGEINDEX_DIR. The live retriever reads
those JSON files — nothing here touches Qdrant, Tantivy, or the hybrid path.

This is a *bounded* indexer by design: tree-building is one LLM call per
section, so you index a curated sample to benchmark quality/speed, not the
whole 5 TB corpus. Per PRD rule #7 — failures are detected, logged, dead-
lettered, and resumable (already-built trees are skipped unless --rebuild).

CLI:
    python -m ingestion.build_pageindex \\
        [--source-file NAME ...] [--limit N] [--rebuild] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from rag_api.config import get_settings
from rag_api.pageindex import PageIndexBuilder, safe_tree_name, tree_path

try:
    import psycopg2  # type: ignore[import-not-found]
    PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover
    psycopg2 = None
    PSYCOPG2_AVAILABLE = False

_LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE = "pageindex.log"
log = logging.getLogger("build_pageindex")


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


# ---- Postgres I/O -------------------------------------------------------


class ChunkStore:
    """Reads the chunk_meta rows the tree builder needs."""

    def __init__(self, dsn: str) -> None:
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not importable — install psycopg2-binary")
        self.conn = psycopg2.connect(dsn, connect_timeout=10)
        self.conn.autocommit = True

    def list_files(self, limit: int | None) -> list[str]:
        sql = "SELECT source_file FROM file_meta ORDER BY source_file"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]

    def load_chunk_rows(self, source_file: str) -> list[dict[str, Any]]:
        """Ordered chunk records for one file. Same ordering as
        `enrich_content_tags.load_transcript`: by start_sec, NULLs last."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, text, start_sec, end_sec, speakers "
                "FROM chunk_meta WHERE source_file = %s "
                "ORDER BY start_sec NULLS LAST, chunk_id",
                (source_file,),
            )
            rows = cur.fetchall()
        return [
            {
                "chunk_id": str(r[0]),
                "text": r[1] or "",
                "start_sec": r[2],
                "end_sec": r[3],
                "speakers": list(r[4]) if r[4] else [],
                "source_file": source_file,
            }
            for r in rows
        ]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001 - best-effort close
            pass


# ---- dead-letter --------------------------------------------------------


def dead_letter(out_dir: Path, source_file: str, reason: str) -> Path:
    """Record a failed build for later forensics."""
    target_dir = out_dir / "_failures"
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / f"{safe_tree_name(source_file)}.txt"
    try:
        dst.write_text(
            f"source_file: {source_file}\nreason: {reason}\n", encoding="utf-8"
        )
    except OSError as e:
        log.error("dead-letter write failed for %s: %s", source_file, e)
    return dst


# ---- core ---------------------------------------------------------------


def process_one_file(
    source_file: str, store: ChunkStore, builder: PageIndexBuilder,
    out_dir: Path, dry_run: bool,
) -> tuple[bool, str | None]:
    """Build + persist one tree. Returns (success, reason_if_failed)."""
    rows = store.load_chunk_rows(source_file)
    if not rows:
        return False, "no chunk_meta rows"
    if dry_run:
        log.info("[dry-run] %s: %d chunks would be indexed", source_file, len(rows))
        return True, None

    t0 = time.monotonic()
    try:
        tree = builder.build(source_file, rows)
    except Exception as e:  # noqa: BLE001 - isolate per-file, surface reason
        return False, f"{type(e).__name__}: {e}"

    dst = tree_path(out_dir, source_file)
    dst.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "%s: %d sections from %d chunks in %.1fs -> %s",
        source_file, len(tree["nodes"]), len(rows), time.monotonic() - t0, dst.name,
    )
    return True, None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build PageIndex trees for transcripts.")
    p.add_argument("--source-file", action="append", default=None,
                   help="Index a specific source_file (repeatable).")
    p.add_argument("--limit", type=int, default=None,
                   help="Index at most N files from file_meta.")
    p.add_argument("--rebuild", action="store_true",
                   help="Rebuild trees that already exist (default: skip them).")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be indexed; no LLM calls or writes.")
    args = p.parse_args(argv)

    setup_logging()
    settings = get_settings()
    out_dir = Path(settings.pageindex_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        store = ChunkStore(settings.pg_dsn)
    except Exception as e:  # noqa: BLE001 - fatal init, report and exit
        log.error("Postgres init failed: %s", e)
        return 2

    builder = PageIndexBuilder(settings)

    files = args.source_file or store.list_files(args.limit)
    if not args.rebuild and not args.dry_run:
        kept = [f for f in files if not tree_path(out_dir, f).is_file()]
        skipped = len(files) - len(kept)
        if skipped:
            log.info("skipping %d already-built tree(s); use --rebuild to force",
                     skipped)
        files = kept

    log.info("queued %d file(s) for PageIndex (model=%s, dir=%s, dry_run=%s)",
             len(files), settings.pageindex_model, out_dir, args.dry_run)

    started = time.monotonic()
    ok = failed = 0
    for idx, src in enumerate(files, start=1):
        try:
            success, reason = process_one_file(
                src, store, builder, out_dir, args.dry_run
            )
            if success:
                ok += 1
            else:
                failed += 1
                dst = dead_letter(out_dir, src, reason or "unknown")
                log.error("[%d/%d] %s: %s — dead-letter %s",
                          idx, len(files), src, reason, dst)
        except Exception as e:  # noqa: BLE001 - never let one file kill the run
            failed += 1
            log.error("[%d/%d] %s: unexpected %s\n%s",
                      idx, len(files), src, type(e).__name__, traceback.format_exc())

    log.info("pageindex summary: ok=%d failed=%d elapsed=%.1fs",
             ok, failed, time.monotonic() - started)
    store.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
