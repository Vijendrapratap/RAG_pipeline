"""Hardened bulk ingestion pipeline. Per PRD §6 Phase 4.

Implements all 16 requirements: health check, SQLite progress, per-file
try/except + SIGALRM timeout, 500 MB cap, encoding recovery, retry decorators
on every HTTP call, NaN/Inf detection, periodic Tantivy commits, signal
handlers, memory monitoring, dead-letter quarantine, UUIDv5 chunk IDs,
guarded Postgres chunk_meta insert, rotating-file structured logging, periodic
progress lines.

CLI:
    python -m ingestion.bulk_ingest_hardened \\
        --chunks-dir /data/processed \\
        --batch-size 32 \\
        [--retry-failed] [--dry-run] [--max-files N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import logging.handlers
import math
import os
import shutil
import signal
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import psutil
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from ingestion.utils.encoding import read_text_robust
from ingestion.utils.health import check_all
from ingestion.utils.progress_db import ProgressDB
from ingestion.utils.retries import retry_with_backoff

# Tantivy is an optional import — the integration test may or may not have it
# installed and the bge-m3 sidecar (Phase 5) is the consumer. If missing, we
# log and continue with Qdrant-only writes.
try:
    import tantivy  # type: ignore[import-not-found]
    TANTIVY_AVAILABLE = True
except ImportError:  # pragma: no cover
    tantivy = None
    TANTIVY_AVAILABLE = False

try:
    import psycopg2  # type: ignore[import-not-found]
    import psycopg2.errors  # type: ignore[import-not-found]
    PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover
    psycopg2 = None
    PSYCOPG2_AVAILABLE = False


# UUIDv5 namespace per PRD §6 Phase 4 #13 — this is the standard
# uuid.NAMESPACE_DNS. Keeping it as a literal so re-running across machines
# produces the same chunk IDs even if uuid module internals shift.
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

MAX_FILE_BYTES = 500 * 1024 * 1024  # PRD #5
FILE_TIMEOUT_SEC = 30 * 60  # PRD #4
TANTIVY_COMMIT_EVERY = 50  # PRD #9
MEM_CHECK_EVERY = 10  # PRD #11
MEM_WARN_PCT = 85.0
MEM_ABORT_PCT = 95.0
PROGRESS_EVERY = 50  # PRD #16

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "transcripts")
TANTIVY_DIR = Path(os.environ.get("TANTIVY_DIR", "./data/tantivy"))
DEAD_LETTER_DIR = Path(os.environ.get("DEAD_LETTER_DIR", "./dead_letter"))
PROGRESS_DB_PATH = os.environ.get("PROGRESS_DB", "./ingest_progress.sqlite")
LOG_FILE = os.environ.get("INGEST_LOG", "ingest.log")
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT_S", "60"))

# Postgres (Phase 6 will create the chunk_meta schema; we attempt writes here
# and gracefully disable on UndefinedTable).
PG_DSN = os.environ.get(
    "PG_DSN",
    "postgresql://owui:{}@localhost:5432/openwebui".format(
        os.environ.get("POSTGRES_PASSWORD", "")
    ),
)


_LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("bulk_ingest")


def setup_logging() -> None:
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(_LOG_FMT)
    # Stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    # Rotating file (100 MB x 10) per PRD #15
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=100 * 1024 * 1024, backupCount=10
    )
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.propagate = False


# ---- stable chunk ids (PRD #13) ----------------------------------------


def chunk_uuid(source_file: str, index: int, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    key = f"{source_file}|{index}|{h}"
    return str(uuid.uuid5(NAMESPACE, key))


# ---- embed + upsert with retry (PRD #7) --------------------------------


@retry_with_backoff(max_tries=5, base=1.0, max_delay=60.0)
def embed(texts: list[str]) -> list[list[float]]:
    """Call Ollama batched embeddings API for bge-m3."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    embs = data.get("embeddings")
    if not embs or len(embs) != len(texts):
        raise ValueError(
            f"Ollama returned {len(embs) if embs else 0} embeddings for "
            f"{len(texts)} inputs"
        )
    return embs


@retry_with_backoff(max_tries=5, base=1.0, max_delay=60.0)
def qdrant_upsert(client: QdrantClient, points: list[PointStruct]) -> None:
    client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)


# ---- NaN/Inf guard (PRD #8) --------------------------------------------


def vec_is_finite(vec: list[float]) -> bool:
    return not any(math.isnan(x) or math.isinf(x) for x in vec)


# ---- Postgres writer (PRD #14, guarded; activates after Phase 6) -------


class PostgresWriter:
    """Inserts into chunk_meta if the table exists; auto-disables on first
    UndefinedTable error so Phase 6 can drop in the schema without code change."""

    def __init__(self) -> None:
        self.enabled = False
        self.conn = None
        if not PSYCOPG2_AVAILABLE:
            log.info("psycopg2 not importable — chunk_meta writes disabled")
            return
        try:
            self.conn = psycopg2.connect(PG_DSN, connect_timeout=5)
            self.conn.autocommit = False
            with self.conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.chunk_meta')")
                exists = cur.fetchone()[0]
            if exists is None:
                log.info(
                    "Postgres chunk_meta table not yet created (Phase 6) "
                    "— chunk_meta writes disabled"
                )
                self.conn.close()
                self.conn = None
            else:
                self.enabled = True
                log.info("Postgres chunk_meta writes enabled")
        except Exception as e:
            log.warning("Postgres connection failed: %s — chunk_meta writes disabled", e)
            self.conn = None

    def write_batch(self, rows: list[tuple[Any, ...]]) -> None:
        if not self.enabled or not self.conn:
            return
        try:
            with self.conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunk_meta
                         (chunk_id, source_file, speakers, start_sec, end_sec,
                          word_count, text)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (chunk_id) DO NOTHING""",
                    rows,
                )
            self.conn.commit()
        except Exception as e:
            log.error("Postgres batch insert failed: %s — disabling writes", e)
            try:
                self.conn.rollback()
            except Exception:
                pass
            self.enabled = False

    def close(self) -> None:
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass


# ---- Tantivy wrapper ---------------------------------------------------


class TantivyWriter:
    """Wrap a tantivy index writer; supports periodic commits and graceful
    shutdown. Schema matches Phase 5 server reader: chunk_id (raw),
    text (default tokenizer), source_file (raw)."""

    def __init__(self, index_dir: Path) -> None:
        self.enabled = TANTIVY_AVAILABLE
        if not self.enabled:
            log.warning("tantivy not importable — Tantivy writes disabled")
            self.index = None
            self.writer = None
            return
        index_dir.mkdir(parents=True, exist_ok=True)
        sb = tantivy.SchemaBuilder()
        sb.add_text_field("chunk_id", stored=True, tokenizer_name="raw")
        sb.add_text_field("text", stored=True, tokenizer_name="default")
        sb.add_text_field("source_file", stored=True, tokenizer_name="raw")
        schema = sb.build()
        self.index = tantivy.Index(schema, path=str(index_dir))
        # 50 MB writer heap is the tantivy-py minimum default; bump higher for
        # bulk loading throughput.
        self.writer = self.index.writer(heap_size=128 * 1024 * 1024, num_threads=1)

    def add(self, chunk_id: str, text: str, source_file: str) -> None:
        if not self.enabled or self.writer is None:
            return
        doc = tantivy.Document()
        doc.add_text("chunk_id", chunk_id)
        doc.add_text("text", text)
        doc.add_text("source_file", source_file)
        self.writer.add_document(doc)

    def commit(self) -> None:
        if not self.enabled or self.writer is None:
            return
        self.writer.commit()
        # writer is one-shot per commit in tantivy-py; reopen for further adds.
        self.writer = self.index.writer(heap_size=128 * 1024 * 1024, num_threads=1)

    def close(self) -> None:
        try:
            self.commit()
        except Exception as e:
            log.error("final tantivy commit failed: %s", e)


# ---- Signal handling (PRD #10) -----------------------------------------


class GracefulExit(Exception):
    pass


_SHUTDOWN_REQUESTED = False


def _install_signal_handlers() -> None:
    def handler(signum: int, _frame: Any) -> None:
        global _SHUTDOWN_REQUESTED
        log.warning("received signal %d — requesting graceful shutdown", signum)
        _SHUTDOWN_REQUESTED = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # Not in main thread (e.g. some pytest setups) — best-effort.
            pass


def _install_file_alarm(seconds: int) -> None:
    if not hasattr(signal, "SIGALRM"):  # Windows fallback — no-op
        return

    def handler(signum: int, _frame: Any) -> None:
        raise TimeoutError(f"per-file timeout after {seconds}s")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)


def _cancel_file_alarm() -> None:
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)


# ---- dead-letter quarantine (PRD #12) ----------------------------------


def quarantine(file_path: Path, reason: str) -> Path:
    """Copy file_path into dead_letter/<reason>/ and return the destination."""
    safe_reason = "".join(c if c.isalnum() or c in "-_" else "_" for c in reason)[:64]
    target_dir = DEAD_LETTER_DIR / safe_reason
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / file_path.name
    try:
        shutil.copy2(file_path, dst)
    except Exception as e:
        log.error("could not copy %s to dead-letter %s: %s", file_path, dst, e)
    return dst


# ---- per-file processing -----------------------------------------------


def _coerce_payload(chunk: dict[str, Any], source_file: str) -> dict[str, Any]:
    """Build the Qdrant payload from a chunker output chunk."""
    return {
        "text": chunk["text"],
        "source_file": chunk.get("source_file", source_file),
        "speakers": chunk.get("speakers", []) or [],
        "start_sec": chunk.get("start_sec"),
        "end_sec": chunk.get("end_sec"),
        "word_count": chunk.get("word_count"),
        "has_timestamps": chunk.get("has_timestamps", True),
    }


def process_file(
    chunks_path: Path,
    qdrant_client: QdrantClient,
    tantivy_writer: TantivyWriter,
    pg_writer: PostgresWriter,
    batch_size: int,
    dry_run: bool,
) -> int:
    """Ingest one .chunks.json file. Returns number of chunks ingested."""
    size = chunks_path.stat().st_size
    if size > MAX_FILE_BYTES:
        dst = quarantine(chunks_path, "oversize")
        log.warning(
            "%s: %.1f MB > %d MB cap — quarantined to %s",
            chunks_path.name, size / 1e6, MAX_FILE_BYTES // (1024 * 1024), dst,
        )
        raise ValueError(f"oversize ({size} bytes)")

    # JSON is the on-disk format; encoding fallback covers edge cases where
    # the chunker received content that smuggled in non-utf-8 bytes.
    text = read_text_robust(chunks_path)
    data = json.loads(text)
    source_file = data.get("source_file", chunks_path.name)
    chunks = data.get("chunks", [])
    if not chunks:
        log.info("%s: 0 chunks — nothing to ingest", chunks_path.name)
        return 0

    total = 0
    for start in range(0, len(chunks), batch_size):
        if _SHUTDOWN_REQUESTED:
            raise GracefulExit()
        batch = chunks[start : start + batch_size]
        texts = [c["text"] for c in batch]
        ids = [chunk_uuid(source_file, start + i, c["text"]) for i, c in enumerate(batch)]

        if dry_run:
            total += len(batch)
            continue

        vectors = embed(texts)

        points: list[PointStruct] = []
        pg_rows: list[tuple[Any, ...]] = []
        for cid, vec, chunk in zip(ids, vectors, batch):
            if not vec_is_finite(vec):
                log.error("%s: NaN/Inf in embedding for chunk %s — skipping",
                          chunks_path.name, cid)
                continue
            payload = _coerce_payload(chunk, source_file)
            points.append(PointStruct(id=cid, vector=vec, payload=payload))
            pg_rows.append((
                cid, source_file, payload["speakers"],
                payload["start_sec"], payload["end_sec"],
                payload["word_count"], payload["text"],
            ))

        if points:
            qdrant_upsert(qdrant_client, points)
            for cid, chunk in zip(ids, batch):
                tantivy_writer.add(cid, chunk["text"], source_file)
            pg_writer.write_batch(pg_rows)
            total += len(points)

    return total


# ---- main loop ---------------------------------------------------------


def _progress_line(idx: int, total: int, fname: str, n_chunks: int,
                   running_total: int, started: float) -> str:
    elapsed = time.monotonic() - started
    rate = running_total / elapsed if elapsed > 0 else 0.0
    mem = psutil.virtual_memory().percent
    return (
        f"[{idx}/{total}] {fname}: +{n_chunks} chunks "
        f"(total {running_total}, {rate:.1f}/s, RAM {mem:.0f}%)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hardened bulk transcript ingestion.")
    parser.add_argument("--chunks-dir", required=True,
                        help="Directory containing *.chunks.json files")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt files previously marked 'failed'.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip embed/upsert; useful for plumbing checks.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Stop after N files (for smoke tests).")
    parser.add_argument("--no-tantivy-health", action="store_true",
                        help="Skip Tantivy in startup health check (sidecar may not be up).")
    args = parser.parse_args(argv)

    setup_logging()

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.is_dir():
        log.error("--chunks-dir not a directory: %s", chunks_dir)
        return 2

    # PRD #1 — fail fast if dependencies down
    health = check_all(include_tantivy=not args.no_tantivy_health)
    if not all(health.get(k, True) for k in ("ollama", "qdrant")):
        log.error("health check failed: %s — exiting", health)
        return 2
    if not args.no_tantivy_health and not health.get("tantivy", True):
        log.warning("Tantivy HTTP sidecar not up — proceeding with direct "
                    "index writes (sidecar is read-only)")

    _install_signal_handlers()

    qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    tantivy_writer = TantivyWriter(TANTIVY_DIR)
    pg_writer = PostgresWriter()
    progress = ProgressDB(PROGRESS_DB_PATH)

    files = sorted(chunks_dir.glob("*.chunks.json"))
    if args.max_files:
        files = files[: args.max_files]
    total_files = len(files)
    log.info("starting ingest: %d candidate files in %s", total_files, chunks_dir)

    started = time.monotonic()
    processed = 0
    skipped = 0
    failed = 0
    grand_total_chunks = 0
    exit_code = 0

    try:
        for idx, fpath in enumerate(files, start=1):
            if _SHUTDOWN_REQUESTED:
                log.warning("shutdown requested before file %d", idx)
                break

            existing = progress.get_status(fpath.name)
            if existing == "ok":
                skipped += 1
                continue
            if existing == "failed" and not args.retry_failed:
                skipped += 1
                continue

            progress.mark_in_progress(fpath.name)

            try:
                _install_file_alarm(FILE_TIMEOUT_SEC)
                n = process_file(
                    fpath, qclient, tantivy_writer, pg_writer,
                    args.batch_size, args.dry_run,
                )
                _cancel_file_alarm()
                progress.mark_ok(fpath.name, n)
                grand_total_chunks += n
                processed += 1

                if idx % PROGRESS_EVERY == 0 or idx == total_files:
                    log.info(_progress_line(
                        idx, total_files, fpath.name, n,
                        grand_total_chunks, started,
                    ))
            except GracefulExit:
                _cancel_file_alarm()
                log.warning("interrupted at file %s", fpath.name)
                progress.mark_failed(fpath.name, "interrupted")
                exit_code = 130
                break
            except TimeoutError as e:
                _cancel_file_alarm()
                log.error("%s: timeout: %s", fpath.name, e)
                quarantine(fpath, "timeout")
                progress.mark_failed(fpath.name, "timeout")
                failed += 1
            except Exception as e:
                _cancel_file_alarm()
                reason = f"{type(e).__name__}: {e}"
                log.error("%s: %s\n%s", fpath.name, reason, traceback.format_exc())
                quarantine(fpath, type(e).__name__)
                progress.mark_failed(fpath.name, reason)
                failed += 1

            if idx % TANTIVY_COMMIT_EVERY == 0:
                try:
                    tantivy_writer.commit()
                    log.info("tantivy commit at file %d", idx)
                except Exception as e:
                    log.error("periodic tantivy commit failed: %s", e)

            if idx % MEM_CHECK_EVERY == 0:
                pct = psutil.virtual_memory().percent
                if pct >= MEM_ABORT_PCT:
                    log.error("memory %.1f%% >= %.1f%% — aborting gracefully",
                              pct, MEM_ABORT_PCT)
                    break
                if pct >= MEM_WARN_PCT:
                    log.warning("memory %.1f%% >= %.1f%%", pct, MEM_WARN_PCT)

    finally:
        try:
            tantivy_writer.close()
        except Exception as e:
            log.error("tantivy close failed: %s", e)
        pg_writer.close()
        stats = progress.stats()
        progress.close()
        elapsed = time.monotonic() - started
        log.info(
            "ingest summary: processed=%d skipped=%d failed=%d total_chunks=%d "
            "elapsed=%.1fs db_status=%s",
            processed, skipped, failed, grand_total_chunks, elapsed, stats,
        )

    if exit_code == 0 and failed > 0:
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
