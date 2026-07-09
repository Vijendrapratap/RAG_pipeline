"""Per-file content-tag enrichment via Qwen 3.5 9B (Ollama). PRD §6 Phase 13.

Runs AFTER bulk_ingest_hardened.py. Selects files with `tagged_at IS NULL`
from file_meta, reconstructs the transcript from chunk_meta rows, calls
Qwen, validates the JSON, writes tags back to file_meta, and propagates
the tags into each Qdrant chunk payload (via filtered set_payload) so
search-time filters work.

Resumable: re-runs skip already-tagged files. Failures go to dead-letter
with the raw model response. The Phase 12 path metadata is untouched.

CLI:
    python -m ingestion.enrich_content_tags \\
        [--limit N] [--retry-failed] [--dry-run] \\
        [--model qwen3.5:9b] [--max-tokens-single-pass 28000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from ingestion.utils.retries import retry_with_backoff
from ingestion.utils.tag_schema import (
    build_mapreduce_chunk_prompt,
    build_prompt,
    parse_model_json,
    validate_tags,
)

try:
    import psycopg2  # type: ignore[import-not-found]
    PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover
    psycopg2 = None
    PSYCOPG2_AVAILABLE = False


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
TAG_MODEL = os.environ.get("TAG_MODEL", "qwen3.5:9b")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "transcripts")
DEAD_LETTER_DIR = Path(os.environ.get("DEAD_LETTER_DIR", "./dead_letter"))
LOG_FILE = os.environ.get("ENRICH_LOG", "enrich.log")
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT_S", "300"))  # tagging is slow

PG_DSN = os.environ.get(
    "PG_DSN",
    "postgresql://owui:{}@localhost:5432/openwebui".format(
        os.environ.get("POSTGRES_PASSWORD", "")
    ),
)

# Single-pass budget. 1 token ≈ 4 chars (rough, OK for English+Hindi mix).
# A 28k-token budget ≈ 112k chars of transcript, well under Qwen's 32k
# context once we add the schema prompt (~600 tokens of overhead).
DEFAULT_MAX_TOKENS_SINGLE_PASS = 28_000
CHARS_PER_TOKEN = 4

# Ollama num_ctx for the tagging call. Has to be ≥ prompt+output tokens.
# 32768 is comfortable for qwen3.5:9b q4 on the RTX 5090 (32 GB VRAM); raising
# it risks spilling the KV cache to system RAM, which makes tagging *slower*.
NUM_CTX = 32_768

# Cap the model's OUTPUT length. The tag JSON is ~10 fields + two short
# summaries — a well-behaved response is well under 2k tokens. Bounding it
# stops a runaway repetition loop (observed on refrain-heavy bhajans) from
# generating until the HTTP read times out, which — with retries — burned
# ~15 min on a single file during calibration. A truncated response just
# fails JSON parse and dead-letters fast instead.
NUM_PREDICT = int(os.environ.get("TAG_NUM_PREDICT", "2048"))
# Discourage the token-level repetition loops that produce those runaways.
REPEAT_PENALTY = float(os.environ.get("TAG_REPEAT_PENALTY", "1.1"))

_LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("enrich_content_tags")


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


# ---- Ollama call --------------------------------------------------------


@retry_with_backoff(max_tries=3, base=2.0, max_delay=120.0)
def ollama_generate_json(prompt: str, model: str) -> str:
    """POST to /api/generate with format=json. Returns raw response text.

    `think=False` is mandatory: qwen3.5:9b is a thinking model, and with
    format=json its output routes into the `thinking` field, leaving
    `response` EMPTY — every file would then dead-letter as "empty response".
    It is also accepted by non-thinking models, so it is safe regardless of
    which TAG_MODEL is configured. (Mirrors rag_api/pageindex.py.)

    A read timeout is re-raised as a non-retryable RuntimeError: it means the
    model is stuck generating (a repetition loop on degenerate input), so
    retrying the identical prompt would just time out again — three times over.
    Failing fast dead-letters the file after one timeout instead of ~3×.
    """
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "think": False,
                "options": {
                    "num_ctx": NUM_CTX,
                    "num_predict": NUM_PREDICT,
                    "temperature": 0.0,
                    "repeat_penalty": REPEAT_PENALTY,
                },
            },
            timeout=HTTP_TIMEOUT,
        )
    except requests.exceptions.ReadTimeout as e:
        raise RuntimeError(
            f"ollama read timeout after {HTTP_TIMEOUT}s — model likely stuck "
            "generating on degenerate input; not retrying"
        ) from e
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


# ---- dead-letter --------------------------------------------------------


def dead_letter(source_file: str, raw: str, reason: str) -> Path:
    """Persist a bad tagging attempt for later forensics. Returns dst path."""
    # Slug for readability, plus a hash of the FULL source_file so two long
    # paths that share a 100-char prefix don't overwrite each other's artifact.
    slug = "".join(c if c.isalnum() or c in "-_." else "_" for c in source_file)[:100]
    digest = hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:12]
    target_dir = DEAD_LETTER_DIR / "tag_failures"
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / f"{slug}.{digest}.txt"
    try:
        dst.write_text(
            f"source_file: {source_file}\nreason: {reason}\n\n--- raw response ---\n{raw}\n",
            encoding="utf-8",
        )
    except Exception as e:
        log.error("dead-letter write failed for %s: %s", source_file, e)
    return dst


# ---- Postgres I/O -------------------------------------------------------


class TagStore:
    """Wraps the file_meta + chunk_meta queries Phase 13 needs."""

    def __init__(self) -> None:
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not importable — install psycopg2-binary")
        self.conn = psycopg2.connect(PG_DSN, connect_timeout=10)
        self.conn.autocommit = False

    def list_untagged(
        self, limit: int | None, retry_failed: bool
    ) -> list[str]:
        """Source files needing tagging. tagged_at NULL = never tagged.
        retry_failed has no special meaning yet — we don't currently mark
        files as 'tag_failed'; failures leave tagged_at NULL so a re-run
        retries automatically. The flag is kept for future use."""
        _ = retry_failed  # reserved
        sql = (
            "SELECT source_file FROM file_meta "
            "WHERE tagged_at IS NULL ORDER BY source_file"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]

    def load_transcript(self, source_file: str) -> str:
        """Reconstruct the file's full transcript from chunk_meta, ordered
        by start_sec. NULL start_sec rows come last (text chunks without
        timestamps)."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT text FROM chunk_meta WHERE source_file = %s "
                "ORDER BY start_sec NULLS LAST, chunk_id",
                (source_file,),
            )
            rows = cur.fetchall()
        return "\n".join(r[0] or "" for r in rows)

    def write_tags(self, source_file: str, tags: dict[str, Any], model: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """UPDATE file_meta SET
                       event_type            = %s,
                       primary_language      = %s,
                       topics                = %s,
                       people_named          = %s,
                       places_named          = %s,
                       scriptures_referenced = %s,
                       timing_clues          = %s,
                       location_clues        = %s,
                       summary_hindi         = %s,
                       summary_english       = %s,
                       tagged_at             = NOW(),
                       tag_model             = %s
                   WHERE source_file = %s""",
                (
                    tags.get("event_type"),
                    tags.get("primary_language"),
                    tags.get("topics") or [],
                    tags.get("people_named") or [],
                    tags.get("places_named") or [],
                    tags.get("scriptures_referenced") or [],
                    tags.get("timing_clues") or [],
                    tags.get("location_clues") or [],
                    tags.get("summary_hindi"),
                    tags.get("summary_english"),
                    model,
                    source_file,
                ),
            )
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


# ---- Qdrant payload propagation -----------------------------------------


# Subset of tag fields we push into chunk payloads for filter-aware search.
# Summaries are large free text — we keep them in Postgres only.
_PAYLOAD_TAG_FIELDS = (
    "event_type", "primary_language", "topics",
    "people_named", "places_named", "scriptures_referenced",
)


@retry_with_backoff(max_tries=3, base=2.0, max_delay=30.0)
def qdrant_set_payload_for_file(
    qclient: QdrantClient, source_file: str, tags: dict[str, Any]
) -> None:
    """Push tag fields into every chunk payload for this source_file."""
    payload: dict[str, Any] = {}
    for k in _PAYLOAD_TAG_FIELDS:
        v = tags.get(k)
        if v is None:
            continue
        if isinstance(v, list) and not v:
            continue
        payload[k] = v
    if not payload:
        return
    # `points` (not `points_selector`) is the qdrant-client kwarg; it accepts a
    # Filter as a selector so this updates every chunk of the file in one call.
    # Same shape as ingestion/backfill_path_meta.py.
    qclient.set_payload(
        collection_name=QDRANT_COLLECTION,
        payload=payload,
        points=Filter(
            must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
        ),
        wait=True,
    )


# ---- core tagging -------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def tag_transcript_single_pass(
    transcript: str, model: str
) -> tuple[dict[str, Any] | None, str, str | None]:
    """Returns (tags_or_none, raw_response, error_reason)."""
    prompt = build_prompt(transcript)
    raw = ollama_generate_json(prompt, model)
    obj, parse_err = parse_model_json(raw)
    if obj is None:
        return None, raw, parse_err
    ok, errs = validate_tags(obj)
    if not ok:
        return None, raw, "; ".join(errs)
    return obj, raw, None


def tag_transcript_mapreduce(
    transcript: str, model: str, max_chars_per_chunk: int = 10_000
) -> tuple[dict[str, Any] | None, str, str | None]:
    """For very long transcripts: chunk → per-chunk summary → final pass.

    The chunking here is *re-chunking* the transcript for tagging purposes
    only. We do not touch the embedding-side chunks in chunk_meta — those
    keep their original boundaries.
    """
    pieces: list[str] = []
    for i in range(0, len(transcript), max_chars_per_chunk):
        pieces.append(transcript[i : i + max_chars_per_chunk])
    log.info("map-reduce: %d fragments", len(pieces))

    summaries: list[str] = []
    for idx, piece in enumerate(pieces, start=1):
        raw = ollama_generate_json(build_mapreduce_chunk_prompt(piece), model)
        obj, perr = parse_model_json(raw)
        if obj is None or "summary" not in obj:
            log.warning("map-reduce fragment %d/%d: bad mini-summary (%s) — "
                        "using truncated raw text", idx, len(pieces), perr)
            summaries.append(piece[:500])
            continue
        summaries.append(str(obj.get("summary", "")))

    reduced = "\n\n".join(f"Fragment {i+1}: {s}" for i, s in enumerate(summaries))
    return tag_transcript_single_pass(reduced, model)


# ---- main loop ----------------------------------------------------------


def process_one_file(
    source_file: str,
    store: TagStore,
    qclient: QdrantClient,
    model: str,
    max_tokens_single_pass: int,
    dry_run: bool,
) -> tuple[bool, str | None]:
    """Returns (success, reason_if_failed)."""
    transcript = store.load_transcript(source_file)
    if not transcript.strip():
        return False, "empty transcript in chunk_meta"
    n_tokens = _estimate_tokens(transcript)
    log.info(
        "tagging %s (chars=%d, est_tokens=%d, strategy=%s)",
        source_file, len(transcript), n_tokens,
        "single-pass" if n_tokens <= max_tokens_single_pass else "map-reduce",
    )

    if dry_run:
        return True, None

    t0 = time.monotonic()
    try:
        if n_tokens <= max_tokens_single_pass:
            tags, raw, err = tag_transcript_single_pass(transcript, model)
        else:
            tags, raw, err = tag_transcript_mapreduce(transcript, model)
    except Exception as e:
        # A transport error escaped the retry decorator (reraise=True). Persist
        # a forensic artifact — otherwise the only trace is a reason string and
        # the raw failure is lost (CLAUDE.md rule 6). tagged_at stays NULL → the
        # file is retried on the next run.
        reason = f"{type(e).__name__}: {e}"
        dst = dead_letter(source_file, traceback.format_exc(), f"tagging call raised: {reason}")
        log.error("%s: tagging call raised after retries (%s) — dead-letter %s",
                  source_file, reason, dst)
        return False, reason

    if tags is None:
        dst = dead_letter(source_file, raw, err or "validation failed")
        log.error("%s: tagging failed (%s) — dead-letter %s", source_file, err, dst)
        return False, err

    # --- Durability: propagate to Qdrant BEFORE marking the file tagged. ---
    # Resume selects `WHERE tagged_at IS NULL`. If we committed tagged_at first
    # and propagation then failed (a Qdrant restart over a multi-day run is not
    # hypothetical), this file's content-tag payloads would be stranded forever.
    # Ordering propagate-then-commit means a transient Qdrant outage just leaves
    # the file NULL → retried next run. set_payload is idempotent (filtered by
    # source_file), so a crash between propagate and commit re-runs both safely.
    try:
        qdrant_set_payload_for_file(qclient, source_file, tags)
    except Exception as e:
        log.error("%s: qdrant propagation failed after retries — leaving "
                  "tagged_at NULL for retry: %s\n%s",
                  source_file, e, traceback.format_exc())
        return False, f"qdrant propagation failed: {e}"

    store.write_tags(source_file, tags, model)

    elapsed = time.monotonic() - t0
    log.info(
        "%s: tagged in %.1fs (event_type=%s, lang=%s, topics=%s)",
        source_file, elapsed,
        tags.get("event_type"), tags.get("primary_language"),
        tags.get("topics"),
    )
    return True, None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-file content tagging via Qwen 3.5 9B.")
    p.add_argument("--limit", type=int, default=None,
                   help="Tag at most N files this run.")
    p.add_argument("--retry-failed", action="store_true",
                   help="Reserved for future use.")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip the LLM call and writes; just log what would be tagged.")
    p.add_argument("--model", default=TAG_MODEL,
                   help=f"Ollama model name (default: {TAG_MODEL}).")
    p.add_argument("--max-tokens-single-pass", type=int,
                   default=DEFAULT_MAX_TOKENS_SINGLE_PASS,
                   help="Token budget above which map-reduce kicks in.")
    args = p.parse_args(argv)

    setup_logging()

    try:
        store = TagStore()
    except Exception as e:
        log.error("Postgres init failed: %s", e)
        return 2

    qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)

    files = store.list_untagged(args.limit, args.retry_failed)
    log.info("untagged files queued: %d (model=%s, dry_run=%s)",
             len(files), args.model, args.dry_run)

    started = time.monotonic()
    ok = 0
    failed = 0
    for idx, src in enumerate(files, start=1):
        try:
            success, reason = process_one_file(
                src, store, qclient, args.model,
                args.max_tokens_single_pass, args.dry_run,
            )
            if success:
                ok += 1
            else:
                failed += 1
                log.error("[%d/%d] %s: %s", idx, len(files), src, reason)
        except Exception as e:
            failed += 1
            log.error("[%d/%d] %s: unexpected %s\n%s",
                      idx, len(files), src, type(e).__name__, traceback.format_exc())

    elapsed = time.monotonic() - started
    log.info("enrich summary: ok=%d failed=%d elapsed=%.1fs", ok, failed, elapsed)
    store.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
