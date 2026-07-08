"""Re-derive path metadata for already-indexed tracks, without re-embedding.

Commit 3a21e7a replaced a position-based folder parser with a shape-based one,
taking session_date coverage from 35.3% to 98.8% *on disk*. Nothing has been
re-chunked, so Postgres and Qdrant still carry the old parser's damage: 5,173
of 9,335 file_meta rows have session_date = NULL.

The date is a pure function of the path, and ``file_meta.source_file`` *is* the
path — ``qualified_source()`` writes exactly the folder chain that ``parse_path``
reads, with the same scaffolding already stripped and the same track filename.
So the correction needs no re-chunking, no re-embedding, and no rebuild: parse
the key, UPDATE the row, set_payload the points.

That equivalence is the whole safety argument, so ``verify`` proves it against
the disk before ``backfill --commit`` is allowed to run:

    python -m ingestion.backfill_path_meta verify --base-dir "D:/Transcription whisperx/Output"
    python -m ingestion.backfill_path_meta backfill                 # dry run, prints the diff
    python -m ingestion.backfill_path_meta backfill --commit

What this does NOT fix: the 877 tracks whose "cleaned" text was an LLM summary.
Their *text* changes, so their embeddings and chunk_uuids change. That needs a
re-chunk and a full index rebuild.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterator

from ingestion.chunker_cleaned import qualified_source
from ingestion.utils.path_parser import PathMetadata, parse_path

log = logging.getLogger("backfill_path_meta")

# The file_meta columns derived from the path. Ordered as the UPDATE writes them.
PG_FIELDS = ("session_date", "track_type", "location", "event_id", "season")

# Everything bulk_ingest_hardened._PATH_FIELDS flattens into a Qdrant payload.
# Kept in sync deliberately: a field indexed for filtering but never corrected
# would silently answer date queries from the old parser's output.
QDRANT_FIELDS = (
    "collection", "year", "event_seq", "event_id", "location",
    "event_start", "event_end",
    "session_date", "session_seq", "session_time",
    "track_no", "track_title", "track_type",
    "season", "primary_speaker",
)


def meta_for_key(source_file: str) -> PathMetadata:
    """Parse a file_meta.source_file identity key back into path metadata.

    No base_dir: the key is already relative, already scaffolding-stripped.
    parse_path's own stripping is idempotent, so re-running it is a no-op.
    """
    return parse_path(source_file)


# --- verify ---------------------------------------------------------------


def _disk_pairs(base_dir: Path) -> Iterator[tuple[str, PathMetadata, PathMetadata]]:
    """(source_file, truth, from_key) for every raw transcript under base_dir.

    `truth` reproduces exactly what the chunker computes at ingest time:
    parse_path of the *disk* path, with the '.raw' infix normalized away
    (chunker_cleaned.process_file builds the same synthetic name).
    """
    for raw_path in sorted(base_dir.rglob("*.raw.json")):
        key = qualified_source(raw_path, base_dir)
        synth = raw_path.with_name(f"{key.rsplit('/', 1)[-1]}")
        yield key, parse_path(synth, base_dir=base_dir), meta_for_key(key)


def _payload_diff(a: PathMetadata, b: PathMetadata, fields: tuple[str, ...]) -> dict[str, Any]:
    pa, pb = a.to_payload(), b.to_payload()
    return {f: (pa.get(f), pb.get(f)) for f in fields if pa.get(f) != pb.get(f)}


def cmd_verify(args: argparse.Namespace) -> int:
    base = Path(args.base_dir)
    if not base.is_dir():
        log.error("base-dir does not exist: %s", base)
        return 2

    n = mismatched = 0
    for key, truth, from_key in _disk_pairs(base):
        n += 1
        diff = _payload_diff(truth, from_key, QDRANT_FIELDS)
        if diff:
            mismatched += 1
            if mismatched <= 20:
                log.error("MISMATCH %s: %s", key, diff)

    if n == 0:
        log.error("no *.raw.json found under %s - nothing verified", base)
        return 2

    print(f"\nchecked {n} transcripts under {base}")
    print(f"mismatches: {mismatched}")
    if mismatched:
        print("\nparse_path(source_file) is NOT equivalent to parse_path(disk_path).")
        print("The backfill is unsafe. Re-chunk instead.")
        return 1
    print("\nparse_path(source_file) == parse_path(disk_path) for every track.")
    print("The backfill can write file_meta and Qdrant payloads directly.")
    return 0


# --- backfill -------------------------------------------------------------


def _connect_pg(dsn: str):
    try:
        import psycopg2  # type: ignore[import-not-found]
    except ImportError:
        log.error("psycopg2 not importable - install psycopg2-binary")
        raise
    return psycopg2.connect(dsn, connect_timeout=10)


def _pg_rows(conn) -> list[tuple[str, ...]]:
    cols = ", ".join(PG_FIELDS)
    with conn.cursor() as cur:
        cur.execute(f"SELECT source_file, {cols} FROM file_meta ORDER BY source_file")
        return cur.fetchall()


def _as_pg_value(meta: PathMetadata, field: str) -> Any:
    """file_meta stores session_date as DATE and the rest as TEXT."""
    if field == "session_date":
        return meta.session_date
    return getattr(meta, field)


def cmd_backfill(args: argparse.Namespace) -> int:
    dsn = os.environ.get("POSTGRES_DSN") or args.dsn
    if not dsn:
        log.error("no DSN: set POSTGRES_DSN or pass --dsn")
        return 2

    conn = _connect_pg(dsn)
    try:
        rows = _pg_rows(conn)
    finally:
        pass
    log.info("file_meta: %d rows", len(rows))

    updates: list[tuple[str, dict[str, tuple[Any, Any]], PathMetadata]] = []
    # Qdrant's work-list is deliberately NOT derived from the Postgres diff: the
    # two stores drift independently, and a crash between the two writes would
    # otherwise leave Qdrant permanently stale with nothing to notice it.
    # set_payload is idempotent, so repayload everything.
    all_metas: dict[str, PathMetadata] = {}
    gained_date = lost_date = unparsed = 0

    for row in rows:
        source_file, current = row[0], dict(zip(PG_FIELDS, row[1:]))
        meta = meta_for_key(source_file)
        all_metas[source_file] = meta
        diff = {
            f: (current[f], _as_pg_value(meta, f))
            for f in PG_FIELDS
            if current[f] != _as_pg_value(meta, f)
        }
        if meta.session_date is None:
            unparsed += 1
        if not diff:
            continue
        if "session_date" in diff:
            before, after = diff["session_date"]
            if before is None and after is not None:
                gained_date += 1
            elif before is not None and after is None:
                lost_date += 1
        updates.append((source_file, diff, meta))

    print(f"\nrows                 {len(rows)}")
    print(f"rows changing        {len(updates)}")
    print(f"  gain session_date  {gained_date}")
    print(f"  LOSE session_date  {lost_date}")
    print(f"still undated after  {unparsed}")

    if lost_date and not args.allow_regression:
        log.error("%d rows would LOSE a session_date they already have. Refusing. "
                  "Inspect them, then re-run with --allow-regression if intended.",
                  lost_date)
        for sf, diff, _ in updates:
            if "session_date" in diff and diff["session_date"][1] is None:
                log.error("  regression: %s %s", sf, diff["session_date"])
        conn.close()
        return 1

    for sf, diff, _ in updates[: args.sample]:
        print(f"  {sf}\n    {diff}")
    if len(updates) > args.sample:
        print(f"  ... and {len(updates) - args.sample} more")

    if not args.commit:
        print("\nDRY RUN - nothing written. Re-run with --commit.")
        conn.close()
        return 0

    n_pg = _write_pg(conn, updates)
    conn.close()
    print(f"\nfile_meta: {n_pg} rows updated")

    if args.qdrant:
        ok, failed = _write_qdrant(args.qdrant_url, args.collection, all_metas,
                                   args.grpc_port)
        print(f"qdrant {args.collection}: {ok} files repayloaded, {failed} failed")
        if failed:
            print("Qdrant is now PARTIALLY corrected. Re-run to converge "
                  "(set_payload is idempotent); check the errors above first.")
            return 1
    else:
        print("qdrant: skipped (--no-qdrant). Date FILTERS in search still lie.")
    return 0


def _write_pg(conn, updates: list[tuple[str, dict, PathMetadata]]) -> int:
    sets = ", ".join(f"{f} = %s" for f in PG_FIELDS)
    n = 0
    try:
        with conn.cursor() as cur:
            for source_file, _, meta in updates:
                cur.execute(
                    f"UPDATE file_meta SET {sets} WHERE source_file = %s",
                    tuple(_as_pg_value(meta, f) for f in PG_FIELDS) + (source_file,),
                )
                n += cur.rowcount
        conn.commit()
    except Exception as e:
        conn.rollback()
        log.error("file_meta UPDATE failed, rolled back: %s: %s", type(e).__name__, e)
        raise
    return n


def _write_qdrant(url: str, collection: str, metas: dict[str, PathMetadata],
                  grpc_port: int = 6334) -> tuple[int, int]:
    """Repayload every file's path metadata. Returns (ok, failed).

    gRPC, not REST: ~19k calls over HTTP/1.1 opens a socket per call, and
    Windows' 16k ephemeral ports all sit in TIME_WAIT for 60 s. The first
    attempt died at WinError 10048 after 8,656 files. gRPC multiplexes the
    whole run over one connection.
    """
    from urllib.parse import urlparse

    from qdrant_client import QdrantClient
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    # https=False is load-bearing: qdrant-client turns on a TLS gRPC channel the
    # moment an api_key is present, and the local Qdrant speaks plaintext h2c.
    client = QdrantClient(host=host, grpc_port=grpc_port, prefer_grpc=True,
                          https=parsed.scheme == "https",
                          api_key=os.environ.get("QDRANT_API_KEY") or None,
                          timeout=60)
    ok = failed = 0
    for i, (source_file, meta) in enumerate(metas.items(), 1):
        payload = {k: v for k, v in meta.to_payload().items()
                   if k in QDRANT_FIELDS and v is not None}
        # Fields the new parser could not derive must be *cleared*, not left at
        # the old parser's guess. A stale event_id is worse than a missing one.
        clear = [k for k in QDRANT_FIELDS if k not in payload]
        flt = Filter(must=[FieldCondition(key="source_file",
                                          match=MatchValue(value=source_file))])
        try:
            if payload:
                client.set_payload(collection_name=collection, payload=payload,
                                   points=flt, wait=False)
            if clear:
                client.delete_payload(collection_name=collection, keys=clear,
                                      points=flt, wait=False)
            ok += 1
        except Exception as e:
            failed += 1
            if failed <= 10:
                log.error("qdrant repayload failed for %s: %s: %s",
                          source_file, type(e).__name__, e)
            elif failed == 11:
                log.error("further qdrant errors suppressed; see the final count")
        if i % 1000 == 0:
            log.info("qdrant: %d/%d files (%d failed)", i, len(metas), failed)
    return ok, failed


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="prove parse_path(source_file) == parse_path(disk)")
    v.add_argument("--base-dir", required=True)
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("backfill", help="re-derive file_meta + Qdrant path metadata")
    b.add_argument("--dsn", default=None)
    b.add_argument("--commit", action="store_true", help="write; default is a dry run")
    b.add_argument("--qdrant", dest="qdrant", action="store_true", default=True)
    b.add_argument("--no-qdrant", dest="qdrant", action="store_false")
    b.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    b.add_argument("--grpc-port", type=int, default=int(os.environ.get("QDRANT_GRPC_PORT", 6334)))
    b.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "transcripts"))
    b.add_argument("--sample", type=int, default=10, help="rows of diff to print")
    b.add_argument("--allow-regression", action="store_true",
                   help="permit rows to lose a session_date they already have")
    b.set_defaults(func=cmd_backfill)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
