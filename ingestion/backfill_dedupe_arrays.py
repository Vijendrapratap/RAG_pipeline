"""De-duplicate the enrichment arrays the ORIGINAL tag run left padded.

Phase 13's decoding grammar bounds each array's *length* but not its *content*:
llama.cpp ignores `uniqueItems`, so a transcript that repeats one name is mirrored
into duplicate padding up to `maxItems`. Stream A added `dedupe_arrays()` to the
producer, but only the files that run re-tagged were cleaned. The rest still carry
the padding in both Postgres (`file_meta`) and every Qdrant chunk payload.

The correction is deterministic — NO model calls. For each tagged file, apply the
producer's `dedupe_list()` (order-preserving, case/whitespace-insensitive) to the
six `file_meta` array columns, then re-propagate the four Qdrant-visible arrays into
every chunk payload via the same `qdrant_set_payload_for_file()` the enrichment uses.

Deliberately NOT `dedupe_arrays()`: that also re-applies the per-array length cap,
which is a *decode-time* runaway guard, not a retention policy. Applied to already-
stored data the cap is destructive — it would delete 2,018 unique `timing_clues`
(664 files) and 251 unique `location_clues` (68 files) that the original run captured
legitimately (distinct timestamp segments like `00:07:52 -> 00:17:44`; the grammar's
cap was higher then, and a re-tag today would lose them too). A backfill removes
duplicates only; it never truncates unique data. The cap never bites the four
Qdrant-visible fields, so payloads are unaffected by this choice.

Single writer, single lane: run this only when no enrichment/ingest is writing.

    python -m ingestion.backfill_dedupe_arrays backfill            # dry run (default)
    python -m ingestion.backfill_dedupe_arrays backfill --commit   # write PG + Qdrant
    python -m ingestion.backfill_dedupe_arrays verify              # PG len == payload len

Idempotent: a second `backfill` finds nothing to change; `verify` re-proves the
invariant from scratch and needs no state from the run that fixed it.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from ingestion.enrich_content_tags import (
    PG_DSN,
    QDRANT_COLLECTION,
    QDRANT_KEY,
    QDRANT_URL,
    _PAYLOAD_TAG_FIELDS,
    qdrant_set_payload_for_file,
)
from ingestion.utils.tag_schema import ARRAY_KEYS, dedupe_list

log = logging.getLogger("backfill_dedupe_arrays")

# Ordered view of the six enrichment arrays (ARRAY_KEYS is an unordered frozenset).
# This is also the column order the UPDATE writes.
PG_ARRAY_FIELDS: tuple[str, ...] = (
    "topics", "people_named", "places_named", "scriptures_referenced",
    "timing_clues", "location_clues",
)
assert frozenset(PG_ARRAY_FIELDS) == ARRAY_KEYS, "PG_ARRAY_FIELDS must cover ARRAY_KEYS"

# The subset that also lives in every Qdrant chunk payload. The other two arrays
# (timing_clues, location_clues) are Postgres-only, so they get deduped in PG but
# need no Qdrant write.
PAYLOAD_ARRAY_FIELDS: tuple[str, ...] = tuple(
    f for f in PG_ARRAY_FIELDS if f in _PAYLOAD_TAG_FIELDS
)


def _connect_pg(dsn: str):
    try:
        import psycopg2  # type: ignore[import-not-found]
    except ImportError:
        log.error("psycopg2 not importable — install psycopg2-binary")
        raise
    return psycopg2.connect(dsn, connect_timeout=10)


def _load_rows(conn) -> list[tuple[str, dict[str, Any]]]:
    """(source_file, {field: array}) for every tagged file, source order stable."""
    cols = ", ".join(PG_ARRAY_FIELDS)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT source_file, {cols} FROM file_meta "
            "WHERE tagged_at IS NOT NULL ORDER BY source_file"
        )
        rows = cur.fetchall()
    return [(r[0], dict(zip(PG_ARRAY_FIELDS, r[1:]))) for r in rows]


def _dedupe(arrays: dict[str, Any]) -> dict[str, Any]:
    """Order-preserving, case/whitespace-insensitive dedupe of the six arrays,
    WITHOUT the length cap (see module docstring — the cap would delete unique
    data). Non-list values (NULL columns) pass through untouched."""
    return {
        f: (dedupe_list(v) if isinstance(v, list) else v)
        for f, v in arrays.items()
    }


def _plan(rows: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """One entry per file whose deduped arrays differ from what is stored.

    A field counts as changed when the stored list is not identical to its
    deduped form — that catches exact-dup padding, case-fold collisions, and
    internal-whitespace normalization, exactly as the producer's dedupe_list does.
    """
    updates: list[dict[str, Any]] = []
    for source_file, arrays in rows:
        deduped = _dedupe(arrays)
        diff = {
            f: (len(arrays[f]), len(deduped[f]))
            for f in PG_ARRAY_FIELDS
            if isinstance(arrays.get(f), list) and arrays[f] != deduped[f]
        }
        if diff:
            updates.append({"source_file": source_file, "diff": diff, "deduped": deduped})
    return updates


# --- reporting ------------------------------------------------------------


def _summarize(updates: list[dict[str, Any]], scanned: int, sample: int) -> None:
    removed = sum(b - a for u in updates for (b, a) in u["diff"].values())
    ws_only = sum(1 for u in updates if all(b == a for (b, a) in u["diff"].values()))
    per_field_files = {f: 0 for f in PG_ARRAY_FIELDS}
    per_field_removed = {f: 0 for f in PG_ARRAY_FIELDS}
    for u in updates:
        for f, (b, a) in u["diff"].items():
            per_field_files[f] += 1
            per_field_removed[f] += b - a
    qdrant_files = sum(
        1 for u in updates if any(f in u["diff"] for f in PAYLOAD_ARRAY_FIELDS)
    )

    print(f"\nfiles scanned (tagged):        {scanned}")
    print(f"files changing:                {len(updates)}")
    print(f"  duplicate entries removed:   {removed}")
    print(f"  files normalized only (0 removed, whitespace/case): {ws_only}")
    print(f"files needing Qdrant re-propagation: {qdrant_files}")
    print("\nby field (files · entries removed):")
    for f in PG_ARRAY_FIELDS:
        tail = "  [PG-only]" if f not in PAYLOAD_ARRAY_FIELDS else ""
        print(f"  {f:<24} {per_field_files[f]:>4} · {per_field_removed[f]:>4}{tail}")

    print(f"\nsample (first {sample}):")
    for u in updates[:sample]:
        parts = ", ".join(f"{f} {b}→{a}" for f, (b, a) in u["diff"].items())
        print(f"  {u['source_file']}\n    {parts}")
    if len(updates) > sample:
        print(f"  ... and {len(updates) - sample} more")


# --- writes ---------------------------------------------------------------


def _write_pg(conn, updates: list[dict[str, Any]]) -> int:
    """UPDATE the six array columns to their deduped form. One transaction."""
    sets = ", ".join(f"{f} = %s" for f in PG_ARRAY_FIELDS)
    n = 0
    try:
        with conn.cursor() as cur:
            for u in updates:
                cur.execute(
                    f"UPDATE file_meta SET {sets} WHERE source_file = %s",
                    tuple(u["deduped"][f] for f in PG_ARRAY_FIELDS) + (u["source_file"],),
                )
                n += cur.rowcount
        conn.commit()
    except Exception as e:
        conn.rollback()
        log.error("file_meta UPDATE failed, rolled back: %s: %s", type(e).__name__, e)
        raise
    return n


def _write_qdrant(updates: list[dict[str, Any]], url: str, key: str,
                  collection: str) -> tuple[int, int]:
    """Re-push the four payload arrays for files whose payload arrays changed.

    Reuses the enrichment's own `qdrant_set_payload_for_file` (retry-wrapped,
    wait=True) so the write is the same one that first propagated these tags.
    set_payload overwrites the key, so a shorter deduped array replaces the padded
    one in every chunk of the file. Returns (ok, failed).
    """
    import ingestion.enrich_content_tags as _enrich
    from qdrant_client import QdrantClient  # noqa: PLC0415  (heavy import, deferred)

    # qdrant_set_payload_for_file resolves QDRANT_COLLECTION in its own module, so
    # redirect it there (a no-op unless --collection overrode the default).
    _enrich.QDRANT_COLLECTION = collection

    todo = [u for u in updates if any(f in u["diff"] for f in PAYLOAD_ARRAY_FIELDS)]
    qclient = QdrantClient(url=url, api_key=key or None)
    ok = failed = 0
    for i, u in enumerate(todo, 1):
        try:
            qdrant_set_payload_for_file(qclient, u["source_file"], u["deduped"])
            ok += 1
        except Exception as e:
            failed += 1
            if failed <= 10:
                log.error("qdrant re-propagate failed for %s: %s: %s",
                          u["source_file"], type(e).__name__, e)
            elif failed == 11:
                log.error("further qdrant errors suppressed; see the final count")
        if i % 100 == 0:
            log.info("qdrant: %d/%d files (%d failed)", i, len(todo), failed)
    return ok, failed


# --- commands -------------------------------------------------------------


def cmd_backfill(args: argparse.Namespace) -> int:
    conn = _connect_pg(args.dsn)
    rows = _load_rows(conn)
    updates = _plan(rows)
    _summarize(updates, scanned=len(rows), sample=args.sample)

    if not updates:
        print("\nNothing to dedupe — file_meta arrays are already unique.")
        conn.close()
        return 0

    if not args.commit:
        print("\nDRY RUN — nothing written. Re-run with --commit.")
        conn.close()
        return 0

    n_pg = _write_pg(conn, updates)
    conn.close()
    print(f"\nfile_meta: {n_pg} rows updated")

    if args.qdrant:
        ok, failed = _write_qdrant(updates, args.qdrant_url, QDRANT_KEY, args.collection)
        print(f"qdrant {args.collection}: {ok} files re-propagated, {failed} failed")
        if failed:
            print("Qdrant is now PARTIALLY corrected. Re-run --commit to converge "
                  "(set_payload is idempotent); inspect the errors above first.")
            return 1
    else:
        print("qdrant: skipped (--no-qdrant). Chunk-payload arrays still carry dups.")
    print("\nDone. Run `verify` to confirm PG length == Qdrant payload length.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Prove the invariant with no state from the backfill run:
      1. no file_meta array still contains duplicates (PG idempotent), and
      2. no Qdrant payload array is LONGER than its PG array (no stale padding).
    A Qdrant array SHORTER than PG is a pre-existing propagation gap (Stage 2.0),
    reported separately — this backfill only ever sets Qdrant == PG for files it
    touched, so it cannot create an under-length payload.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    conn = _connect_pg(args.dsn)
    rows = _load_rows(conn)
    conn.close()

    residual = _plan(rows)
    print(f"file_meta arrays with remaining duplicates: {len(residual)}")
    for u in residual[:10]:
        print(f"  DUP {u['source_file']}: {u['diff']}")

    if args.limit:
        rows = rows[: args.limit]
    qclient = QdrantClient(url=args.qdrant_url, api_key=QDRANT_KEY or None)
    stale: list[str] = []
    under = missing = checked = 0
    for i, (source_file, arrays) in enumerate(rows, 1):
        pg_len = {f: len(arrays[f]) for f in PAYLOAD_ARRAY_FIELDS
                  if isinstance(arrays.get(f), list)}
        if not pg_len:
            continue
        pts, _ = qclient.scroll(
            collection_name=args.collection,
            scroll_filter=Filter(must=[FieldCondition(
                key="source_file", match=MatchValue(value=source_file))]),
            limit=1, with_payload=list(PAYLOAD_ARRAY_FIELDS), with_vectors=False,
        )
        checked += 1
        if not pts:
            missing += 1
            continue
        payload = pts[0].payload or {}
        for f, n in pg_len.items():
            q = len(payload.get(f) or [])
            if q > n:
                stale.append(f"{source_file}:{f} qdrant={q} pg={n}")
            elif q < n:
                under += 1
        if i % 1000 == 0:
            log.info("verify: %d/%d files", i, len(rows))

    print(f"\nfiles checked against Qdrant:  {checked}")
    print(f"  stale (qdrant LONGER than pg): {len(stale)}")
    print(f"  under (qdrant shorter — pre-existing propagation gap): {under}")
    print(f"  missing in qdrant:             {missing}")
    for s in stale[:20]:
        print(f"  STALE {s}")

    ok = not residual and not stale
    print("\nPASS — PG arrays unique and no Qdrant payload carries stale padding."
          if ok else "\nFAIL — see DUP/STALE lines above.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stderr)
    # Qdrant's client logs every REST call at INFO; over thousands of per-file
    # calls that buries our own progress lines. Keep only its warnings.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dsn", default=PG_DSN)
    common.add_argument("--qdrant-url", default=QDRANT_URL)
    common.add_argument("--collection", default=QDRANT_COLLECTION)

    b = sub.add_parser("backfill", parents=[common],
                       help="dedupe file_meta arrays + re-propagate Qdrant payloads")
    b.add_argument("--commit", action="store_true", help="write; default is a dry run")
    b.add_argument("--qdrant", dest="qdrant", action="store_true", default=True)
    b.add_argument("--no-qdrant", dest="qdrant", action="store_false")
    b.add_argument("--sample", type=int, default=10, help="rows of diff to print")
    b.set_defaults(func=cmd_backfill)

    v = sub.add_parser("verify", parents=[common],
                       help="prove PG arrays unique and Qdrant payloads not stale")
    v.add_argument("--limit", type=int, default=0,
                   help="check only the first N files against Qdrant (0 = all)")
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
