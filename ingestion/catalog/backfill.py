"""Load the curated catalog into Postgres and (optionally) enrich Qdrant.

Two outputs, both gated behind ``--commit`` (dry-run is the default — it prints
a plan and writes nothing):

  1. **Postgres** — upsert `catalog_sitting` + `catalog_track` (the standalone
     authoritative tables from Phase 14). This never touches `chunk_meta` /
     `file_meta`, so retrieval is unaffected by the load itself.

  2. **Qdrant** — for each catalog track, `set_payload` adds `performers`,
     `catalog_title` (the canonical Devanagari song title), and the normalized
     `location` onto the matching chunk points. Matching is by the
     **date-anchored join key** expressed as a Qdrant filter
     (session_date == day AND session_seq == seq AND track_no == no) — NOT the
     bare filename, which repeats across sittings ("06 SAMBODHAN" is in every
     camp). Points that were never ingested simply match nothing, so this is
     safe to run against a corpus where only a slice of audio is present.

Pass ``--audio-root`` to restrict the Qdrant pass to tracks that actually have
a transcript on disk (cheap, recommended). Without it, every keyed catalog
track is pushed (idempotent, but ~22k filter calls — a warning is logged).

Dry-run example (offline, no services needed):
    python -m ingestion.catalog.backfill \
        --xlsx ".../Camp Record Export.xlsx" \
        --audio-root "/mnt/d/Transcription whisperx/Output" \
        --audio-glob "**/*.cleaned.json"

Commit example:
    python -m ingestion.catalog.backfill --xlsx ... \
        --postgres "$PG_DSN" --qdrant http://localhost:6333 \
        --collection transcripts --audio-root ... --commit
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from ingestion.catalog.normalize import (
    CatalogTrack,
    join_key_from_fields,
    join_key_from_path,
    rows_to_tracks,
)

log = logging.getLogger("ingestion.catalog.backfill")


def _read_rows(xlsx_path: Path) -> list[dict[str, Any]]:
    import pandas as pd  # noqa: PLC0415 — lazy

    df = pd.read_excel(xlsx_path)
    return df.where(df.notna(), None).to_dict("records")


def _sitting_key(t: CatalogTrack) -> str | None:
    return join_key_from_fields(t.session_date, t.session_seq, None)


def _scan_audio_keys(audio_root: Path, glob: str) -> set[str]:
    keys: set[str] = set()
    for p in audio_root.glob(glob):
        if p.is_file():
            k = join_key_from_path(p, base_dir=audio_root)
            if k:
                keys.add(k)
    return keys


# --- Postgres ---------------------------------------------------------------

def _upsert_postgres(tracks: list[CatalogTrack], dsn: str) -> tuple[int, int]:
    """Upsert catalog_sitting + catalog_track. Returns (n_sittings, n_tracks)."""
    import psycopg2  # noqa: PLC0415
    from psycopg2.extras import execute_values  # noqa: PLC0415

    # Collapse to sittings (dedupe DetailContents, union performers).
    sittings: dict[str, dict[str, Any]] = {}
    for t in tracks:
        sk = _sitting_key(t)
        if sk is None:
            continue
        s = sittings.get(sk)
        if s is None:
            s = {
                "sitting_key": sk, "location": t.location, "camp_place": t.camp_place,
                "camp_year": t.camp_year, "year_reliable": t.year_reliable,
                "session_date": t.session_date, "session_seq": t.session_seq,
                "session_time": t.session_time.strftime("%H:%M") if t.session_time else None,
                "season": t.season, "track_count": 0, "performers": [],
                "release_ref": t.release_ref, "venue": t.venue,
                "detail_contents": t.detail_contents,
            }
            sittings[sk] = s
        s["track_count"] += 1
        for p in t.performers:
            if p not in s["performers"]:
                s["performers"].append(p)
        if not s["detail_contents"] and t.detail_contents:
            s["detail_contents"] = t.detail_contents
        if not s["release_ref"] and t.release_ref:
            s["release_ref"] = t.release_ref

    track_rows = [t for t in tracks if t.join_key]

    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO catalog_sitting (
                    sitting_key, location, camp_place, camp_year, year_reliable,
                    session_date, session_seq, session_time, season, track_count,
                    performers, release_ref, venue, detail_contents)
                VALUES %s
                ON CONFLICT (sitting_key) DO UPDATE SET
                    location=EXCLUDED.location, camp_place=EXCLUDED.camp_place,
                    camp_year=EXCLUDED.camp_year, year_reliable=EXCLUDED.year_reliable,
                    session_date=EXCLUDED.session_date, session_seq=EXCLUDED.session_seq,
                    session_time=EXCLUDED.session_time, season=EXCLUDED.season,
                    track_count=EXCLUDED.track_count, performers=EXCLUDED.performers,
                    release_ref=EXCLUDED.release_ref, venue=EXCLUDED.venue,
                    detail_contents=EXCLUDED.detail_contents, loaded_at=NOW()
            """, [(
                s["sitting_key"], s["location"], s["camp_place"], s["camp_year"],
                s["year_reliable"], s["session_date"], s["session_seq"],
                s["session_time"], s["season"], s["track_count"], s["performers"],
                s["release_ref"], s["venue"], s["detail_contents"],
            ) for s in sittings.values()])

            execute_values(cur, """
                INSERT INTO catalog_track (
                    join_key, sitting_key, location, session_date, session_seq,
                    track_no, track_title, track_type, duration_sec, performers)
                VALUES %s
                ON CONFLICT (join_key) DO UPDATE SET
                    sitting_key=EXCLUDED.sitting_key, location=EXCLUDED.location,
                    session_date=EXCLUDED.session_date, session_seq=EXCLUDED.session_seq,
                    track_no=EXCLUDED.track_no, track_title=EXCLUDED.track_title,
                    track_type=EXCLUDED.track_type, duration_sec=EXCLUDED.duration_sec,
                    performers=EXCLUDED.performers, loaded_at=NOW()
            """, [(
                t.join_key, _sitting_key(t), t.location, t.session_date,
                t.session_seq, t.track_no, t.track_title, t.track_type,
                t.duration_sec, t.performers,
            ) for t in track_rows])
        return len(sittings), len(track_rows)
    finally:
        conn.close()


# --- Qdrant -----------------------------------------------------------------

def _qdrant_filter_for(t: CatalogTrack) -> dict[str, Any]:
    """Date-anchored filter selecting the chunk points for one catalog track.
    session_date is a DATETIME payload index, so it is matched as a same-day
    range; seq + track_no pin the exact track."""
    d = t.session_date.isoformat()
    must: list[dict[str, Any]] = [
        {"key": "session_date", "range": {"gte": d, "lte": d}},
        {"key": "track_no", "match": {"value": t.track_no}},
    ]
    if t.session_seq is not None:
        must.append({"key": "session_seq", "match": {"value": t.session_seq}})
    return {"must": must}


def _enrich_qdrant(
    tracks: list[CatalogTrack], qdrant_url: str, collection: str,
    api_key: str | None, timeout: float,
) -> int:
    """set_payload performers / catalog_title / location onto matching points.
    Returns the number of tracks pushed."""
    import requests  # noqa: PLC0415

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    url = f"{qdrant_url}/collections/{collection}/points/payload"
    pushed = 0
    for t in tracks:
        if t.track_no is None or t.session_date is None:
            continue
        payload = {"performers": t.performers}
        if t.track_title:
            payload["catalog_title"] = t.track_title
        if t.location:
            payload["location"] = t.location
        body = {"payload": payload, "filter": _qdrant_filter_for(t)}
        r = requests.post(url, json=body, headers=headers, timeout=timeout)
        r.raise_for_status()
        pushed += 1
    return pushed


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Backfill catalog into Postgres + Qdrant.")
    ap.add_argument("--xlsx", required=True, type=Path)
    ap.add_argument("--postgres", default=None, help="Postgres DSN. Omit to skip the PG load.")
    ap.add_argument("--qdrant", default=None, help="Qdrant base URL. Omit to skip enrichment.")
    ap.add_argument("--collection", default="transcripts")
    ap.add_argument("--qdrant-key", default=None)
    ap.add_argument("--audio-root", type=Path, default=None,
                    help="Restrict Qdrant enrichment to tracks with a transcript on disk.")
    ap.add_argument("--audio-glob", default="**/*.cleaned.json")
    ap.add_argument("--http-timeout", type=float, default=30.0)
    ap.add_argument("--commit", action="store_true",
                    help="Actually write. Without it, prints a plan and exits.")
    args = ap.parse_args(argv)

    if not args.xlsx.exists():
        log.error("xlsx not found: %s", args.xlsx)
        return 2

    tracks = rows_to_tracks(_read_rows(args.xlsx))
    keyed = [t for t in tracks if t.join_key]
    log.info("normalised %d tracks (%d keyed)", len(tracks), len(keyed))

    # Scope the Qdrant pass to tracks that have audio, if a root was given.
    qdrant_targets = keyed
    if args.audio_root is not None and args.audio_root.exists():
        audio_keys = _scan_audio_keys(args.audio_root, args.audio_glob)
        qdrant_targets = [t for t in keyed if t.join_key in audio_keys]
        log.info("audio scan: %d on-disk keys -> %d catalog tracks have audio",
                 len(audio_keys), len(qdrant_targets))
    elif args.qdrant and not args.commit:
        log.warning("no --audio-root: a commit would push all %d keyed tracks", len(keyed))

    if not args.commit:
        log.info("DRY-RUN (no writes). Pass --commit to apply.")
        log.info("  postgres: %s", "would upsert catalog_sitting + catalog_track"
                 if args.postgres else "skipped (no --postgres)")
        log.info("  qdrant:   %s", f"would set_payload on {len(qdrant_targets)} tracks "
                 f"(performers/catalog_title/location)" if args.qdrant else "skipped (no --qdrant)")
        sample = qdrant_targets[0] if qdrant_targets else (keyed[0] if keyed else None)
        if sample is not None:
            log.info("  sample filter: %s", _qdrant_filter_for(sample)
                     if sample.session_date else "n/a")
            log.info("  sample payload: performers=%s title=%r location=%s",
                     sample.performers, sample.track_title, sample.location)
        return 0

    if args.postgres:
        n_s, n_t = _upsert_postgres(tracks, args.postgres)
        log.info("postgres: upserted %d sittings, %d tracks", n_s, n_t)
    if args.qdrant:
        pushed = _enrich_qdrant(qdrant_targets, args.qdrant, args.collection,
                                args.qdrant_key, args.http_timeout)
        log.info("qdrant: pushed payload for %d tracks", pushed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
