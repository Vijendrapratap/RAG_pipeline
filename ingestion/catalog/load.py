"""Load the 'Camp Record Export' spreadsheet into clean, sync-ready tables.

Reads the .xlsx, normalises every row (`ingestion.catalog.normalize`), and
writes three artifacts to an output directory:

  * ``catalog_tracks.csv``   — one row per track (the full grain).
  * ``catalog_sittings.csv`` — one row per sitting (DetailContents deduped;
    this is where the hand transcription lives, once per session).
  * ``catalog_report.json``  — parse health + a coverage report.

This step is **read-only with respect to the running stack**. It touches no
Qdrant / Postgres / Tantivy state — it produces files a later phase ingests.

Optional ``--audio-root <dir>`` turns on the *sync* check: every audio /
transcript file under that root is keyed with the production `path_parser`,
and the report tells you how many catalog rows align to real folders
(``matched``), how many name a folder that isn't present yet (``catalog_only``
— the "dark" archive waiting on more audio), and how many on-disk files have
no catalog entry (``audio_only``). As more of the 5 TB becomes accessible,
re-running raises the match rate with zero code changes.

Usage (inside WSL, per project convention):
    python -m ingestion.catalog.load \
        --xlsx "/mnt/c/Users/Pc/Downloads/Camp Record Export.xlsx" \
        --out-dir data/catalog \
        [--audio-root /mnt/d/GuruAudio/Output] \
        [--audio-glob '**/*.json']
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ingestion.catalog.normalize import (
    CatalogTrack,
    clean_text,
    join_key_from_path,
    rows_to_tracks,
)

log = logging.getLogger("ingestion.catalog.load")

_SITTING_FIELDS = [
    "sitting_key", "location", "camp_place", "camp_year", "year_reliable",
    "session_date", "session_seq", "session_time", "season",
    "track_count", "performers", "release_ref", "venue",
    "camp_date_raw", "sitting_raw", "detail_contents", "warnings",
]


def _read_rows(xlsx_path: Path) -> list[dict[str, Any]]:
    """Read the spreadsheet into a list of column-name -> value dicts.
    pandas is imported lazily so the rest of the package stays dependency-light
    and importable in environments without pandas (e.g. unit tests)."""
    import pandas as pd  # noqa: PLC0415 — lazy by design

    df = pd.read_excel(xlsx_path)
    # NaN -> None so clean_text sees a uniform 'absent' value.
    return df.where(df.notna(), None).to_dict("records")


def _sitting_key(t: CatalogTrack) -> str | None:
    """Key identifying the sitting a track belongs to (track number dropped)."""
    from ingestion.catalog.normalize import join_key_from_fields

    return join_key_from_fields(t.session_date, t.session_seq, None)


def _write_tracks(tracks: list[CatalogTrack], path: Path) -> None:
    rows = [t.to_row() for t in tracks]
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_sittings(tracks: list[CatalogTrack], path: Path) -> int:
    """Collapse tracks to one row per sitting. DetailContents (the hand
    transcription) is identical across a sitting's tracks in the export, so it
    is stored once here. Performers are unioned across the sitting."""
    grouped: dict[str, list[CatalogTrack]] = {}
    order: list[str] = []
    for t in tracks:
        key = _sitting_key(t)
        if key is None:
            continue
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(t)

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_SITTING_FIELDS)
        w.writeheader()
        for key in order:
            members = grouped[key]
            head = members[0]
            performers: list[str] = []
            for m in members:
                for p in m.performers:
                    if p not in performers:
                        performers.append(p)
            detail = next((m.detail_contents for m in members if m.detail_contents), None)
            release = next((m.release_ref for m in members if m.release_ref), None)
            warns = sorted({w for m in members for w in m.warnings})
            w.writerow({
                "sitting_key": key,
                "location": head.location,
                "camp_place": head.camp_place,
                "camp_year": head.camp_year,
                "year_reliable": head.year_reliable,
                "session_date": head.session_date.isoformat() if head.session_date else None,
                "session_seq": head.session_seq,
                "session_time": head.session_time.strftime("%H:%M") if head.session_time else None,
                "season": head.season,
                "track_count": len(members),
                "performers": "; ".join(performers),
                "release_ref": release,
                "venue": head.venue,
                "camp_date_raw": head.camp_date_raw,
                "sitting_raw": clean_text(head.sitting_raw),
                "detail_contents": detail,
                "warnings": "; ".join(warns),
            })
    return len(order)


def _scan_audio_keys(audio_root: Path, glob: str) -> tuple[set[str], int]:
    """Key every file under audio_root via the production path_parser. Returns
    (set_of_join_keys, files_scanned)."""
    keys: set[str] = set()
    scanned = 0
    for p in audio_root.glob(glob):
        if not p.is_file():
            continue
        scanned += 1
        k = join_key_from_path(p, base_dir=audio_root)
        if k:
            keys.add(k)
    return keys, scanned


def _build_report(
    tracks: list[CatalogTrack],
    n_sittings: int,
    audio_keys: set[str] | None,
    audio_scanned: int,
) -> dict[str, Any]:
    total = len(tracks)
    keyed = [t for t in tracks if t.join_key]
    dark = total - len(keyed)
    year_unreliable = sum(1 for t in tracks if not t.year_reliable)
    no_date = sum(1 for t in tracks if t.session_date is None)
    with_detail = sum(1 for t in tracks if t.detail_contents)
    track_types = Counter(t.track_type for t in tracks if t.track_type)
    years = Counter(t.camp_year for t in tracks if t.year_reliable)
    warn_kinds: Counter[str] = Counter()
    for t in tracks:
        for w in t.warnings:
            warn_kinds[w.split(":", 1)[0]] += 1

    report: dict[str, Any] = {
        "tracks_total": total,
        "sittings_total": n_sittings,
        "tracks_keyed": len(keyed),
        "tracks_dark_no_key": dark,
        "year_unreliable_1900": year_unreliable,
        "tracks_without_session_date": no_date,
        "tracks_with_hand_transcription": with_detail,
        "track_type_distribution": dict(track_types.most_common()),
        "year_distribution": {str(y): c for y, c in sorted(years.items())},
        "warning_kinds": dict(warn_kinds.most_common()),
    }

    if audio_keys is not None:
        catalog_keys = {t.join_key for t in keyed}
        matched = catalog_keys & audio_keys
        report["sync"] = {
            "audio_files_scanned": audio_scanned,
            "audio_keys": len(audio_keys),
            "catalog_keys": len(catalog_keys),
            "matched": len(matched),
            "catalog_only_dark": len(catalog_keys - audio_keys),
            "audio_only_uncatalogued": len(audio_keys - catalog_keys),
            "catalog_match_pct": round(100 * len(matched) / max(1, len(catalog_keys)), 1),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    ap = argparse.ArgumentParser(description="Normalise the Camp Record Export spreadsheet.")
    ap.add_argument("--xlsx", required=True, type=Path, help="Path to the .xlsx export.")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    ap.add_argument("--audio-root", type=Path, default=None,
                    help="Optional audio/transcript root to compute a sync coverage report.")
    ap.add_argument("--audio-glob", default="**/*",
                    help="Glob under --audio-root selecting files to key (default: all files).")
    args = ap.parse_args(argv)

    if not args.xlsx.exists():
        log.error("xlsx not found: %s", args.xlsx)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("reading %s", args.xlsx)
    rows = _read_rows(args.xlsx)
    log.info("normalising %d rows", len(rows))
    tracks = rows_to_tracks(rows)

    tracks_path = args.out_dir / "catalog_tracks.csv"
    sittings_path = args.out_dir / "catalog_sittings.csv"
    _write_tracks(tracks, tracks_path)
    n_sittings = _write_sittings(tracks, sittings_path)
    log.info("wrote %s (%d tracks) and %s (%d sittings)",
             tracks_path, len(tracks), sittings_path, n_sittings)

    audio_keys: set[str] | None = None
    audio_scanned = 0
    if args.audio_root is not None:
        if not args.audio_root.exists():
            log.warning("audio-root does not exist, skipping sync: %s", args.audio_root)
        else:
            log.info("scanning audio root %s (glob %s)", args.audio_root, args.audio_glob)
            audio_keys, audio_scanned = _scan_audio_keys(args.audio_root, args.audio_glob)
            log.info("scanned %d files -> %d distinct keys", audio_scanned, len(audio_keys))

    report = _build_report(tracks, n_sittings, audio_keys, audio_scanned)
    report_path = args.out_dir / "catalog_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
