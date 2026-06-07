"""Normalise raw 'Camp Record Export' rows into structured catalog records.

Pure functions only — no I/O, no pandas. The loader (`load.py`) reads the
spreadsheet and hands plain dicts here; tests exercise these functions with
literal dicts. Every record carries a `join_key` built with the SAME grammar
`ingestion.utils.path_parser` derives from audio folder names, so a catalog
row and its transcript folder collapse onto one key (see `join_key_from_path`).

Design decisions (mirroring the locked Phase 12 conventions):
  * Season + track-type come from `path_parser.season_for` /
    `track_type_for` — never re-implemented here, so the two sources can
    never drift apart in their vocabularies.
  * `session_date` = date(CampYear, month-from-Sitting, day-from-Sitting).
    The Sitting cell already encodes day + month; CampYear supplies the year.
  * CampYear == 1900 is the export's "year unknown" sentinel — kept, but
    flagged `year_reliable=False` so downstream callers never trust the date.
  * Durations are mm.ss floats (23.42 == 23 min 42 s), NOT decimal minutes.
  * Devanagari is NFC-normalised so titles match across catalog and ASR.

Per CLAUDE.md rule 6: nothing is silently dropped. A row that fails to parse
its Sitting still yields a CatalogTrack with `session_date=None` and a
human-readable note in `warnings`; the caller logs / reports it.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import Any

from ingestion.utils.path_parser import (
    parse_path,
    season_for,
    track_type_for,
)

# Three-letter month keys (uppercased first 3 chars of whatever the cell uses:
# "Apr", "June", "NOV" all fold to JAN..DEC). Mirrors path_parser._MONTHS.
_MONTHS: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# The export encodes carriage returns from the original cells as this token.
_CR_TOKEN = "_x000D_"

# "Year unknown" sentinel used throughout the export.
UNKNOWN_YEAR = 1900

# Sitting grammar. Tolerant of the variants seen in the real sheet:
#   "13 APR - 2$ - 10:30 AM"   full form
#   "16 Nov - 8 PM"            no sitting-sequence (older daily sittings)
#   "16 APR - 3$ - 630 AM"     time without a colon
#   "7 June - 6$ - 10:00 AM"   4-letter month
# The sequence group ("2$ - ") is optional; time meridiem is required so a
# bare "1$" does not masquerade as a session.
_SITTING_RE = re.compile(
    r"^(?P<day>\d{1,2})\s+(?P<mon>[A-Za-z]{3,})\s*-\s*"
    r"(?:(?P<seq>\d+)\s*\$?\s*-\s*)?"
    r"(?P<hour>\d{1,4})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<mer>[AP]M)\s*$",
    re.IGNORECASE,
)

# Fallback: a sitting cell that is only a sequence marker ("1$", "6$"). Carries
# no date — useful only to disambiguate tracks within an already-known camp.
_SEQ_ONLY_RE = re.compile(r"^(?P<seq>\d+)\s*\$\s*$")


def clean_text(value: Any) -> str | None:
    """Strip the export's `_x000D_` carriage-return tokens, collapse the
    resulting blank runs, NFC-normalise, and trim. Returns None for empty /
    NaN-like input so downstream code can treat 'absent' uniformly."""
    if value is None:
        return None
    s = str(value)
    if s.strip().lower() in ("", "nan", "none"):
        return None
    s = s.replace(_CR_TOKEN, "\n")
    s = unicodedata.normalize("NFC", s)
    # Collapse 3+ newlines to 2 (paragraph breaks) and trailing spaces per line.
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip() or None


def _month_num(token: str) -> int | None:
    return _MONTHS.get(token[:3].upper())


# Metros whose camps appear under neighbourhood-qualified folder names
# ("PITAMPURA DELHI", "DWARKA DELHI"). When such a token is present the
# location folds to the metro so the catalog ("Delhi") and the folder agree on
# a single city-level facet. Grow this set as new multi-area cities surface.
_METRO_TOKENS: frozenset[str] = frozenset({"DELHI"})


def normalize_location(raw: Any) -> str | None:
    """Fold a raw place string (catalog `CampPlace` or a path-derived
    location) to a canonical city-level facet, uppercase.

    Handles the drift observed between the two sources:
      * comma forms      "Chhattarpur, Delhi" -> "DELHI"
      * neighbourhoods   "PITAMPURA DELHI"     -> "DELHI"
      * spelling/case    "Noida" / "NOIDA"     -> "NOIDA"

    Heuristic by design — a metro token collapses its neighbourhood; otherwise
    the cleaned string stands. Returns None for blank input. The raw place is
    always preserved alongside (CatalogTrack.camp_place) so nothing is lost.
    """
    text = clean_text(raw)
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip().upper()
    if "," in text:
        # City is conventionally the last comma-separated segment.
        text = text.split(",")[-1].strip()
    tokens = text.split()
    for metro in _METRO_TOKENS:
        if metro in tokens and len(tokens) > 1:
            return metro
    return text or None


def _to_24h(hour_12: int, meridiem: str) -> int | None:
    """12-hour to 24-hour. Returns None on an out-of-range hour rather than
    raising — a bad time should degrade the record, not crash the batch."""
    if not 1 <= hour_12 <= 12:
        return None
    mer = meridiem.upper()
    if mer == "AM":
        return 0 if hour_12 == 12 else hour_12
    return 12 if hour_12 == 12 else hour_12 + 12


@dataclass
class ParsedSitting:
    """The structured pieces pulled out of a raw 'Sitting' cell."""
    session_date: date | None = None
    session_seq: int | None = None
    session_time: time | None = None
    warnings: list[str] = field(default_factory=list)


def parse_sitting(raw: Any, camp_year: int | None) -> ParsedSitting:
    """Parse a 'Sitting' cell (e.g. '13 APR - 2$ - 10:30 AM') against a
    CampYear. Best-effort: an unparseable cell returns an all-None
    ParsedSitting with a warning, never raises."""
    out = ParsedSitting()
    text = clean_text(raw)
    if text is None:
        out.warnings.append("sitting: empty")
        return out
    # Some cells smuggle a trailing year or stray note after the time; take the
    # first line only.
    text = text.splitlines()[0].strip()

    m = _SITTING_RE.match(text)
    if not m:
        seq_only = _SEQ_ONLY_RE.match(text)
        if seq_only:
            out.session_seq = int(seq_only.group("seq"))
            out.warnings.append(f"sitting: sequence-only, no date in {text!r}")
        else:
            out.warnings.append(f"sitting: unparseable {text!r}")
        return out

    if m.group("seq") is not None:
        out.session_seq = int(m.group("seq"))

    month = _month_num(m.group("mon"))
    if month is None:
        out.warnings.append(f"sitting: unknown month in {text!r}")
    elif camp_year is None:
        out.warnings.append(f"sitting: no CampYear for {text!r}")
    else:
        try:
            out.session_date = date(camp_year, month, int(m.group("day")))
        except (ValueError, OverflowError) as exc:
            out.warnings.append(f"sitting: bad date in {text!r}: {exc}")

    # Time.
    hour_raw = m.group("hour")
    try:
        if len(hour_raw) >= 3:  # "630" -> 6:30, "1030" -> 10:30
            hour_12, minute = int(hour_raw[:-2]), int(hour_raw[-2:])
        else:
            hour_12 = int(hour_raw)
            minute = int(m.group("minute")) if m.group("minute") else 0
        hour_24 = _to_24h(hour_12, m.group("mer"))
        if hour_24 is None or not 0 <= minute < 60:
            out.warnings.append(f"sitting: bad time in {text!r}")
        else:
            out.session_time = time(hour_24, minute)
    except (ValueError, OverflowError):
        out.warnings.append(f"sitting: bad time in {text!r}")

    return out


def parse_duration_to_seconds(value: Any) -> int | None:
    """Convert a mm.ss duration float (23.42 -> 1422 s; 30.3 -> 1830 s) to
    whole seconds. The fractional part is *seconds*, not a decimal fraction of
    a minute, so 0.3 means :30 and 0.07 means :07. Returns None for missing /
    malformed input (e.g. a seconds field >= 60)."""
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x or x < 0:  # NaN or negative
        return None
    minutes = int(x)
    # Round to 2 dp to recover ':30' from a float that stored 30.3, not 30.30.
    seconds = round((x - minutes) * 100)
    if seconds >= 60:
        return None
    return minutes * 60 + seconds


def split_performers(raw: Any) -> list[str]:
    """Split a 'Chorus' cell ('ABHIPSA, SUMAN, Prachi Ji') into a clean,
    title-cased name list. Drops honorific-only fragments and dedupes
    case-insensitively while preserving order."""
    text = clean_text(raw)
    if text is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,/&]| and ", text):
        name = re.sub(r"\s+", " ", part).strip(" .")
        # Strip a trailing honorific so "Suman Ji" and "Suman" collapse.
        name = re.sub(r"\s+ji$", "", name, flags=re.IGNORECASE).strip()
        if not name:
            continue
        norm = name.casefold()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(name.title() if name.isupper() or name.islower() else name)
    return out


def join_key_from_fields(
    session_date: date | None,
    session_seq: int | None,
    track_no: int | None,
) -> str | None:
    """Canonical join key shared with the audio/transcript side.

    Format: ``YYYY-MM-DD|SEQ|TRACKNO``. Deliberately **date-anchored, not
    location-anchored**: a camp happens on specific dates, so date+seq
    uniquely identifies a sitting and track_no the track — globally, without
    relying on the location string. Validated on the accessible corpus: the
    date key matches 63/63 audio tracks, whereas adding location dropped that
    to 30/64 because folders carry a neighbourhood ("PITAMPURA DELHI") while
    the catalog carries the city ("Delhi"). Location is reconciled separately
    (`normalize_location`) as a filter/display field, never as a join axis.

    Returns None when no session_date is known — such a row cannot be safely
    aligned to a folder and is reported as 'dark' (catalog-only) by sync.
    """
    if session_date is None:
        return None
    seq = "" if session_seq is None else str(session_seq)
    tno = "" if track_no is None else str(track_no)
    return f"{session_date.isoformat()}|{seq}|{tno}"


def join_key_from_path(path: Path | str, base_dir: Path | str | None = None) -> str | None:
    """Derive the same join key from an audio / transcript file path, using
    the production `path_parser`. This is the bridge that lets the sync step
    match a catalog row to a real folder without duplicating parse logic."""
    meta = parse_path(path, base_dir)
    return join_key_from_fields(meta.session_date, meta.session_seq, meta.track_no)


@dataclass
class CatalogTrack:
    """One normalised spreadsheet row (grain = one track)."""
    # Raw provenance (kept verbatim for audit / display).
    camp_date_raw: str | None = None
    camp_year: int | None = None
    camp_place: str | None = None
    sitting_raw: str | None = None
    venue: str | None = None

    # Normalised event / session.
    location: str | None = None
    session_date: date | None = None
    session_seq: int | None = None
    session_time: time | None = None
    season: str | None = None
    year_reliable: bool = True

    # Track.
    track_no: int | None = None
    track_title: str | None = None      # canonical, NFC-normalised (Content)
    track_type: str | None = None
    duration_sec: int | None = None

    # Catalog-only enrichment (no audio-path equivalent).
    performers: list[str] = field(default_factory=list)
    release_ref: str | None = None      # Comment / MainComm
    detail_contents: str | None = None  # per-sitting hand transcription

    # Linkage + diagnostics.
    join_key: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        """Flat, CSV/DB-friendly dict (dates -> ISO strings, lists -> joined)."""
        return {
            "join_key": self.join_key,
            "location": self.location,
            "camp_place": self.camp_place,
            "camp_year": self.camp_year,
            "year_reliable": self.year_reliable,
            "session_date": self.session_date.isoformat() if self.session_date else None,
            "session_seq": self.session_seq,
            "session_time": self.session_time.strftime("%H:%M") if self.session_time else None,
            "season": self.season,
            "track_no": self.track_no,
            "track_title": self.track_title,
            "track_type": self.track_type,
            "duration_sec": self.duration_sec,
            "performers": "; ".join(self.performers),
            "release_ref": self.release_ref,
            "venue": self.venue,
            "camp_date_raw": self.camp_date_raw,
            "sitting_raw": clean_text(self.sitting_raw),
            "has_detail": self.detail_contents is not None,
            "warnings": "; ".join(self.warnings),
        }


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        f = float(value)
        if f != f:  # NaN
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


def row_to_track(row: dict[str, Any]) -> CatalogTrack:
    """Normalise one raw spreadsheet row (column-name -> cell value) into a
    CatalogTrack. Column names match the export headers exactly."""
    camp_year = _coerce_int(row.get("CampYear"))
    year_reliable = camp_year is not None and camp_year != UNKNOWN_YEAR

    sitting = parse_sitting(row.get("Sitting"), camp_year if year_reliable else None)

    location = clean_text(row.get("CampPlace"))
    location_norm = normalize_location(location)

    track_no = _coerce_int(row.get("TrackNo"))
    title = clean_text(row.get("Content"))

    track = CatalogTrack(
        camp_date_raw=clean_text(row.get("CampDate")),
        camp_year=camp_year,
        camp_place=location,
        sitting_raw=clean_text(row.get("Sitting")),
        venue=clean_text(row.get("Venue")),
        location=location_norm,
        session_date=sitting.session_date,
        session_seq=sitting.session_seq,
        session_time=sitting.session_time,
        year_reliable=year_reliable,
        track_no=track_no,
        track_title=title,
        track_type=track_type_for(title) if title else None,
        duration_sec=parse_duration_to_seconds(row.get("Duration")),
        performers=split_performers(row.get("Chorus")),
        release_ref=clean_text(row.get("Comment")) or clean_text(row.get("MainComm")),
        detail_contents=clean_text(row.get("DetailContents")),
        warnings=list(sitting.warnings),
    )
    if track.session_date is not None:
        track.season = season_for(track.session_date)
    if not year_reliable and camp_year == UNKNOWN_YEAR:
        track.warnings.append("camp_year: 1900 sentinel (year unknown)")
    track.join_key = join_key_from_fields(
        track.session_date, track.session_seq, track.track_no
    )
    return track


def rows_to_tracks(rows: list[dict[str, Any]]) -> list[CatalogTrack]:
    """Normalise an iterable of raw rows. Convenience wrapper over
    `row_to_track` so the loader stays a thin shell."""
    return [row_to_track(r) for r in rows]
