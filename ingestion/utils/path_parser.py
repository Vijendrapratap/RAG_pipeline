"""Parse the source-audio folder hierarchy into structured chunk metadata.

Per PRD §6 Phase 12.

Expected layout (one path level per directory):

    <base_dir>/
      <Collection> <YYYY>/                      e.g. "Live Masters 2010"
        <NN> <LOCATION> <D1> - <D2> <MON> <YYYY>/   e.g. "01 NOIDA 7 - 10 JAN 2010"
          <D> <MON> - <SEQ>$ - <H> <AM|PM>/         e.g. "7 JAN - 1$ - 6 PM"
            <NN> <TITLE>.<ext>                     e.g. "04 PRAVACHAN.wav"

Every level is parsed best-effort: a malformed level still yields a
PathMetadata object with the unparsed fields set to None and a human-readable
warning appended to `parse_warnings`. The caller (chunker / ingester) decides
whether to log, quarantine, or proceed — this module never raises on bad
input. Per CLAUDE.md rule 6 (no silent swallowing) every failed-to-parse
level produces a warning string.

Public surface:
    parse_path(path, base_dir=None) -> PathMetadata
    season_for(d: date) -> str
    track_type_for(title: str) -> str
    PRIMARY_SPEAKER: str
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path, PurePath

PRIMARY_SPEAKER: str = "Swami ji"

# Track-type vocabulary. Keys are normalized (upper, single-spaced). Default
# for unmatched titles is "bhajan" — i.e. anything that is not a recognized
# activity type is assumed to be a song/bhajan track.
TRACK_TYPE_VOCAB: dict[str, str] = {
    "PRAVACHAN": "discourse",
    "SAMBODHAN": "address",
    "MEDITATION": "meditation",
    "OM GURUVE NAMAH": "invocation",
    "ENTRY MUSIC": "music",
    "RETURN MUSIC": "music",
}
DEFAULT_TRACK_TYPE: str = "bhajan"

# IMD-standard 4-season mapping for India. See PRD §6 Phase 12.
SEASON_BY_MONTH: dict[int, str] = {
    1: "winter", 2: "winter",
    3: "summer", 4: "summer", 5: "summer",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post-monsoon", 11: "post-monsoon", 12: "post-monsoon",
}

# Three-letter month abbreviations as used in folder names (uppercase).
_MONTHS: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Suffixes appended by the upstream vocal-isolation / whisper pipelines.
# These get tacked onto folder names ("Live Masters 2010_isolation") and must
# be stripped before the level regexes run, otherwise the year/AM-PM anchors
# at end-of-string never match.
_FOLDER_SUFFIX_RE = re.compile(r"(_isolation|_model-.*)$")

# Folders matching this pattern are inserted by the whisper pipeline between
# the session-level folder and the actual transcript files. They carry no
# semantic information for retrieval; drop them when walking levels so the
# parser sees the intended 4-level structure.
_MODEL_FOLDER_RE = re.compile(r"_model-")

# Exact folder names produced by the whisper CLI as a leaf scaffolding
# directory (one per inference model size). Drop these from the parent
# chain. Keep this list narrow — only names we've seen in this pipeline.
_SCAFFOLDING_FOLDERS: frozenset[str] = frozenset({"turbo"})


def _strip_folder_suffix(name: str) -> str:
    """Strip pipeline-added suffixes ('_isolation', '_model-...') from a
    folder name so it parses against the PRD-spec regexes."""
    return _FOLDER_SUFFIX_RE.sub("", name).strip()


def _is_pipeline_scaffolding(name: str) -> bool:
    """True for folders inserted by the upstream whisper pipeline that
    carry no semantic info (model-name folders, turbo/ leaf folders)."""
    return name in _SCAFFOLDING_FOLDERS or bool(_MODEL_FOLDER_RE.search(name))


# Regex grammars. Liberal on whitespace; case-insensitive where it matters.
# Collection: any text ending with a 4-digit year.
_COLLECTION_RE = re.compile(r"^(?P<name>.+?)\s+(?P<year>\d{4})$")

# Event: "<NN> <LOC...> <D1> - <D2> <MON> <YYYY>"
# Location may contain spaces ("NEW DELHI"); month is uppercase 3-letter.
_EVENT_RE = re.compile(
    r"^(?P<seq>\d+)\s+"
    r"(?P<loc>.+?)\s+"
    r"(?P<d1>\d{1,2})\s*-\s*(?P<d2>\d{1,2})\s+"
    r"(?P<mon>[A-Z]{3})\s+"
    r"(?P<year>\d{4})$",
    re.IGNORECASE,
)

# Session: "<D> <MON> - <SEQ>$ - <H> <AM|PM>"
# The $ marker after SEQ is optional (older folders may use a hyphen alone).
# Time may be written as "6 PM" (H), "10:30 AM" (H:MM), or "1030 AM" (HHMM
# without colon — observed in upstream whisper pipeline output).
_SESSION_RE = re.compile(
    r"^(?P<day>\d{1,2})\s+(?P<mon>[A-Z]{3})\s*-\s*"
    r"(?P<seq>\d+)\$?\s*-\s*"
    r"(?P<hour>\d{1,4})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<mer>AM|PM)$",
    re.IGNORECASE,
)

# Track: "<NN> <TITLE>" (extension already stripped). TITLE may contain
# spaces and special characters; keep entire remainder as raw title.
_TRACK_RE = re.compile(r"^(?P<no>\d+)\s+(?P<title>.+?)$")


@dataclass
class PathMetadata:
    """Structured metadata extracted from an audio file path.

    Every field is optional — a malformed input still produces a usable
    PathMetadata where unparsed fields are None and `parse_warnings`
    lists the levels that failed. The chunker / ingester decides what to
    do with partial metadata.
    """
    # Collection level
    collection: str | None = None
    year: int | None = None

    # Event level
    event_seq: int | None = None
    event_id: str | None = None        # raw folder name, e.g. "01 NOIDA 7 - 10 JAN 2010"
    location: str | None = None
    event_start: date | None = None
    event_end: date | None = None

    # Session level
    session_date: date | None = None
    session_seq: int | None = None
    session_time: time | None = None

    # Track level
    track_no: int | None = None
    track_title: str | None = None
    track_type: str | None = None

    # Derived
    season: str | None = None
    primary_speaker: str = PRIMARY_SPEAKER

    # Diagnostics
    parse_warnings: list[str] = field(default_factory=list)
    source_path: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Flat dict for Qdrant payload / Postgres insert. Dates and times
        serialized to ISO strings so they round-trip cleanly through JSON."""
        return {
            "collection": self.collection,
            "year": self.year,
            "event_seq": self.event_seq,
            "event_id": self.event_id,
            "location": self.location,
            "event_start": self.event_start.isoformat() if self.event_start else None,
            "event_end": self.event_end.isoformat() if self.event_end else None,
            "session_date": self.session_date.isoformat() if self.session_date else None,
            "session_seq": self.session_seq,
            "session_time": self.session_time.isoformat() if self.session_time else None,
            "track_no": self.track_no,
            "track_title": self.track_title,
            "track_type": self.track_type,
            "season": self.season,
            "primary_speaker": self.primary_speaker,
        }

    def header_fragment(self) -> str:
        """Compact human-readable fragment for the chunk header line.
        Returned empty when no metadata parsed (so the chunker can decide
        whether to include it). Embedding model sees this verbatim — it's
        a free precision boost for queries that mention location/date/type.
        """
        parts: list[str] = []
        if self.event_id:
            parts.append(f"Event: {self.event_id}")
        elif self.location:
            parts.append(f"Location: {self.location}")
        if self.session_date:
            parts.append(f"Date: {self.session_date.isoformat()}")
        if self.session_time:
            parts.append(f"Time: {self.session_time.strftime('%H:%M')}")
        if self.track_title:
            parts.append(f"Track: {self.track_title}")
        if self.track_type:
            parts.append(f"Type: {self.track_type}")
        if self.season:
            parts.append(f"Season: {self.season}")
        return " | ".join(parts)


def season_for(d: date) -> str:
    """Return the IMD-standard Indian season for a given date."""
    return SEASON_BY_MONTH[d.month]


def track_type_for(title: str) -> str:
    """Map a track title to a controlled-vocab type. Unknown -> 'bhajan'."""
    normalized = re.sub(r"\s+", " ", title.strip().upper())
    return TRACK_TYPE_VOCAB.get(normalized, DEFAULT_TRACK_TYPE)


def _parse_collection(name: str, meta: PathMetadata) -> None:
    m = _COLLECTION_RE.match(_strip_folder_suffix(name))
    if not m:
        meta.parse_warnings.append(f"collection: could not parse {name!r}")
        return
    meta.collection = m.group("name").strip()
    try:
        meta.year = int(m.group("year"))
    except ValueError:
        meta.parse_warnings.append(f"collection: bad year in {name!r}")


def _parse_event(name: str, meta: PathMetadata) -> None:
    cleaned = _strip_folder_suffix(name)
    m = _EVENT_RE.match(cleaned)
    if not m:
        meta.parse_warnings.append(f"event: could not parse {name!r}")
        return
    meta.event_id = cleaned
    try:
        meta.event_seq = int(m.group("seq"))
    except ValueError:
        meta.parse_warnings.append(f"event: bad seq in {name!r}")
    meta.location = m.group("loc").strip().upper()
    mon_key = m.group("mon").upper()
    if mon_key not in _MONTHS:
        meta.parse_warnings.append(f"event: unknown month {mon_key!r} in {name!r}")
        return
    month = _MONTHS[mon_key]
    try:
        year = int(m.group("year"))
        d1 = int(m.group("d1"))
        d2 = int(m.group("d2"))
        meta.event_start = date(year, month, d1)
        meta.event_end = date(year, month, d2)
    except (ValueError, OverflowError) as e:
        meta.parse_warnings.append(f"event: bad date in {name!r}: {e}")


def _parse_session(name: str, meta: PathMetadata) -> None:
    m = _SESSION_RE.match(_strip_folder_suffix(name))
    if not m:
        meta.parse_warnings.append(f"session: could not parse {name!r}")
        return
    mon_key = m.group("mon").upper()
    if mon_key not in _MONTHS:
        meta.parse_warnings.append(f"session: unknown month {mon_key!r} in {name!r}")
        return
    month = _MONTHS[mon_key]
    # Year for session comes from the event (or collection) — sessions don't
    # carry their own year. Fall back to collection year if event missing.
    year = meta.event_start.year if meta.event_start else meta.year
    if year is None:
        meta.parse_warnings.append(
            f"session: no year context for {name!r} (event + collection both missing)"
        )
    try:
        meta.session_seq = int(m.group("seq"))
    except ValueError:
        meta.parse_warnings.append(f"session: bad seq in {name!r}")
    if year is not None:
        try:
            meta.session_date = date(year, month, int(m.group("day")))
        except (ValueError, OverflowError) as e:
            meta.parse_warnings.append(f"session: bad date in {name!r}: {e}")
    try:
        hour_raw = m.group("hour")
        # HHMM without colon: "1030" → 10:30, "930" → 9:30
        if len(hour_raw) >= 3:
            hour_12 = int(hour_raw[:-2])
            minute = int(hour_raw[-2:])
        else:
            hour_12 = int(hour_raw)
            minute = int(m.group("minute")) if m.group("minute") else 0
        hour_24 = _to_24h(hour_12, m.group("mer").upper())
        meta.session_time = time(hour_24, minute)
    except (ValueError, OverflowError) as e:
        meta.parse_warnings.append(f"session: bad time in {name!r}: {e}")


def _to_24h(hour_12: int, meridiem: str) -> int:
    """Convert 12-hour to 24-hour. 12 AM -> 0; 12 PM -> 12; others +12 if PM."""
    if not 1 <= hour_12 <= 12:
        raise ValueError(f"hour {hour_12} outside 1-12")
    if meridiem == "AM":
        return 0 if hour_12 == 12 else hour_12
    if meridiem == "PM":
        return 12 if hour_12 == 12 else hour_12 + 12
    raise ValueError(f"bad meridiem {meridiem!r}")


def _parse_track(name: str, meta: PathMetadata) -> None:
    # Strip a file extension if present so callers can pass the raw filename.
    stem = PurePath(name).stem if "." in name else name
    m = _TRACK_RE.match(stem.strip())
    if not m:
        # Permit no leading number — the filename itself is the title.
        meta.track_title = stem.strip() or None
        if meta.track_title:
            meta.track_type = track_type_for(meta.track_title)
        meta.parse_warnings.append(f"track: no leading number in {name!r}")
        return
    try:
        meta.track_no = int(m.group("no"))
    except ValueError:
        meta.parse_warnings.append(f"track: bad track number in {name!r}")
    meta.track_title = m.group("title").strip()
    meta.track_type = track_type_for(meta.track_title)


def parse_path(path: Path | str, base_dir: Path | str | None = None) -> PathMetadata:
    """Parse a source-audio file path into structured metadata.

    Args:
        path: Full path to an audio file (or transcript file derived from
            one). May be absolute or relative. Extension is ignored.
        base_dir: Optional base directory to strip before parsing. If
            provided, only the path components below `base_dir` are
            inspected. If `path` is not under `base_dir`, all components
            are inspected from the leftmost (drive root excluded).

    Returns:
        PathMetadata. Always non-None; check `parse_warnings` to discover
        which levels failed. `source_path` is set to the original path
        for downstream logging.
    """
    p = Path(path)
    meta = PathMetadata(source_path=str(p))

    if base_dir is not None:
        try:
            rel_parts = p.relative_to(Path(base_dir)).parts
        except ValueError:
            rel_parts = p.parts
    else:
        rel_parts = p.parts

    # Drop drive root / Windows leading "\" entries: only meaningful named
    # components contribute. We also drop the bare drive ("D:\\") if present.
    parts = [seg for seg in rel_parts if seg and seg not in ("/", "\\")
             and not (len(seg) == 3 and seg.endswith(":\\"))
             and seg != Path(p.anchor).name]
    # Path.anchor handles "C:\\" and "/" already, but defensively re-strip:
    if p.anchor and parts and parts[0] == p.anchor.rstrip("\\/"):
        parts = parts[1:]

    if not parts:
        meta.parse_warnings.append("empty path after stripping anchor/base_dir")
        return meta

    # Drop whisper model-name folders from the parent chain (e.g.
    # "04 PRAVACHAN_model-1_mel_roformer_kim_ft"). They sit between the
    # session folder and the actual transcript file and would otherwise
    # shift every level by one. Only filter parents — never the track itself.
    if len(parts) > 1:
        parents_kept = [p for p in parts[:-1] if not _is_pipeline_scaffolding(p)]
        parts = parents_kept + [parts[-1]]

    # The track is always the last component (the file itself). The three
    # parents above it (if present) map to session, event, collection.
    track_name = parts[-1]
    parents = parts[:-1]

    # Walk from rightmost upward so that a path shallower than 4 levels
    # still parses the deepest meaningful folders. The known order
    # (collection > event > session > track) is encoded by index.
    if len(parents) >= 3:
        collection_name = parents[-3]
        event_name = parents[-2]
        session_name = parents[-1]
    elif len(parents) == 2:
        collection_name = None
        event_name = parents[-2]
        session_name = parents[-1]
        meta.parse_warnings.append("collection level missing (path < 4 levels)")
    elif len(parents) == 1:
        collection_name = None
        event_name = None
        session_name = parents[-1]
        meta.parse_warnings.append("collection and event levels missing")
    else:
        collection_name = None
        event_name = None
        session_name = None
        meta.parse_warnings.append("only track name present (no enclosing folders)")

    if collection_name:
        _parse_collection(collection_name, meta)
    if event_name:
        _parse_event(event_name, meta)
    if session_name:
        _parse_session(session_name, meta)
    _parse_track(track_name, meta)

    if meta.session_date:
        meta.season = season_for(meta.session_date)
    elif meta.event_start:
        meta.season = season_for(meta.event_start)

    return meta
