"""Parse the source-audio folder hierarchy into structured chunk metadata.

Per PRD §6 Phase 12.

Canonical layout (the PRD example)::

    <base_dir>/
      <Collection> <YYYY>/                          "Live Masters 2010"
        <NN> <LOCATION> <D1> - <D2> <MON> <YYYY>/   "01 NOIDA 7 - 10 JAN 2010"
          <D> <MON> - <SEQ>$ - <H> <AM|PM>/         "7 JAN - 1$ - 6 PM"
            <NN> <TITLE>.<ext>                      "04 PRAVACHAN.wav"

The real corpus is messier. The Dagshai tree inserts a month-bucket level
("01 JAN - 2001"), sometimes omits the event level entirely, and sometimes
inserts a media-container folder ("MD RECORDING OF THIS CAMP") between event
and session. Directory depth therefore ranges from 2 to 5. Levels are
consequently identified **by the shape of the folder name, not by its position
in the path** — see `_classify_folder`. A folder that matches no level grammar
contributes nothing and is skipped (media containers, "New folder", whisper's
`turbo/` and `_model-*` scaffolding all fall out through the same door).

Every level is parsed best-effort: a malformed level still yields a
PathMetadata object with the unparsed fields set to None and a human-readable
warning appended to `parse_warnings`. The caller (chunker / ingester) decides
whether to log, quarantine, or proceed — this module never raises on bad
input. Per CLAUDE.md rule 6 (no silent swallowing) every missing or failed
level produces a warning string.

Public surface:
    parse_path(path, base_dir=None) -> PathMetadata
    season_for(d: date) -> str
    track_type_for(title: str) -> str
    month_num(token: str) -> int | None
    PRIMARY_SPEAKER: str
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path, PurePath

PRIMARY_SPEAKER: str = "Swami ji"

# A genuine second speaker. 113 SAMBODHAN titles name him; of those 38 name Swami
# ji too. Only a title naming Rishi ji AND NOT Swami ji is stamped as his — a
# dual-speaker track keeps the primary (Swami ji is legitimately present), the
# same single-valued-field limitation that makes a two-type track 'combined'.
SECOND_SPEAKER: str = "Rishi ji"
_RISHI_JI = re.compile(r"\bRISHI\s*JI\b")
_SWAMI_JI = re.compile(r"\bSWAMI\s*JI\b")

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

# Every controlled value track_type_for can emit, for schema validation elsewhere.
TRACK_TYPES: frozenset[str] = frozenset(TRACK_TYPE_VOCAB.values()) | {
    DEFAULT_TRACK_TYPE, "qa", "session", "combined",
}

# Vocab keys ordered longest-first so "OM GURUVE NAMAH" is tried before any prefix
# of it could match. Used by the head-anchored matcher below.
_VOCAB_BY_LEN: tuple[tuple[str, str], ...] = tuple(
    sorted(TRACK_TYPE_VOCAB.items(), key=lambda kv: -len(kv[0]))
)

# A vocab word at the HEAD of the title, followed by end-of-title, a separator
# ( "(" or "-" ), or a bare trailing number. Anything else after the word means it
# is part of a song lyric ("MEDITATION KI MASTIYON MEIN"), so the title stays a
# bhajan. This is head-anchored on purpose: substring matching reclassifies that
# song, and "MERI SHAM MEDITATION MEIN" (a song) has MEDITATION mid-title.
_HEAD_TAIL = re.compile(r"\s*($|[(\-]|\d+\s*$)")

# IMD-standard 4-season mapping for India. See PRD §6 Phase 12.
SEASON_BY_MONTH: dict[int, str] = {
    1: "winter", 2: "winter",
    3: "summer", 4: "summer", 5: "summer",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post-monsoon", 11: "post-monsoon", 12: "post-monsoon",
}

# Month keys are the uppercased first three characters, so "Apr", "APRIL",
# "June" and "SEPT" all fold onto JAN..DEC. Folder names in the corpus mix all
# four styles within a single tree.
_MONTHS: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def month_num(token: str) -> int | None:
    """Fold a month token of any length ('Apr', 'APRIL', 'June') to 1..12.
    Returns None for anything that isn't a month — this is what stops words
    like 'NOON' from being read as a month by the `[A-Za-z]{3,}` grammars."""
    return _MONTHS.get(token[:3].upper()) if len(token) >= 3 else None


# Suffixes appended by the upstream vocal-isolation / whisper pipelines.
# These get tacked onto folder names ("Live Masters 2010_isolation") and must
# be stripped before the level regexes run, otherwise the year/AM-PM anchors
# at end-of-string never match.
_FOLDER_SUFFIX_RE = re.compile(r"(_isolation|_model-.*)$")

# Folders matching this pattern are inserted by the whisper pipeline between
# the session-level folder and the actual transcript files.
_MODEL_FOLDER_RE = re.compile(r"_model-")

# Exact folder names produced by the whisper CLI as a leaf scaffolding
# directory (one per inference model size).
_SCAFFOLDING_FOLDERS: frozenset[str] = frozenset({"turbo"})


def _strip_folder_suffix(name: str) -> str:
    """Strip pipeline-added suffixes ('_isolation', '_model-...') from a
    folder name so it parses against the PRD-spec regexes."""
    return _FOLDER_SUFFIX_RE.sub("", name).strip()


def _is_pipeline_scaffolding(name: str) -> bool:
    """True for folders inserted by the upstream whisper pipeline that
    carry no semantic info (model-name folders, turbo/ leaf folders)."""
    return name in _SCAFFOLDING_FOLDERS or bool(_MODEL_FOLDER_RE.search(name))


# --- Level grammars ---------------------------------------------------------
#
# Tried in this order. The order is load-bearing: "01 JAN - 2001" satisfies
# the collection grammar's "<text> <year>" shape, and "DAGSHAI 15 - 19 JAN
# 2001" satisfies both, so the most specific grammar must win.

# Event: an optional sequence number, a location, and a day RANGE.
# Handles every observed variant:
#   "01 NOIDA 7 - 10 JAN 2010"                seq + loc, month/year on d2 only
#   "DAGSHAI 15 - 19 JAN 2001"                no seq
#   "DAGSHAI 10 - 13 JUNE CHILDREN CAMP"      no year (inherited), trailing tail
#   "08 BATHINDA 30 NOV - 2 DEC 2013"         month on both ends
#   "08 CHHATTARPUR DELHI 30 DEC 2014 - 1 JAN 2015"   cross-year camp
_EVENT_RE = re.compile(
    r"^(?:(?P<seq>\d{1,3})\s+)?"
    r"(?P<loc>[A-Za-z][^\d]*?)\s+"
    r"(?P<d1>\d{1,2})(?:\s+(?P<mon1>[A-Za-z]{3,}))?(?:\s+(?P<y1>\d{4}))?"
    r"\s*-\s*"
    r"(?P<d2>\d{1,2})\s+(?P<mon2>[A-Za-z]{3,})(?:\s+(?P<y2>\d{4}))?"
    r"(?:\s+(?P<tail>\S.*))?$"
)

# Month bucket: the Dagshai tree groups sittings by calendar month before
# (or instead of) the event level. Carries a year, nothing else.
#   "01 JAN - 2001"
_MONTH_BUCKET_RE = re.compile(
    r"^\d{1,2}\s+(?P<mon>[A-Za-z]{3,})\s*-\s*(?P<year>\d{4})$"
)

# Collection: a name (no digits) followed by a 4-digit year.
#   "Live Masters 2010", "Dagshai 2001"
_COLLECTION_RE = re.compile(r"^(?P<name>[A-Za-z][^\d]*?)\s+(?P<year>\d{4})$")

# Session: a day + month, then anything. The sequence marker and the clock
# time are pulled out of the remainder separately because both are optional
# and appear with wildly inconsistent punctuation:
#   "7 JAN - 1$ - 6 PM"      "10 JAN - 6 PM"        "8 AUG 12 NOON"
#   "25 MAR-2$-12 NOON"      "13 FEB - 6 PM (CASS REC)"
#   "18 MAY 1 PM - SCHOOL CHILDREN"                 "30 JUNE - NO TIME"
#   "26 JAN MORNING"         "16 JAN"               "08 APRIL"
_SESSION_RE = re.compile(r"^(?P<day>\d{1,2})\s+(?P<mon>[A-Za-z]{3,})\b(?P<rest>.*)$")

# "- 1$ -", " 2$ ", "-2$-". The '$' must directly follow digits, so a bare
# "$" marker ("13 JAN - LOHRI CELEBRATION $") is correctly read as no seq.
_SESSION_SEQ_RE = re.compile(r"(?:^|[-\s])(?P<seq>\d{1,2})\s*\$")

# "6 PM", "10:30 AM", "630 PM", "1230 PM". Anchored on the meridiem.
_SESSION_TIME_RE = re.compile(
    r"\b(?P<hour>\d{1,4})(?::(?P<minute>\d{2}))?\s*(?P<mer>AM|PM)\b", re.IGNORECASE
)
_SESSION_NOON_RE = re.compile(r"\bNOON\b", re.IGNORECASE)

# Track: "<NN> <TITLE>" (extension already stripped).
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


def primary_speaker_for(title: str) -> str:
    """The speaker to stamp, from the track title. Defaults to `PRIMARY_SPEAKER`.

    Diarization is 0/9,335 corpus-wide (every chunk is stamped Swami ji), so the
    title is the only speaker signal there is. A title naming ONLY Rishi ji (e.g.
    `SAMBODHAN - RISHI JI`, 73 tracks) is his; a title naming both (38 tracks) keeps
    Swami ji, because he is genuinely present and the field holds one value.
    """
    t = re.sub(r"\s+", " ", (title or "").upper())
    if _RISHI_JI.search(t) and not _SWAMI_JI.search(t):
        return SECOND_SPEAKER
    return PRIMARY_SPEAKER


def track_type_for(title: str) -> str:
    """Map a track title to a controlled-vocab type. Unknown -> 'bhajan'.

    Head-anchored, not substring. Over the 9,335-track corpus this recovers 1,049
    tracks the old exact-match dropped to 'bhajan' (4,212 -> 5,261 typed) while
    reclassifying zero songs. Three carve-outs, checked before the vocab:

    - a "QUES"/"QUESTION" prefix -> 'qa'      (209 tracks: audience Q&A)
    - the substring "SITTING"    -> 'session' (83 tracks: COMPLETE SITTING ...)
    - two or more distinct vocab TYPES present -> 'combined' (108 tracks; a
      "SAMBODHAN & PRAVACHAN" is genuinely both an address and a discourse, and a
      single-valued field cannot say so).

    Only after those does a head-anchored vocab word decide the type.
    """
    t = re.sub(r"\s+", " ", title.strip().upper())
    if not t:
        return DEFAULT_TRACK_TYPE

    if t.startswith("QUES"):  # QUESTION 1, QUES-01, QUES - 3
        return "qa"
    if "SITTING" in t:  # COMPLETE SITTING (SONY RECORDER), WELCOME SITTING
        return "session"

    types_present = {
        typ for key, typ in TRACK_TYPE_VOCAB.items()
        if re.search(rf"\b{re.escape(key)}\b", t)
    }
    if len(types_present) >= 2:
        return "combined"

    for key, typ in _VOCAB_BY_LEN:
        if t == key or (t.startswith(key) and _HEAD_TAIL.match(t[len(key):])):
            return typ
    return DEFAULT_TRACK_TYPE


def _to_24h(hour_12: int, meridiem: str) -> int:
    """Convert 12-hour to 24-hour. 12 AM -> 0; 12 PM -> 12; others +12 if PM."""
    if not 1 <= hour_12 <= 12:
        raise ValueError(f"hour {hour_12} outside 1-12")
    if meridiem == "AM":
        return 0 if hour_12 == 12 else hour_12
    if meridiem == "PM":
        return 12 if hour_12 == 12 else hour_12 + 12
    raise ValueError(f"bad meridiem {meridiem!r}")


# --- Shape classification ---------------------------------------------------

def _classify_folder(name: str) -> tuple[str, re.Match[str]] | None:
    """Identify which level a folder name belongs to from its shape alone.

    Returns (level, match) for 'event' | 'month_bucket' | 'collection' |
    'session', or None when the name matches no grammar — media containers
    ("MD RECORDING OF THIS CAMP"), undated sittings ("EVENING $"), and
    pipeline scaffolding all land here and are skipped by the caller.

    Order is significant: the event grammar and the month-bucket grammar both
    produce strings that also satisfy the (looser) collection and session
    grammars, so the specific ones are tried first.
    """
    m = _EVENT_RE.match(name)
    if m and month_num(m.group("mon2")) is not None:
        mon1 = m.group("mon1")
        if mon1 is None or month_num(mon1) is not None:
            return "event", m

    m = _MONTH_BUCKET_RE.match(name)
    if m and month_num(m.group("mon")) is not None:
        return "month_bucket", m

    m = _COLLECTION_RE.match(name)
    if m:
        return "collection", m

    m = _SESSION_RE.match(name)
    if m and month_num(m.group("mon")) is not None:
        return "session", m

    return None


def _parse_collection(name: str, m: re.Match[str], meta: PathMetadata) -> None:
    meta.collection = m.group("name").strip()
    meta.year = int(m.group("year"))


def _parse_month_bucket(name: str, m: re.Match[str], meta: PathMetadata) -> None:
    """The month bucket contributes a year only. It is a grouping folder, not
    an event: it has no location and no day range."""
    if meta.year is None:
        meta.year = int(m.group("year"))


def _parse_event(name: str, m: re.Match[str], meta: PathMetadata) -> None:
    meta.event_id = name
    if m.group("seq"):
        meta.event_seq = int(m.group("seq"))
    meta.location = re.sub(r"\s+", " ", m.group("loc").strip()).upper()

    mon2 = month_num(m.group("mon2"))
    mon1 = month_num(m.group("mon1")) if m.group("mon1") else mon2
    # "DAGSHAI 10 - 13 JUNE CHILDREN CAMP" carries no year — inherit the
    # collection's. The end year defaults to the start year, and vice versa.
    y2 = int(m.group("y2")) if m.group("y2") else meta.year
    y1_explicit = m.group("y1") is not None
    y1 = int(m.group("y1")) if y1_explicit else y2
    if y2 is None:
        meta.parse_warnings.append(f"event: no year context for {name!r}")
        return
    try:
        start = date(y1, mon1, int(m.group("d1")))
        end = date(y2, mon2, int(m.group("d2")))
    except (ValueError, OverflowError) as e:
        meta.parse_warnings.append(f"event: bad date in {name!r}: {e}")
        return
    if start > end:
        # "17 NEW YEAR DELHI 30 DEC - 1 JAN 2012" writes the year once, at the
        # end. Inheriting it for both ends puts 30 DEC *after* 1 JAN; the camp
        # actually opened in the previous December.
        if y1_explicit:
            meta.parse_warnings.append(f"event: start after end in {name!r}")
        else:
            start = date(y1 - 1, mon1, start.day)
    meta.event_start, meta.event_end = start, end


def _session_year(month: int, day: int, meta: PathMetadata) -> int | None:
    """Pick the year for a session date. Sessions carry a day and a month but
    never a year, so it comes from the enclosing event span.

    A New Year camp ("30 DEC 2014 - 1 JAN 2015") spans two years, and taking
    `event_start.year` unconditionally dated its 1 JAN sitting to 2014 — one
    year early, and colliding with a genuine 1 JAN 2014 camp on the
    date-anchored catalog join key. Choose the year that puts the sitting
    inside the camp's own span.
    """
    start, end = meta.event_start, meta.event_end
    if start and end:
        for y in dict.fromkeys((start.year, end.year)):
            try:
                candidate = date(y, month, day)
            except ValueError:
                continue
            if start <= candidate <= end:
                return y
        return start.year
    if start:
        return start.year
    return meta.year


def _parse_session(name: str, m: re.Match[str], meta: PathMetadata) -> None:
    month = month_num(m.group("mon"))
    day = int(m.group("day"))
    rest = m.group("rest")

    seq_m = _SESSION_SEQ_RE.search(rest)
    if seq_m:
        meta.session_seq = int(seq_m.group("seq"))

    year = _session_year(month, day, meta)
    if year is None:
        meta.parse_warnings.append(
            f"session: no year context for {name!r} (event + collection both missing)"
        )
    else:
        try:
            meta.session_date = date(year, month, day)
        except (ValueError, OverflowError) as e:
            meta.parse_warnings.append(f"session: bad date in {name!r}: {e}")

    time_m = _SESSION_TIME_RE.search(rest)
    if time_m:
        try:
            hour_raw = time_m.group("hour")
            # HHMM without colon: "1030" → 10:30, "930" → 9:30
            if len(hour_raw) >= 3:
                hour_12, minute = int(hour_raw[:-2]), int(hour_raw[-2:])
            else:
                hour_12 = int(hour_raw)
                minute = int(time_m.group("minute")) if time_m.group("minute") else 0
            meta.session_time = time(_to_24h(hour_12, time_m.group("mer").upper()), minute)
        except (ValueError, OverflowError) as e:
            meta.parse_warnings.append(f"session: bad time in {name!r}: {e}")
    elif _SESSION_NOON_RE.search(rest):
        meta.session_time = time(12, 0)


def _parse_track(name: str, meta: PathMetadata) -> None:
    # Strip a file extension if present so callers can pass the raw filename.
    stem = PurePath(name).stem if "." in name else name
    m = _TRACK_RE.match(stem.strip())
    if not m:
        # Permit no leading number — the filename itself is the title.
        meta.track_title = stem.strip() or None
        if meta.track_title:
            meta.track_type = track_type_for(meta.track_title)
            meta.primary_speaker = primary_speaker_for(meta.track_title)
        meta.parse_warnings.append(f"track: no leading number in {name!r}")
        return
    try:
        meta.track_no = int(m.group("no"))
    except ValueError:
        meta.parse_warnings.append(f"track: bad track number in {name!r}")
    meta.track_title = m.group("title").strip()
    meta.track_type = track_type_for(meta.track_title)
    meta.primary_speaker = primary_speaker_for(meta.track_title)


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

    # The track is always the last component (the file itself). Every folder
    # above it is classified by shape; unrecognized folders (whisper
    # scaffolding, media containers, "New folder") contribute nothing.
    track_name = parts[-1]
    parents = [_strip_folder_suffix(x) for x in parts[:-1]
               if not _is_pipeline_scaffolding(x)]

    # Left-to-right so the collection year is in hand before the event needs
    # it, and the event span before the session needs it. A repeated level
    # keeps the deepest (rightmost) folder, which is the more specific one.
    found: dict[str, tuple[str, re.Match[str]]] = {}
    for name in parents:
        hit = _classify_folder(name)
        if hit is None:
            meta.parse_warnings.append(f"folder: unrecognized level {name!r} - skipped")
            continue
        found[hit[0]] = (name, hit[1])

    for level, parser in (
        ("collection", _parse_collection),
        ("month_bucket", _parse_month_bucket),
        ("event", _parse_event),
        ("session", _parse_session),
    ):
        if level in found:
            name, m = found[level]
            parser(name, m, meta)
        elif level in ("collection", "event", "session"):
            meta.parse_warnings.append(f"{level}: no {level}-level folder found in path")

    _parse_track(track_name, meta)

    if meta.session_date:
        meta.season = season_for(meta.session_date)
    elif meta.event_start:
        meta.season = season_for(meta.event_start)

    return meta
