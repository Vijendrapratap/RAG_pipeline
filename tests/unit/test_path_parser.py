"""Unit tests for ingestion.utils.path_parser.

Per PRD §6 Phase 12 acceptance criteria:
- parse_path() on the example user path produces the expected fields.
- Each track-type vocab entry maps correctly; unknown defaults to bhajan.
- Season boundaries match IMD 4-season Indian mapping.
- Malformed inputs at every level degrade gracefully (warnings, no raise).
- Re-parsing the same path is idempotent.
"""
from __future__ import annotations

from datetime import date, time
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from ingestion.utils.path_parser import (
    DEFAULT_TRACK_TYPE,
    PRIMARY_SPEAKER,
    SEASON_BY_MONTH,
    PathMetadata,
    parse_path,
    season_for,
    track_type_for,
)

# Use PurePosixPath for tests so they pass on Windows + Linux identically.
# parse_path takes Path|str; passing strings sidesteps platform pathing.
EXAMPLE_PATH = (
    "/mnt/audio/Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/"
    "7 JAN - 1$ - 6 PM/04 PRAVACHAN.wav"
)
EXAMPLE_BASE = "/mnt/audio"


# ---- season_for --------------------------------------------------------


def test_season_for_winter():
    assert season_for(date(2026, 1, 15)) == "winter"
    assert season_for(date(2026, 2, 28)) == "winter"


def test_season_for_summer():
    assert season_for(date(2026, 3, 1)) == "summer"
    assert season_for(date(2026, 5, 31)) == "summer"


def test_season_for_monsoon():
    # The "barsat" months — the trigger query for Phase 12.
    assert season_for(date(2026, 6, 1)) == "monsoon"
    assert season_for(date(2026, 7, 15)) == "monsoon"
    assert season_for(date(2026, 9, 30)) == "monsoon"


def test_season_for_post_monsoon():
    assert season_for(date(2026, 10, 1)) == "post-monsoon"
    assert season_for(date(2026, 12, 31)) == "post-monsoon"


def test_season_by_month_complete():
    # Every month must map to a non-empty season.
    for m in range(1, 13):
        assert m in SEASON_BY_MONTH
        assert SEASON_BY_MONTH[m]


# ---- track_type_for ----------------------------------------------------


@pytest.mark.parametrize(
    "title, expected",
    [
        ("PRAVACHAN", "discourse"),
        ("SAMBODHAN", "address"),
        ("MEDITATION", "meditation"),
        ("OM GURUVE NAMAH", "invocation"),
        ("ENTRY MUSIC", "music"),
        ("RETURN MUSIC", "music"),
        # Case insensitivity + whitespace tolerance
        ("pravachan", "discourse"),
        ("  Pravachan  ", "discourse"),
        ("Om   Guruve   Namah", "invocation"),
    ],
)
def test_track_type_for_known(title: str, expected: str):
    assert track_type_for(title) == expected


@pytest.mark.parametrize(
    "title",
    ["NA HARA HAI ISHQ", "Tu Hi Tu", "BHAJAN INTRO", "अमृत वाणी", "Anonymous Song 42"],
)
def test_track_type_for_unknown_defaults_to_bhajan(title: str):
    assert track_type_for(title) == DEFAULT_TRACK_TYPE
    assert DEFAULT_TRACK_TYPE == "bhajan"


# ---- parse_path: the example from the trigger --------------------------


def test_parse_example_path_full():
    meta = parse_path(EXAMPLE_PATH, base_dir=EXAMPLE_BASE)
    assert meta.collection == "Live Masters"
    assert meta.year == 2010
    assert meta.event_seq == 1
    assert meta.event_id == "01 NOIDA 7 - 10 JAN 2010"
    assert meta.location == "NOIDA"
    assert meta.event_start == date(2010, 1, 7)
    assert meta.event_end == date(2010, 1, 10)
    assert meta.session_date == date(2010, 1, 7)
    assert meta.session_seq == 1
    assert meta.session_time == time(18, 0)
    assert meta.track_no == 4
    assert meta.track_title == "PRAVACHAN"
    assert meta.track_type == "discourse"
    assert meta.season == "winter"
    assert meta.primary_speaker == "Swami ji"
    assert meta.parse_warnings == []
    assert meta.source_path is not None


def test_parse_example_path_no_base_dir():
    """Without base_dir, the parser still walks up from the file name."""
    meta = parse_path(EXAMPLE_PATH)
    assert meta.location == "NOIDA"
    assert meta.session_date == date(2010, 1, 7)
    assert meta.track_type == "discourse"


def test_parse_path_idempotent():
    """Parsing the same path twice produces equal metadata."""
    a = parse_path(EXAMPLE_PATH, base_dir=EXAMPLE_BASE)
    b = parse_path(EXAMPLE_PATH, base_dir=EXAMPLE_BASE)
    assert a.to_payload() == b.to_payload()


def test_to_payload_serializable():
    """All values in to_payload() must JSON-serialize cleanly."""
    import json

    meta = parse_path(EXAMPLE_PATH, base_dir=EXAMPLE_BASE)
    payload = meta.to_payload()
    # Should not raise.
    text = json.dumps(payload)
    rt = json.loads(text)
    assert rt["track_type"] == "discourse"
    assert rt["session_date"] == "2010-01-07"
    assert rt["session_time"] == "18:00:00"
    assert rt["season"] == "winter"


def test_header_fragment_includes_key_fields():
    meta = parse_path(EXAMPLE_PATH, base_dir=EXAMPLE_BASE)
    frag = meta.header_fragment()
    assert "NOIDA" in frag
    assert "2010-01-07" in frag
    assert "PRAVACHAN" in frag
    assert "discourse" in frag
    assert "winter" in frag


# ---- parse_path: bhajan / other track types ---------------------------


def test_parse_bhajan_track():
    p = (
        "/mnt/audio/Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/"
        "7 JAN - 1$ - 6 PM/05 NA HARA HAI ISHQ.wav"
    )
    meta = parse_path(p, base_dir="/mnt/audio")
    assert meta.track_no == 5
    assert meta.track_title == "NA HARA HAI ISHQ"
    assert meta.track_type == "bhajan"
    # Other levels still parse:
    assert meta.location == "NOIDA"
    assert meta.session_date == date(2010, 1, 7)


def test_parse_meditation_track():
    p = (
        "/mnt/audio/Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/"
        "7 JAN - 1$ - 6 PM/02 MEDITATION.wav"
    )
    meta = parse_path(p, base_dir="/mnt/audio")
    assert meta.track_type == "meditation"


def test_parse_om_guruve_namah_track():
    p = (
        "/mnt/audio/Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/"
        "7 JAN - 1$ - 6 PM/03 OM GURUVE NAMAH.wav"
    )
    meta = parse_path(p, base_dir="/mnt/audio")
    assert meta.track_type == "invocation"


# ---- parse_path: monsoon / "barsat" example ---------------------------


def test_parse_monsoon_session_for_barsat_query():
    """The trigger query was 'what swami ji said on a monsoon day about
    barsat'. Verify a July session gets season=monsoon so the date filter
    can find it."""
    p = (
        "/mnt/audio/Live Masters 2015/03 RISHIKESH 12 - 18 JUL 2015/"
        "14 JUL - 2$ - 9 AM/04 PRAVACHAN.wav"
    )
    meta = parse_path(p, base_dir="/mnt/audio")
    assert meta.session_date == date(2015, 7, 14)
    assert meta.season == "monsoon"
    assert meta.location == "RISHIKESH"
    assert meta.track_type == "discourse"
    assert meta.session_time == time(9, 0)


def test_parse_pm_meridiem_conversion():
    p = (
        "/mnt/audio/Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/"
        "7 JAN - 1$ - 6 PM/04 PRAVACHAN.wav"
    )
    meta = parse_path(p, base_dir="/mnt/audio")
    assert meta.session_time == time(18, 0)


def test_parse_am_meridiem_conversion():
    p = (
        "/mnt/audio/Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/"
        "8 JAN - 1$ - 6 AM/04 PRAVACHAN.wav"
    )
    meta = parse_path(p, base_dir="/mnt/audio")
    assert meta.session_time == time(6, 0)


def test_parse_noon_and_midnight_edge_cases():
    p_noon = (
        "/mnt/audio/Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/"
        "8 JAN - 1$ - 12 PM/04 PRAVACHAN.wav"
    )
    p_midnight = (
        "/mnt/audio/Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/"
        "8 JAN - 1$ - 12 AM/04 PRAVACHAN.wav"
    )
    assert parse_path(p_noon, base_dir="/mnt/audio").session_time == time(12, 0)
    assert parse_path(p_midnight, base_dir="/mnt/audio").session_time == time(0, 0)


# ---- parse_path: malformed inputs degrade gracefully ------------------


def test_parse_missing_session_level():
    """Path with only 3 levels — collection, event, track. Session missing."""
    p = (
        "/mnt/audio/Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/04 PRAVACHAN.wav"
    )
    meta = parse_path(p, base_dir="/mnt/audio")
    # The "event" position now holds NOIDA folder and the "session" position
    # holds what looks like an event-formatted string. That's expected
    # imperfection — parser fills in what it can, warnings for the rest.
    # The track parses cleanly either way:
    assert meta.track_title == "PRAVACHAN"
    assert meta.track_type == "discourse"
    # And we got at least one warning because at least one level didn't match.
    # (Don't over-constrain which one; the heuristic is rightmost-3.)
    assert meta.parse_warnings  # non-empty


def test_parse_completely_unstructured_path():
    """A path with no recognizable folder structure still returns a
    PathMetadata (never raises) with the track at minimum."""
    meta = parse_path("/random/junk/file_name_with_no_number.wav")
    # Track has no leading number — title falls back to the stem.
    assert meta.track_title == "file_name_with_no_number"
    # No date, no location, no season.
    assert meta.session_date is None
    assert meta.location is None
    assert meta.season is None
    # Default speaker still set.
    assert meta.primary_speaker == "Swami ji"
    # Warnings present.
    assert meta.parse_warnings


def test_parse_malformed_event_keeps_session_and_track():
    p = (
        "/mnt/audio/Live Masters 2010/JUNK FOLDER NAME/"
        "7 JAN - 1$ - 6 PM/04 PRAVACHAN.wav"
    )
    meta = parse_path(p, base_dir="/mnt/audio")
    assert meta.collection == "Live Masters"
    assert meta.year == 2010
    assert meta.event_id is None  # event failed
    assert meta.location is None
    # Session still parses, using collection year as fallback for the
    # session_date year:
    assert meta.session_date == date(2010, 1, 7)
    assert meta.session_time == time(18, 0)
    assert meta.track_type == "discourse"
    assert any("event" in w for w in meta.parse_warnings)


def test_parse_empty_path_returns_empty_meta_with_warning():
    meta = parse_path("")
    assert meta.parse_warnings
    assert meta.collection is None
    assert meta.track_title is None


def test_parse_path_never_raises_on_bad_input():
    """The contract: parse_path returns PathMetadata; it does not raise.
    This is the safety net that lets the ingester continue on weird inputs.
    """
    bad_inputs = [
        "/",
        "//",
        "single_file_no_dirs.wav",
        "/path/with/leading/slash/04 PRAVACHAN.wav",
        "no/extension/04 PRAVACHAN",
        "/mnt/x/2010/event/8 ABC - 1$ - 6 PM/04 PRAVACHAN.wav",  # bad month
        "/mnt/x/2010/event/32 JAN - 1$ - 6 PM/04 PRAVACHAN.wav",  # bad day
        "/mnt/x/2010/event/8 JAN - X$ - 6 PM/04 PRAVACHAN.wav",  # bad seq
        "/mnt/x/2010/event/8 JAN - 1$ - 25 PM/04 PRAVACHAN.wav",  # bad hour
    ]
    for p in bad_inputs:
        meta = parse_path(p)
        assert isinstance(meta, PathMetadata)
        # Always-true invariant: primary speaker is preserved.
        assert meta.primary_speaker == PRIMARY_SPEAKER


def test_parse_path_accepts_pathlib_inputs():
    posix_meta = parse_path(PurePosixPath(EXAMPLE_PATH), base_dir=EXAMPLE_BASE)
    assert posix_meta.location == "NOIDA"

    # Windows path equivalent. Use absolute Windows path so anchor handling
    # exercises the parts-stripping code.
    win = (
        r"D:\Audio Data\Live Masters 2010\01 NOIDA 7 - 10 JAN 2010"
        r"\7 JAN - 1$ - 6 PM\04 PRAVACHAN.wav"
    )
    win_meta = parse_path(PureWindowsPath(win), base_dir=r"D:\Audio Data")
    # PureWindowsPath on POSIX hosts may not fully resolve relative_to; we
    # only require that the parser returned non-trivial metadata.
    assert win_meta.track_title == "PRAVACHAN"
    assert win_meta.track_type == "discourse"


def test_parse_track_extension_stripped():
    meta = parse_path("/x/y/z/session/04 PRAVACHAN.mp3", base_dir="/x")
    assert meta.track_title == "PRAVACHAN"
    meta2 = parse_path("/x/y/z/session/04 PRAVACHAN.flac", base_dir="/x")
    assert meta2.track_title == "PRAVACHAN"
    # No extension also fine:
    meta3 = parse_path("/x/y/z/session/04 PRAVACHAN", base_dir="/x")
    assert meta3.track_title == "PRAVACHAN"


# ---- PRIMARY_SPEAKER as single source of truth ------------------------


def test_primary_speaker_is_module_constant():
    """Renaming the speaker requires a single-line change."""
    assert PRIMARY_SPEAKER == "Swami ji"


def test_primary_speaker_propagates_to_metadata():
    meta = parse_path(EXAMPLE_PATH, base_dir=EXAMPLE_BASE)
    assert meta.primary_speaker == PRIMARY_SPEAKER
    # Even on failure paths:
    meta_bad = parse_path("/nonsense.wav")
    assert meta_bad.primary_speaker == PRIMARY_SPEAKER
