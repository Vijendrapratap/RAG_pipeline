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
    month_num,
    parse_path,
    primary_speaker_for,
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


# ---- Phase 17: head-anchored track_type + qa/session/combined -----------
#
# Head-anchoring recovers 1,050 of 9,335 tracks the old exact match dropped to
# 'bhajan' (typed 4,212 -> 5,262), reclassifying zero songs.


@pytest.mark.parametrize(
    "title, expected",
    [
        # A vocab word at the head, followed by "(", "-", a number, or end.
        ("MEDITATION (WITHOUT OM)", "meditation"),   # 258 tracks
        ("MEDITATION (INCOMPLETE)", "meditation"),
        ("MEDITATION - ONLY OM", "meditation"),
        ("MEDITATION (1)", "meditation"),
        ("OM GURUVE NAMAH (INCOMPLETE)", "invocation"),
        ("SAMBODHAN - RISHI JI", "address"),         # 67 tracks
        ("SAMBODHAN (2)", "address"),
        ("PRAVACHAN (ZINDAGI KA SAFAR)", "discourse"),
        ("PRAVACHAN - THIRD STAGE", "discourse"),
    ],
)
def test_track_type_head_anchored_recovers_variants(title, expected):
    assert track_type_for(title) == expected


@pytest.mark.parametrize(
    "title",
    [
        # A vocab word mid-title is part of a song lyric, not the type.
        "MEDITATION KI MASTIYON MEIN",   # 13 tracks — a song
        "MERI SHAM MEDITATION MEIN",     # 12 tracks — MEDITATION is mid-title
        "STANDING MEDITATION",           # 10 tracks
        "DHYAN MEIN UTRO",
    ],
)
def test_track_type_does_not_reclassify_songs(title):
    """The false-positive gate: a real bhajan whose title merely contains a vocab
    word must stay 'bhajan'. This is why the match is head-anchored, not substring."""
    assert track_type_for(title) == "bhajan"


@pytest.mark.parametrize(
    "title, expected",
    [
        ("QUESTION 1", "qa"),
        ("QUES-01", "qa"),
        ("QUES - 3", "qa"),
        ("QUESTION 14 & CHAL MERE DIL", "qa"),   # QUES prefix wins over the song tail
        ("COMPLETE SITTING", "session"),
        ("COMPLETE SITTING (SONY RECORDER)", "session"),
        ("WELCOME SITTING (ONLY VOICE)", "session"),
    ],
)
def test_track_type_qa_and_session_carveouts(title, expected):
    assert track_type_for(title) == expected


@pytest.mark.parametrize(
    "title",
    [
        "SAMBODHAN & PRAVACHAN",       # 35 — address + discourse
        "PRAVACHAN & SAMBODHAN",       # 33 — order must not matter
        "MEDITATION & PRAVACHAN",      # 9
        "SAMBODHAN&PRAVACHAN",         # no spaces, still two types
    ],
)
def test_track_type_combined_for_multi_type_titles(title):
    """A title naming two distinct activity TYPES is 'combined' — a single-valued
    field cannot say it is both an address and a discourse."""
    assert track_type_for(title) == "combined"


def test_track_type_two_songs_and_one_type_is_not_combined():
    """`RUHANI GEET & SAMBODHAN` names a song and an address; only one vocab TYPE
    is present, so it is not 'combined' — it stays bhajan (the head is a song)."""
    assert track_type_for("RUHANI GEET & SAMBODHAN") == "bhajan"


def test_track_types_frozenset_covers_every_output():
    """path_parser.TRACK_TYPES is the validation vocabulary; it must contain every
    value track_type_for can emit and nothing it cannot."""
    from ingestion.utils.path_parser import TRACK_TYPES
    for v in ("bhajan", "discourse", "address", "meditation", "invocation",
              "music", "qa", "session", "combined"):
        assert v in TRACK_TYPES
    assert len(TRACK_TYPES) == 9


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


# ---- Real-world: user's vocal-isolation + whisper folder layout -------
#
# Upstream (D:\GuruAudio) appends "_isolation" to every level and inserts an
# extra "<NN> <TITLE>_model-..." folder between the session and the actual
# transcript file. The parser must tolerate both without losing metadata.


_USER_BASE = "/mnt/d/GuruAudio/Output Transcribe"
# Real 6-level path observed in the user's tree: whisper drops files inside
# a `turbo/` (inference-size) folder under the `_model-...` folder.
_USER_PATH = (
    f"{_USER_BASE}/Live Masters 2010_isolation/"
    "01 NOIDA 7 - 10 JAN 2010_isolation/"
    "7 JAN - 1$ - 6 PM_isolation/"
    "04 PRAVACHAN_model-1_mel_roformer_kim_ft/"
    "turbo/"
    "04 PRAVACHAN.json"
)


def test_parse_user_real_path_full_metadata():
    """The user's actual folder layout (with `_isolation` suffixes and
    an extra `_model-...` folder above the file) must yield full metadata,
    identical to the canonical PRD layout."""
    meta = parse_path(_USER_PATH, base_dir=_USER_BASE)
    assert meta.collection == "Live Masters"
    assert meta.year == 2010
    assert meta.event_id == "01 NOIDA 7 - 10 JAN 2010"  # suffix stripped
    assert meta.event_seq == 1
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
    assert meta.parse_warnings == []


def test_parse_user_real_path_no_base_dir():
    """Without base_dir the parser still walks rightmost levels and
    recovers the same fields."""
    meta = parse_path(_USER_PATH)
    assert meta.location == "NOIDA"
    assert meta.session_date == date(2010, 1, 7)
    assert meta.track_type == "discourse"


def test_isolation_suffix_stripped_from_event_id():
    """event_id is the cleaned (suffix-stripped) folder name, not the raw
    one — downstream filter UIs surface this string verbatim."""
    meta = parse_path(_USER_PATH, base_dir=_USER_BASE)
    assert "_isolation" not in (meta.event_id or "")


def test_model_folder_alone_above_track_still_resolves_track():
    """Stray top-level model folder (e.g.
    "03 PRAVACHAN IN MEDITATION_model-..." sitting directly under the
    Output Transcribe root) drops out cleanly — the file's track parses,
    higher levels gracefully warn."""
    p = (
        f"{_USER_BASE}/03 PRAVACHAN IN MEDITATION_model-1_mel_roformer_kim_ft/"
        "turbo/03 PRAVACHAN IN MEDITATION.json"
    )
    meta = parse_path(p, base_dir=_USER_BASE)
    # Track still parses (the file itself):
    assert meta.track_title == "PRAVACHAN IN MEDITATION"
    assert meta.track_no == 3
    # "PRAVACHAN IN MEDITATION" names two distinct vocab types (discourse +
    # meditation), so Phase 17 classifies it 'combined' — it is genuinely both.
    assert meta.track_type == "combined"
    # No collection/event/session info recoverable — that's acceptable:
    assert meta.collection is None
    assert meta.event_id is None
    assert meta.session_date is None


def test_parse_hhmm_time_format_1030am():
    """Session folder written as '1030 AM' (no colon) must parse as 10:30."""
    p = (
        f"{_USER_BASE}/Live Masters 2010_isolation/"
        "01 NOIDA 7 - 10 JAN 2010_isolation/"
        "10 JAN - 6$ - 1030 AM_isolation/"
        "06 SAMBODHAN_model-1_mel_roformer_kim_ft/turbo/06 SAMBODHAN.json"
    )
    meta = parse_path(p, base_dir=_USER_BASE)
    assert meta.session_date == date(2010, 1, 10)
    assert meta.session_time == time(10, 30)
    assert meta.session_seq == 6
    assert meta.parse_warnings == []


def test_parse_hhmm_time_format_930am():
    """3-digit HHMM: '930 AM' → 9:30 AM."""
    p = (
        f"{_USER_BASE}/Live Masters 2010_isolation/"
        "01 NOIDA 7 - 10 JAN 2010_isolation/"
        "8 JAN - 3$ - 930 AM_isolation/"
        "04 PRAVACHAN_model-1/turbo/04 PRAVACHAN.json"
    )
    meta = parse_path(p, base_dir=_USER_BASE)
    assert meta.session_time == time(9, 30)
    assert meta.parse_warnings == []


def test_only_isolation_suffix_still_parses():
    """If only the `_isolation` suffix appears (no model folder), each
    level should still parse."""
    p = (
        f"{_USER_BASE}/Live Masters 2010_isolation/"
        "01 NOIDA 7 - 10 JAN 2010_isolation/"
        "7 JAN - 1$ - 6 PM_isolation/"
        "04 PRAVACHAN.txt"
    )
    meta = parse_path(p, base_dir=_USER_BASE)
    assert meta.year == 2010
    assert meta.location == "NOIDA"
    assert meta.session_time == time(18, 0)
    assert meta.track_title == "PRAVACHAN"
    assert meta.parse_warnings == []


def test_model_folder_dropped_does_not_break_3level_path():
    """Path missing the collection level (only event/session/model-folder/
    file) still recovers event + session + track."""
    p = (
        f"{_USER_BASE}/01 NOIDA 7 - 10 JAN 2010_isolation/"
        "7 JAN - 1$ - 6 PM_isolation/"
        "04 PRAVACHAN_model-1_mel_roformer_kim_ft/"
        "04 PRAVACHAN.json"
    )
    meta = parse_path(p, base_dir=_USER_BASE)
    assert meta.event_id == "01 NOIDA 7 - 10 JAN 2010"
    assert meta.session_date == date(2010, 1, 7)
    assert meta.track_title == "PRAVACHAN"


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


# ---- Phase 17: Rishi ji is a genuine second speaker -------------------


@pytest.mark.parametrize(
    "title, expected",
    [
        ("SAMBODHAN - RISHI JI", "Rishi ji"),           # 67 tracks — his
        ("SAMBODHAN (RISHI JI)", "Rishi ji"),
        ("SAMBODHAN BY RISHI JI", "Rishi ji"),
        ("SAMBODHAN - RISHI JI (INCOMPLETE)", "Rishi ji"),
        # Dual-speaker: Swami ji is genuinely present, field holds one value.
        ("SAMBODHAN - SWAMI JI - RISHI JI", "Swami ji"),  # 18 tracks
        ("SAMBODHAN - SWAMI JI & RISHI JI", "Swami ji"),  # 18 tracks
        # No Rishi ji named -> the default.
        ("SAMBODHAN", "Swami ji"),
        ("PRAVACHAN", "Swami ji"),
        ("MEDITATION (WITHOUT OM)", "Swami ji"),
    ],
)
def test_primary_speaker_for(title, expected):
    assert primary_speaker_for(title) == expected


def test_rishi_speaker_reaches_the_metadata_object():
    """A SAMBODHAN - RISHI JI track must carry primary_speaker='Rishi ji' so the
    speaker facet (chunker_json falls back to it) returns his tracks."""
    p = (f"{_USER_BASE}/Live Masters 2010_isolation/01 NOIDA 7 - 10 JAN 2010_isolation/"
         "7 JAN - 1$ - 6 PM_isolation/06 SAMBODHAN - RISHI JI.json")
    meta = parse_path(p, base_dir=_USER_BASE)
    assert meta.track_title == "SAMBODHAN - RISHI JI"
    assert meta.primary_speaker == "Rishi ji"


# ---- Real corpus shapes: levels identified by shape, not position -----
#
# Every folder name below was taken verbatim from
# `D:\Transcription whisperx\Output`. The position-based parser these tests
# replaced resolved session_date on 35.3% of distinct directory chains; this
# suite pins the shapes that took it to 98.8% of files.

_D = "/mnt/d/Transcription whisperx/Output"


def _sd(chain: str, track: str = "04 PRAVACHAN.raw.json"):
    return parse_path(f"{_D}/{chain}/{track}", base_dir=_D)


def test_dagshai_month_bucket_is_not_an_event():
    """The Dagshai tree groups sittings under a calendar-month folder that
    carries a year but no location and no day range. Reading it as the event
    level (position -2) dropped the session year for 11,350 chunks."""
    meta = _sd("Dagshai 2001_isolation/01 JAN - 2001_isolation/10 JAN - 6 PM_isolation")
    assert meta.collection == "Dagshai"
    assert meta.year == 2001
    assert meta.event_id is None          # month bucket is NOT an event
    assert meta.location is None
    assert meta.session_date == date(2001, 1, 10)
    assert meta.session_time == time(18, 0)


def test_dagshai_five_level_path_with_media_container():
    """A media-container folder ('MD RECORDING OF THIS CAMP') sits between the
    event and the session. It carries no metadata and must not shift levels."""
    meta = _sd(
        "Dagshai 2001_isolation/01 JAN - 2001_isolation/"
        "DAGSHAI 15 - 19 JAN 2001_isolation/MD RECORDING OF THIS CAMP_isolation/"
        "16 JAN - 1$ - 8 AM_isolation"
    )
    assert meta.event_id == "DAGSHAI 15 - 19 JAN 2001"
    assert meta.location == "DAGSHAI"
    assert meta.event_seq is None         # this event has no leading sequence
    assert meta.session_date == date(2001, 1, 16)
    assert meta.session_seq == 1
    assert meta.session_time == time(8, 0)


@pytest.mark.parametrize(
    "folder, expected_date, expected_time, expected_seq",
    [
        # Four-letter months: "JUNE"/"JULY"/"APRIL" against the old [A-Z]{3}.
        ("12 JUNE - 3$ - 6 PM", date(2011, 6, 12), time(18, 0), 3),
        ("08 APRIL", date(2011, 4, 8), None, None),
        # No sequence marker at all.
        ("10 JAN - 6 PM", date(2011, 1, 10), time(18, 0), None),
        # No dash between date and time.
        ("8 AUG 7 PM", date(2011, 8, 8), time(19, 0), None),
        ("2 JUNE 630 PM", date(2011, 6, 2), time(18, 30), None),
        # No spaces around the dashes; NOON as the clock.
        ("25 MAR-2$-12 NOON", date(2011, 3, 25), time(12, 0), 2),
        ("14 AUG 2$ 12 NOON", date(2011, 8, 14), time(12, 0), 2),
        # Trailing parentheticals and free-text notes after the time.
        ("13 FEB - 6 PM (CASS REC)", date(2011, 2, 13), time(18, 0), None),
        ("14 NOV - 530 PM (DEEPAWALI)", date(2011, 11, 14), time(17, 30), None),
        ("18 MAY 1 PM - SCHOOL CHILDREN", date(2011, 5, 18), time(13, 0), None),
        ("7 SEP 1230 PM ARMY PUBLIC SCHOOL", date(2011, 9, 7), time(12, 30), None),
        # Named sittings: dated, but with no clock time.
        ("28 NOV - WELCOME - 8 PM", date(2011, 11, 28), time(20, 0), None),
        ("11 APR - WELCOME - EVENING", date(2011, 4, 11), None, None),
        ("30 JUNE - NO TIME", date(2011, 6, 30), None, None),
        ("26 JAN MORNING", date(2011, 1, 26), None, None),
        ("2 JULY GURUPURNIMA", date(2011, 7, 2), None, None),
        # A bare '$' with no digits is not a sequence number.
        ("13 JAN - LOHRI CELEBRATION $", date(2011, 1, 13), None, None),
        ("8 JUN 7$ INDIVIDUAL MEDITATION ON 26 NO. HILLS",
         date(2011, 6, 8), None, 7),
    ],
)
def test_real_session_folder_shapes(folder, expected_date, expected_time, expected_seq):
    meta = _sd(f"Dagshai 2011_isolation/{folder}_isolation")
    assert meta.session_date == expected_date
    assert meta.session_time == expected_time
    assert meta.session_seq == expected_seq


@pytest.mark.parametrize(
    "folder, seq, loc, start, end",
    [
        ("01 NOIDA 7 - 10 JAN 2010", 1, "NOIDA", date(2010, 1, 7), date(2010, 1, 10)),
        # No leading sequence number.
        ("DAGSHAI 15 - 19 JAN 2001", None, "DAGSHAI",
         date(2001, 1, 15), date(2001, 1, 19)),
        # Dotted / multi-word location.
        ("05 MCF AUDI. FARIDABAD 22 - 25 JAN 2010", 5, "MCF AUDI. FARIDABAD",
         date(2010, 1, 22), date(2010, 1, 25)),
        # Month on both ends of the range.
        ("08 BATHINDA 30 NOV - 2 DEC 2013", 8, "BATHINDA",
         date(2013, 11, 30), date(2013, 12, 2)),
        # Trailing camp label, and no year at all (inherited from collection).
        ("DAGSHAI 10 - 13 JUNE CHILDREN CAMP", None, "DAGSHAI",
         date(2010, 6, 10), date(2010, 6, 13)),
        ("DAGSHAI 10 - 13 MAR 2002 SHIVRATRI CAMP", None, "DAGSHAI",
         date(2002, 3, 10), date(2002, 3, 13)),
    ],
)
def test_real_event_folder_shapes(folder, seq, loc, start, end):
    meta = _sd(f"Live Masters 2010_isolation/{folder}_isolation/7 JAN - 1$ - 6 PM_isolation")
    assert meta.event_seq == seq
    assert meta.location == loc
    assert meta.event_start == start
    assert meta.event_end == end


# ---- Cross-year camps: the catalog join-key collision ------------------


def test_new_year_camp_session_dated_from_the_span_not_the_start_year():
    """'30 DEC 2014 - 1 JAN 2015' spans two years. Taking event_start.year
    unconditionally dated the 1 JAN sitting to 2014-01-01 — one year early,
    and identical to the join_key of a genuine 1 JAN 2014 camp, so the catalog
    backfill wrote Siri Fort's performers onto Chhattarpur's chunks."""
    chain = ("Live Masters 2014_isolation/"
             "17 CHHATTARPUR DELHI 30 DEC 2014 - 1 JAN 2015_isolation")
    assert _sd(f"{chain}/1 JAN - 4$ - 7 PM_isolation").session_date == date(2015, 1, 1)
    assert _sd(f"{chain}/30 DEC - 1$ - 7 PM_isolation").session_date == date(2014, 12, 30)
    # A sitting outside the span (arrival day) keeps the start year.
    assert _sd(f"{chain}/29 DEC - 6 PM_isolation").session_date == date(2014, 12, 29)


def test_new_year_camp_with_year_written_once_backs_off_the_start_year():
    """'30 DEC - 1 JAN 2012' writes the year only at the end. Inheriting it for
    both ends yields start=2012-12-30 > end=2012-01-01."""
    meta = _sd("Live Masters 2011_isolation/17 NEW YEAR DELHI 30 DEC - 1 JAN 2012_isolation/"
               "30 DEC - 1$ - 7 PM_isolation")
    assert meta.event_start == date(2011, 12, 30)
    assert meta.event_end == date(2012, 1, 1)
    assert meta.session_date == date(2011, 12, 30)
    assert meta.parse_warnings == []


def test_session_outside_a_normal_span_keeps_the_event_year():
    meta = _sd("Live Masters 2010_isolation/23 JIND 19 - 21 NOV 2010_isolation/"
               "22 NOV - 6 PM_isolation")
    assert meta.session_date == date(2010, 11, 22)


# ---- Unrecognized folders contribute nothing (and say so) --------------


@pytest.mark.parametrize(
    "folder",
    ["MD RECORDING", "MD-42 (This MD is not playing properly)", "MISC DAGSHAI RECORDING",
     "CASS RECORDING", "UNKNOWN RECORDING FROM CASSETTE", "New folder",
     "EVENING $", "MORNING $", "UNKNOWN DATE", "No Date No Time",
     "export from cassette recording"],
)
def test_unrecognized_folders_are_skipped_not_misread(folder):
    """These carry no date. The contract is that they contribute nothing —
    never a wrong session_date — and that the omission is warned about."""
    meta = _sd(f"Dagshai 2003_isolation/06 JUNE - 2003_isolation/{folder}_isolation")
    assert meta.session_date is None
    assert meta.session_time is None
    assert meta.collection == "Dagshai"      # the levels that DO parse survive
    assert meta.year == 2003
    assert meta.parse_warnings


def test_noon_is_not_a_month():
    """`[A-Za-z]{3,}` matches 'NOON'; month_num must reject it so the session
    grammar does not read '12 NOON' as a date."""
    assert month_num("NOON") is None
    assert month_num("JUNE") == 6
    assert month_num("APRIL") == 4
    assert month_num("Sept") == 9
    assert month_num("no") is None
