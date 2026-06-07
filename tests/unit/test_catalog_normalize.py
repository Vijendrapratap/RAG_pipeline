"""Unit tests for ingestion.catalog.normalize.

The load-bearing test is `test_join_key_parity_with_path_parser`: it proves a
catalog row and the equivalent audio folder path collapse onto the same join
key. If that ever breaks, the spreadsheet stops syncing to the corpus.
"""
from __future__ import annotations

from datetime import date, time

import pytest

from ingestion.catalog.normalize import (
    clean_text,
    join_key_from_fields,
    join_key_from_path,
    normalize_location,
    parse_duration_to_seconds,
    parse_sitting,
    row_to_track,
    split_performers,
)


# --- clean_text ------------------------------------------------------------

def test_clean_text_strips_cr_token_and_nfc():
    raw = "Sangat Bhajan_x000D_\n_x000D_\nMeditation_x000D_"
    out = clean_text(raw)
    assert "_x000D_" not in out
    assert "Sangat Bhajan" in out and "Meditation" in out


@pytest.mark.parametrize("empty", [None, "", "   ", "nan", "NaN", "None"])
def test_clean_text_treats_blanklike_as_none(empty):
    assert clean_text(empty) is None


def test_clean_text_collapses_blank_runs():
    assert clean_text("a_x000D_\n_x000D_\n_x000D_\n_x000D_\nb") == "a\n\nb"


# --- parse_sitting ---------------------------------------------------------

def test_parse_sitting_full_form():
    p = parse_sitting("13 APR - 2$ - 10:30 AM", 2005)
    assert p.session_date == date(2005, 4, 13)
    assert p.session_seq == 2
    assert p.session_time == time(10, 30)
    assert p.warnings == []


def test_parse_sitting_four_letter_month_and_pm():
    p = parse_sitting("7 June - 6$ - 6:30 PM", 2005)
    assert p.session_date == date(2005, 6, 7)
    assert p.session_time == time(18, 30)


def test_parse_sitting_time_without_colon():
    p = parse_sitting("16 APR - 3$ - 630 AM", 1998)
    assert p.session_time == time(6, 30)


def test_parse_sitting_without_sequence():
    p = parse_sitting("16 Nov - 8 PM", 1995)
    assert p.session_date == date(1995, 11, 16)
    assert p.session_seq is None
    assert p.session_time == time(20, 0)


def test_parse_sitting_sequence_only_has_no_date():
    p = parse_sitting("1$", 2005)
    assert p.session_date is None
    assert p.session_seq == 1
    assert any("sequence-only" in w for w in p.warnings)


def test_parse_sitting_unparseable_warns_not_raises():
    p = parse_sitting("garbage cell", 2005)
    assert p.session_date is None
    assert any("unparseable" in w for w in p.warnings)


def test_parse_sitting_no_year_context():
    p = parse_sitting("13 APR - 2$ - 10:30 AM", None)
    assert p.session_date is None
    assert p.session_seq == 2  # still recovers seq + time
    assert p.session_time == time(10, 30)


# --- parse_duration_to_seconds --------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (23.42, 23 * 60 + 42),
    (5.47, 5 * 60 + 47),
    (0.59, 59),
    (30.3, 30 * 60 + 30),   # 30.3 == 30:30, not 30:03
    (52.49, 52 * 60 + 49),
    (0.07, 7),
])
def test_parse_duration(raw, expected):
    assert parse_duration_to_seconds(raw) == expected


@pytest.mark.parametrize("bad", [None, "abc", -1, 5.61])  # 5.61 -> 61s invalid
def test_parse_duration_bad_returns_none(bad):
    assert parse_duration_to_seconds(bad) is None


# --- split_performers ------------------------------------------------------

def test_split_performers_basic():
    assert split_performers("ABHIPSA, SUMAN, PRACHI") == ["Abhipsa", "Suman", "Prachi"]


def test_split_performers_dedupes_honorific_variants():
    # "Suman Ji" and "SUMAN" should collapse to one.
    assert split_performers("Suman Ji, SUMAN, Neelam Richa Ji") == ["Suman", "Neelam Richa"]


def test_split_performers_empty():
    assert split_performers(None) == []


# --- join keys -------------------------------------------------------------

def test_join_key_from_fields_canonical():
    k = join_key_from_fields(date(2005, 4, 13), 2, 4)
    assert k == "2005-04-13|2|4"


def test_join_key_none_when_no_date():
    # Date-anchored: without a session_date a row cannot be aligned to a folder.
    assert join_key_from_fields(None, 1, 4) is None


def test_join_key_is_location_independent():
    """The whole point of date-anchoring: a folder neighbourhood
    ("PITAMPURA DELHI") and the catalog city ("Delhi") must still produce the
    same key for the same date+seq+track."""
    folder = (
        r"Live Masters 2010\02 PITAMPURA DELHI 12 - 14 JAN 2010"
        r"\12 JAN - 1$ - 6 PM\04 PRAVACHAN.json"
    )
    catalog = row_to_track({
        "CampYear": 2010, "CampPlace": "Delhi",
        "Sitting": "12 JAN - 1$ - 6 PM", "TrackNo": 4, "Content": "PRAVACHAN",
    })
    assert catalog.join_key == join_key_from_path(folder) == "2010-01-12|1|4"


def test_join_key_parity_with_path_parser():
    """A catalog row and the equivalent audio folder path MUST produce the
    same join key — this is what makes the spreadsheet sync to the corpus."""
    row = {
        "CampYear": 2010, "CampPlace": "NOIDA",
        "Sitting": "7 JAN - 1$ - 6 PM", "TrackNo": 4, "Content": "PRAVACHAN",
    }
    track = row_to_track(row)
    path = (
        r"Live Masters 2010\01 NOIDA 7 - 10 JAN 2010"
        r"\7 JAN - 1$ - 6 PM\04 PRAVACHAN.json"
    )
    assert track.join_key == join_key_from_path(path)
    assert track.join_key == "2010-01-07|1|4"


@pytest.mark.parametrize("raw,expected", [
    ("Noida", "NOIDA"),
    ("NOIDA", "NOIDA"),
    ("PITAMPURA DELHI", "DELHI"),        # neighbourhood folds to metro
    ("Chhattarpur, Delhi", "DELHI"),     # comma form -> city
    ("Dwarka, Delhi", "DELHI"),
    ("Panchkula", "PANCHKULA"),
    (None, None),
])
def test_normalize_location(raw, expected):
    assert normalize_location(raw) == expected


# --- row_to_track end-to-end ----------------------------------------------

def test_row_to_track_full():
    row = {
        "CampDate": "12 - 13 Apr", "CampYear": 2005, "CampPlace": "Panchkula",
        "Sitting": "13 APR - 2$ - 10:30 AM", "Venue": None,
        "Chorus": "ABHIPSA, SUMAN, PRACHI",
        "Comment": "Released in Vishvas Vibrations Vol - 245",
        "DetailContents": "YELLOW ROBE CELEBRATION_x000D_\nMeditation",
        "TrackNo": 4.0, "Content": "हे मेरे गुरुदेव करुणा", "Duration": 23.42,
    }
    t = row_to_track(row)
    assert t.location == "PANCHKULA"
    assert t.session_date == date(2005, 4, 13)
    assert t.season == "summer"           # April -> summer (IMD mapping)
    assert t.track_no == 4
    assert t.track_title == "हे मेरे गुरुदेव करुणा"
    assert t.track_type == "bhajan"       # Devanagari title -> default
    assert t.duration_sec == 23 * 60 + 42
    assert t.performers == ["Abhipsa", "Suman", "Prachi"]
    assert t.release_ref.startswith("Released in Vishvas")
    assert t.detail_contents is not None and "_x000D_" not in t.detail_contents
    assert t.year_reliable is True
    assert t.join_key == "2005-04-13|2|4"


def test_row_to_track_unknown_year_sentinel():
    row = {"CampYear": 1900, "CampPlace": "Hisar",
           "Sitting": "16 Nov - 8 PM", "TrackNo": 1, "Content": "meditation"}
    t = row_to_track(row)
    assert t.year_reliable is False
    assert t.session_date is None         # year not trusted -> no date
    assert any("1900 sentinel" in w for w in t.warnings)


def test_row_to_track_discourse_type():
    row = {"CampYear": 2010, "CampPlace": "NOIDA",
           "Sitting": "7 JAN - 1$ - 6 PM", "TrackNo": 4, "Content": "PRAVACHAN"}
    assert row_to_track(row).track_type == "discourse"
