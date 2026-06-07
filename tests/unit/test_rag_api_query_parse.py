"""Unit tests for rag_api.query_parse — deterministic filter extraction.

All pure: no Postgres / Qdrant / Ollama needed.
"""
from __future__ import annotations

from rag_api.query_parse import detect_signals, merge_filters, signals_to_filters

# Shape matches rag_api.db.get_filter_options output.
_VOCAB = {
    "seasons": ["monsoon", "winter"],
    "locations": ["NOIDA", "RISHIKESH"],
    "event_ids": ["shivir-2015"],
    "track_types": ["discourse", "qa"],
    "event_types": ["satsang"],
    "primary_languages": ["hindi", "english"],
    "topics": ["karma-yoga", "self-inquiry"],
    "scriptures_referenced": ["Bhagavad Gita"],
    "performers": ["Abhipsa", "Suman"],
}


def test_performer_name_detected_as_soft_list_signal():
    sig = [s for s in detect_signals("bhajans sung by Abhipsa", _VOCAB)
           if s["field"] == "performers"]
    assert len(sig) == 1
    assert sig[0]["value"] == "Abhipsa"
    assert sig[0]["confidence"] == "soft"
    # List-valued: include_soft accumulates into a list filter.
    filters = signals_to_filters(sig, include_soft=True)
    assert filters["performers"] == ["Abhipsa"]


# ---- date signals (strong) -----------------------------------------------


def test_bare_year_becomes_strong_date_range():
    dates = [s for s in detect_signals("discourses from 2015 on dharma")
             if s["field"] == "date_range"]
    assert len(dates) == 1
    assert dates[0]["confidence"] == "strong"
    assert dates[0]["value"] == ["2015-01-01", "2015-12-31"]


def test_multiple_years_span_a_range():
    d = [s for s in detect_signals("between 2010 and 2014")
         if s["field"] == "date_range"][0]
    assert d["value"] == ["2010-01-01", "2014-12-31"]


def test_iso_date_detected_and_year_not_double_counted():
    dates = [s for s in detect_signals("the satsang on 2015-08-09")
             if s["field"] == "date_range"]
    assert len(dates) == 1
    assert dates[0]["value"] == ["2015-08-09", "2015-08-09"]


def test_invalid_iso_date_yields_no_date_signal():
    # 2015-13-40 is not a real date; the regex still spans it, so the 2015
    # inside is blanked and not read as a bare year either.
    assert [s for s in detect_signals("nonsense 2015-13-40 here")
            if s["field"] == "date_range"] == []


def test_five_digit_number_is_not_a_year():
    assert [s for s in detect_signals("track 20155 of the set")
            if s["field"] == "date_range"] == []


# ---- vocab signals (soft) ------------------------------------------------


def test_season_word_is_soft():
    seasons = [s for s in detect_signals("what about monsoon discourses", _VOCAB)
               if s["field"] == "season"]
    assert len(seasons) == 1
    assert seasons[0]["value"] == "monsoon"
    assert seasons[0]["confidence"] == "soft"


def test_kebab_topic_matches_spaced_query():
    topics = [s for s in detect_signals("teachings on karma yoga", _VOCAB)
              if s["field"] == "topics"]
    assert topics[0]["value"] == "karma-yoga"


def test_multiword_scripture_matches():
    sc = [s for s in detect_signals("what does the Bhagavad Gita say", _VOCAB)
          if s["field"] == "scriptures_referenced"]
    assert sc[0]["value"] == "Bhagavad Gita"


def test_location_match_is_case_insensitive():
    locs = [s for s in detect_signals("the noida shivir", _VOCAB)
            if s["field"] == "location"]
    assert locs[0]["value"] == "NOIDA"


def test_short_vocab_value_is_skipped():
    # 'qa' is 2 chars — below the minimum, never matched.
    assert [s for s in detect_signals("a qa session", _VOCAB)
            if s["field"] == "track_type"] == []


def test_whole_word_matching_only():
    # 'winter' must not match inside 'wintergreen'.
    assert [s for s in detect_signals("wintergreen oil", _VOCAB)
            if s["field"] == "season"] == []


def test_no_vocab_still_detects_dates_but_no_soft():
    d = detect_signals("karma yoga in 2015")
    assert any(s["field"] == "date_range" for s in d)
    assert all(s["field"] == "date_range" for s in d)


# ---- signals_to_filters --------------------------------------------------


def test_strong_only_by_default():
    f = signals_to_filters(detect_signals("monsoon discourses 2015", _VOCAB))
    assert f == {"date_range": ("2015-01-01", "2015-12-31")}


def test_include_soft_adds_vocab_filters():
    f = signals_to_filters(
        detect_signals("monsoon discourses 2015", _VOCAB), include_soft=True
    )
    assert f["season"] == "monsoon"
    assert f["date_range"] == ("2015-01-01", "2015-12-31")


def test_list_fields_accumulate():
    f = signals_to_filters(
        detect_signals("karma yoga and self inquiry", _VOCAB), include_soft=True
    )
    assert set(f["topics"]) == {"karma-yoga", "self-inquiry"}


# ---- merge_filters -------------------------------------------------------


def test_merge_explicit_wins_over_auto():
    merged = merge_filters(
        {"season": "winter"}, {"season": "monsoon", "location": "NOIDA"}
    )
    assert merged == {"season": "winter", "location": "NOIDA"}


def test_merge_drops_none_valued_explicit_keys():
    merged = merge_filters(
        {"season": None, "topics": None}, {"season": "monsoon"}
    )
    assert merged == {"season": "monsoon"}


def test_merge_handles_none_inputs():
    assert merge_filters(None, None) == {}
