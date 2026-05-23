"""Unit tests for the pure helpers in ingestion.build_summary_index."""
from __future__ import annotations

import datetime

from ingestion.build_summary_index import build_payload, build_summary_text


def test_summary_text_combines_both_languages_and_tags():
    text = build_summary_text({
        "summary_english": "A discourse on karma yoga.",
        "summary_hindi": "कर्म योग पर प्रवचन।",
        "topics": ["karma-yoga", "self-inquiry"],
        "scriptures_referenced": ["Bhagavad Gita"],
        "people_named": ["Arjuna"],
    })
    assert "A discourse on karma yoga." in text
    assert "कर्म योग पर प्रवचन।" in text
    assert "Topics: karma-yoga, self-inquiry" in text
    assert "Scriptures: Bhagavad Gita" in text
    assert "People: Arjuna" in text


def test_summary_text_handles_only_hindi():
    text = build_summary_text({"summary_hindi": "केवल हिंदी सारांश।"})
    assert text == "केवल हिंदी सारांश।"


def test_summary_text_empty_row_is_empty_string():
    assert build_summary_text({}) == ""
    assert build_summary_text({"topics": []}) == ""


def test_build_payload_drops_none_and_empty():
    payload = build_payload({
        "source_file": "f.json",
        "summary_english": "x",
        "summary_hindi": None,
        "topics": [],
        "people_named": ["A"],
        "event_type": None,
    })
    assert payload["source_file"] == "f.json"
    assert payload["summary_english"] == "x"
    assert "summary_hindi" not in payload
    assert "topics" not in payload          # empty list dropped
    assert payload["people_named"] == ["A"]
    assert "event_type" not in payload      # None dropped


def test_build_payload_serialises_session_date():
    payload = build_payload({
        "source_file": "f.json",
        "session_date": datetime.date(2010, 1, 7),
    })
    assert payload["session_date"] == "2010-01-07"
