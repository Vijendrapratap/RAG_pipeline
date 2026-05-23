"""Unit tests for Phase 13 content-tag helpers.

Covers the pure-function module (tag_schema.py) and the small pure parts
of enrich_content_tags.py. The Ollama/Postgres/Qdrant integration is out
of scope for unit tests — the smoke/integration story lives in the
manual end-to-end path described in PRD §6 Phase 13 acceptance criteria.
"""
from __future__ import annotations

import json

import pytest

from ingestion.utils.tag_schema import (
    ALLOWED_EVENT_TYPES,
    ALLOWED_LANGUAGES,
    REQUIRED_KEYS,
    build_mapreduce_chunk_prompt,
    build_prompt,
    parse_model_json,
    validate_tags,
)


# ---- build_prompt -------------------------------------------------------


def test_build_prompt_includes_all_schema_keys() -> None:
    prompt = build_prompt("hello world")
    for k in REQUIRED_KEYS:
        assert f'"{k}"' in prompt, f"prompt missing key: {k}"


def test_build_prompt_appends_transcript_last() -> None:
    transcript = "UNIQUE_MARKER_12345"
    prompt = build_prompt(transcript)
    # Transcript must appear AFTER the schema and rules so it stays in the
    # model's most-recent attention window for long inputs.
    schema_pos = prompt.find("event_type")
    transcript_pos = prompt.find(transcript)
    assert schema_pos > 0
    assert transcript_pos > schema_pos


def test_build_prompt_demands_devanagari_for_hindi_summary() -> None:
    prompt = build_prompt("x")
    assert "Devanagari" in prompt
    assert "WHOLE transcript" in prompt


def test_build_prompt_demands_verbatim_clues() -> None:
    prompt = build_prompt("x")
    assert "verbatim" in prompt.lower()


def test_build_mapreduce_chunk_prompt_returns_json_schema() -> None:
    p = build_mapreduce_chunk_prompt("fragment text")
    assert '"summary"' in p
    assert '"people"' in p
    assert '"places"' in p
    assert '"scriptures"' in p
    assert "fragment text" in p


# ---- validate_tags ------------------------------------------------------


def _valid_tags() -> dict:
    return {
        "event_type": "discourse",
        "primary_language": "hindi",
        "topics": ["karma-yoga", "self-inquiry", "guru-bhakti"],
        "people_named": ["Anush"],
        "places_named": ["Noida"],
        "scriptures_referenced": ["Bhagavad Gita"],
        "timing_clues": ["आज सुबह"],
        "location_clues": ["यह नोएडा शिविर है"],
        "summary_hindi": "गुरु जी ने कर्म योग पर प्रवचन दिया।",
        "summary_english": "Guruji gave a discourse on karma yoga.",
    }


def test_validate_accepts_well_formed_object() -> None:
    ok, errs = validate_tags(_valid_tags())
    assert ok, errs
    assert errs == []


def test_validate_rejects_non_dict() -> None:
    ok, errs = validate_tags(["not", "a", "dict"])
    assert not ok
    assert any("not a dict" in e for e in errs)


def test_validate_rejects_missing_keys() -> None:
    tags = _valid_tags()
    del tags["summary_hindi"]
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("missing required keys" in e for e in errs)
    assert any("summary_hindi" in e for e in errs)


def test_validate_rejects_extra_keys() -> None:
    tags = _valid_tags()
    tags["unexpected"] = "value"
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("unexpected keys" in e for e in errs)


def test_validate_rejects_unknown_event_type() -> None:
    tags = _valid_tags()
    tags["event_type"] = "lecture"  # not in enum
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("event_type" in e and "lecture" in e for e in errs)


def test_validate_rejects_unknown_language() -> None:
    tags = _valid_tags()
    tags["primary_language"] = "english"  # not in enum
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("primary_language" in e for e in errs)


def test_validate_rejects_empty_summaries() -> None:
    tags = _valid_tags()
    tags["summary_hindi"] = "   "
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("summary_hindi" in e and "non-empty" in e for e in errs)


def test_validate_rejects_array_with_non_strings() -> None:
    tags = _valid_tags()
    tags["topics"] = ["karma-yoga", 42, "self-inquiry"]
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("topics" in e for e in errs)


def test_validate_accepts_empty_arrays() -> None:
    tags = _valid_tags()
    tags["scriptures_referenced"] = []
    tags["timing_clues"] = []
    tags["location_clues"] = []
    ok, errs = validate_tags(tags)
    assert ok, errs


def test_allowed_enums_match_doc_contract() -> None:
    # Sanity check: the enums the validator enforces match what we tell
    # the model in the prompt. Catches accidental drift if someone edits
    # one and forgets the other.
    assert ALLOWED_EVENT_TYPES == frozenset({
        "satsang", "bhajan", "meditation", "qa",
        "discourse", "mixed", "unknown",
    })
    assert ALLOWED_LANGUAGES == frozenset({"hindi", "sanskrit", "mixed"})


# ---- parse_model_json ---------------------------------------------------


def test_parse_clean_json() -> None:
    raw = json.dumps(_valid_tags())
    obj, err = parse_model_json(raw)
    assert err is None
    assert obj is not None
    assert obj["event_type"] == "discourse"


def test_parse_strips_markdown_fences() -> None:
    raw = "```json\n" + json.dumps(_valid_tags()) + "\n```"
    obj, err = parse_model_json(raw)
    assert err is None, err
    assert obj is not None
    assert obj["primary_language"] == "hindi"


def test_parse_strips_bare_fences() -> None:
    raw = "```\n" + json.dumps(_valid_tags()) + "\n```"
    obj, err = parse_model_json(raw)
    assert err is None, err
    assert obj is not None


def test_parse_rejects_empty() -> None:
    obj, err = parse_model_json("")
    assert obj is None
    assert err == "empty response"


def test_parse_rejects_garbage() -> None:
    obj, err = parse_model_json("this is not JSON at all")
    assert obj is None
    assert err is not None
    assert "json decode failed" in err


def test_parse_rejects_non_object() -> None:
    obj, err = parse_model_json('["array", "at", "top"]')
    assert obj is None
    assert err is not None
    assert "not a JSON object" in err


# ---- end-to-end pure pipeline (parse + validate) ------------------------


def test_full_pipeline_clean_model_output() -> None:
    """A realistic clean model output should parse and validate."""
    raw = json.dumps(_valid_tags())
    obj, perr = parse_model_json(raw)
    assert perr is None
    ok, errs = validate_tags(obj)
    assert ok, errs


def test_full_pipeline_dead_letters_bad_json() -> None:
    """Bad JSON → parse returns (None, reason). Validator never sees it."""
    raw = '{"event_type": "discourse", "topics": [unfinished'
    obj, perr = parse_model_json(raw)
    assert obj is None
    assert perr is not None


def test_full_pipeline_dead_letters_validation_failure() -> None:
    """Parseable JSON but bad schema → validate returns (False, errs)."""
    bad = _valid_tags()
    bad["event_type"] = "definitely_not_in_enum"
    raw = json.dumps(bad)
    obj, perr = parse_model_json(raw)
    assert perr is None  # JSON is fine
    ok, errs = validate_tags(obj)
    assert not ok
    assert errs  # at least one validation error
