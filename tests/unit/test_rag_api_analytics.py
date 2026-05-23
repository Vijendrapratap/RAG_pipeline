"""Unit tests for rag_api.analytics pure helpers.

The Postgres-touching methods need a live DB; these cover the pure row
shaping and the Hindi-correct full-text config.
"""
from __future__ import annotations

from rag_api.analytics import (
    _FTS_MATCH,
    FTS_CONFIG,
    shape_speaker_counts,
    shape_transcript_counts,
)


def test_fts_config_is_simple_for_hindi():
    assert FTS_CONFIG == "simple"
    assert "to_tsvector('simple', text)" in _FTS_MATCH
    assert "plainto_tsquery('simple', %s)" in _FTS_MATCH
    # The English config mangles Devanagari — it must not appear.
    assert "english" not in _FTS_MATCH


def test_shape_speaker_counts():
    rows = [("SPEAKER_00", 14), ("SPEAKER_02", 9)]
    assert shape_speaker_counts(rows) == [
        {"speaker": "SPEAKER_00", "chunk_count": 14},
        {"speaker": "SPEAKER_02", "chunk_count": 9},
    ]


def test_shape_transcript_counts():
    rows = [("a.json", 12), ("b.json", 3)]
    assert shape_transcript_counts(rows) == [
        {"source_file": "a.json", "chunk_count": 12},
        {"source_file": "b.json", "chunk_count": 3},
    ]


def test_shape_helpers_handle_empty():
    assert shape_speaker_counts([]) == []
    assert shape_transcript_counts([]) == []


def test_shape_coerces_count_to_int():
    # psycopg2 can hand back non-int numerics; shaping normalises them.
    assert shape_speaker_counts([("X", "7")]) == [
        {"speaker": "X", "chunk_count": 7}
    ]
