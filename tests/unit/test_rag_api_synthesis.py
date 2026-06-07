"""Unit tests for rag_api.synthesis prompt builders (no network)."""
from __future__ import annotations

from rag_api.lang import LANG_ENGLISH, LANG_HINDI
from rag_api.synthesis import (
    NO_CONTEXT_MESSAGE,
    build_context_block,
    build_system_prompt,
    build_user_prompt,
)

SAMPLE = [
    {
        "chunk_id": "c1", "score": 0.91,
        "text": "Karma yoga means action performed without attachment.",
        "source_file": "04 PRAVACHAN.json",
        "start_sec": 12.0, "end_sec": 45.0, "speakers": ["Swami ji"],
        "metadata": {"event_id": "01 NOIDA", "session_date": "2010-01-07"},
    },
    {
        "chunk_id": "c2", "score": 0.80,
        "text": "Plain-text passage with no timestamps.",
        "source_file": "notes.txt",
        "start_sec": None, "end_sec": None, "speakers": [],
        "metadata": {},
    },
]


def test_context_block_is_numbered_and_carries_sources():
    block = build_context_block(SAMPLE)
    assert "[1]" in block and "[2]" in block
    assert "04 PRAVACHAN.json" in block
    assert "Karma yoga means action" in block
    assert "12.0s-45.0s" in block


def test_context_block_marks_missing_timestamps():
    block = build_context_block(SAMPLE)
    assert "timestamps unavailable" in block


def test_context_block_empty_results():
    assert "no transcript passages" in build_context_block([]).lower()


def test_system_prompt_hindi_demands_devanagari_and_no_invention():
    p = build_system_prompt(LANG_HINDI)
    assert "Devanagari" in p
    assert "invent" in p.lower()
    assert "citation" in p.lower() or "cite" in p.lower()


def test_system_prompt_english():
    assert "English" in build_system_prompt(LANG_ENGLISH)


def test_user_prompt_contains_query_and_context():
    up = build_user_prompt("what is karma yoga?", "CONTEXT_PLACEHOLDER", LANG_HINDI)
    assert "what is karma yoga?" in up
    assert "CONTEXT_PLACEHOLDER" in up
    assert "Hindi" in up


def test_no_context_messages_exist_for_both_languages():
    assert NO_CONTEXT_MESSAGE[LANG_HINDI].strip()
    assert NO_CONTEXT_MESSAGE[LANG_ENGLISH].strip()
    # Hindi message must actually be in Devanagari.
    assert any("ऀ" <= c <= "ॿ" for c in NO_CONTEXT_MESSAGE[LANG_HINDI])
