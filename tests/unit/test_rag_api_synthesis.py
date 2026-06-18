"""Unit tests for rag_api.synthesis prompt builders (no network)."""
from __future__ import annotations

from rag_api.lang import LANG_ENGLISH, LANG_HINDI
from rag_api.synthesis import (
    NO_CONTEXT_MESSAGE,
    build_context_block,
    build_system_prompt,
    build_user_prompt,
    trim_by_relevance,
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


# ---- trim_by_relevance ---------------------------------------------------

def _r(score):
    return {"chunk_id": "x", "score": score, "text": "t"}


def test_trim_keeps_only_dominant_hit_on_a_located_quote():
    # find_quote shape: one strong match, the rest noise.
    results = [_r(0.80), _r(0.047), _r(0.044), _r(0.041)]
    kept = trim_by_relevance(results, 0.2)
    assert [r["score"] for r in kept] == [0.80]


def test_trim_keeps_the_cluster_on_a_semantic_query():
    # Several genuinely-relevant passages clustered near the top: keep them all.
    results = [_r(0.93), _r(0.79), _r(0.78), _r(0.20)]
    kept = trim_by_relevance(results, 0.2)
    assert [r["score"] for r in kept] == [0.93, 0.79, 0.78, 0.20]


def test_trim_is_a_prefix_so_citation_numbers_align():
    results = [_r(0.5), _r(0.4), _r(0.01)]
    kept = trim_by_relevance(results, 0.2)
    assert kept == results[:2]  # contiguous prefix, order preserved


def test_trim_disabled_when_ratio_zero():
    results = [_r(0.8), _r(0.01)]
    assert trim_by_relevance(results, 0.0) == results


def test_trim_keeps_top_when_top_score_is_zero_or_missing():
    assert trim_by_relevance([_r(0.0), _r(0.0)], 0.2) == [_r(0.0), _r(0.0)]


def test_trim_single_result_unchanged():
    assert trim_by_relevance([_r(0.8)], 0.2) == [_r(0.8)]
