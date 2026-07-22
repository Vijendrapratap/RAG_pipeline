"""Unit tests for rag_api.synthesis prompt builders (no network)."""
from __future__ import annotations

from rag_api.lang import LANG_ENGLISH, LANG_HINDI
from rag_api.synthesis import (
    CHITCHAT_MESSAGE,
    NO_CONTEXT_MESSAGE,
    build_best_question,
    build_chitchat_system,
    build_context_block,
    build_list_question,
    build_system_prompt,
    build_user_prompt,
    build_verbatim_question,
    chitchat_reply,
    filter_min_score,
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


def test_context_block_carries_best_ranking_stats():
    # Best-sitting results (rag_api.best) stash their ranking rationale in
    # metadata; the context block must surface it so the model can explain
    # why a sitting ranks where it does.
    results = [{
        "chunk_id": None, "score": 0.83, "text": "sitting summary",
        "source_file": "f.json", "start_sec": None, "end_sec": None,
        "speakers": [],
        "metadata": {"mention_count": 12, "duration_min": 61.5,
                     "play_count": 3},
    }]
    block = build_context_block(results)
    assert "Mentions: 12" in block
    assert "Length (min): 61.5" in block
    assert "Plays: 3" in block


def test_verbatim_question_demands_exact_untranslated_quotes():
    q = build_verbatim_question("summary of the sitting on rain", LANG_ENGLISH)
    assert 'The user asked: "summary of the sitting on rain"' in q
    assert "word-for-word" in q
    assert "no translation" in q
    assert "[N]" in q


def test_best_question_keeps_rank_order_and_cites_signals():
    q = build_best_question("best sittings on rain", 5, LANG_ENGLISH)
    assert 'The user asked: "best sittings on rain"' in q
    assert "5" in q and "BEST FIRST" in q
    assert "Mentions" in q and "Plays" in q
    assert "English" in q and "[N]" in q


def test_chitchat_system_prompt_language_and_no_invention():
    s = build_chitchat_system(LANG_ENGLISH)
    assert "English" in s
    assert "Never invent" in s
    h = build_chitchat_system(LANG_HINDI)
    assert "Hindi" in h


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


# ---- filter_min_score (citation floor) -----------------------------------

def test_floor_keeps_only_passages_above_threshold():
    results = [_r(0.91), _r(0.80), _r(0.74), _r(0.20)]
    kept = filter_min_score(results, 0.75)
    assert [r["score"] for r in kept] == [0.91, 0.80]


def test_floor_is_strictly_greater_than():
    # Exactly 0.75 does not clear a 0.75 floor.
    kept = filter_min_score([_r(0.75), _r(0.76)], 0.75)
    assert [r["score"] for r in kept] == [0.76]


def test_floor_keeps_top_when_nothing_clears_it():
    # Weak query / reranker-down (raw RRF) — never go blank, keep the best hit.
    results = [_r(0.40), _r(0.05), _r(0.01)]
    assert filter_min_score(results, 0.75) == results[:1]


def test_floor_is_a_prefix_so_citation_numbers_align():
    results = [_r(0.90), _r(0.78), _r(0.10)]
    assert filter_min_score(results, 0.75) == results[:2]


def test_floor_disabled_when_zero():
    results = [_r(0.40), _r(0.01)]
    assert filter_min_score(results, 0.0) == results


def test_floor_empty_results_unchanged():
    assert filter_min_score([], 0.75) == []


# ---- 1.5: abstention (RAG_ALLOW_ABSTAIN) ---------------------------------

def test_abstain_returns_empty_when_nothing_clears_floor():
    # allow_abstain=True drops the safety net: a genuine negative (nothing above
    # the floor) yields [] so the caller returns NO_CONTEXT_MESSAGE, 0 citations.
    results = [_r(0.02), _r(0.01), _r(0.005)]
    assert filter_min_score(results, 0.03, allow_abstain=True) == []


def test_abstain_still_keeps_passages_above_floor():
    # Abstention only removes the net; real hits above the floor are still kept.
    results = [_r(0.91), _r(0.80), _r(0.02)]
    kept = filter_min_score(results, 0.03, allow_abstain=True)
    assert [r["score"] for r in kept] == [0.91, 0.80]


def test_abstain_false_keeps_single_best_hit():
    # Default (net on): nothing clears the floor -> keep the single best hit so
    # "puran singh" and other weak-but-real queries still answer instead of blank.
    results = [_r(0.02), _r(0.01)]
    assert filter_min_score(results, 0.03, allow_abstain=False) == results[:1]


def test_calibrated_floor_keeps_cross_script_top_hit():
    # §0.2 calibration point: the lowest legit cross-script top hit (0.040) must
    # survive the new 0.03 floor (the old 0.75 dropped it entirely).
    results = [_r(0.040), _r(0.02), _r(0.01)]
    kept = filter_min_score(results, 0.03, allow_abstain=True)
    assert [r["score"] for r in kept] == [0.040]


def test_list_question_wraps_query_with_numbered_list_instruction():
    q = build_list_question("10 sitting for rain", 10, LANG_ENGLISH)
    assert "10 sitting for rain" in q          # original ask preserved
    assert "numbered list" in q and "10" in q  # explicit enumeration + count
    assert "METADATA" in q                     # entries grounded in metadata
    assert "English" in q                      # language directive carried


def test_chitchat_reply_bilingual_with_english_fallback():
    assert chitchat_reply(LANG_HINDI) == CHITCHAT_MESSAGE[LANG_HINDI]
    assert chitchat_reply(LANG_ENGLISH) == CHITCHAT_MESSAGE[LANG_ENGLISH]
    # Unknown language falls back to English rather than raising.
    assert chitchat_reply("klingon") == CHITCHAT_MESSAGE[LANG_ENGLISH]
