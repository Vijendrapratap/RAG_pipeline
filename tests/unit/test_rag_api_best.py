"""Unit tests for rag_api.best — the best-sitting composite ranking.

Only the pure core (`_normalize`, `score_candidates`) is tested; the Postgres
stat lookups are fail-open plumbing exercised in live probes.
"""
from __future__ import annotations

from rag_api.best import _normalize, score_candidates, stat_terms


def _cand(sf: str, score: float) -> dict:
    return {
        "result_type": "summary", "chunk_id": None, "source_file": sf,
        "score": score, "text": "sitting summary",
        "metadata": {"track_title": sf},
    }


def test_normalize_min_max():
    assert _normalize([1.0, 3.0, 2.0]) == [0.0, 1.0, 0.5]
    assert _normalize([]) == []


def test_normalize_constant_column_contributes_nothing():
    # A column that cannot discriminate must not shift the ranking.
    assert _normalize([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]


def test_semantic_order_survives_missing_stats():
    # Postgres down / migration absent -> stats empty -> pure semantic rank.
    ranked = score_candidates([_cand("a", 0.9), _cand("b", 0.5)], {}, top_k=10)
    assert [r["source_file"] for r in ranked] == ["a", "b"]
    assert ranked[0]["metadata"]["semantic_score"] == 0.9
    assert "mention_count" not in ranked[0]["metadata"]


def test_mentions_break_a_semantic_tie():
    stats = {"a": {"mention_count": 2}, "b": {"mention_count": 40}}
    ranked = score_candidates(
        [_cand("a", 0.8), _cand("b", 0.8)], stats, top_k=10)
    assert [r["source_file"] for r in ranked] == ["b", "a"]
    assert ranked[0]["metadata"]["mention_count"] == 40


def test_duration_promotes_longer_sittings():
    stats = {"a": {"duration_sec": 600.0}, "b": {"duration_sec": 3600.0}}
    ranked = score_candidates(
        [_cand("a", 0.8), _cand("b", 0.8)], stats, top_k=10)
    assert [r["source_file"] for r in ranked] == ["b", "a"]
    assert ranked[0]["metadata"]["duration_min"] == 60.0


def test_plays_promote_played_sittings():
    stats = {"b": {"play_count": 10}}
    ranked = score_candidates(
        [_cand("a", 0.8), _cand("b", 0.8)], stats, top_k=10)
    assert [r["source_file"] for r in ranked] == ["b", "a"]
    assert ranked[0]["metadata"]["play_count"] == 10


def test_semantic_outweighs_any_single_stat():
    # 0.50 semantic > any one 0.25/0.15/0.10 component: sweeping mentions
    # alone cannot flip a clear semantic winner.
    stats = {"b": {"mention_count": 100}}
    ranked = score_candidates(
        [_cand("a", 0.95), _cand("b", 0.05)], stats, top_k=10)
    assert [r["source_file"] for r in ranked] == ["a", "b"]


def test_top_k_caps_the_ranked_list():
    cands = [_cand(f"f{i}", 0.5 + i / 100.0) for i in range(5)]
    ranked = score_candidates(cands, {}, top_k=2)
    assert [r["source_file"] for r in ranked] == ["f4", "f3"]


def test_candidates_are_not_mutated():
    c = _cand("a", 0.9)
    score_candidates([c], {"a": {"mention_count": 3}}, top_k=1)
    assert c["score"] == 0.9
    assert "mention_count" not in c["metadata"]
    assert "semantic_score" not in c["metadata"]


def test_empty_candidates():
    assert score_candidates([], {}, top_k=5) == []


def test_stat_terms_extracts_topic_words_per_word():
    # Whole variants AND-match ~nothing in FTS (live probe 2026-07-15 came
    # back all-zero); the counting unit is the individual topic word, with
    # ranking-meta / sitting-nouns / function words dropped and synonyms from
    # every variant kept.
    terms = stat_terms([
        "best 5 sittings on rain",
        "बारिश पर सत्संग",
        "rain satsang sessions",
        "वर्षा प्रवचन",
    ])
    assert terms == ["rain", "बारिश", "वर्षा"]


def test_stat_terms_dedupes_and_caps():
    terms = stat_terms(["rain rain barish", "monsoon rain"], max_terms=2)
    assert terms == ["rain", "barish"]


def test_stat_terms_empty():
    assert stat_terms([]) == []
    assert stat_terms(["best sittings"]) == []
