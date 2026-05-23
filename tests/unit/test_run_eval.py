"""Unit tests for eval.run_eval — exercises the pure functions
(_match_quote_or_qa, _match_topic, _aggregate, YAML loading) without
requiring the live stack."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import run_eval


def _ranked(*items: tuple[str, str]) -> list:
    """Helper: build a [(cid, score, payload)] list from (source_file, text) pairs.
    Scores descend from 1.0 to mark order."""
    out = []
    n = len(items)
    for i, (sf, text) in enumerate(items):
        out.append((f"cid-{i}", 1.0 - i * 0.05,
                    {"source_file": sf, "text": text, "speakers": []}))
    return out


def test_match_quote_finds_substring_in_correct_source():
    ranked = _ranked(
        ("other.json", "irrelevant content"),
        ("right.json", "this chunk contains the target phrase here"),
        ("right.json", "also right but doesn't have it"),
    )
    rank = run_eval._match_quote_or_qa(
        ranked, "right.json", "target phrase",
    )
    assert rank == 2


def test_match_quote_returns_none_when_substring_absent():
    ranked = _ranked(
        ("right.json", "nothing useful here"),
        ("right.json", "still nothing"),
    )
    assert run_eval._match_quote_or_qa(ranked, "right.json", "missing thing") is None


def test_match_quote_ignores_substring_in_wrong_source():
    ranked = _ranked(
        ("wrong.json", "this chunk contains the target phrase but wrong file"),
        ("right.json", "right file but no phrase"),
    )
    assert run_eval._match_quote_or_qa(ranked, "right.json", "target phrase") is None


def test_match_quote_is_case_insensitive():
    ranked = _ranked(("a.json", "WELL HELLO There Friend"))
    assert run_eval._match_quote_or_qa(ranked, "a.json", "hello there") == 1


def test_match_topic_all_present():
    ranked = _ranked(
        ("a.json", "x"), ("b.json", "y"), ("a.json", "z"),
    )
    passed, missing = run_eval._match_topic(ranked, ["a.json", "b.json"], k=3)
    assert passed
    assert missing == []


def test_match_topic_partial_in_top_k():
    ranked = _ranked(
        ("a.json", "x"), ("c.json", "y"), ("b.json", "z"),
    )
    # k=2 sees only a.json + c.json — missing b.json
    passed, missing = run_eval._match_topic(ranked, ["a.json", "b.json"], k=2)
    assert not passed
    assert missing == ["b.json"]
    # k=3 sees b.json too
    passed, _ = run_eval._match_topic(ranked, ["a.json", "b.json"], k=3)
    assert passed


def test_aggregate_quote_metrics():
    per_q = [
        {"id": "q1", "type": "quote", "pass_at_1": True, "pass_at_5": True,
         "pass_at_10": True, "reciprocal_rank": 1.0},
        {"id": "q2", "type": "quote", "pass_at_1": False, "pass_at_5": True,
         "pass_at_10": True, "reciprocal_rank": 1 / 3},
        {"id": "q3", "type": "quote", "pass_at_1": False, "pass_at_5": False,
         "pass_at_10": True, "reciprocal_rank": 1 / 7},
        {"id": "q4", "type": "quote", "pass_at_1": False, "pass_at_5": False,
         "pass_at_10": False, "reciprocal_rank": 0.0},
    ]
    s = run_eval._aggregate(per_q)
    assert s["quote"]["n"] == 4
    assert s["quote"]["hit_at_1"] == 0.25
    assert s["quote"]["hit_at_5"] == 0.5
    assert s["quote"]["hit_at_10"] == 0.75
    assert abs(s["quote"]["mrr"] - (1.0 + 1/3 + 1/7 + 0) / 4) < 1e-9


def test_aggregate_analytics_pass_rate():
    per_q = [
        {"id": "a1", "type": "analytics", "passed": True},
        {"id": "a2", "type": "analytics", "passed": False},
        {"id": "a3", "type": "analytics", "passed": True},
    ]
    s = run_eval._aggregate(per_q)
    assert s["analytics"]["pass_rate"] == pytest.approx(2 / 3)


def test_aggregate_handles_multiple_types():
    per_q = [
        {"id": "q1", "type": "quote", "pass_at_1": True, "pass_at_5": True,
         "pass_at_10": True, "reciprocal_rank": 1.0},
        {"id": "t1", "type": "topic", "pass_at_1": False, "pass_at_5": True,
         "pass_at_10": True, "reciprocal_rank": 0.5},
        {"id": "a1", "type": "analytics", "passed": True},
    ]
    s = run_eval._aggregate(per_q)
    assert set(s.keys()) == {"quote", "topic", "analytics"}
    assert s["quote"]["n"] == 1
    assert s["topic"]["hit_at_5"] == 1.0
    assert s["analytics"]["pass_rate"] == 1.0


def test_score_analytics_count_in_range():
    class _FakeTool:
        def count_mentions(self, term, speaker=None):
            return f'Term "{term}" mentioned in 7 chunks (across all speakers).'
    q = {"tool": "count_mentions", "args": {"term": "x"},
         "expected_count_min": 5, "expected_count_max": 10}
    passed, reason, extra = run_eval._score_analytics(_FakeTool(), q)
    assert passed
    assert extra["count"] == 7


def test_score_analytics_count_below_min():
    class _FakeTool:
        def count_mentions(self, term, speaker=None):
            return f'Term "{term}" mentioned in 0 chunks (across all speakers).'
    q = {"tool": "count_mentions", "args": {"term": "x"},
         "expected_count_min": 1, "expected_count_max": 10}
    passed, reason, _ = run_eval._score_analytics(_FakeTool(), q)
    assert not passed
    assert "0 < min 1" in reason


def test_score_analytics_list_transcripts_missing_file():
    class _FakeTool:
        def list_transcripts_mentioning(self, term, limit=20):
            return f'Transcripts mentioning "{term}" (1):\n  other.json (3)'
    q = {"tool": "list_transcripts_mentioning", "args": {"term": "x"},
         "expected_source_files": ["wanted.json"]}
    passed, reason, _ = run_eval._score_analytics(_FakeTool(), q)
    assert not passed
    assert "wanted.json" in reason


def test_score_analytics_list_transcripts_all_present():
    class _FakeTool:
        def list_transcripts_mentioning(self, term, limit=20):
            return ('Transcripts mentioning "x" (2):\n'
                    '  a.json (3)\n  b.json (1)')
    q = {"tool": "list_transcripts_mentioning", "args": {"term": "x"},
         "expected_source_files": ["a.json"]}
    passed, _reason, extra = run_eval._score_analytics(_FakeTool(), q)
    assert passed
    assert "a.json" in extra["found"]


def test_golden_queries_yaml_loads_and_has_30_queries():
    """Sanity: the shipped YAML must be valid and hit the PRD count."""
    import yaml
    repo_root = Path(__file__).resolve().parents[2]
    queries = yaml.safe_load(
        (repo_root / "eval" / "golden_queries.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(queries, list)
    assert len(queries) == 30
    by_type = {}
    for q in queries:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    assert by_type == {"quote": 12, "qa": 8, "topic": 5, "analytics": 5}
    # IDs unique + sequential q001..q030
    ids = [q["id"] for q in queries]
    assert len(set(ids)) == 30
    assert ids[0] == "q001" and ids[-1] == "q030"
