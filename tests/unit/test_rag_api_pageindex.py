"""Unit tests for the pure helpers in rag_api.pageindex.

These cover section grouping, provenance collapsing, the reasoning-prompt
builder, node-selection parsing, and result shaping — all without Ollama, a
filesystem, or a live stack.
"""
from __future__ import annotations

from rag_api.pageindex import (
    build_tree_reasoning_prompt,
    estimate_tokens,
    group_chunks_into_sections,
    node_to_result,
    parse_node_selection,
    safe_tree_name,
    section_provenance,
    tree_path,
)


def _row(cid, text, start=None, end=None, speakers=None, source="a.json"):
    return {
        "chunk_id": cid, "text": text, "start_sec": start, "end_sec": end,
        "speakers": speakers or [], "source_file": source,
    }


# ---- estimate_tokens / safe_tree_name / tree_path ------------------------


def test_estimate_tokens_floor_is_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 40) == 10


def test_safe_tree_name_sanitises_and_caps():
    assert safe_tree_name("04 PRAVACHAN.json") == "04_PRAVACHAN.json"
    assert "/" not in safe_tree_name("a/b\\c:d*e.json")
    assert len(safe_tree_name("x" * 500)) == 120


def test_tree_path_is_deterministic_for_a_source_file():
    p1 = tree_path("/data/pi", "04 PRAVACHAN.json")
    p2 = tree_path("/data/pi", "04 PRAVACHAN.json")
    assert p1 == p2
    assert p1.name == "04_PRAVACHAN.json.json"


# ---- group_chunks_into_sections ------------------------------------------


def test_grouping_packs_until_budget_then_splits():
    # 40 chars ~= 10 tokens each; budget 25 tokens fits two rows per section.
    rows = [_row(str(i), "x" * 40) for i in range(5)]
    sections = group_chunks_into_sections(rows, max_tokens=25)
    assert [len(s) for s in sections] == [2, 2, 1]


def test_oversized_single_chunk_is_its_own_section():
    rows = [_row("0", "x" * 4000), _row("1", "y" * 4000)]
    sections = group_chunks_into_sections(rows, max_tokens=100)
    assert [len(s) for s in sections] == [1, 1]


def test_empty_input_yields_no_sections():
    assert group_chunks_into_sections([], max_tokens=100) == []


# ---- section_provenance --------------------------------------------------


def test_provenance_spans_time_and_unions_speakers():
    rows = [
        _row("a", "hello", start=1.0, end=3.0, speakers=["S0"]),
        _row("b", "world", start=3.0, end=9.5, speakers=["S1", "S0"]),
    ]
    prov = section_provenance(rows)
    assert prov["start_sec"] == 1.0
    assert prov["end_sec"] == 9.5
    assert prov["speakers"] == ["S0", "S1"]
    assert prov["chunk_ids"] == ["a", "b"]
    assert prov["text"] == "hello\nworld"
    assert prov["source_file"] == "a.json"


def test_provenance_timeless_chunks_have_null_span():
    prov = section_provenance([_row("a", "t", start=None, end=None)])
    assert prov["start_sec"] is None and prov["end_sec"] is None


# ---- build_tree_reasoning_prompt -----------------------------------------


def test_reasoning_prompt_lists_every_node_and_the_query():
    nodes = [
        {"node_id": "0001", "title": "Karma", "summary": "on action"},
        {"node_id": "0002", "title": "Dhyana", "summary": "on meditation"},
    ]
    prompt = build_tree_reasoning_prompt("what is karma?", "a discourse", nodes)
    assert "[0001] Karma — on action" in prompt
    assert "[0002] Dhyana — on meditation" in prompt
    assert "what is karma?" in prompt
    assert "node_ids" in prompt


# ---- parse_node_selection ------------------------------------------------


def test_parse_keeps_only_valid_ids_in_order_dedup():
    raw = '{"node_ids": ["0002", "0001", "0002", "0099"]}'
    assert parse_node_selection(raw, {"0001", "0002"}) == ["0002", "0001"]


def test_parse_tolerates_prose_around_json():
    raw = 'Sure! {"node_ids": ["0001"]} hope that helps'
    assert parse_node_selection(raw, {"0001"}) == ["0001"]


def test_parse_empty_selection_and_garbage_return_empty():
    assert parse_node_selection('{"node_ids": []}', {"0001"}) == []
    assert parse_node_selection("not json at all", {"0001"}) == []
    assert parse_node_selection('{"wrong_key": ["0001"]}', {"0001"}) == []


# ---- node_to_result ------------------------------------------------------


def test_node_to_result_matches_chunk_result_shape():
    node = {
        "node_id": "0003", "title": "Karma", "summary": "on action",
        "source_file": "a.json", "start_sec": 12.0, "end_sec": 30.0,
        "speakers": ["S0"], "text": "the section text",
    }
    r = node_to_result(node, 0.8123456)
    assert r["result_type"] == "chunk"
    assert r["chunk_id"] == "a.json#0003"
    assert r["score"] == 0.8123  # rounded to 4dp like to_result
    assert r["text"] == "the section text"
    assert r["source_file"] == "a.json"
    assert r["start_sec"] == 12.0 and r["end_sec"] == 30.0
    assert r["speakers"] == ["S0"]
    assert r["metadata"]["backend"] == "pageindex"
    assert r["metadata"]["section_title"] == "Karma"
    assert r["metadata"]["node_id"] == "0003"
