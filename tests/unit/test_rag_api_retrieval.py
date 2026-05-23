"""Unit tests for the pure retrieval helpers in rag_api.retrieval.

These exercise filter building, RRF fusion, post-fusion filtering, and result
shaping — all without a live stack (no Qdrant / Ollama / Tantivy needed).
"""
from __future__ import annotations

from rag_api.retrieval import (
    apply_post_filters,
    build_qdrant_filter,
    mean_pool,
    rrf_fuse,
    summary_text_of,
    to_result,
    to_summary_result,
)


# ---- build_qdrant_filter -------------------------------------------------


def test_empty_filters_yield_none():
    assert build_qdrant_filter({}) is None
    assert build_qdrant_filter({"speaker": None, "topics": None}) is None


def test_scalar_filter_builds_match_value():
    qf = build_qdrant_filter({"source_file": "04 PRAVACHAN.json"})
    assert qf == {"must": [
        {"key": "source_file", "match": {"value": "04 PRAVACHAN.json"}}
    ]}


def test_location_is_uppercased():
    qf = build_qdrant_filter({"location": "noida"})
    assert qf["must"][0]["match"]["value"] == "NOIDA"


def test_array_filter_uses_match_any():
    qf = build_qdrant_filter({"topics": ["karma-yoga", "self-inquiry"]})
    assert qf["must"][0] == {
        "key": "topics", "match": {"any": ["karma-yoga", "self-inquiry"]}
    }


def test_track_type_string_is_wrapped_in_list():
    qf = build_qdrant_filter({"track_type": "discourse"})
    assert qf["must"][0]["match"] == {"any": ["discourse"]}


def test_date_range_builds_range_clause():
    qf = build_qdrant_filter({"date_range": ("2010-01-01", "2010-01-31")})
    assert qf["must"][0] == {
        "key": "session_date",
        "range": {"gte": "2010-01-01", "lte": "2010-01-31"},
    }


# ---- rrf_fuse ------------------------------------------------------------


def test_rrf_fuse_merges_and_sorts():
    dense = [{"id": "a", "score": 0.9, "payload": {"text": "alpha"}}]
    bm25 = [{"chunk_id": "b", "text": "bravo", "source_file": "f.json",
             "score": 5.0}]
    fused = rrf_fuse(dense, bm25, bm25_weight=0.65)
    ids = [cid for cid, _, _ in fused]
    assert set(ids) == {"a", "b"}
    # With bm25_weight 0.65, a rank-0 BM25-only hit outscores a rank-0
    # dense-only hit.
    assert ids[0] == "b"


def test_rrf_fuse_overlap_adds_scores_and_keeps_dense_payload():
    dense = [{"id": "x", "score": 0.5, "payload": {"text": "from-dense"}}]
    bm25 = [{"chunk_id": "x", "text": "from-bm25", "source_file": "f.json",
             "score": 1.0}]
    fused = rrf_fuse(dense, bm25, bm25_weight=0.5)
    assert len(fused) == 1
    cid, score, payload = fused[0]
    assert cid == "x"
    # Both arms contributed at rank 0 -> score is the sum.
    assert score == (0.5 / 61) + (0.5 / 61)
    # Dense payload wins (dense loop runs first, sets payload).
    assert payload["text"] == "from-dense"


def test_rrf_fuse_bm25_only_hit_gets_synthetic_payload():
    fused = rrf_fuse(
        [], [{"chunk_id": "z", "text": "t", "source_file": "f.json",
              "score": 2.0}],
        bm25_weight=0.65,
    )
    _, _, payload = fused[0]
    assert payload["text"] == "t"
    assert payload["source_file"] == "f.json"
    assert payload["speakers"] == []


# ---- apply_post_filters --------------------------------------------------


def test_post_filter_drops_non_matching_season():
    fused = [
        ("a", 1.0, {"text": "x", "season": "monsoon"}),
        ("b", 0.9, {"text": "y", "season": "winter"}),
    ]
    kept = apply_post_filters(fused, {"season": "monsoon"})
    assert [cid for cid, _, _ in kept] == ["a"]


def test_post_filter_drops_bm25_only_hits_when_filter_active():
    # A BM25-only hit has no 'season' key -> dropped when season is filtered.
    fused = [("bm25only", 1.0, {"text": "x", "source_file": "f.json"})]
    assert apply_post_filters(fused, {"season": "winter"}) == []


def test_post_filter_topics_match_any():
    fused = [
        ("a", 1.0, {"text": "x", "topics": ["karma-yoga", "dharma"]}),
        ("b", 0.9, {"text": "y", "topics": ["bhakti"]}),
    ]
    kept = apply_post_filters(fused, {"topics": ["karma-yoga"]})
    assert [cid for cid, _, _ in kept] == ["a"]


# ---- to_result -----------------------------------------------------------


def test_to_result_shapes_payload_and_omits_null_metadata():
    res = to_result("cid1", 0.87654, {
        "text": "hello", "source_file": "f.json",
        "start_sec": 1.0, "end_sec": 9.0, "speakers": ["Swami ji"],
        "season": "winter", "track_title": None,
    })
    assert res["chunk_id"] == "cid1"
    assert res["result_type"] == "chunk"
    assert res["score"] == 0.8765
    assert res["text"] == "hello"
    assert res["speakers"] == ["Swami ji"]
    assert res["metadata"] == {"season": "winter"}
    assert "track_title" not in res["metadata"]


# ---- source_file list filtering (two-stage retrieval) --------------------


def test_source_file_string_builds_match_value():
    qf = build_qdrant_filter({"source_file": "a.json"})
    assert qf["must"][0] == {"key": "source_file", "match": {"value": "a.json"}}


def test_source_file_list_builds_match_any():
    qf = build_qdrant_filter({"source_file": ["a.json", "b.json"]})
    assert qf["must"][0] == {
        "key": "source_file", "match": {"any": ["a.json", "b.json"]}
    }


def test_post_filter_source_file_list_keeps_only_listed():
    fused = [
        ("a", 1.0, {"text": "x", "source_file": "a.json"}),
        ("b", 0.9, {"text": "y", "source_file": "b.json"}),
        ("c", 0.8, {"text": "z", "source_file": "c.json"}),
    ]
    kept = apply_post_filters(fused, {"source_file": ["a.json", "c.json"]})
    assert [cid for cid, _, _ in kept] == ["a", "c"]


# ---- summary results -----------------------------------------------------


def test_summary_text_of_prefers_english():
    assert summary_text_of(
        {"summary_english": "EN text", "summary_hindi": "हिंदी"}
    ) == "EN text"
    assert summary_text_of({"summary_hindi": "हिंदी"}) == "हिंदी"
    assert summary_text_of({}) == ""


def test_to_summary_result_shape():
    res = to_summary_result("file-uuid", 0.77, {
        "source_file": "04 PRAVACHAN.json",
        "summary_english": "A discourse on karma yoga.",
        "summary_hindi": "कर्म योग पर एक प्रवचन।",
        "event_type": "discourse", "topics": ["karma-yoga"],
    })
    assert res["result_type"] == "summary"
    assert res["source_file"] == "04 PRAVACHAN.json"
    assert res["text"] == "A discourse on karma yoga."
    assert res["start_sec"] is None and res["end_sec"] is None
    assert res["summary_hindi"] == "कर्म योग पर एक प्रवचन।"
    assert res["metadata"]["event_type"] == "discourse"
    assert res["metadata"]["topics"] == ["karma-yoga"]


# ---- mean_pool (HyDE vector blending) ------------------------------------


def test_mean_pool_averages_elementwise():
    assert mean_pool([[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]]) == [1.0, 3.0, 5.0]


def test_mean_pool_single_vector_is_identity():
    assert mean_pool([[1.0, 2.0, 3.0]]) == [1.0, 2.0, 3.0]


def test_mean_pool_empty_is_empty():
    assert mean_pool([]) == []
