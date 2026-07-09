"""Unit tests for the catalog separate-source indexer.

Pure logic only — no Qdrant / Ollama / reranker. The HTTP paths are exercised
by integration tests; here we lock the chunking, document building, and
point-id determinism.
"""
from __future__ import annotations

from ingestion.catalog.index_catalog import (
    CatalogDoc,
    build_documents,
    chunk_detail,
    _point_id,
)
from ingestion.catalog.normalize import row_to_track
from ingestion.chunker_text import split_sentences


# --- chunk_detail ----------------------------------------------------------

def test_chunk_detail_empty():
    assert chunk_detail("") == []
    assert chunk_detail("   ") == []


def test_chunk_detail_short_is_one_chunk():
    out = chunk_detail("Meditation begins. Silence settles. Peace.")
    assert len(out) == 1
    assert "Meditation begins" in out[0]


def test_chunk_detail_long_text_splits():
    # ~120 sentences of ~8 words each -> well over one TARGET window.
    text = " ".join(f"This is sentence number {i} about meditation today." for i in range(120))
    chunks = chunk_detail(text)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


# --- 1.1: danda-aware split (Devanagari sentence terminators) --------------

def test_split_sentences_on_danda():
    # Hindi prose ends sentences with । (danda) and ॥ (double danda). Before 1.1
    # the split regex knew only .!? so this was ONE sentence; now it is three.
    parts = split_sentences("पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य॥ चौथा वाक्य।")
    assert len(parts) == 4
    assert parts[0] == "पहला वाक्य।"


def test_chunk_detail_splits_devanagari_danda():
    # A long DetailContents in Devanagari: before 1.1 it collapsed to a single
    # "sentence" -> one oversize chunk; danda-aware splitting now packs it into
    # multiple TARGET-sized chunks (the precondition 1.2's subdivider relies on).
    text = " ".join(f"यह ध्यान का {i} वाँ वाक्य है।" for i in range(300))
    chunks = chunk_detail(text)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


# --- build_documents -------------------------------------------------------

def _track(**over):
    row = {
        "CampYear": 2005, "CampPlace": "Panchkula",
        "Sitting": "13 APR - 2$ - 10:30 AM", "TrackNo": 4,
        "Content": "हे मेरे गुरुदेव करुणा", "Chorus": "Abhipsa, Suman",
        "DetailContents": None, "Comment": None,
    }
    row.update(over)
    return row_to_track(row)


def test_build_documents_makes_title_doc():
    docs = build_documents([_track()])
    titles = [d for d in docs if d.payload["doc_type"] == "track_title"]
    assert len(titles) == 1
    d = titles[0]
    assert d.payload["source_type"] == "catalog"
    assert d.payload["track_title"] == "हे मेरे गुरुदेव करुणा"
    assert d.payload["performers"] == ["Abhipsa", "Suman"]
    assert d.payload["join_key"] == "2005-04-13|2|4"
    assert d.payload["location"] == "PANCHKULA"
    assert "हे मेरे गुरुदेव करुणा" in d.text


def test_build_documents_makes_sitting_detail_docs():
    t = _track(DetailContents="Meditation begins now. The mind grows still. Peace arrives.")
    docs = build_documents([t])
    details = [d for d in docs if d.payload["doc_type"] == "sitting_detail"]
    assert len(details) >= 1
    d = details[0]
    assert d.payload["sitting_key"] == "2005-04-13|2|"
    assert d.payload["performers"] == ["Abhipsa", "Suman"]
    assert d.text.startswith("[Catalog | PANCHKULA | 2005-04-13")


def test_sitting_detail_deduped_across_tracks():
    # Two tracks of the SAME sitting carry identical DetailContents in the
    # export; only one sitting_detail set should be produced.
    detail = "One. Two. Three."
    t1 = _track(TrackNo=1, DetailContents=detail)
    t2 = _track(TrackNo=2, DetailContents=detail)
    docs = build_documents([t1, t2])
    details = [d for d in docs if d.payload["doc_type"] == "sitting_detail"]
    sitting_keys = {d.payload["sitting_key"] for d in details}
    assert sitting_keys == {"2005-04-13|2|"}


def test_point_ids_are_deterministic():
    docs1 = build_documents([_track(DetailContents="A. B. C.")])
    docs2 = build_documents([_track(DetailContents="A. B. C.")])
    assert [d.point_id for d in docs1] == [d.point_id for d in docs2]


def test_point_id_distinct_per_doc():
    assert _point_id("title", "k1") != _point_id("title", "k2")
    assert _point_id("detail", "k", 0) != _point_id("detail", "k", 1)


def test_track_without_title_or_key_skipped():
    docs = build_documents([_track(Content=None)])
    assert all(d.payload["doc_type"] != "track_title" for d in docs)
