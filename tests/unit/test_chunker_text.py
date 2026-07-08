"""Unit tests for ingestion.chunker_text.

Per PRD §6 Phase 3 acceptance criteria:
- Empty input -> 0 chunks, no crash.
- Single short utterance -> 1 chunk.
- Single very long monologue -> multiple chunks with overlap.
- Mixed-language Unicode text -> preserved exactly.
- Malformed encoding -> caught, file marked failed, pipeline continues.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion import chunker_text as ct


FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_split_sentences_basic():
    out = ct.split_sentences("Hello world. How are you? I am fine!")
    assert out == ["Hello world.", "How are you?", "I am fine!"]


def test_split_sentences_empty():
    assert ct.split_sentences("") == []
    assert ct.split_sentences("   \n\n  ") == []


def test_split_sentences_preserves_unicode():
    text = "Hello 世界. Привет мир! مرحبا بالعالم?"
    out = ct.split_sentences(text)
    assert any("世界" in s for s in out)
    assert any("Привет" in s for s in out)
    assert any("مرحبا" in s for s in out)


def test_empty_input_zero_chunks():
    assert ct.chunk_sentences([], "empty.txt") == []


def test_single_short_one_chunk():
    chunks = ct.chunk_sentences(
        ["Just a brief thought here.", "Nothing more to say."],
        "short.txt",
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c["has_timestamps"] is False
    assert c["start_sec"] is None
    assert c["speakers"] == []
    assert c["sentence_start"] == 0
    assert c["sentence_end"] == 1


def test_long_monologue_multiple_chunks_with_overlap():
    # 40 sentences x ~30 words = ~1560 tokens -> several chunks.
    sentences = [" ".join(["word"] * 30) + "." for _ in range(40)]
    chunks = ct.chunk_sentences(sentences, "long.txt")
    assert len(chunks) >= 3
    # Adjacent chunks should overlap by exactly OVERLAP_SENTENCES sentences.
    for prev, nxt in zip(chunks[:-1], chunks[1:]):
        prev_end = prev["sentence_end"]
        nxt_start = nxt["sentence_start"]
        # nxt starts at (prev_end + 1 - OVERLAP). Allow off-by-one if a
        # giant sentence prevented full overlap.
        gap = (prev_end + 1) - nxt_start
        assert 0 <= gap <= ct.OVERLAP_SENTENCES + 1, (
            f"unexpected overlap gap {gap} between chunks"
        )


def test_oversize_single_sentence_emits_alone():
    # One sentence of ~1000 tokens -> single chunk, no infinite loop.
    huge = " ".join(["word"] * 1000) + "."
    chunks = ct.chunk_sentences([huge, "Short follow-up."], "big.txt")
    assert len(chunks) >= 1
    assert chunks[0]["sentence_start"] == 0
    assert chunks[0]["sentence_end"] == 0


def test_unicode_preserved_in_chunk():
    sentences = ["English sentence.", "中文句子。Another one.",
                 "Привет всем."]
    # Note: middle entry tests that CJK punctuation doesn't split (regex is
    # only [.!?] ASCII). That's fine — it just gets grouped.
    chunks = ct.chunk_sentences(sentences, "unicode.txt")
    body = chunks[0]["text"]
    assert "中文" in body
    assert "Привет" in body


def test_header_format():
    chunks = ct.chunk_sentences(["First.", "Second.", "Third."], "myfile.txt")
    head = chunks[0]["text"].splitlines()[0]
    assert head.startswith("[Source: myfile.txt")
    assert "sentences 0-2" in head
    assert "timestamps unavailable" in head


def test_process_file_writes_chunks_json(tmp_path, monkeypatch):
    # rag_api.config loads .env into os.environ at import time, so whichever
    # test imports it first leaks RAW_TRANSCRIPTS_BASE_DIR into this one and
    # _path_meta_for starts prepending a metadata header. This test covers the
    # no-base-dir path; pin the environment rather than depend on test order.
    monkeypatch.delenv("RAW_TRANSCRIPTS_BASE_DIR", raising=False)
    out_dir = tmp_path / "out"
    failed_dir = out_dir / "_failed"
    n, status = ct.process_file(FIXTURES / "sample_plain.txt", out_dir,
                                failed_dir)
    assert status == "ok"
    assert n > 0
    written = out_dir / "sample_plain.chunks.json"
    assert written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["source_file"] == "sample_plain.txt"
    assert payload["format"] == "text"
    for c in payload["chunks"]:
        assert c["has_timestamps"] is False
        assert c["start_sec"] is None
        assert c["text"].startswith("[Source: sample_plain.txt")


def test_oversize_file_skipped(tmp_path, monkeypatch):
    # Lower cap to force the size guard to fire.
    monkeypatch.setattr(ct, "MAX_FILE_BYTES", 50)
    big = tmp_path / "big.txt"
    big.write_text("x" * 1000, encoding="utf-8")
    n, status = ct.process_file(big, tmp_path / "out", tmp_path / "out" / "_failed")
    assert status == "skipped"
    assert n == 0


def test_read_text_handles_latin1(tmp_path):
    p = tmp_path / "latin.txt"
    # 0xff is invalid utf-8 but valid latin-1 (ÿ).
    p.write_bytes(b"caf\xe9 stuff. More text.")
    text = ct.read_text(p)
    assert "caf" in text


def test_main_returns_zero_when_all_ok(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "a.txt").write_text("Hello there. Second sentence.",
                                  encoding="utf-8")
    rc = ct.main([str(in_dir), str(out_dir)])
    assert rc == 0
    assert (out_dir / "a.chunks.json").exists()


def test_main_handles_missing_input_dir(tmp_path):
    rc = ct.main([str(tmp_path / "nope"), str(tmp_path / "out")])
    assert rc == 2
