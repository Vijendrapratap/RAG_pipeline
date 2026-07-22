"""Unit tests for ingestion.chunker_json.

Per PRD §6 Phase 3 acceptance criteria:
- Empty input -> 0 chunks, no crash.
- Single short utterance -> 1 chunk.
- Single very long monologue -> multiple chunks with overlap (overlap not
  required for JSON path; PRD calls this out only for text path).
- Mixed-language Unicode text -> preserved exactly.
- Speaker change mid-utterance -> chunk boundary respected.
- Malformed JSON -> caught, file marked failed, pipeline continues.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion import chunker_json as cj


FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_estimate_tokens_nonempty():
    assert cj.estimate_tokens("hello world") >= 1
    assert cj.estimate_tokens("") == 1  # safety floor


def test_normalize_segments_rejects_bad_shape():
    with pytest.raises(ValueError):
        cj.normalize_segments([1, 2, 3], "whisperx")
    with pytest.raises(ValueError):
        cj.normalize_segments({"foo": "bar"}, "whisperx")
    with pytest.raises(ValueError):
        cj.normalize_segments({"segments": "not a list"}, "whisperx")


def test_normalize_segments_drops_empty_text():
    data = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "", "speaker": "A"},
        {"start": 1.0, "end": 2.0, "text": "  ", "speaker": "A"},
        {"start": 2.0, "end": 3.0, "text": "hello", "speaker": "A"},
    ]}
    out = cj.normalize_segments(data, "whisperx")
    assert len(out) == 1
    assert out[0]["text"] == "hello"


def test_normalize_segments_ignores_speaker_for_whisper_format():
    data = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "A"}]}
    out = cj.normalize_segments(data, "whisper")
    assert out[0]["speaker"] is None


def test_empty_input_zero_chunks():
    chunks = cj.chunk_segments([], "empty.json", "whisperx")
    assert chunks == []


def test_single_short_utterance_one_chunk():
    segs = [{"text": "Just a brief thought.", "start": 0.0, "end": 2.0,
             "speaker": "A"}]
    chunks = cj.chunk_segments(segs, "short.json", "whisperx")
    assert len(chunks) == 1
    assert "Just a brief thought" in chunks[0]["text"]
    assert chunks[0]["speakers"] == ["A"]
    assert chunks[0]["start_sec"] == 0.0
    assert chunks[0]["end_sec"] == 2.0
    assert chunks[0]["has_timestamps"] is True


def test_long_monologue_split_within_single_segment():
    # A single oversize segment of DISTINCT (non-loop) sentences. Pre-1.2 this
    # emitted as one giant chunk; 1.2 subdivides it on sentence boundaries into
    # multiple chunks, each within MAX_TOKENS. Assert on word_count (the real
    # body length) so the check doesn't depend on the header layout.
    long_text = ". ".join(" ".join(f"word{i}_{k}" for k in range(6)) for i in range(300))
    segs = [{"text": long_text, "start": 0.0, "end": 600.0, "speaker": "A"}]
    chunks = cj.chunk_segments(segs, "long.json", "whisperx")
    assert len(chunks) >= 2
    for c in chunks:
        assert max(1, round(c["word_count"] * 1.3)) <= cj.MAX_TOKENS


def test_long_monologue_split_across_many_segments():
    # 30 segments of distinct words = ~1500 tokens — must yield multiple chunks.
    segs = [
        {"text": " ".join(f"seg{i}word{j}" for j in range(40)), "start": float(i),
         "end": float(i + 1), "speaker": "A"}
        for i in range(30)
    ]
    chunks = cj.chunk_segments(segs, "split.json", "whisperx")
    assert len(chunks) >= 2
    # Every chunk should sit between MIN and MAX (with some tail tolerance).
    for c in chunks[:-1]:  # all except final tail
        assert c["word_count"] >= 100  # roughly MIN_TOKENS / 1.3 in words


# --- 1.2: chunk-quality guards (ASR-loop drop + oversize subdivide) --------

def test_asr_loop_chunk_dropped_not_embedded():
    # A single segment repeating one phrase for thousands of words is an ASR
    # loop (unique-word ratio ~0). It must be dropped and dead-lettered, never
    # embedded.
    loop = " ".join(["राम"] * 5000)
    segs = [{"text": loop, "start": 0.0, "end": 300.0, "speaker": "A"}]
    dead: list = []
    chunks = cj.chunk_segments(segs, "loop.json", "whisperx", dead_letters=dead)
    assert chunks == []
    assert len(dead) == 1
    assert "asr_loop" in dead[0]["reason"]
    assert dead[0]["source_file"] == "loop.json"


def test_loop_across_many_segments_dropped():
    # The loop can also be spread across many short segments; the flushed chunk
    # is still degenerate and must be dropped.
    segs = [
        {"text": "हरि ॐ हरि ॐ", "start": float(i), "end": float(i + 1),
         "speaker": "A"}
        for i in range(400)
    ]
    dead: list = []
    chunks = cj.chunk_segments(segs, "loop2.json", "whisperx", dead_letters=dead)
    assert chunks == []
    assert dead and all("asr_loop" in d["reason"] for d in dead)


def test_legit_long_segment_subdivides_to_max_tokens():
    # A long, healthy (high unique-word) single segment must subdivide into
    # multiple chunks each within MAX_TOKENS — never one oversize chunk. Distinct
    # tokens per sentence keep the unique-word ratio well above the loop floor.
    body = "। ".join(" ".join(f"शब्द{i}क{k}" for k in range(6)) for i in range(400))
    segs = [{"text": body, "start": 0.0, "end": 500.0, "speaker": "A"}]
    chunks = cj.chunk_segments(segs, "legit.json", "whisperx")
    assert len(chunks) >= 2
    for c in chunks:
        assert max(1, round(c["word_count"] * 1.3)) <= cj.MAX_TOKENS


def test_single_oversize_sentence_window_splits_without_truncating(caplog):
    # Phase 17: a single "sentence" with no terminator that alone exceeds
    # MAX_TOKENS is a Whisper artifact, not prose. It used to raise and
    # dead-letter the whole file. It is now window-split on word boundaries and
    # logged loudly — the file is degraded, not discarded. Nothing is truncated.
    words = [f"w{i}" for i in range(2000)]  # ~2600 est-tokens, no terminator
    segs = [{"text": " ".join(words), "start": 0.0, "end": 100.0, "speaker": "A"}]
    with caplog.at_level("WARNING"):
        chunks = cj.chunk_segments(segs, "runon.json", "whisperx")

    assert len(chunks) >= 2
    for c in chunks:
        assert max(1, round(c["word_count"] * 1.3)) <= cj.MAX_TOKENS
    assert "window-splitting" in caplog.text

    # Every input word survives exactly once, in order, across the pieces.
    # (Chunk bodies carry a "[Source: …]" header line; compare the body only.)
    emitted: list[str] = []
    for c in chunks:
        emitted.extend(c["text"].split("\n", 1)[1].split())
    assert emitted == words


def test_single_token_chunk_dropped_and_dead_lettered():
    # Phase 17: a lone `ओ` spanning 16 s escapes the loop guard (wc=1 is below
    # MIN_LOOP_WORDS, an exemption that exists to protect short naturally-
    # repetitive chunks). It carries nothing retrievable and must be dropped.
    segs = [{"text": "ओ", "start": 10.0, "end": 26.0, "speaker": "A"}]
    dead: list = []
    chunks = cj.chunk_segments(segs, "lone.json", "whisperx", dead_letters=dead)
    assert chunks == []
    assert len(dead) == 1
    assert "single_token" in dead[0]["reason"]


def test_long_asr_loop_keeps_its_loop_classification():
    # The single-token guard must not preempt the loop guard: a 5000-word "राम"
    # loop is one distinct token, but audit_chunks.py keys on the loop ratio, so
    # it must still dead-letter as `asr_loop`, not `single_token`.
    segs = [{"text": " ".join(["राम"] * 5000), "start": 0.0, "end": 300.0,
             "speaker": "A"}]
    dead: list = []
    assert cj.chunk_segments(segs, "loop.json", "whisperx", dead_letters=dead) == []
    assert "asr_loop" in dead[0]["reason"]


def test_unicode_preserved_exactly():
    segs = [
        {"text": "Hello 世界, привет мир, مرحبا بالعالم.", "start": 0.0,
         "end": 3.0, "speaker": "A"},
    ]
    chunks = cj.chunk_segments(segs, "unicode.json", "whisperx")
    assert "世界" in chunks[0]["text"]
    assert "привет" in chunks[0]["text"]
    assert "مرحبا" in chunks[0]["text"]


def test_speaker_change_after_target_flushes():
    # 50 segments of speaker A (~plenty above TARGET), then 5 from speaker B.
    # Distinct words so nothing trips the ASR-loop guard. Boundary should land
    # at the speaker change.
    segs = [
        {"text": " ".join(f"alpha{i}x{k}" for k in range(20)), "start": float(i),
         "end": float(i + 1), "speaker": "A"}
        for i in range(50)
    ]
    segs += [
        {"text": " ".join(f"bravo{i}y{k}" for k in range(20)), "start": float(50 + i),
         "end": float(51 + i), "speaker": "B"}
        for i in range(5)
    ]
    chunks = cj.chunk_segments(segs, "multi.json", "whisperx")
    # At least one chunk should be A-only and at least one should include B.
    a_only = [c for c in chunks if c["speakers"] == ["A"]]
    has_b = [c for c in chunks if "B" in c["speakers"]]
    assert a_only, "expected at least one A-only chunk before the boundary"
    assert has_b, "expected at least one chunk containing speaker B"


def test_speaker_change_below_min_does_not_flush():
    # Just one short A utterance (~14 tokens), then B continues. A's chunk
    # is below MIN, so it absorbs B rather than flushing prematurely.
    segs = [
        {"text": "A speaker says a quick thing here.",
         "start": 0.0, "end": 1.0, "speaker": "A"},
        {"text": " ".join(["bravo"] * 30), "start": 1.0, "end": 5.0,
         "speaker": "B"},
    ]
    chunks = cj.chunk_segments(segs, "quickmix.json", "whisperx")
    # First chunk should include both A and B.
    assert "A" in chunks[0]["speakers"]
    assert "B" in chunks[0]["speakers"]


def test_header_format():
    # Two words, not one: Phase 17 drops single-token chunks as unretrievable,
    # and the header format under test here is unrelated to that guard.
    segs = [{"text": "hello world", "start": 65.0, "end": 70.0, "speaker": "A"}]
    chunks = cj.chunk_segments(segs, "myfile.json", "whisperx")
    head = chunks[0]["text"].splitlines()[0]
    assert head.startswith("[Source: myfile.json")
    assert "00:01:05" in head  # 65s -> 00:01:05
    assert "00:01:10" in head
    assert "Speakers: A" in head


def test_no_speaker_label_when_whisper_format():
    segs = [{"text": "hi there", "start": 0.0, "end": 1.0, "speaker": None}]
    chunks = cj.chunk_segments(segs, "plain.json", "whisper")
    head = chunks[0]["text"].splitlines()[0]
    assert "Speakers: unknown" in head


def test_process_file_writes_chunks_json(tmp_path, monkeypatch):
    # rag_api.config loads .env into os.environ at import time, so whichever
    # test imports it first leaks RAW_TRANSCRIPTS_BASE_DIR into this one and
    # _path_meta_for starts prepending a metadata header. This test covers the
    # no-base-dir path; pin the environment rather than depend on test order.
    monkeypatch.delenv("RAW_TRANSCRIPTS_BASE_DIR", raising=False)
    out_dir = tmp_path / "out"
    failed_dir = out_dir / "_failed"
    n, status = cj.process_file(FIXTURES / "sample_whisperx.json", out_dir,
                                "whisperx", failed_dir)
    assert status == "ok"
    assert n > 0
    written = out_dir / "sample_whisperx.chunks.json"
    assert written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["source_file"] == "sample_whisperx.json"
    assert payload["format"] == "whisperx"
    assert len(payload["chunks"]) == n
    for c in payload["chunks"]:
        assert c["has_timestamps"] is True
        assert isinstance(c["speakers"], list)
        assert c["text"].startswith("[Source: sample_whisperx.json")


def test_short_file_tail_flush_preserves_path_metadata(tmp_path):
    """Regression: a single-chunk file (so short that only the tail flush
    emits it) must still carry path_metadata. Earlier code dropped it for
    the tail-only path."""
    # Build a path that matches the PRD-spec hierarchy so parse_path
    # returns a fully-populated PathMetadata.
    base = tmp_path / "audio"
    track_dir = (
        base
        / "Live Masters 2010"
        / "01 NOIDA 7 - 10 JAN 2010"
        / "7 JAN - 1$ - 6 PM"
    )
    track_dir.mkdir(parents=True)
    src = track_dir / "04 PRAVACHAN.json"
    src.write_text(
        json.dumps({"segments": [
            # Tiny content: well below MIN_TOKENS so only tail flush fires.
            {"start": 0.0, "end": 1.5, "text": "hello world", "speaker": "A"}
        ]}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    n, status = cj.process_file(src, out_dir, "whisperx",
                                failed_dir=out_dir / "_failed",
                                base_dir=base)
    assert status == "ok"
    assert n == 1
    payload = json.loads(
        (out_dir / "04 PRAVACHAN.chunks.json").read_text(encoding="utf-8")
    )
    ch = payload["chunks"][0]
    pm = ch.get("path_metadata")
    assert pm is not None, "tail-flushed chunk lost path_metadata"
    assert pm["event_id"] == "01 NOIDA 7 - 10 JAN 2010"
    assert pm["location"] == "NOIDA"
    assert pm["session_date"] == "2010-01-07"
    assert pm["track_title"] == "PRAVACHAN"
    assert pm["track_type"] == "discourse"


def test_malformed_json_logs_to_failed_continues(tmp_path):
    out_dir = tmp_path / "out"
    failed_dir = out_dir / "_failed"
    n, status = cj.process_file(FIXTURES / "corrupted_files" / "malformed.json",
                                out_dir, "whisperx", failed_dir)
    assert status == "failed"
    assert n == 0
    err = failed_dir / "malformed.json.error.txt"
    assert err.exists()
    assert "JSONDecodeError" in err.read_text(encoding="utf-8") or \
        "Expecting" in err.read_text(encoding="utf-8")


def test_main_returns_nonzero_on_failure(tmp_path):
    # Mix of good and bad files in one input dir.
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "good.json").write_text(
        json.dumps({"segments": [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "A"}
        ]}),
        encoding="utf-8",
    )
    (in_dir / "bad.json").write_text("{not valid json", encoding="utf-8")
    rc = cj.main([str(in_dir), str(out_dir), "--format", "whisperx"])
    assert rc == 1  # one failure
    assert (out_dir / "good.chunks.json").exists()
    assert (out_dir / "_failed" / "bad.json.error.txt").exists()


def test_main_returns_zero_when_all_ok(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "a.json").write_text(
        json.dumps({"segments": [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "A"}
        ]}),
        encoding="utf-8",
    )
    rc = cj.main([str(in_dir), str(out_dir), "--format", "whisperx"])
    assert rc == 0
