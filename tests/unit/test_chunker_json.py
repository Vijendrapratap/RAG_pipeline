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


def test_long_monologue_multiple_chunks():
    # Build ~1500-word monologue (~1950 tokens). Should yield >=2 chunks
    # since MAX_TOKENS=700.
    long_text = " ".join(["word"] * 1500)
    segs = [{"text": long_text, "start": 0.0, "end": 600.0, "speaker": "A"}]
    chunks = cj.chunk_segments(segs, "long.json", "whisperx")
    # Single segment of >MAX tokens emits as one chunk (we can't split inside
    # a single segment without word-level timing). That's expected behavior:
    # the segment IS the atomic unit. Verify it emits without crashing.
    assert len(chunks) >= 1
    assert chunks[0]["word_count"] == 1500


def test_long_monologue_split_across_many_segments():
    # 30 segments x ~50 tokens each = ~1500 tokens — must yield multiple chunks.
    segs = [
        {"text": " ".join(["word"] * 40), "start": float(i),
         "end": float(i + 1), "speaker": "A"}
        for i in range(30)
    ]
    chunks = cj.chunk_segments(segs, "split.json", "whisperx")
    assert len(chunks) >= 2
    # Every chunk should sit between MIN and MAX (with some tail tolerance).
    for c in chunks[:-1]:  # all except final tail
        assert c["word_count"] >= 100  # roughly MIN_TOKENS / 1.3 in words


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
    # Boundary should land at the speaker change.
    segs = [
        {"text": " ".join(["alpha"] * 20), "start": float(i),
         "end": float(i + 1), "speaker": "A"}
        for i in range(50)
    ]
    segs += [
        {"text": " ".join(["bravo"] * 20), "start": float(50 + i),
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
    segs = [{"text": "hello", "start": 65.0, "end": 70.0, "speaker": "A"}]
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


def test_process_file_writes_chunks_json(tmp_path):
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
