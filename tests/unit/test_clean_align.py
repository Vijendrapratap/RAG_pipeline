"""Unit tests for ingestion.clean_align.

The aligner redistributes LLM-cleaned text across the raw WhisperX segments
while preserving each segment's timestamp. Properties under test:
- every cleaned word is placed exactly once, in order;
- raw timestamps are preserved on the segments that survive;
- spelling corrections in the cleaned stream replace the raw words;
- word-count drift (merges/splits/drops) does not crash or duplicate words;
- degenerate inputs fall back safely (empty raw -> [], empty cleaned -> raw).
"""
from __future__ import annotations

from ingestion.clean_align import align_cleaned_to_segments


def _seg(text, start, end, speaker=None):
    return {"text": text, "start": start, "end": end, "speaker": speaker}


def _all_words(segs):
    return [w for s in segs for w in s["text"].split()]


def test_identity_alignment_preserves_words_and_times():
    segs = [_seg("hello world", 0.0, 1.0), _seg("foo bar baz", 1.0, 2.5)]
    out = align_cleaned_to_segments(segs, "hello world foo bar baz")
    assert _all_words(out) == ["hello", "world", "foo", "bar", "baz"]
    assert out[0]["start"] == 0.0 and out[0]["end"] == 1.0
    assert out[1]["start"] == 1.0 and out[1]["end"] == 2.5


def test_spelling_correction_replaces_words_keeps_timestamps():
    # Mirrors the real data: same word count, only spellings change.
    segs = [_seg("medishan ise muhabbat", 6.3, 9.0),
            _seg("apne aapse mulakat", 9.0, 12.0)]
    cleaned = "meditation ise muhabbat apne aapse mulakat"
    out = align_cleaned_to_segments(segs, cleaned)
    assert out[0]["text"] == "meditation ise muhabbat"
    assert out[0]["start"] == 6.3 and out[0]["end"] == 9.0
    assert out[1]["text"] == "apne aapse mulakat"
    # No raw garbled token survives.
    assert "medishan" not in " ".join(s["text"] for s in out)


def test_word_order_preserved_under_drift():
    # Cleaned has fewer words (a drop) — order must still be exact.
    segs = [_seg("a b c d", 0.0, 1.0), _seg("e f g h", 1.0, 2.0)]
    cleaned = "a b c e f g h"  # 'd' dropped
    out = align_cleaned_to_segments(segs, cleaned)
    assert _all_words(out) == ["a", "b", "c", "e", "f", "g", "h"]


def test_insertion_attaches_without_losing_words():
    # Cleaned has an extra word (a split/insert). All cleaned words land once.
    segs = [_seg("one two", 0.0, 1.0), _seg("three four", 1.0, 2.0)]
    cleaned = "one two extra three four"
    out = align_cleaned_to_segments(segs, cleaned)
    assert _all_words(out) == ["one", "two", "extra", "three", "four"]


def test_devanagari_unicode_preserved():
    segs = [_seg("मेडिशन इसे", 0.0, 1.0), _seg("मुहब्बत", 1.0, 2.0)]
    out = align_cleaned_to_segments(segs, "मेडिटेशन इसे मुहब्बत")
    joined = " ".join(s["text"] for s in out)
    assert "मेडिटेशन" in joined
    assert "मुहब्बत" in joined


def test_empty_segments_returns_empty():
    assert align_cleaned_to_segments([], "anything here") == []


def test_empty_cleaned_falls_back_to_raw():
    segs = [_seg("keep this", 0.0, 1.0), _seg("   ", 1.0, 2.0)]
    out = align_cleaned_to_segments(segs, "   ")
    # Falls back to raw text, dropping the whitespace-only segment.
    assert len(out) == 1
    assert out[0]["text"] == "keep this"
    assert out[0]["start"] == 0.0


def test_segment_emptied_by_cleanup_is_dropped():
    # All of segment 2's words are deleted by the cleanup; it should vanish,
    # and its timestamp should not appear.
    segs = [_seg("alpha beta", 0.0, 1.0), _seg("noise noise", 1.0, 2.0),
            _seg("gamma", 2.0, 3.0)]
    cleaned = "alpha beta gamma"
    out = align_cleaned_to_segments(segs, cleaned)
    assert _all_words(out) == ["alpha", "beta", "gamma"]
    assert all(s["start"] != 1.0 for s in out)


def test_every_cleaned_word_placed_exactly_once():
    segs = [_seg(" ".join(f"r{i}" for i in range(j, j + 5)), float(j), float(j + 5))
            for j in range(0, 40, 5)]
    cleaned = " ".join(f"c{i}" for i in range(40))  # full replace, same count
    out = align_cleaned_to_segments(segs, cleaned)
    placed = _all_words(out)
    assert placed == [f"c{i}" for i in range(40)]
    assert len(placed) == len(set(placed))  # no duplication
