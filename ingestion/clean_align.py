"""Align LLM-cleaned transcript text back onto WhisperX raw segments.

The newer transcription pipeline (D:\\Transcription whisperx\\Output) emits,
per track:

    <stem>.raw.json     WhisperX large-v3 segments + word-level timestamps
    <stem>.cleaned.txt  the same transcript, spelling-normalized by Qwen 3.5
    <stem>.cleaned.json  cleaned_text + extracted metadata

The cleanup is order-preserving, word-level spelling normalization: on a
measured sample 1746 raw words -> 1726 cleaned words, i.e. ~99% survive 1:1,
with the rest being merges/splits/drops. That property lets us keep the exact
WhisperX segment timestamps (good for audio seek + citations) while swapping
in the corrected text (good for dense/BM25 retrieval, since garbled ASR tokens
never match a query).

This module is pure — no IO, no network. It redistributes the cleaned word
stream across the raw segments via difflib and returns segments in the same
{text, start, end, speaker} shape `chunker_json.chunk_segments` consumes, so
the rest of the ingestion path is unchanged.
"""
from __future__ import annotations

import difflib
from typing import Any


def _flatten_words(segments: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    """Flatten segment texts into a word list and a parallel list giving the
    owning segment index for each word."""
    words: list[str] = []
    seg_of_word: list[int] = []
    for idx, seg in enumerate(segments):
        for w in (seg.get("text") or "").split():
            words.append(w)
            seg_of_word.append(idx)
    return words, seg_of_word


def align_cleaned_to_segments(
    segments: list[dict[str, Any]], cleaned_text: str
) -> list[dict[str, Any]]:
    """Redistribute ``cleaned_text`` across ``segments``, preserving each
    segment's start/end timestamp.

    Every cleaned word is assigned to exactly one segment — the segment of the
    raw word it matches or replaces; cleaned words with no raw counterpart
    (insertions) attach to the most recent segment seen. Word order is
    preserved end-to-end. Segments that receive no cleaned words are dropped
    (the cleanup removed their content).

    Returns new segment dicts ``{text, start, end, speaker}``. Degenerate
    inputs fall back to the raw segment text so the caller never silently
    loses a transcript:
      * empty ``segments``           -> ``[]``
      * empty/whitespace cleaned_text -> raw segments (stripped, non-empty)
    """
    raw_words, seg_of_word = _flatten_words(segments)
    cleaned_words = cleaned_text.split()

    if not raw_words:
        return []
    if not cleaned_words:
        return [
            {
                "text": (s.get("text") or "").strip(),
                "start": s.get("start"),
                "end": s.get("end"),
                "speaker": s.get("speaker"),
            }
            for s in segments
            if (s.get("text") or "").strip()
        ]

    assigned: list[list[str]] = [[] for _ in segments]
    sm = difflib.SequenceMatcher(a=raw_words, b=cleaned_words, autojunk=False)
    last_seg = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "delete":
            # Raw words dropped by the cleanup. Place nothing, but remember the
            # segment so a following insertion attaches sensibly.
            last_seg = seg_of_word[i1]
            continue
        if tag == "insert":
            # Cleaned words with no raw counterpart -> the current segment.
            assigned[last_seg].extend(cleaned_words[j1:j2])
            continue
        # 'equal' or 'replace': both spans are non-empty. Map each cleaned word
        # to a raw word proportionally, then to that raw word's segment.
        raw_span = i2 - i1
        cl_span = j2 - j1
        for k, j in enumerate(range(j1, j2)):
            off = min(i1 + k * raw_span // cl_span, i2 - 1)
            last_seg = seg_of_word[off]
            assigned[last_seg].append(cleaned_words[j])

    out: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        words = assigned[idx]
        if not words:
            continue
        out.append(
            {
                "text": " ".join(words),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "speaker": seg.get("speaker"),
            }
        )
    return out
