"""Unit tests for ingestion.chunker_cleaned.

Focus on the identity/naming layer that prevents cross-session stem collisions
(e.g. '06 SAMBODHAN' appearing in many session folders), plus the end-to-end
process_file contract: cleaned body, preserved timestamps, qualified
source_file, and a flat unique output filename.
"""
from __future__ import annotations

import json
from pathlib import Path

from ingestion import chunker_cleaned as cc


def test_normalized_stem_strips_dual_output_infix():
    assert cc.normalized_stem(Path("x/04 PRAVACHAN.raw.json")) == "04 PRAVACHAN"
    assert cc.normalized_stem(Path("x/04 PRAVACHAN.cleaned.json")) == "04 PRAVACHAN"
    assert cc.normalized_stem(Path("x/04 PRAVACHAN.cleaned.txt")) == "04 PRAVACHAN"


def test_qualified_source_is_unique_across_sessions():
    base = Path("/data/out")
    a = base / "Live Masters 2010_isolation" / "01 NOIDA 7 - 10 JAN 2010_isolation" \
        / "7 JAN - 1$ - 6 PM_isolation" / "06 SAMBODHAN.raw.json"
    b = base / "Live Masters 2010_isolation" / "01 NOIDA 7 - 10 JAN 2010_isolation" \
        / "8 JAN - 3$ - 6 PM_isolation" / "06 SAMBODHAN.raw.json"
    sa = cc.qualified_source(a, base)
    sb = cc.qualified_source(b, base)
    assert sa != sb, "same bare stem in different sessions must not collide"
    # '_isolation' suffixes stripped -> canonical names in the key.
    assert sa == ("Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/"
                  "7 JAN - 1$ - 6 PM/06 SAMBODHAN.json")
    assert "_isolation" not in sa


def test_output_basename_flat_unique_and_safe():
    base = Path("/data/out")
    a = base / "Live Masters 2010_isolation" / "01 NOIDA 7 - 10 JAN 2010_isolation" \
        / "7 JAN - 1$ - 6 PM_isolation" / "06 SAMBODHAN.raw.json"
    b = base / "Live Masters 2010_isolation" / "01 NOIDA 7 - 10 JAN 2010_isolation" \
        / "8 JAN - 3$ - 6 PM_isolation" / "06 SAMBODHAN.raw.json"
    na = cc.output_basename(a, base)
    nb = cc.output_basename(b, base)
    assert na != nb
    assert na.endswith(".chunks.json")
    assert "/" not in na and "\\" not in na  # flat: no separators


def test_no_base_dir_degrades_to_bare_stem():
    p = Path("/whatever/04 PRAVACHAN.raw.json")
    assert cc.qualified_source(p, None) == "04 PRAVACHAN.json"
    assert cc.output_basename(p, None) == "04 PRAVACHAN.chunks.json"


def test_process_file_end_to_end(tmp_path):
    # Build a realistic nested layout with raw.json + cleaned.txt for one track.
    base = tmp_path / "Output"
    track_dir = (base / "Live Masters 2010_isolation"
                 / "01 NOIDA 7 - 10 JAN 2010_isolation"
                 / "7 JAN - 1$ - 6 PM_isolation")
    track_dir.mkdir(parents=True)
    raw = track_dir / "06 SAMBODHAN.raw.json"
    raw.write_text(json.dumps({"segments": [
        {"start": 6.3, "end": 9.0, "text": "medishan ise muhabbat"},
        {"start": 9.0, "end": 12.0, "text": "apne aapse mulakat"},
    ]}), encoding="utf-8")
    (track_dir / "06 SAMBODHAN.cleaned.txt").write_text(
        "meditation ise muhabbat apne aapse mulakat", encoding="utf-8")

    out_dir = tmp_path / "chunks"
    n, status = cc.process_file(raw, out_dir, out_dir / "_failed", base_dir=base)
    assert status == "ok" and n == 1

    out_files = list(out_dir.glob("*.chunks.json"))
    assert len(out_files) == 1
    payload = json.loads(out_files[0].read_text(encoding="utf-8"))
    assert payload["source_file"] == ("Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/"
                                      "7 JAN - 1$ - 6 PM/06 SAMBODHAN.json")
    ch = payload["chunks"][0]
    # Cleaned body (corrected token in, garbled token out), raw timestamps kept.
    assert "meditation" in ch["text"]
    assert "medishan" not in ch["text"]
    assert ch["start_sec"] == 6.3 and ch["end_sec"] == 12.0
    assert ch["has_timestamps"] is True
    # Path metadata resolved off the real folder names despite the qualified key.
    assert ch["path_metadata"]["track_title"] == "SAMBODHAN"
    assert ch["path_metadata"]["session_date"] == "2010-01-07"


def test_rerun_skips_already_processed(tmp_path):
    """Incremental: a second run over the same folder (same --base-dir) does no
    work and produces no duplicate output files."""
    base = tmp_path / "Output"
    track_dir = base / "Live Masters 2010_isolation" / "01 NOIDA 7 - 10 JAN 2010_isolation" \
        / "7 JAN - 1$ - 6 PM_isolation"
    track_dir.mkdir(parents=True)
    raw = track_dir / "04 PRAVACHAN.raw.json"
    raw.write_text(json.dumps({"segments": [
        {"start": 0.0, "end": 2.0, "text": "alpha beta gamma"}]}), encoding="utf-8")
    (track_dir / "04 PRAVACHAN.cleaned.txt").write_text("alpha beta gamma", encoding="utf-8")
    out_dir = tmp_path / "chunks"

    # First run processes it.
    rc1 = cc.main([str(base), str(out_dir), "--base-dir", str(base), "-r"])
    assert rc1 == 0
    outs = list(out_dir.glob("*.chunks.json"))
    assert len(outs) == 1
    mtime1 = outs[0].stat().st_mtime_ns

    # Second run skips it (default skip-existing) — no rewrite, no duplicate.
    rc2 = cc.main([str(base), str(out_dir), "--base-dir", str(base), "-r"])
    assert rc2 == 0
    outs2 = list(out_dir.glob("*.chunks.json"))
    assert len(outs2) == 1, "re-run must not create a duplicate output"
    assert outs2[0].stat().st_mtime_ns == mtime1, "re-run must not rewrite skipped file"


# ---- load_cleaned_text: reject summaries, keep cleanups ----------------
#
# The cleanup LLM is told never to cut content. On 877 of 7,413 real tracks it
# summarized instead, and the summary was indexed as a verbatim transcript.


def _prose(n_words: int, salt: str = "") -> str:
    """Non-degenerate filler: every word distinct, like real prose."""
    return " ".join(f"{salt}w{i}" for i in range(n_words))


def _write_pair(tmp_path: Path, raw_words: str, cleaned_text: str) -> Path:
    raw = tmp_path / "04 PRAVACHAN.raw.json"
    raw.write_text(json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": raw_words}],
                               "text": raw_words}), encoding="utf-8")
    (tmp_path / "04 PRAVACHAN.cleaned.txt").write_text(cleaned_text, encoding="utf-8")
    return raw


def test_load_cleaned_text_accepts_a_normal_cleanup(tmp_path):
    """The median real track keeps 99% of its words; p25 keeps 95%."""
    raw = _write_pair(tmp_path, _prose(200), _prose(190))
    assert cc.load_cleaned_text(raw, _prose(200)) is not None


def test_load_cleaned_text_rejects_a_summary(tmp_path):
    """2,689 raw words -> 38 cleaned words was indexed as Swami ji's verbatim
    words. The chunker must fall back to raw instead."""
    raw = _write_pair(tmp_path, _prose(2689), _prose(38))
    assert cc.load_cleaned_text(raw, _prose(2689)) is None


def test_load_cleaned_text_rejects_just_below_the_threshold(tmp_path):
    raw = _write_pair(tmp_path, _prose(1000), _prose(840))
    assert cc.load_cleaned_text(raw, _prose(1000)) is None
    raw2 = _write_pair(tmp_path, _prose(1000), _prose(860))
    assert cc.load_cleaned_text(raw2, _prose(1000)) is not None


def test_load_cleaned_text_keeps_cleanup_of_a_whisper_repetition_loop(tmp_path):
    """When raw is 'om om om om ...' a drastically shorter cleaned text is the
    correct output, not a summary. Falling back to raw would re-index the loop."""
    loop = "om " * 400
    raw = _write_pair(tmp_path, loop, "om namah")
    assert cc.load_cleaned_text(raw, loop) == "om namah"


def test_load_cleaned_text_ignores_ratio_on_very_short_tracks(tmp_path):
    """A 12-word invocation losing filler is not a summarization event."""
    raw = _write_pair(tmp_path, _prose(12), _prose(6))
    assert cc.load_cleaned_text(raw, _prose(12)) is not None


def test_load_cleaned_text_without_raw_text_skips_the_check(tmp_path):
    raw = _write_pair(tmp_path, _prose(2000), _prose(20))
    assert cc.load_cleaned_text(raw) is not None


def test_process_file_falls_back_to_raw_when_cleaned_is_a_summary(tmp_path):
    """End-to-end: the summarized body must never reach the chunks."""
    base = tmp_path / "Output"
    track_dir = (base / "Live Masters 2010_isolation"
                 / "01 NOIDA 7 - 10 JAN 2010_isolation"
                 / "7 JAN - 1$ - 6 PM_isolation")
    track_dir.mkdir(parents=True)
    raw_text = _prose(400)
    raw = track_dir / "04 PRAVACHAN.raw.json"
    raw.write_text(json.dumps({"segments": [{"start": 0.0, "end": 60.0, "text": raw_text}],
                               "text": raw_text}), encoding="utf-8")
    (track_dir / "04 PRAVACHAN.cleaned.txt").write_text(
        "swami ji spoke about meditation and love", encoding="utf-8")

    out_dir = tmp_path / "chunks"
    n, status = cc.process_file(raw, out_dir, out_dir / "_failed", base_dir=base)
    assert status == "ok" and n >= 1

    payload = json.loads(next(out_dir.glob("*.chunks.json")).read_text(encoding="utf-8"))
    body = " ".join(c["text"] for c in payload["chunks"])
    assert "w399" in body, "raw content must survive when the cleaned text is a summary"
    assert "swami ji spoke about meditation" not in body
