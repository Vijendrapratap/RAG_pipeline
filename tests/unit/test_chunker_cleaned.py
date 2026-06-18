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
