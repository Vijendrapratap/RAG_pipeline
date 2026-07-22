"""Detect transcripts edited AFTER they were last chunked/embedded (Stage 0 of
``add_transcripts.sh``).

A track is STALE when any chunker input (``<stem>.raw.json`` / ``.cleaned.txt`` /
``.cleaned.json``) has an mtime NEWER than the ``<...>.chunks.json`` the chunker
previously emitted for it. That happens whenever an upstream pass rewrites a
transcript in place after it was already ingested -- e.g. the WhisperX VAD-profile
re-transcribe, the energy-gated recall pass, or the hallucination/loop scrubs. The
stores then hold the OLD text, and a plain incremental run would never fix it:
``chunker_cleaned`` skips any track whose ``.chunks.json`` already exists (existence
only, no mtime), and ``bulk_ingest_hardened`` skips any file marked ``ok``.

This module is the missing staleness signal. It is **READ-ONLY** -- it writes to no
store and deletes nothing. It emits two worklists for the caller to act on:

  --out-sources : one ``qualified_source`` per line -> feed to ``reindex_file``
                  (``--source-list``), which deletes the old data across every
                  store and resets the progress row.
  --out-chunks  : one ``<...>.chunks.json`` basename per line -> the caller ``rm``s
                  these so the chunker regenerates them from the current transcript.

It reuses ``chunker_cleaned.{qualified_source, output_basename, normalized_stem}``
verbatim so the keys match, byte-for-byte, what the ingest path wrote. Never
hand-derive these names -- a mismatch silently reindexes the wrong track or none.

Usage:
    python -m ingestion.find_stale_transcripts \
        --source-root "/mnt/d/Transcription whisperx/Output" \
        --base-dir    "/mnt/d/Transcription whisperx/Output" \
        --chunks-dir  data/processed \
        --out-sources /tmp/stale_sources.txt \
        --out-chunks  /tmp/stale_chunks.txt

Prints one line ``STALE=<n>`` (plus a short human summary on stderr). Always exit 0
unless the inputs are unusable (missing dirs -> exit 2).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ingestion.chunker_cleaned import (
    normalized_stem,
    output_basename,
    qualified_source,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
log = logging.getLogger("find_stale_transcripts")

# Filesystem mtime jitter guard: only flag a real, meaningful edit, not a
# sub-second touch or a copy that reset timestamps by a hair.
MTIME_TOLERANCE_SEC = 2.0


def _input_mtime(raw_path: Path) -> float:
    """Newest mtime among the inputs that actually determine THIS track's chunk
    content -- mirroring the chunker's ``load_cleaned_text`` precedence exactly:

      * ``.raw.json``     -- always (segment timing source)
      * ``.cleaned.txt``  -- the body, PREFERRED when present
      * ``.cleaned.json`` -- body fallback, used ONLY when ``.cleaned.txt`` is absent

    Critically, ``.cleaned.json`` is NOT counted when ``.cleaned.txt`` exists: a
    metadata/tag enrichment pass rewrites ``.cleaned.json`` (its content tags are
    out of scope for the chunker) without changing the embedded text, and counting
    it would false-positive a track as stale and trigger a needless DESTRUCTIVE
    reindex. Only a change to the inputs the chunker truly reads means stale chunks.
    """
    stem = normalized_stem(raw_path)
    candidates = [raw_path]  # timing source, always read
    cleaned_txt = raw_path.with_name(f"{stem}.cleaned.txt")
    cleaned_json = raw_path.with_name(f"{stem}.cleaned.json")
    if cleaned_txt.exists():
        candidates.append(cleaned_txt)          # chunker's preferred body
    elif cleaned_json.exists():
        candidates.append(cleaned_json)         # fallback body (txt absent)
    mtimes = [p.stat().st_mtime for p in candidates if p.exists()]
    return max(mtimes) if mtimes else raw_path.stat().st_mtime


def find_stale(source_root: Path, base_dir: Path,
               chunks_dir: Path) -> list[tuple[str, str, float, str]]:
    """Return (qualified_source, chunks_basename, hours_newer, raw_json_path) for
    every track whose transcript is newer than its already-emitted chunks. Tracks
    with no ``.chunks.json`` yet (never ingested) are NOT stale -- the normal
    incremental path picks those up -- so they are skipped here."""
    stale: list[tuple[str, str, float, str]] = []
    for raw in source_root.rglob("*.raw.json"):
        chunks_name = output_basename(raw, base_dir)
        cj = chunks_dir / chunks_name
        if not cj.exists():
            continue  # never chunked -> handled by the normal incremental scan
        src_mtime = _input_mtime(raw)
        delta = src_mtime - cj.stat().st_mtime
        if delta > MTIME_TOLERANCE_SEC:
            stale.append((qualified_source(raw, base_dir), chunks_name,
                          delta / 3600.0, str(raw)))
    stale.sort(key=lambda t: t[2], reverse=True)
    return stale


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Find transcripts edited since they were last chunked (read-only).")
    p.add_argument("--source-root", required=True,
                   help="Directory to scan for *.raw.json (the chunker's INPUT_DIR).")
    p.add_argument("--base-dir", required=True,
                   help="Base dir for qualified_source/output_basename -- MUST match "
                        "the --base-dir the chunker/ingest ran with (SOURCE_ROOT).")
    p.add_argument("--chunks-dir", default="data/processed",
                   help="Directory of previously-emitted *.chunks.json.")
    p.add_argument("--out-sources", default=None,
                   help="Write stale qualified_source values here (one per line).")
    p.add_argument("--out-chunks", default=None,
                   help="Write stale .chunks.json basenames here (one per line).")
    p.add_argument("--out-raw", default=None,
                   help="Write stale *.raw.json absolute paths here (one per line).")
    p.add_argument("--limit-preview", type=int, default=20,
                   help="How many stale tracks to list on stderr.")
    args = p.parse_args(argv)

    source_root = Path(args.source_root)
    base_dir = Path(args.base_dir)
    chunks_dir = Path(args.chunks_dir)
    for label, d in (("source-root", source_root), ("chunks-dir", chunks_dir)):
        if not d.is_dir():
            log.error("%s does not exist or is not a directory: %s", label, d)
            return 2

    stale = find_stale(source_root, base_dir, chunks_dir)

    if args.out_sources is not None:
        Path(args.out_sources).write_text(
            "".join(f"{s}\n" for s, _, _, _ in stale), encoding="utf-8")
    if args.out_chunks is not None:
        Path(args.out_chunks).write_text(
            "".join(f"{c}\n" for _, c, _, _ in stale), encoding="utf-8")
    if args.out_raw is not None:
        Path(args.out_raw).write_text(
            "".join(f"{r}\n" for _, _, _, r in stale), encoding="utf-8")

    if stale:
        log.info("Stale transcripts (edited after last chunk): %d", len(stale))
        for s, _, hrs, _ in stale[: args.limit_preview]:
            log.info("   +%6.1fh  %s", hrs, s)
        if len(stale) > args.limit_preview:
            log.info("   ... and %d more", len(stale) - args.limit_preview)
    else:
        log.info("No stale transcripts: every chunked track is up to date.")

    # Machine-readable line for the shell caller.
    print(f"STALE={len(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
