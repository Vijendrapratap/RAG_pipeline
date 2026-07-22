"""Chunk the WhisperX 'cleaned' transcription output into retrieval-ready
chunks: LLM-cleaned text as the body, raw WhisperX segment timestamps for
timing.

Input layout (per track), as produced by D:\\Transcription whisperx\\Output:

    <stem>.raw.json     WhisperX segments + word timestamps  (timing source)
    <stem>.cleaned.txt  spelling-normalized transcript        (body source)
    <stem>.cleaned.json cleaned_text + metadata               (body fallback)

For each ``*.raw.json`` we align the cleaned words back onto the raw segments
(see clean_align) and then reuse ``chunker_json.chunk_segments`` unchanged, so
the emitted ``.chunks.json`` is the exact contract bulk_ingest_hardened.py
already consumes. The only differences from the old chunker are: (1) the chunk
body holds corrected words, and (2) ``source_file`` is normalized to
``<stem>.json`` (the ``.raw`` / ``.cleaned`` infix dropped) so a track keeps one
stable identity across re-chunks.

Note that a stable ``source_file`` does **not** make re-ingestion idempotent:
Qdrant point IDs hash the chunk text, and the Tantivy writer is append-only, so
changing a transcript orphans its old points and duplicates its BM25 documents.
Re-ingest into a rebuilt index, not on top of a live one.

Cleaned text is trusted only when it is a *cleanup* of the raw transcript, not a
summary of it — see ``load_cleaned_text``.

Content tags (``cleaned.json.metadata``) are intentionally OUT OF SCOPE here —
that feeds the Phase 13 enrichment path and is handled separately.

CLI mirrors chunker_json:
    python -m ingestion.chunker_cleaned <input_dir> <output_dir> \\
        [--base-dir DIR] [--recursive] [--no-skip-existing]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import traceback
from pathlib import Path

from ingestion import chunker_json as cj
from ingestion.clean_align import align_cleaned_to_segments
from ingestion.utils import path_parser as pp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("chunker_cleaned")

# This pipeline has no speaker diarization (single speaker per file), so the
# chunker runs in 'whisper' format and speakers fall back to PRIMARY_SPEAKER.
FMT = "whisper"


def normalized_stem(raw_path: Path) -> str:
    """'04 PRAVACHAN.raw.json' -> '04 PRAVACHAN'. Strips the .raw/.cleaned
    infix the dual-output pipeline appends so the result matches the original
    track name (and the older transcripts already in the index)."""
    stem = raw_path.name
    for suffix in (".raw.json", ".cleaned.json", ".raw.txt", ".cleaned.txt"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return raw_path.stem


def _rel_parents(raw_path: Path, base_dir: Path | None) -> list[str]:
    """Folder names between base_dir and the file, with pipeline scaffolding
    (model-name / 'turbo' dirs) dropped and '_isolation'/'_model-' suffixes
    stripped — i.e. the canonical collection/event/session names. Empty when
    no base_dir is given (caller falls back to the bare track stem)."""
    if base_dir is None:
        return []
    try:
        parents = list(raw_path.relative_to(base_dir).parts)[:-1]
    except ValueError:
        # File not under base_dir: keep the nearest 3 parents for uniqueness.
        parents = list(raw_path.parts)[:-1][-3:]
    return [pp._strip_folder_suffix(x) for x in parents
            if not pp._is_pipeline_scaffolding(x)]


def qualified_source(raw_path: Path, base_dir: Path | None) -> str:
    """Globally-unique, human-readable identity key for a track, e.g.
    'Live Masters 2010/01 NOIDA 7 - 10 JAN 2010/7 JAN - 1$ - 6 PM/04 PRAVACHAN.json'.

    This becomes the chunk ``source_file`` — the join key for Qdrant point IDs,
    Tantivy, and the Postgres file_meta/chunk_meta tables — so two tracks that
    share a bare stem (e.g. '06 SAMBODHAN' in different sessions) no longer
    collide. With no base_dir it degrades to the bare '<stem>.json'."""
    parts = _rel_parents(raw_path, base_dir) + [normalized_stem(raw_path)]
    return "/".join(parts) + ".json"


def output_basename(raw_path: Path, base_dir: Path | None) -> str:
    """Flat, filesystem-safe output filename derived from qualified_source.
    bulk_ingest_hardened.py globs the output dir non-recursively, so names must
    be flat and unique. Path separators -> '__'; over-long names get a short
    hash suffix to stay under filesystem limits while remaining unique."""
    key = "/".join(_rel_parents(raw_path, base_dir) + [normalized_stem(raw_path)])
    safe = re.sub(r'[\\/:*?"<>|]+', "__", key)
    if len(safe) > 180:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[:170]}__{digest}"
    return f"{safe}.chunks.json"


# The LLM cleanup pass is instructed to normalize spelling and cut nothing
# ("DO NOT cut any content" — CLEAN_SYSTEM_PROMPT rule 5). On 877 of 7,413
# tracks it summarized instead, and the summary was indexed as if it were a
# verbatim transcript: one 2,689-word discourse survives as 38 words. The
# upstream runaway detector only trips on *inflation*, so nothing caught it.
#
# Legitimate cleaning trims filler — the median track keeps 99% of its words
# and the 25th percentile keeps 95%. Below 85% the text is being summarized,
# not cleaned. Falling back to the raw ASR text costs spelling normalization;
# accepting a summary costs content that no query can ever retrieve. Prefer
# the noisy-but-complete text.
MIN_CLEANED_WORD_RATIO = 0.85

# ...except when the raw text is a Whisper repetition loop ("ॐ ॐ ॐ ॐ …"),
# where a drastically shorter cleaned text is the correct output. Normal Hindi
# prose repeats far less: the 1st percentile of distinct/total words across the
# corpus is 0.079, while every degenerate track sits below 0.10.
DEGENERATE_RAW_DISTINCT_RATIO = 0.10

# The guard above is one-sided, and the cleaner fails in *both* directions. On
# 453 of 7,040 raw/cleaned pairs the cleaned text is *longer* than the raw ASR,
# up to 3.913x — it fabricated content rather than dropping it (4,010 injected
# Latin glosses like "(Third Eye)"; amplified repetition loops). Measured over
# those pairs (raw >= 50 words): p75=1.000, p90=1.010, p95=1.088. Legitimate
# cleaning essentially never adds words, so 1.15 is symmetric with the 0.85
# floor and fires on 268 files (3.8%). Accepting fabricated text costs content
# that no query should ever retrieve; falling back to raw only costs spelling
# normalization. Prefer raw, same trade as the low side.
MAX_CLEANED_WORD_RATIO = 1.15

# Below this the ratio is dominated by rounding — a 12-word invocation losing
# two filler words is not a summarization event.
MIN_RAW_WORDS_FOR_RATIO_CHECK = 50


def _is_degenerate(words: list[str]) -> bool:
    """True when the text is a Whisper repetition loop rather than prose."""
    return bool(words) and len(set(words)) / len(words) < DEGENERATE_RAW_DISTINCT_RATIO


def _cleaned_is_unusable(cleaned: str, raw_text: str, name: str) -> bool:
    """True when the cleanup LLM dropped or invented content instead of
    normalizing it. Logs the reason with the file name — never silently rejects.

    The expansion check runs *before* the degenerate-raw exemption on purpose:
    when raw is a repetition loop and cleaned is several times longer, the
    cleaner amplified the loop rather than condensing it, and raw is the honest
    text. Downstream `chunker_json._emit_guarded` then drops it as an ASR loop,
    which is the right outcome for a track with no retrievable speech.
    """
    raw_words = raw_text.split()
    if len(raw_words) < MIN_RAW_WORDS_FOR_RATIO_CHECK:
        return False
    ratio = len(cleaned.split()) / len(raw_words)
    if ratio > MAX_CLEANED_WORD_RATIO:
        log.warning("%s: cleaned text is %.0f%% of raw (%d -> %d words) - LLM "
                    "invented content instead of cleaning; falling back to raw text",
                    name, ratio * 100, len(raw_words), len(cleaned.split()))
        return True
    if ratio >= MIN_CLEANED_WORD_RATIO:
        return False
    if _is_degenerate(raw_words):
        log.info("%s: cleaned text is %.0f%% of raw, but raw is a repetition "
                 "loop - keeping cleaned", name, ratio * 100)
        return False
    log.warning("%s: cleaned text is %.0f%% of raw (%d -> %d words) - LLM "
                "summarized instead of cleaning; falling back to raw text",
                name, ratio * 100, len(raw_words), len(cleaned.split()))
    return True


def load_cleaned_text(raw_path: Path, raw_text: str | None = None) -> str | None:
    """Find the cleaned transcript for a raw.json. Prefers the sibling
    .cleaned.txt; falls back to cleaned.json's 'cleaned_text'.

    Returns None when no usable cleaned text exists — either because the files
    are absent, or because the cleanup LLM summarized the transcript rather
    than cleaning it (see MIN_CLEANED_WORD_RATIO). The caller then chunks the
    raw text, which is lossy in spelling but complete in content.

    `raw_text` is the raw ASR transcript. Omit it to skip the loss check.
    """
    stem = normalized_stem(raw_path)
    cleaned: str | None = None

    txt = raw_path.with_name(f"{stem}.cleaned.txt")
    if txt.exists():
        cleaned = txt.read_text(encoding="utf-8")
    else:
        cj_json = raw_path.with_name(f"{stem}.cleaned.json")
        if cj_json.exists():
            data = json.loads(cj_json.read_text(encoding="utf-8"))
            text = data.get("cleaned_text")
            if isinstance(text, str) and text.strip():
                cleaned = text
            else:
                log.warning("%s: cleaned.json has no usable 'cleaned_text'", cj_json.name)

    if cleaned is None or not cleaned.strip():
        return None
    if raw_text and _cleaned_is_unusable(cleaned, raw_text, raw_path.name):
        return None
    return cleaned


def process_file(
    raw_path: Path,
    out_dir: Path,
    failed_dir: Path,
    base_dir: Path | None = None,
    skip_existing: bool = True,
) -> tuple[int, str]:
    """Returns (n_chunks, status) with status in {'ok','skipped','failed'}."""
    size = raw_path.stat().st_size
    if size > cj.MAX_FILE_BYTES:
        log.warning("%s: %.1f MB exceeds cap — skipping", raw_path, size / 1e6)
        return 0, "skipped"

    track_stem = normalized_stem(raw_path)
    source_name = qualified_source(raw_path, base_dir)
    out_path = out_dir / output_basename(raw_path, base_dir)
    if skip_existing and out_path.exists():
        log.info("%s: output already exists (%s) — skipping", raw_path.name, out_path.name)
        return 0, "skipped"

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with raw_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        segs = cj.normalize_segments(data, FMT)

        raw_text = data.get("text") or " ".join(s.get("text", "") for s in segs)
        cleaned = load_cleaned_text(raw_path, raw_text)
        if cleaned is None:
            log.warning("%s: no usable cleaned text - chunking raw text instead",
                        raw_path.name)
            aligned = segs
        else:
            aligned = align_cleaned_to_segments(segs, cleaned)

        # Parse path metadata against the normalized track name so the track
        # title isn't polluted by the '.raw' infix (parse_path keys off the
        # filename for the track level). Uses the bare track stem, not the
        # slash-bearing qualified source_name.
        synth_path = raw_path.with_name(f"{track_stem}.json")
        path_meta = cj._path_meta_for(synth_path, base_dir)

        chunks = cj.chunk_segments(aligned, source_name, FMT, path_meta)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"source_file": source_name, "format": FMT, "chunks": chunks},
                      f, ensure_ascii=False, indent=2)
        log.info("%s -> %s (%d chunks)", raw_path.name, out_path.name, len(chunks))
        return len(chunks), "ok"
    except Exception as e:
        failed_dir.mkdir(parents=True, exist_ok=True)
        err_path = failed_dir / f"{raw_path.name}.error.txt"
        with err_path.open("w", encoding="utf-8") as f:
            f.write(f"file: {raw_path}\nreason: {type(e).__name__}: {e}\n\n")
            f.write(traceback.format_exc())
        log.error("%s: %s: %s (logged to %s)", raw_path.name, type(e).__name__, e,
                  err_path)
        return 0, "failed"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Chunk WhisperX cleaned transcripts (cleaned body + raw timestamps)."
    )
    p.add_argument("input_dir", help="Directory of *.raw.json files (use -r for nesting).")
    p.add_argument("output_dir", help="Directory for .chunks.json output.")
    p.add_argument("--base-dir", default=None,
                   help="Root above the audio folder hierarchy for Phase 12 path "
                        "metadata. Falls back to RAW_TRANSCRIPTS_BASE_DIR; omit both to skip.")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="Walk input_dir recursively (needed for the nested layout).")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-chunk even when .chunks.json output already exists.")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = out_dir / "_failed"
    base_dir = Path(args.base_dir) if args.base_dir else None

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        log.error("input_dir does not exist or is not a directory: %s", in_dir)
        return 2

    pattern = "*.raw.json"
    files = sorted(in_dir.rglob(pattern) if args.recursive else in_dir.glob(pattern))
    skip_existing = not args.no_skip_existing

    # Incremental scan: classify discovered tracks as already-processed (an
    # output .chunks.json exists) vs new, so a re-run over a folder only does
    # the new work. The skip is keyed off output_basename(), which is derived
    # from --base-dir; pass the SAME --base-dir every run or the qualified key
    # shifts and previously-processed tracks look "new" (and would duplicate).
    if skip_existing:
        already = sum(1 for f in files if (out_dir / output_basename(f, base_dir)).exists())
        new = len(files) - already
        log.info("incremental scan: %d tracks found — %d already processed (skip), "
                 "%d new to process", len(files), already, new)
        if base_dir is None:
            log.warning("no --base-dir: skip uses bare stems; cross-session stem "
                        "collisions are possible. Pass --base-dir for stable keys.")

    totals = {"files": 0, "chunks": 0, "failed": 0, "skipped": 0}
    for f in files:
        totals["files"] += 1
        n, status = process_file(f, out_dir, failed_dir,
                                 base_dir=base_dir, skip_existing=skip_existing)
        totals["chunks"] += n
        if status == "failed":
            totals["failed"] += 1
        elif status == "skipped":
            totals["skipped"] += 1

    processed_new = totals["files"] - totals["skipped"] - totals["failed"]
    print(
        f"\nSummary: {totals['files']} tracks found | {processed_new} processed (new) "
        f"-> {totals['chunks']} chunks | {totals['skipped']} skipped (already processed) "
        f"| {totals['failed']} failed"
    )
    return 0 if totals["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
