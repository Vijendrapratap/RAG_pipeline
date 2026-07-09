"""Chunk whisperX / whisper JSON transcripts into retrieval-ready segments.

Per PRD §6 Phase 3:
- TARGET_TOKENS=450, MAX_TOKENS=700, MIN_TOKENS=200
- Two input formats: 'whisperx' (with diarization) and 'whisper' (no speakers)
- Header prepended to each chunk:
    [Source: <file> | HH:MM:SS -> HH:MM:SS | Speakers: <list>]
- Output: one <stem>.chunks.json per input
- Failures logged to <OUTPUT_DIR>/_failed/<file>.error.txt; pipeline continues
- Files > MAX_FILE_BYTES skipped with a warning
- Reads from stdin when no INPUT_DIR is supplied (single-file streaming mode)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

from ingestion.chunker_text import split_sentences
from ingestion.utils.path_parser import PRIMARY_SPEAKER, PathMetadata, parse_path

TARGET_TOKENS = 450
MAX_TOKENS = 700
MIN_TOKENS = 200
MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB

# Chunk-quality guards (Stage 1.2). An ASR loop is Whisper repeating one phrase
# for thousands of words; its unique-word ratio collapses. Drop such chunks (they
# are noise and poison enrichment), never silently. A minimum word count keeps a
# legitimately short, naturally-repetitive chunk from being flagged.
LOOP_UNIQ_RATIO = 0.15
MIN_LOOP_WORDS = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("chunker_json")


def estimate_tokens(text: str) -> int:
    """Heuristic token count: words * 1.3 (English baseline).

    Tokenizer is not locked by PRD; this avoids a heavy transformers/tiktoken
    dep at chunk time. Conservative for non-Latin scripts (produces slightly
    smaller chunks, which only helps retrieval quality).
    """
    words = len(text.split())
    return max(1, round(words * 1.3))


def fmt_hms(sec: float | None) -> str:
    if sec is None:
        return "??:??:??"
    s = int(sec)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def normalize_segments(data: Any, fmt: str) -> list[dict[str, Any]]:
    """Coerce whisperX/whisper JSON into a uniform list of segments.

    Each output segment has: {text: str, start: float|None, end: float|None,
    speaker: str|None}. Raises ValueError on unrecognized shape.
    """
    if not isinstance(data, dict) or "segments" not in data:
        raise ValueError("input JSON missing top-level 'segments' list")
    segs = data["segments"]
    if not isinstance(segs, list):
        raise ValueError("'segments' is not a list")

    out: list[dict[str, Any]] = []
    for i, s in enumerate(segs):
        if not isinstance(s, dict):
            raise ValueError(f"segment[{i}] is not an object")
        text = (s.get("text") or "").strip()
        if not text:
            continue
        start = s.get("start")
        end = s.get("end")
        speaker = s.get("speaker") if fmt == "whisperx" else None
        out.append(
            {
                "text": text,
                "start": float(start) if start is not None else None,
                "end": float(end) if end is not None else None,
                "speaker": str(speaker) if speaker else None,
            }
        )
    return out


def uniq_word_ratio(text: str) -> tuple[int, float]:
    """(word_count, unique/total). Empty text => (0, 1.0) so it never flags."""
    words = text.split()
    if not words:
        return 0, 1.0
    return len(words), len(set(words)) / len(words)


def _render_chunk(
    body: str,
    buf: list[dict[str, Any]],
    source_name: str,
    fmt: str,
    path_meta: PathMetadata | None,
) -> dict[str, Any]:
    """Build one chunk dict for `body`, deriving timing/speakers from `buf`."""
    starts = [s["start"] for s in buf if s["start"] is not None]
    ends = [s["end"] for s in buf if s["end"] is not None]
    speakers = sorted({s["speaker"] for s in buf if s["speaker"]})
    # Per PRD §6 Phase 12: every file in this corpus is Swami ji's voice.
    # If diarization produced no labels, fall back to PRIMARY_SPEAKER so
    # speaker-based filters still match.
    if not speakers and path_meta is not None:
        speakers = [PRIMARY_SPEAKER]
    start_sec = min(starts) if starts else None
    end_sec = max(ends) if ends else None
    speakers_label = ", ".join(speakers) if speakers else "unknown"
    base = (
        f"[Source: {source_name} | {fmt_hms(start_sec)} -> {fmt_hms(end_sec)} "
        f"| Speakers: {speakers_label}]"
    )
    extra = path_meta.header_fragment() if path_meta else ""
    header = f"[{extra}]\n{base}" if extra else base
    full_text = f"{header}\n{body}"
    chunk: dict[str, Any] = {
        "text": full_text,
        "source_file": source_name,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "speakers": speakers,
        "format": fmt,
        "has_timestamps": True,
        "word_count": len(body.split()),
    }
    if path_meta is not None:
        chunk["path_metadata"] = path_meta.to_payload()
    return chunk


def _subdivide_body(body: str, source_name: str) -> list[str]:
    """Split an oversize body into <= MAX_TOKENS pieces on sentence boundaries
    (danda-aware via ``split_sentences``). Raises ``ValueError`` if a single
    sentence still exceeds MAX_TOKENS — never truncate silently; the file
    dead-letters loudly instead (caught by ``process_file``)."""
    pieces: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for sent in split_sentences(body):
        s_tok = estimate_tokens(sent)
        if s_tok > MAX_TOKENS:
            raise ValueError(
                f"{source_name}: a single sentence of ~{s_tok} est-tokens "
                f"exceeds MAX_TOKENS={MAX_TOKENS} and cannot be split on "
                f"sentence boundaries (likely no punctuation / an ASR artifact)"
            )
        if cur and cur_tok + s_tok > MAX_TOKENS:
            pieces.append(" ".join(cur))
            cur, cur_tok = [], 0
        cur.append(sent)
        cur_tok += s_tok
    if cur:
        pieces.append(" ".join(cur))
    return pieces or [body]


def _emit_guarded(
    chunks: list[dict[str, Any]],
    buf: list[dict[str, Any]],
    source_name: str,
    fmt: str,
    path_meta: PathMetadata | None,
    dead_letters: list[dict[str, Any]] | None,
) -> None:
    """Materialize guarded chunk(s) from `buf` and append to `chunks`:

    * **Degenerate ASR loop** — drop a chunk whose unique-word ratio is below
      ``LOOP_UNIQ_RATIO`` (with >= ``MIN_LOOP_WORDS`` words) and record a
      chunk-level dead-letter. Never silently.
    * **Oversize** — subdivide a chunk exceeding ``MAX_TOKENS`` into sentence-
      packed pieces (each <= MAX_TOKENS), so no chunk overruns bge-m3's context.
    """
    if not buf:
        return
    body = " ".join(s["text"] for s in buf).strip()
    if not body:
        return
    wc, ratio = uniq_word_ratio(body)
    if wc >= MIN_LOOP_WORDS and ratio < LOOP_UNIQ_RATIO:
        start = next((s["start"] for s in buf if s["start"] is not None), None)
        log.warning(
            "%s: dropping degenerate ASR-loop chunk (start=%s uniq_ratio=%.4f "
            "words=%d) — not embedded", source_name, start, ratio, wc)
        if dead_letters is not None:
            dead_letters.append({
                "source_file": source_name, "start_sec": start,
                "reason": f"asr_loop uniq_ratio={round(ratio, 4)} words={wc}",
                "head": body[:200].replace("\n", " "),
            })
        return
    if estimate_tokens(body) > MAX_TOKENS:
        for piece in _subdivide_body(body, source_name):
            chunks.append(_render_chunk(piece, buf, source_name, fmt, path_meta))
        return
    chunks.append(_render_chunk(body, buf, source_name, fmt, path_meta))


def chunk_segments(
    segments: Iterable[dict[str, Any]],
    source_name: str,
    fmt: str,
    path_meta: PathMetadata | None = None,
    dead_letters: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Walk segments, flushing on size/speaker boundaries per PRD spec.

    Every emitted chunk passes through ``_emit_guarded`` (Stage 1.2): degenerate
    ASR-loop chunks are dropped to ``dead_letters`` (never silently) and oversize
    chunks are subdivided to <= MAX_TOKENS. Pass a list for ``dead_letters`` to
    capture dropped-chunk records; omit it and drops are still logged.
    """
    chunks: list[dict[str, Any]] = []
    buf: list[dict[str, Any]] = []
    buf_tokens = 0
    buf_speakers: set[str] = set()

    for seg in segments:
        seg_tokens = estimate_tokens(seg["text"])
        spk = seg["speaker"]

        # Hard cap: flush before adding if this segment would push past MAX
        # and we already have at least MIN tokens accumulated.
        if buf_tokens + seg_tokens > MAX_TOKENS and buf_tokens >= MIN_TOKENS:
            _emit_guarded(chunks, buf, source_name, fmt, path_meta, dead_letters)
            buf = []
            buf_tokens = 0
            buf_speakers = set()

        # Soft boundary: at/past TARGET and the speaker changed -> flush.
        # "Speaker change" here means new speaker that wasn't in the current
        # buffer; protects against mixing distinct speakers across the
        # TARGET threshold once a clean breakpoint is available.
        elif (
            buf_tokens >= TARGET_TOKENS
            and spk
            and buf_speakers
            and spk not in buf_speakers
        ):
            _emit_guarded(chunks, buf, source_name, fmt, path_meta, dead_letters)
            buf = []
            buf_tokens = 0
            buf_speakers = set()

        buf.append(seg)
        buf_tokens += seg_tokens
        if spk:
            buf_speakers.add(spk)

        # Pathological single segment > MAX_TOKENS: emit on its own (the guard
        # subdivides it, or drops it if it is a degenerate loop).
        if buf_tokens >= MAX_TOKENS:
            _emit_guarded(chunks, buf, source_name, fmt, path_meta, dead_letters)
            buf = []
            buf_tokens = 0
            buf_speakers = set()

    # Tail flush — emit whatever remains even if below MIN; better to keep
    # the trailing content than drop it. Must propagate path_meta or short
    # single-chunk files would lose their event/date/track metadata.
    _emit_guarded(chunks, buf, source_name, fmt, path_meta, dead_letters)
    return chunks


def _path_meta_for(in_path: Path, base_dir: Path | None) -> PathMetadata | None:
    """Wrap parse_path with a try/except so a parser bug never kills the
    chunker. Logs warnings collected during parsing per CLAUDE.md rule 6."""
    if base_dir is None and not os.environ.get("RAW_TRANSCRIPTS_BASE_DIR"):
        return None
    bd = base_dir or Path(os.environ["RAW_TRANSCRIPTS_BASE_DIR"])
    try:
        meta = parse_path(in_path, base_dir=bd)
    except Exception as e:  # pragma: no cover — parser contract is no-raise
        log.error("path_parser raised on %s: %s", in_path, e)
        return None
    for w in meta.parse_warnings:
        log.warning("path_parse %s: %s", in_path.name, w)
    return meta


def process_file(
    in_path: Path,
    out_dir: Path,
    fmt: str,
    failed_dir: Path,
    base_dir: Path | None = None,
    skip_existing: bool = True,
) -> tuple[int, str]:
    """Returns (n_chunks, status). status in {'ok','skipped','failed'}."""
    size = in_path.stat().st_size
    if size > MAX_FILE_BYTES:
        log.warning("%s: %.1f MB exceeds %d MB cap — skipping", in_path, size / 1e6,
                    MAX_FILE_BYTES // (1024 * 1024))
        return 0, "skipped"
    out_path = out_dir / f"{in_path.stem}.chunks.json"
    if skip_existing and out_path.exists():
        log.info("%s: output already exists (%s) — skipping", in_path.name, out_path.name)
        return 0, "skipped"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        segs = normalize_segments(data, fmt)
        path_meta = _path_meta_for(in_path, base_dir)
        dead_letters: list[dict[str, Any]] = []
        chunks = chunk_segments(segs, in_path.name, fmt, path_meta, dead_letters)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"source_file": in_path.name, "format": fmt, "chunks": chunks},
                      f, ensure_ascii=False, indent=2)
        if dead_letters:
            failed_dir.mkdir(parents=True, exist_ok=True)
            dl_path = failed_dir / f"{in_path.name}.dropped_chunks.json"
            dl_path.write_text(json.dumps(dead_letters, ensure_ascii=False, indent=2),
                               encoding="utf-8")
            log.warning("%s: dropped %d degenerate ASR-loop chunk(s) -> %s",
                        in_path.name, len(dead_letters), dl_path.name)
        log.info("%s -> %s (%d chunks)", in_path.name, out_path.name, len(chunks))
        return len(chunks), "ok"
    except Exception as e:
        failed_dir.mkdir(parents=True, exist_ok=True)
        err_path = failed_dir / f"{in_path.name}.error.txt"
        with err_path.open("w", encoding="utf-8") as f:
            f.write(f"file: {in_path}\nreason: {type(e).__name__}: {e}\n\n")
            f.write(traceback.format_exc())
        log.error("%s: %s: %s (logged to %s)", in_path.name, type(e).__name__, e,
                  err_path)
        return 0, "failed"


def run_stdin(out_dir: Path, fmt: str, failed_dir: Path) -> tuple[int, str]:
    try:
        data = json.load(sys.stdin)
        segs = normalize_segments(data, fmt)
        chunks = chunk_segments(segs, "stdin", fmt)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "stdin.chunks.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"source_file": "stdin", "format": fmt, "chunks": chunks},
                      f, ensure_ascii=False, indent=2)
        log.info("stdin -> %s (%d chunks)", out_path, len(chunks))
        return len(chunks), "ok"
    except Exception as e:
        failed_dir.mkdir(parents=True, exist_ok=True)
        err_path = failed_dir / "stdin.error.txt"
        with err_path.open("w", encoding="utf-8") as f:
            f.write(f"reason: {type(e).__name__}: {e}\n\n")
            f.write(traceback.format_exc())
        log.error("stdin: %s: %s (logged to %s)", type(e).__name__, e, err_path)
        return 0, "failed"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Chunk whisperX/whisper JSON transcripts.")
    p.add_argument("input_dir", nargs="?", default=None,
                   help="Directory of input .json files. Omit to read one JSON from stdin.")
    p.add_argument("output_dir", help="Directory for .chunks.json output.")
    p.add_argument("--format", choices=("whisperx", "whisper"), default="whisperx",
                   help="Input format. whisperx has speaker diarization; whisper does not.")
    p.add_argument("--base-dir", default=None,
                   help="Root above the audio folder hierarchy. When set, "
                        "every file's path is parsed for metadata (event, date, "
                        "track type, season) per PRD §6 Phase 12. Falls back to "
                        "RAW_TRANSCRIPTS_BASE_DIR env var; omit both to skip.")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="Walk input_dir recursively (needed for the nested "
                        "audio folder layout that Phase 12 parses).")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-chunk files even when .chunks.json output already "
                        "exists. Default: skip existing outputs (Phase 13). "
                        "Use this to force a re-chunk after tuning parameters.")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = out_dir / "_failed"
    base_dir = Path(args.base_dir) if args.base_dir else None

    totals = {"files": 0, "chunks": 0, "failed": 0, "skipped": 0}

    if args.input_dir is None:
        n, status = run_stdin(out_dir, args.format, failed_dir)
        totals["files"] = 1
        totals["chunks"] = n
        totals[status] = totals.get(status, 0) + (1 if status != "ok" else 0)
    else:
        in_dir = Path(args.input_dir)
        if not in_dir.is_dir():
            log.error("input_dir does not exist or is not a directory: %s", in_dir)
            return 2
        files = sorted(in_dir.rglob("*.json") if args.recursive else in_dir.glob("*.json"))
        skip_existing = not args.no_skip_existing
        for f in files:
            totals["files"] += 1
            n, status = process_file(f, out_dir, args.format, failed_dir,
                                     base_dir=base_dir,
                                     skip_existing=skip_existing)
            totals["chunks"] += n
            if status == "failed":
                totals["failed"] += 1
            elif status == "skipped":
                totals["skipped"] += 1

    print(
        f"\nSummary: {totals['files']} files, {totals['chunks']} chunks, "
        f"{totals['failed']} failed, {totals['skipped']} skipped"
    )
    return 0 if totals["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
