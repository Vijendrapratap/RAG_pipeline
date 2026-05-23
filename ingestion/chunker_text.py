"""Chunk plain-text transcripts into retrieval-ready segments with overlap.

Per PRD §6 Phase 3:
- TARGET_TOKENS=450, MAX_TOKENS=700, OVERLAP_SENTENCES=2
- Sentence split via regex `(?<=[.!?])\\s+`
- Header prepended to each chunk:
    [Source: <file> | Approx position: sentences X-Y |
     NOTE: plain-text source, timestamps unavailable]
- Output: one <stem>.chunks.json per input
- Each chunk's metadata sets `has_timestamps: false`
- Failures logged to <OUTPUT_DIR>/_failed/<file>.error.txt; pipeline continues
- Files > MAX_FILE_BYTES skipped with a warning
- Reads from stdin when no INPUT_DIR is supplied
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path

from ingestion.utils.path_parser import PRIMARY_SPEAKER, PathMetadata, parse_path

TARGET_TOKENS = 450
MAX_TOKENS = 700
OVERLAP_SENTENCES = 2
MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("chunker_text")


def estimate_tokens(text: str) -> int:
    """Heuristic token count: words * 1.3 (English baseline). See chunker_json."""
    words = len(text.split())
    return max(1, round(words * 1.3))


def read_text(path: Path) -> str:
    """Try utf-8, utf-8-sig, latin-1 in order. Phase 4 will replace this with
    the shared utils/encoding.py helper."""
    last: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError as e:
            last = e
            continue
    raise last if last else UnicodeDecodeError("?", b"", 0, 0, "all encodings failed")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text) if s and s.strip()]
    return parts


def _build_chunk(
    sentences: list[str],
    idx_start: int,
    source_name: str,
    path_meta: PathMetadata | None = None,
) -> dict[str, object]:
    body = " ".join(sentences).strip()
    idx_end = idx_start + len(sentences) - 1
    base = (
        f"[Source: {source_name} | Approx position: sentences "
        f"{idx_start}-{idx_end} | "
        f"NOTE: plain-text source, timestamps unavailable]"
    )
    # Embedding model sees the header verbatim — prepending parsed path
    # metadata (event/date/track/season) is a free precision boost for
    # filter-aware semantic queries. Phase 12.
    extra = path_meta.header_fragment() if path_meta else ""
    header = f"[{extra}]\n{base}" if extra else base
    full_text = f"{header}\n{body}"

    # speakers defaults to [PRIMARY_SPEAKER] when path metadata is present
    # so existing speaker filters work. Per PRD §6 Phase 12 / user
    # confirmation: every transcript in this corpus is Swami ji's voice.
    speakers = [PRIMARY_SPEAKER] if path_meta else []

    chunk: dict[str, object] = {
        "text": full_text,
        "source_file": source_name,
        "start_sec": None,
        "end_sec": None,
        "speakers": speakers,
        "format": "text",
        "has_timestamps": False,
        "word_count": len(body.split()),
        "sentence_start": idx_start,
        "sentence_end": idx_end,
    }
    if path_meta is not None:
        chunk["path_metadata"] = path_meta.to_payload()
    return chunk


def chunk_sentences(
    sentences: list[str],
    source_name: str,
    path_meta: PathMetadata | None = None,
) -> list[dict[str, object]]:
    """Walk sentences left to right, emitting chunks of ~TARGET tokens with
    2-sentence overlap. Hard cap at MAX (mid-sentence boundaries never split
    sentences; a single oversize sentence becomes its own chunk)."""
    if not sentences:
        return []

    chunks: list[dict[str, object]] = []
    i = 0
    n = len(sentences)
    while i < n:
        buf: list[str] = []
        buf_tokens = 0
        chunk_start = i
        while i < n:
            s = sentences[i]
            s_tok = estimate_tokens(s)
            # If buffer is non-empty and adding would exceed MAX, stop.
            if buf and buf_tokens + s_tok > MAX_TOKENS:
                break
            buf.append(s)
            buf_tokens += s_tok
            i += 1
            # Hit TARGET — close chunk and continue (caller handles overlap).
            if buf_tokens >= TARGET_TOKENS:
                break
        if not buf:
            break
        chunks.append(_build_chunk(buf, chunk_start, source_name, path_meta))
        # Overlap: rewind by OVERLAP_SENTENCES so the next chunk starts with
        # the last N sentences. Skip overlap if it would prevent forward
        # progress (oversize single sentence already consumed).
        if i < n:
            i = max(i - OVERLAP_SENTENCES, chunk_start + 1)
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
    failed_dir: Path,
    base_dir: Path | None = None,
    skip_existing: bool = True,
) -> tuple[int, str]:
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
        text = read_text(in_path)
        sentences = split_sentences(text)
        path_meta = _path_meta_for(in_path, base_dir)
        chunks = chunk_sentences(sentences, in_path.name, path_meta)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"source_file": in_path.name, "format": "text", "chunks": chunks},
                f, ensure_ascii=False, indent=2,
            )
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


def run_stdin(out_dir: Path, failed_dir: Path) -> tuple[int, str]:
    try:
        text = sys.stdin.read()
        sentences = split_sentences(text)
        chunks = chunk_sentences(sentences, "stdin")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "stdin.chunks.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"source_file": "stdin", "format": "text", "chunks": chunks},
                f, ensure_ascii=False, indent=2,
            )
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
    p = argparse.ArgumentParser(description="Chunk plain-text transcripts.")
    p.add_argument("input_dir", nargs="?", default=None,
                   help="Directory of input .txt files. Omit to read one document from stdin.")
    p.add_argument("output_dir", help="Directory for .chunks.json output.")
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
        n, status = run_stdin(out_dir, failed_dir)
        totals["files"] = 1
        totals["chunks"] = n
        totals[status] = totals.get(status, 0) + (1 if status != "ok" else 0)
    else:
        in_dir = Path(args.input_dir)
        if not in_dir.is_dir():
            log.error("input_dir does not exist or is not a directory: %s", in_dir)
            return 2
        files = sorted(in_dir.rglob("*.txt") if args.recursive else in_dir.glob("*.txt"))
        skip_existing = not args.no_skip_existing
        for f in files:
            totals["files"] += 1
            n, status = process_file(f, out_dir, failed_dir,
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
