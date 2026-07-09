"""§0.3 — Chunk-quality audit over the full `chunk_meta` table.

Full scan of every stored chunk, reporting two independent defects:

1. **Degenerate ASR loops** — a chunk whose unique-word ratio is below
   ``LOOP_UNIQ_RATIO`` (0.15). These are upstream Whisper failures (the same
   phrase repeated for tens of thousands of words); embedding them is pure noise
   and, worse, poisons §2.3 enrichment, which reconstructs each file's transcript
   from ``chunk_meta.text``. A minimum word count guards against flagging a
   legitimately short, naturally repetitive chunk.

2. **Oversize chunks** — real token count above ``MAX_TOKENS`` (8192, bge-m3's
   context limit). Token count uses the **real bge-m3 tokenizer**
   (``BAAI/bge-m3``, the same XLM-RoBERTa SentencePiece Ollama serves), NOT
   ``words x 1.3`` — that heuristic under-counts Devanagari badly (a 7-word Hindi
   phrase is 16 real tokens, not 9).

Oversize-but-healthy chunks are "legit-long" (they need subdividing in §1.2);
oversize-and-degenerate chunks are loops (they get dropped). The emitted
work-list — every ``source_file`` with >=1 flagged chunk — is what §1.4 re-chunks.

This is a **read-only** audit. It never writes to any store.

Usage:
    python -m ingestion.audit_chunks [--out ingestion/chunk_audit_worklist.json]
        [--loop-ratio 0.15] [--max-tokens 8192] [--no-tokens]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import psycopg2

_REPO_ROOT = Path(__file__).resolve().parents[1]

# --- defect thresholds ------------------------------------------------------
LOOP_UNIQ_RATIO = 0.15      # unique_words / total_words below this => ASR loop
MIN_LOOP_WORDS = 50         # don't flag naturally-repetitive short chunks
MAX_TOKENS = 8192           # bge-m3 context limit
# Only tokenize chunks longer than this many chars; shorter chunks cannot reach
# MAX_TOKENS tokens even at bge-m3's densest observed Devanagari packing, so
# skipping them makes the scan fast without missing an oversize chunk.
CHAR_TOKEN_SCAN_THRESHOLD = 2000
OVERSIZE_CHARS = 6000       # reported for continuity with the plan's framing

TOKENIZER_NAME = "BAAI/bge-m3"

def resolve_dsn() -> str:
    """Build the Postgres DSN from the environment. Call *after* `_load_dotenv`
    so POSTGRES_PASSWORD is populated when running from a host shell."""
    return os.environ.get("PG_DSN") or (
        "postgresql://owui:{}@localhost:5432/openwebui".format(
            os.environ.get("POSTGRES_PASSWORD", "")
        )
    )


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("audit_chunks")


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from `.env` for keys not already set, so the audit
    runs from a host shell (where docker hasn't injected POSTGRES_PASSWORD)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def uniq_word_ratio(text: str) -> tuple[int, float]:
    """(word_count, unique/total). Empty text => (0, 1.0) so it never flags."""
    words = text.split()
    if not words:
        return 0, 1.0
    return len(words), len(set(words)) / len(words)


def load_tokenizer() -> Any | None:
    """The real bge-m3 tokenizer, offline. Returns None (with a loud warning) if
    it can't load — the audit still emits the loop work-list, just without a real
    token count. Never silently substitute words x 1.3."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as e:  # noqa: BLE001 - report the reason, do not hide it
        log.warning("could not load %s tokenizer (%s: %s) — token counts "
                    "disabled; oversize detection will be skipped",
                    TOKENIZER_NAME, type(e).__name__, e)
        return None


def audit(dsn: str, tokenizer: Any | None, loop_ratio: float,
          max_tokens: int) -> dict[str, Any]:
    conn = psycopg2.connect(dsn, connect_timeout=10)
    try:
        cur = conn.cursor(name="audit_scan")  # server-side; stream, don't buffer
        cur.itersize = 500
        cur.execute("SELECT chunk_id, source_file, start_sec, text FROM chunk_meta")

        total = 0
        flagged: list[dict[str, Any]] = []
        per_file: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"loop_chunks": 0, "over_char_chunks": 0, "over_token_chunks": 0,
                     "max_chars": 0, "max_tokens": 0, "min_uniq_ratio": 1.0})
        worst = {"chars": 0, "tokens": 0, "min_ratio": 1.0,
                 "worst_ratio_words": 0, "worst_ratio_uniq": 0}
        # cross-tabs so the counts map onto the plan's framing
        n_loop = n_over_char = n_over_token = 0
        n_loop_and_over_char = n_legit_long_char = n_legit_long_token = 0

        for chunk_id, source_file, start_sec, text in cur:
            total += 1
            text = text or ""
            char_len = len(text)
            wc, ratio = uniq_word_ratio(text)

            tokens: int | None = None
            if tokenizer is not None and char_len > CHAR_TOKEN_SCAN_THRESHOLD:
                # add_special_tokens=False -> count content tokens only
                tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])

            is_loop = ratio < loop_ratio and wc >= MIN_LOOP_WORDS
            over_char = char_len > OVERSIZE_CHARS
            over_token = tokens is not None and tokens > max_tokens

            pf = per_file[source_file]
            pf["max_chars"] = max(pf["max_chars"], char_len)
            if tokens is not None:
                pf["max_tokens"] = max(pf["max_tokens"], tokens)
            if wc >= MIN_LOOP_WORDS:
                pf["min_uniq_ratio"] = min(pf["min_uniq_ratio"], ratio)

            if char_len > worst["chars"]:
                worst["chars"] = char_len
            if tokens is not None and tokens > worst["tokens"]:
                worst["tokens"] = tokens
            if wc >= MIN_LOOP_WORDS and ratio < worst["min_ratio"]:
                worst.update(min_ratio=ratio, worst_ratio_words=wc,
                             worst_ratio_uniq=int(round(ratio * wc)))

            n_loop += is_loop
            n_over_char += over_char
            n_over_token += over_token
            n_loop_and_over_char += is_loop and over_char
            n_legit_long_char += over_char and not is_loop      # plan's "legit-long"
            n_legit_long_token += over_token and not is_loop

            if is_loop or over_char or over_token:
                defects = ([d for d, on in
                            (("loop", is_loop), ("over6kchar", over_char),
                             ("over8192tok", over_token)) if on])
                pf["loop_chunks"] += is_loop
                pf["over_char_chunks"] += over_char
                pf["over_token_chunks"] += over_token
                flagged.append({
                    "chunk_id": str(chunk_id), "source_file": source_file,
                    "start_sec": start_sec, "chars": char_len, "words": wc,
                    "uniq_ratio": round(ratio, 4), "tokens": tokens,
                    "reason": "+".join(defects),
                    "head": text[:200].replace("\n", " "),
                })
        cur.close()

        worklist = sorted(
            source for source, s in per_file.items()
            if s["loop_chunks"] or s["over_char_chunks"] or s["over_token_chunks"]
        )
        return {
            "total_chunks": total,
            "total_files": len(per_file),
            # corpus-wide (the rule §1.2 actually applies everywhere)
            "loop_chunk_count": n_loop,
            "over_char_count": n_over_char,          # chunks > OVERSIZE_CHARS
            "over_token_count": n_over_token,        # chunks > max_tokens (real bge-m3)
            # cross-tabs to reconcile with the plan's oversize-scoped numbers
            "loops_among_oversize_char": n_loop_and_over_char,
            "legit_long_char": n_legit_long_char,    # plan's "14 legit-long" (chars>6k, healthy)
            "legit_long_token": n_legit_long_token,  # healthy chunks over the real 8192 limit
            "flagged_files": len(worklist),
            "worst": worst,
            "params": {"loop_ratio": loop_ratio, "min_loop_words": MIN_LOOP_WORDS,
                       "max_tokens": max_tokens, "oversize_chars": OVERSIZE_CHARS,
                       "tokenizer": TOKENIZER_NAME if tokenizer is not None else None},
            "worklist": [{"source_file": s, **per_file[s]} for s in worklist],
            "flagged_chunks": sorted(flagged, key=lambda f: f["chars"], reverse=True),
        }
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Full chunk_meta quality audit (§0.3).")
    p.add_argument("--out", default=str(_REPO_ROOT / "ingestion" / "chunk_audit_worklist.json"))
    p.add_argument("--loop-ratio", type=float, default=LOOP_UNIQ_RATIO)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--no-tokens", action="store_true",
                   help="Skip real-token counting (loop detection only).")
    args = p.parse_args(argv)

    _load_dotenv(_REPO_ROOT / ".env")
    dsn = resolve_dsn()
    tokenizer = None if args.no_tokens else load_tokenizer()
    if tokenizer is not None:
        log.info("tokenizer: %s (real bge-m3 tokens)", TOKENIZER_NAME)

    log.info("scanning chunk_meta ...")
    report = audit(dsn, tokenizer, args.loop_ratio, args.max_tokens)

    w = report["worst"]
    log.info("total chunks=%d files=%d", report["total_chunks"], report["total_files"])
    log.info("ASR-loop chunks (uniq-ratio < %.2f, >=%d words): %d  "
             "(of which > %d chars: %d)", args.loop_ratio, MIN_LOOP_WORDS,
             report["loop_chunk_count"], OVERSIZE_CHARS,
             report["loops_among_oversize_char"])
    log.info("chunks > %d chars: %d  (legit-long, non-loop: %d)",
             OVERSIZE_CHARS, report["over_char_count"], report["legit_long_char"])
    log.info("chunks > %d real tokens: %d  (legit-long, non-loop: %d)",
             args.max_tokens, report["over_token_count"], report["legit_long_token"])
    log.info("FILES needing re-chunk (§1.4 work-list): %d", report["flagged_files"])
    log.info("worst chunk: %d chars, %d real tokens; worst uniq-ratio %.4f "
             "(%d words / %d unique)", w["chars"], w["tokens"], w["min_ratio"],
             w["worst_ratio_words"], w["worst_ratio_uniq"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("work-list + flagged-chunk detail: %s", args.out)

    print("\nTop 10 flagged chunks (by chars):")
    print(f"{'chars':>8} {'tokens':>7} {'ratio':>7}  reason        source_file")
    for f in report["flagged_chunks"][:10]:
        print(f"{f['chars']:>8} {str(f['tokens']):>7} {f['uniq_ratio']:>7}  "
              f"{f['reason']:<13} {f['source_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
