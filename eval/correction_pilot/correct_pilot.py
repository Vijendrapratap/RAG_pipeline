#!/usr/bin/env python
"""Pilot: LLM correction of Whisper-generated Hindi transcripts via OpenRouter.

Reads ``<stem>.cleaned.json`` files and, per transcript, produces TWO corrected
variants of ``cleaned_text`` from a cloud model:

  * ``conservative`` - fix only clear ASR errors (garbled/misheard words,
    spelling, broken code-switched English) and collapse verbatim
    hallucination-loop repetitions. Word order and content are preserved, so
    ``ingestion.clean_align``'s word->timestamp mapping still holds.
  * ``aggressive``   - a fluent, readable Hindi rewrite. Better prose, but it
    reorders/merges words, so timestamp alignment no longer holds.

Originals (``.raw.json`` / ``.cleaned.json``) are NEVER modified. Output lands in
a separate pilot dir as ``<stem>__<hash>.corrected.json`` plus ``report.md``.

Pilot scope: the PRD's strictly-local posture is deliberately relaxed here -
transcript text leaves the machine to OpenRouter - per an explicit operator
decision (accept the cloud trade-off, pilot ~30 files, produce both variants).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_ROOT = Path(r"d:\Transcription whisperx\Output")

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)

log = logging.getLogger("correct_pilot")


# --------------------------------------------------------------------------- #
# env                                                                         #
# --------------------------------------------------------------------------- #
def load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file without overriding existing vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# --------------------------------------------------------------------------- #
# prompts                                                                      #
# --------------------------------------------------------------------------- #
CONSERVATIVE_SYSTEM = (
    "You are a meticulous transcript corrector for Hindi spiritual discourses "
    "(satsang / pravachan) transcribed by an automatic speech recognizer "
    "(Whisper). The text is mostly Hindi in Devanagari with occasional real "
    "English phrases the speaker actually spoke.\n\n"
    "Correct ONLY clear transcription errors:\n"
    "- Fix misheard or garbled Hindi words and obvious misspellings.\n"
    "- Fix garbled proper nouns and place names when the intended word is "
    "unambiguous from context (e.g. बंचकुला -> पंचकूला).\n"
    "- Repair garbled code-switched English words back to correct English.\n"
    "- Collapse a phrase that is repeated many times in a row (a Whisper "
    "hallucination loop on silence/music) down to a SINGLE occurrence.\n\n"
    "Hard constraints:\n"
    "- Do NOT paraphrase, summarize, translate, reorder, or change wording or "
    "style.\n"
    "- Do NOT add or remove content beyond the fixes above.\n"
    "- Preserve real English sentences the speaker actually said, unchanged.\n"
    "- If a garbled word's intended form is genuinely unrecoverable from "
    "context, LEAVE IT AS-IS rather than guessing. Never invent a word.\n"
    "- Output ONLY the corrected transcript text - no preamble, no notes, no "
    "markdown, no quotation marks around the whole thing."
)

AGGRESSIVE_SYSTEM = (
    "You are an expert Hindi editor. You are given an automatic (Whisper) "
    "transcript of a Hindi spiritual discourse (satsang / pravachan), mixed "
    "with some English. Produce a clean, fluent, readable Hindi rendering of "
    "the SAME content:\n"
    "- Fix all transcription errors, spelling, and garbled words.\n"
    "- Fix punctuation and sentence boundaries; produce well-formed sentences.\n"
    "- Preserve the full meaning and every idea faithfully; do not add "
    "opinions, commentary, or invented facts.\n"
    "- Keep genuine English quotes the speaker said (correct their spelling).\n"
    "- Do NOT summarize or condense - keep a full rendering of the content.\n"
    "- Output ONLY the edited transcript text - no preamble, no notes, no "
    "markdown."
)

_USER_TEMPLATE = "Correct this transcript excerpt:\n\n{chunk}"


# --------------------------------------------------------------------------- #
# chunking                                                                     #
# --------------------------------------------------------------------------- #
def chunk_paragraphs(text: str, max_chars: int) -> list[str]:
    """Group blank-line-separated paragraphs into chunks <= max_chars.

    Word/paragraph order is preserved. A single paragraph longer than
    ``max_chars`` becomes its own (oversized) chunk rather than being split
    mid-sentence.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paras:
        add = len(p) + (2 if cur else 0)
        if cur and cur_len + add > max_chars:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + (2 if len(cur) > 1 else 0)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# --------------------------------------------------------------------------- #
# OpenRouter                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost += other.cost


class OpenRouterError(RuntimeError):
    pass


class Corrector:
    """Thin OpenRouter chat client for the correction task."""

    def __init__(self, model: str, max_tokens: int, timeout_s: float) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.url = os.environ.get(
            "OPENROUTER_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/") + "/chat/completions"
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.key:
            raise OpenRouterError("OPENROUTER_API_KEY is empty")
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        title = os.environ.get("OPENROUTER_TITLE")
        if title:
            h["X-Title"] = title
        return h

    def correct_chunk(
        self, system: str, chunk: str, temperature: float
    ) -> tuple[str, Usage, bool]:
        """Return (corrected_text, usage, truncated) for one chunk.

        Retries transient (429/5xx/network) failures a few times; raises
        OpenRouterError on a hard failure so the caller can isolate this file.
        """
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": _USER_TEMPLATE.format(chunk=chunk)},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
            # qwen3.6-27b (and other Qwen3/DeepSeek-R1 reasoning models) otherwise
            # spend the whole token budget on a hidden <think> chain and hit the
            # cap before emitting any answer -> empty, truncated output at full
            # cost. Disable reasoning for this correction task. Same "thinking-
            # model pollution" guard the HyDE/followup paths use (think=false).
            "reasoning": {"enabled": False},
        }
        last_err: Exception | None = None
        for attempt in range(1, 5):
            try:
                r = self._session.post(
                    self.url, headers=self._headers(), json=body, timeout=self.timeout_s
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    raise OpenRouterError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
            except (requests.RequestException, OpenRouterError) as e:
                last_err = e
                wait = 2 * attempt
                log.warning("OpenRouter attempt %d failed (%s) - retrying in %ds",
                            attempt, e, wait)
                time.sleep(wait)
                continue
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise OpenRouterError(f"no choices in response: {list(data.keys())}")
            msg = choices[0].get("message") or {}
            content = _THINK_BLOCK_RE.sub("", msg.get("content") or "").strip()
            truncated = choices[0].get("finish_reason") == "length"
            u = data.get("usage") or {}
            usage = Usage(
                prompt_tokens=int(u.get("prompt_tokens") or 0),
                completion_tokens=int(u.get("completion_tokens") or 0),
                cost=float(u.get("cost") or 0.0),
            )
            return content, usage, truncated
        raise OpenRouterError(f"exhausted retries: {last_err}")

    def correct_text(
        self, system: str, text: str, temperature: float, max_chunk_chars: int
    ) -> tuple[str, Usage, bool]:
        """Correct a full transcript by chunking, then rejoining with blank lines."""
        chunks = chunk_paragraphs(text, max_chunk_chars)
        out_parts: list[str] = []
        total = Usage()
        any_trunc = False
        for i, ch in enumerate(chunks, 1):
            corrected, usage, trunc = self.correct_chunk(system, ch, temperature)
            out_parts.append(corrected)
            total.add(usage)
            any_trunc = any_trunc or trunc
            log.info("    chunk %d/%d ok (%d->%d chars%s)", i, len(chunks),
                     len(ch), len(corrected), ", TRUNCATED" if trunc else "")
        return "\n\n".join(out_parts), total, any_trunc


# --------------------------------------------------------------------------- #
# selection                                                                    #
# --------------------------------------------------------------------------- #
def select_sample(root: Path, count: int) -> list[Path]:
    """Pick ~count cleaned.json files stratified across file size.

    Sorting by size and sampling evenly gives a spread of short bhajans through
    long pravachans; the largest bucket tends to include hallucination-loop
    files (a repeated phrase inflates size). Deterministic - no randomness.
    """
    files = sorted(root.rglob("*.cleaned.json"), key=lambda p: p.stat().st_size)
    if not files:
        return []
    if len(files) <= count:
        return files
    step = len(files) / count
    picked = [files[min(int(i * step), len(files) - 1)] for i in range(count)]
    # Ensure the already-inspected pravachan is in the sample if present.
    pravachan = root / "03 PRAVACHAN IN MEDITATION.cleaned.json"
    if pravachan.exists() and pravachan not in picked:
        picked[-1] = pravachan
    # De-dup while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for p in picked:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def repetition_score(text: str) -> float:
    """Fraction of paragraphs that are exact duplicates - a hallucination-loop
    smell for the report (not used for filtering)."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) < 2:
        return 0.0
    return 1.0 - (len(set(paras)) / len(paras))


# --------------------------------------------------------------------------- #
# driver                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class FileResult:
    source: str
    stem: str
    chars_in: int
    rep_score: float
    conservative: str = ""
    aggressive: str = ""
    usage: Usage = field(default_factory=Usage)
    truncated: bool = False
    error: str = ""


def process_file(
    corrector: Corrector, path: Path, root: Path, modes: list[str],
    max_chunk_chars: int,
) -> FileResult:
    rel = path.relative_to(root).as_posix()
    stem = path.name[: -len(".cleaned.json")]
    data = json.loads(path.read_text(encoding="utf-8"))
    text = (data.get("cleaned_text") or "").strip()
    res = FileResult(
        source=rel, stem=stem, chars_in=len(text), rep_score=repetition_score(text)
    )
    if not text:
        res.error = "empty cleaned_text"
        return res
    if "conservative" in modes:
        res.conservative, u, t = corrector.correct_text(
            CONSERVATIVE_SYSTEM, text, 0.1, max_chunk_chars)
        res.usage.add(u)
        res.truncated = res.truncated or t
    if "aggressive" in modes:
        res.aggressive, u, t = corrector.correct_text(
            AGGRESSIVE_SYSTEM, text, 0.2, max_chunk_chars)
        res.usage.add(u)
        res.truncated = res.truncated or t
    return res


def write_outputs(
    results: list[FileResult], out_dir: Path, model: str, orig_texts: dict[str, str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = Usage()
    for r in results:
        total.add(r.usage)
        h = hashlib.md5(r.source.encode("utf-8")).hexdigest()[:8]
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", r.stem)[:60]
        payload = {
            "source_file": r.source,
            "model": model,
            "chars_in": r.chars_in,
            "repetition_score": round(r.rep_score, 3),
            "truncated": r.truncated,
            "error": r.error,
            "original_cleaned_text": orig_texts.get(r.source, ""),
            "conservative_text": r.conservative,
            "aggressive_text": r.aggressive,
            "usage": {
                "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens,
                "cost_usd": round(r.usage.cost, 6),
            },
        }
        (out_dir / f"{safe}__{h}.corrected.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Correction pilot report",
        "",
        f"- Model: `{model}`",
        f"- Files: {len(results)}",
        f"- Tokens: {total.prompt_tokens:,} in / {total.completion_tokens:,} out",
        f"- Reported cost: ${total.cost:.4f}",
        "",
        "| # | file | chars | rep | trunc | cost | status |",
        "|---|------|------:|----:|:-----:|-----:|--------|",
    ]
    for i, r in enumerate(results, 1):
        status = r.error or "ok"
        lines.append(
            f"| {i} | {r.source} | {r.chars_in:,} | {r.rep_score:.2f} | "
            f"{'Y' if r.truncated else ''} | ${r.usage.cost:.4f} | {status} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Full sample cost projection.
    n_ok = sum(1 for r in results if not r.error)
    if n_ok:
        per_file = total.cost / n_ok
        proj = per_file * 8400
        print(f"\nReported cost so far: ${total.cost:.4f} over {n_ok} files "
              f"(${per_file:.4f}/file)")
        print(f"Rough full-corpus (~8,400 files) projection: ${proj:.2f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--model", default=os.environ.get(
        "OPENROUTER_MODEL", "qwen/qwen3.6-27b"))
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--files", nargs="*", type=Path,
                    help="explicit cleaned.json paths (overrides auto-select)")
    ap.add_argument("--modes", nargs="+", default=["conservative", "aggressive"],
                    choices=["conservative", "aggressive"])
    ap.add_argument("--max-chunk-chars", type=int, default=1800)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(_REPO_ROOT / ".env")

    if args.files:
        sample = [p if p.is_absolute() else (args.root / p) for p in args.files]
    else:
        sample = select_sample(args.root, args.count)
    if not sample:
        log.error("no cleaned.json files found under %s", args.root)
        return 2

    log.info("model=%s files=%d modes=%s", args.model, len(sample), args.modes)
    try:
        corrector = Corrector(args.model, args.max_tokens, args.timeout)
    except OpenRouterError as e:
        log.error("cannot init OpenRouter: %s", e)
        return 2

    results: list[FileResult] = []
    orig_texts: dict[str, str] = {}
    for i, path in enumerate(sample, 1):
        rel = path.relative_to(args.root).as_posix() if args.root in path.parents \
            or path.parent == args.root else path.as_posix()
        log.info("[%d/%d] %s", i, len(sample), rel)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            orig_texts[rel] = (data.get("cleaned_text") or "").strip()
            r = process_file(corrector, path, args.root, args.modes,
                             args.max_chunk_chars)
            results.append(r)
        except Exception as e:  # isolate per-file: log, record, continue
            log.exception("failed on %s: %s", rel, e)
            results.append(FileResult(source=rel, stem=path.stem, chars_in=0,
                                      rep_score=0.0, error=str(e)))

    write_outputs(results, args.out, args.model, orig_texts)
    log.info("wrote %d outputs + report.md to %s", len(results), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
