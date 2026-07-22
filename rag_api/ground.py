"""Post-synthesis groundedness check (opt-in, RAG_GROUNDEDNESS).

The synthesis prompt already instructs "answer only from the passages", but
nothing verifies the instruction held. This layer does: after the answer is
generated, one extra think=false chat call judges each answer sentence against
the retrieved passages, and sentences the judge marks unsupported are dropped
before the answer reaches the user.

Scope and posture:
  * Non-streaming `/api/query` only — a token stream cannot be post-edited.
    The faithfulness eval (`eval/faithfulness.py`) uses the same judge to
    SCORE answers instead of editing them.
  * Fail-open: any judge failure (unreachable model, malformed verdict) keeps
    the answer unchanged and logs the reason. A grounding check must never be
    the thing that breaks answering.
  * Never edits down to an empty answer: a verdict that rejects EVERY sentence
    is far more likely a judge failure than a fully fabricated answer, so it
    is logged and ignored.

Prompt builder / sentence splitter / verdict parser are pure functions
(unit-tested); the network call lives in `GroundChecker` via the shared
`ollama_chat.chat_text` (which carries the think=false guard).
"""
from __future__ import annotations

import json
import logging
import re

import requests

from rag_api.config import Settings
from rag_api.ollama_chat import chat_text
from rag_api.synthesis import build_context_block

log = logging.getLogger("rag_api.ground")

_GROUND_SYSTEM = (
    "You are a strict fact-checking judge. You compare an answer against the "
    "transcript passages it was written from and report which answer "
    "sentences are NOT supported by the passages. You output ONLY JSON."
)

# Sentence boundary: Devanagari danda or ./?/! followed by whitespace. The
# answers are short (a few sentences) and citation markers [N] sit before the
# terminator, so this cheap split is adequate — no NLP dependency.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.?!])\s+")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def split_sentences(answer: str) -> list[str]:
    """Split an answer into sentences (bilingual: danda or ./?/!)."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer or "") if s.strip()]


def build_ground_prompt(sentences: list[str], context_block: str) -> str:
    """Judge prompt: numbered passages, numbered answer sentences, JSON ask.

    Pure function — unit-tested. The verdict lists UNSUPPORTED sentence
    numbers (usually the short list), not supported ones.
    """
    numbered = "\n".join(f"S{i}: {s}" for i, s in enumerate(sentences, start=1))
    return (
        "Transcript passages:\n"
        "----------------------------------------\n"
        f"{context_block}\n"
        "----------------------------------------\n\n"
        "Answer sentences:\n"
        f"{numbered}\n\n"
        "For each sentence decide: is its claim supported by the passages "
        "above (including their METADATA lines)? A sentence that only cites, "
        "hedges, or says the answer is absent counts as supported.\n"
        'Reply with ONLY this JSON, nothing else: {"unsupported": [<sentence '
        'numbers>]} — an empty list if every sentence is supported.'
    )


def parse_verdict(text: str, n_sentences: int) -> list[int] | None:
    """Extract the unsupported-sentence numbers from the judge's reply.

    Returns None when the reply is unusable (no JSON, wrong shape, numbers out
    of range) — the caller fails open. Duplicates are collapsed; order kept.
    """
    m = _JSON_OBJECT_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    raw = obj.get("unsupported")
    if not isinstance(raw, list):
        return None
    out: list[int] = []
    for v in raw:
        if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= n_sentences):
            return None
        if v not in out:
            out.append(v)
    return out


def apply_verdict(sentences: list[str], unsupported: list[int]) -> str:
    """Rejoin the sentences the judge kept (1-based `unsupported` dropped)."""
    drop = set(unsupported)
    return " ".join(s for i, s in enumerate(sentences, start=1) if i not in drop)


class GroundChecker:
    """Judges a generated answer against its passages via the local model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = requests.Session()

    def check(self, answer: str, results: list[dict]) -> tuple[str, int]:
        """Return ``(answer_after_check, dropped_sentence_count)``.

        Fail-open by design: judge unreachable / unusable verdict / all-drop
        verdict → the original answer comes back with ``dropped=0`` and a log
        line saying why.
        """
        sentences = split_sentences(answer)
        if not sentences or not results:
            return answer, 0
        prompt = build_ground_prompt(sentences, build_context_block(results))
        reply = chat_text(
            self._session, self.settings.ollama_url, self.settings.chat_model,
            _GROUND_SYSTEM, prompt,
            timeout=self.settings.chat_timeout_s,
            temperature=0.0,
            num_ctx=self.settings.chat_num_ctx, num_predict=128,
            log=log, label="groundedness check", subject=answer[:80],
        )
        if not reply:
            return answer, 0  # chat_text already logged the failure
        unsupported = parse_verdict(reply, len(sentences))
        if unsupported is None:
            log.warning(
                "groundedness verdict unusable (%r) — keeping answer", reply[:200]
            )
            return answer, 0
        if not unsupported:
            return answer, 0
        if len(unsupported) >= len(sentences):
            log.warning(
                "groundedness judge rejected ALL %d sentences — keeping answer "
                "(treating a total rejection as judge failure)", len(sentences),
            )
            return answer, 0
        return apply_verdict(sentences, unsupported), len(unsupported)
