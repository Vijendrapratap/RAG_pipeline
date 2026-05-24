"""Bilingual, citation-grounded answer synthesis over retrieved chunks.

After retrieval, this layer turns the retrieved transcript passages into a
prompt and calls the Ollama chat model to write the final answer. It is the
piece Open WebUI used to own — now under our control, so:
  * the answer is written in the query's language (Hindi or English),
  * the model is instructed to answer ONLY from the passages and cite them,
  * an empty retrieval short-circuits to a canned message — no GPU call.

Prompt builders are pure functions (unit-testable); the network call lives
in `Synthesizer`.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import requests

from rag_api.config import Settings
from rag_api.lang import LANG_ENGLISH, LANG_HINDI, language_label

log = logging.getLogger("rag_api.synthesis")

# Shown when retrieval returns nothing — keyed by answer language.
NO_CONTEXT_MESSAGE: dict[str, str] = {
    LANG_HINDI: (
        "क्षमा करें — इस प्रश्न का उत्तर देने योग्य कोई प्रासंगिक अंश "
        "ट्रांसक्रिप्ट में नहीं मिला। कृपया प्रश्न को दूसरे शब्दों में पूछें।"
    ),
    LANG_ENGLISH: (
        "Sorry — I could not find any relevant passage in the transcripts "
        "to answer this question. Try rephrasing it."
    ),
}


def build_context_block(results: list[dict[str, Any]]) -> str:
    """Render retrieved results into a numbered context the model reads.

    Each passage is tagged ``[N]`` so the model can cite it as ``[N]``.
    """
    if not results:
        return "(no transcript passages were retrieved)"
    blocks: list[str] = []
    for i, r in enumerate(results, start=1):
        src = r.get("source_file") or "unknown"
        start, end = r.get("start_sec"), r.get("end_sec")
        ts = (f"{start}s-{end}s" if start is not None and end is not None
              else "timestamps unavailable")
        speakers = r.get("speakers") or []
        spk = ", ".join(speakers) if speakers else "unknown"
        meta = r.get("metadata") or {}
        meta_bits: list[str] = []
        for key, label in (
            ("event_id", "Event"), ("session_date", "Date"),
            ("track_title", "Track"), ("track_type", "Type"),
            ("location", "Location"),
        ):
            if meta.get(key):
                meta_bits.append(f"{label}: {meta[key]}")
        lines = [f"[{i}] Source: {src} | {ts} | Speakers: {spk}"]
        if meta_bits:
            lines.append("    " + " | ".join(meta_bits))
        lines.append(f"    {(r.get('text') or '').strip()}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_system_prompt(answer_lang: str) -> str:
    """System prompt: role, language, grounding + citation rules."""
    label = language_label(answer_lang)
    return (
        "You are a research assistant for the recorded discourses "
        "(satsang / pravachan) of Swami ji. You answer strictly from the "
        "transcript passages given in the user message.\n\n"
        "Rules:\n"
        f"- Write your ENTIRE answer in {label}.\n"
        "- Use ONLY the numbered passages as your source of truth. Do not "
        "use outside knowledge and do not invent quotes.\n"
        "- After each claim, cite the passage number(s) it came from, like "
        "[1] or [2][3].\n"
        "- When you quote Swami ji, quote verbatim from the passage.\n"
        "- Cite the source file and timestamp for important claims when they "
        "are available. If a passage says 'timestamps unavailable', do not "
        "invent a time.\n"
        f"- If the passages do not contain the answer, say so plainly in "
        f"{label} — do not guess.\n"
        "- Be concise and faithful to what Swami ji actually said."
    )


def build_user_prompt(query: str, context_block: str) -> str:
    """User-turn prompt: the passages, then the question."""
    return (
        "Transcript passages:\n"
        "----------------------------------------\n"
        f"{context_block}\n"
        "----------------------------------------\n\n"
        f"Question: {query}\n\n"
        "Answer using only the passages above, with [N] citations."
    )


class Synthesizer:
    """Calls the Ollama chat model to produce the final answer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = requests.Session()

    def _messages(
        self, query: str, results: list[dict[str, Any]], answer_lang: str,
    ) -> list[dict[str, str]]:
        context = build_context_block(results)
        return [
            {"role": "system", "content": build_system_prompt(answer_lang)},
            {"role": "user", "content": build_user_prompt(query, context)},
        ]

    def _options(self) -> dict[str, Any]:
        return {
            "temperature": self.settings.chat_temperature,
            "num_ctx": self.settings.chat_num_ctx,
        }

    def _no_context(self, answer_lang: str) -> str:
        return NO_CONTEXT_MESSAGE.get(answer_lang, NO_CONTEXT_MESSAGE[LANG_ENGLISH])

    def generate(
        self, query: str, results: list[dict[str, Any]], answer_lang: str,
    ) -> str:
        """Non-streaming: return the full answer string."""
        if not results:
            return self._no_context(answer_lang)
        r = self._session.post(
            f"{self.settings.ollama_url}/api/chat",
            json={
                "model": self.settings.chat_model,
                "messages": self._messages(query, results, answer_lang),
                "stream": False,
                "options": self._options(),
            },
            timeout=self.settings.chat_timeout_s,
        )
        r.raise_for_status()
        content = ((r.json().get("message") or {}).get("content") or "").strip()
        if not content:
            log.warning("chat model returned empty content for %r", query)
        return content

    def stream(
        self, query: str, results: list[dict[str, Any]], answer_lang: str,
    ) -> Iterator[str]:
        """Streaming: yield answer text deltas as the model produces them."""
        if not results:
            yield self._no_context(answer_lang)
            return
        with self._session.post(
            f"{self.settings.ollama_url}/api/chat",
            json={
                "model": self.settings.chat_model,
                "messages": self._messages(query, results, answer_lang),
                "stream": True,
                "options": self._options(),
            },
            timeout=self.settings.chat_timeout_s,
            stream=True,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("ollama stream: skipped malformed json line")
                    continue
                delta = (obj.get("message") or {}).get("content") or ""
                if delta:
                    yield delta
                if obj.get("done"):
                    break
