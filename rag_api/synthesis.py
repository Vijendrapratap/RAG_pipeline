"""Bilingual, citation-grounded answer synthesis over retrieved chunks.

After retrieval, this layer turns the retrieved transcript passages into a
prompt and calls a chat model to write the final answer. It is the piece
Open WebUI used to own — now under our control, so:
  * the answer is written in the query's language (Hindi or English),
  * the model is instructed to answer ONLY from the passages and cite them,
  * an empty retrieval short-circuits to a canned message — no GPU call.

Two providers are supported, selected by `settings.chat_provider`:
  * "ollama"     — local Ollama (default; preserves the PRD's strictly-local
                   posture). Native NDJSON streaming.
  * "openrouter" — OpenAI-compatible cloud API. Opt-in alternate provider
                   for testing larger models (e.g. qwen/qwen3.6-27b). SSE
                   streaming; `<think>...</think>` blocks are stripped from
                   reasoning models before the answer reaches the UI.

Prompt builders are pure functions (unit-testable); the network calls live
in `Synthesizer`.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

import requests

from rag_api.config import Settings
from rag_api.lang import LANG_ENGLISH, LANG_HINDI, language_label

# Reasoning models (Qwen3.x, DeepSeek-R1, etc.) emit a <think>...</think>
# block before the user-visible answer. The dashboard should not show that.
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)

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
        meta_bits: list[str] = [
            f"Source: {src}", f"Timestamp: {ts}", f"Speakers: {spk}",
        ]
        for key, label in (
            ("event_id", "Event"), ("session_date", "Date"),
            ("session_time", "Time"), ("track_title", "Track"),
            ("track_type", "Type"), ("location", "Location"),
            ("season", "Season"),
        ):
            if meta.get(key):
                meta_bits.append(f"{label}: {meta[key]}")
        lines = [
            f"[{i}] METADATA: {' | '.join(meta_bits)}",
            f"    TEXT: {(r.get('text') or '').strip()}",
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_system_prompt(answer_lang: str) -> str:
    """System prompt: role, language, grounding + citation rules.

    The language directive is deliberately front-loaded and emphatic. With
    reasoning disabled (CHAT_THINK=off) the model otherwise drifts to the
    question's language instead of the requested one — a strong leading rule
    keeps Hindi/English selection reliable without paying for a think chain.
    """
    label = language_label(answer_lang)
    return (
        "You are a research assistant for the recorded discourses "
        "(satsang / pravachan) of Swami ji.\n\n"
        f"LANGUAGE — your single most important rule: write your ENTIRE "
        f"answer in {label}. This holds even when the question or the "
        f"passages are in a different language. Never answer in any language "
        f"other than {label}.\n\n"
        "You answer strictly from the transcript passages given in the user "
        "message. Each passage has two parts and BOTH are part of the "
        "passage:\n"
        "  * METADATA — Event, Date, Time, Track, Type, Location, Season,\n"
        "    Speakers, Source, Timestamp. This is authoritative ground truth.\n"
        "  * TEXT — what Swami ji actually said.\n\n"
        "Rules:\n"
        "- Use ONLY the numbered passages (METADATA + TEXT) as your source "
        "of truth. Do not use outside knowledge and do not invent quotes.\n"
        "- For 'when / where / who / what time / which event / which track' "
        "questions, read the answer from the METADATA line. Do NOT search "
        "the TEXT body for these facts — they live in METADATA.\n"
        "- For 'what does Swami ji say / teach about X' questions, read the "
        "answer from the TEXT body and cite the source from METADATA.\n"
        "- Cite inline: after each claim, add the passage number(s) like [1] "
        "or [2][3]. Do NOT append a list of the passages or echo their "
        "METADATA/TEXT — cite inline only.\n"
        "- When you quote Swami ji, quote verbatim from the TEXT body.\n"
        "- If a passage's METADATA says 'Timestamp: timestamps unavailable', "
        "do not invent a time.\n"
        f"- If the passages do not contain the answer, say so plainly in "
        f"{label} — do not guess.\n"
        "- Be concise and faithful to what Swami ji actually said."
    )


def build_user_prompt(query: str, context_block: str, answer_lang: str) -> str:
    """User-turn prompt: the passages, the question, then a closing language +
    citation reminder that reinforces the system prompt for non-reasoning runs.
    """
    label = language_label(answer_lang)
    return (
        "Transcript passages:\n"
        "----------------------------------------\n"
        f"{context_block}\n"
        "----------------------------------------\n\n"
        f"Question: {query}\n\n"
        f"Answer in {label}, using only the passages above. Cite inline as "
        "[N]; do not repeat the passages."
    )


def _strip_think_block(text: str) -> str:
    """Remove any complete <think>...</think> block from a full response.

    Used for the non-streaming path where the entire answer is in hand.
    """
    return _THINK_BLOCK_RE.sub("", text).lstrip()


class _ThinkFilter:
    """Streaming-safe filter that drops content inside <think>...</think>.

    Reasoning models emit tokens like `<th`, `ink>`, `reason...`, `</think>`
    in arbitrary boundaries across SSE deltas. Buffering a small tail across
    deltas lets us find the open/close tags reliably while still streaming
    the post-think answer to the UI token-by-token.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, delta: str) -> str:
        """Consume one delta, return any post-think text that's safe to emit."""
        if not delta:
            return ""
        self._buffer += delta
        out = ""
        while self._buffer:
            if self._inside:
                m = _THINK_CLOSE_RE.search(self._buffer)
                if not m:
                    # Keep a tail in case `</think>` spans the next delta.
                    if len(self._buffer) > 16:
                        self._buffer = self._buffer[-16:]
                    return out
                self._buffer = self._buffer[m.end():]
                self._inside = False
                continue
            m = _THINK_OPEN_RE.search(self._buffer)
            if not m:
                # Hold back the last few chars in case `<think>` is splitting.
                safe = max(0, len(self._buffer) - 8)
                out += self._buffer[:safe]
                self._buffer = self._buffer[safe:]
                return out
            out += self._buffer[:m.start()]
            self._buffer = self._buffer[m.end():]
            self._inside = True
        return out

    def flush(self) -> str:
        """At end-of-stream, emit whatever's left outside a think block."""
        if self._inside:
            return ""
        out = self._buffer
        self._buffer = ""
        return out


class Synthesizer:
    """Generates the final answer.

    Branches on `settings.chat_provider`:
      * "ollama"     — local, NDJSON streaming (`/api/chat`)
      * "openrouter" — cloud, OpenAI-compatible (`/chat/completions`)
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = requests.Session()
        self._provider = (settings.chat_provider or "ollama").lower()
        if self._provider not in ("ollama", "openrouter"):
            log.warning(
                "unknown CHAT_PROVIDER=%r — falling back to ollama",
                settings.chat_provider,
            )
            self._provider = "ollama"
        if self._provider == "openrouter" and not settings.openrouter_api_key:
            log.error(
                "CHAT_PROVIDER=openrouter but OPENROUTER_API_KEY is empty — "
                "answers will fail with 401"
            )

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def active_model(self) -> str:
        """Model id of the currently selected provider (for /api/health)."""
        if self._provider == "openrouter":
            return self.settings.openrouter_model
        return self.settings.chat_model

    def _messages(
        self, query: str, results: list[dict[str, Any]], answer_lang: str,
    ) -> list[dict[str, str]]:
        context = build_context_block(results)
        return [
            {"role": "system", "content": build_system_prompt(answer_lang)},
            {"role": "user",
             "content": build_user_prompt(query, context, answer_lang)},
        ]

    def _ollama_options(self) -> dict[str, Any]:
        return {
            "temperature": self.settings.chat_temperature,
            "num_ctx": self.settings.chat_num_ctx,
            "num_batch": self.settings.chat_num_batch,
        }

    def _ollama_think(self) -> bool | None:
        """Resolve CHAT_THINK to the ollama `think` flag (None = omit it).

        Returning None leaves the request without a `think` key, so the model
        keeps its own default — the safe choice for models that don't support
        the flag at all.
        """
        v = (self.settings.chat_think or "auto").lower()
        if v in ("off", "false", "no", "0"):
            return False
        if v in ("on", "true", "yes", "1"):
            return True
        return None

    def _ollama_body(
        self, query: str, results: list[dict[str, Any]], answer_lang: str,
        model: str, *, stream: bool,
    ) -> dict[str, Any]:
        """Build the /api/chat request body, adding `think` only when set."""
        body: dict[str, Any] = {
            "model": model,
            "messages": self._messages(query, results, answer_lang),
            "stream": stream,
            "options": self._ollama_options(),
        }
        think = self._ollama_think()
        if think is not None:
            body["think"] = think
        return body

    def _openrouter_headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.openrouter_referer:
            h["HTTP-Referer"] = self.settings.openrouter_referer
        if self.settings.openrouter_title:
            h["X-Title"] = self.settings.openrouter_title
        return h

    def _openrouter_endpoint(self, path: str) -> str:
        return self.settings.openrouter_url.rstrip("/") + path

    def _no_context(self, answer_lang: str) -> str:
        return NO_CONTEXT_MESSAGE.get(answer_lang, NO_CONTEXT_MESSAGE[LANG_ENGLISH])

    def _resolve(
        self, provider: str | None, model: str | None,
    ) -> tuple[str, str]:
        """Pick the provider+model to use for this call.

        Per-request overrides win when both fields are set; otherwise the
        env-configured default for the active provider is used. The caller
        (FastAPI route) is responsible for validating the override against
        ``Settings.is_model_allowed`` *before* getting here.
        """
        if provider and model:
            return provider.lower(), model
        if self._provider == "openrouter":
            return "openrouter", self.settings.openrouter_model
        return "ollama", self.settings.chat_model

    # ----- non-streaming -----

    def generate(
        self, query: str, results: list[dict[str, Any]], answer_lang: str,
        *, provider: str | None = None, model: str | None = None,
    ) -> str:
        """Non-streaming: return the full answer string."""
        if not results:
            return self._no_context(answer_lang)
        prov, mdl = self._resolve(provider, model)
        if prov == "openrouter":
            content = self._generate_openrouter(query, results, answer_lang, mdl)
        else:
            content = self._generate_ollama(query, results, answer_lang, mdl)
        content = _strip_think_block(content).strip()
        if not content:
            log.warning(
                "chat model returned empty content for %r (provider=%s model=%s)",
                query, prov, mdl,
            )
        return content

    def _generate_ollama(
        self, query: str, results: list[dict[str, Any]], answer_lang: str,
        model: str,
    ) -> str:
        r = self._session.post(
            f"{self.settings.ollama_url}/api/chat",
            json=self._ollama_body(query, results, answer_lang, model,
                                   stream=False),
            timeout=self.settings.chat_timeout_s,
        )
        r.raise_for_status()
        return ((r.json().get("message") or {}).get("content") or "")

    def _generate_openrouter(
        self, query: str, results: list[dict[str, Any]], answer_lang: str,
        model: str,
    ) -> str:
        r = self._session.post(
            self._openrouter_endpoint("/chat/completions"),
            headers=self._openrouter_headers(),
            json={
                "model": model,
                "messages": self._messages(query, results, answer_lang),
                "stream": False,
                "temperature": self.settings.chat_temperature,
                "max_tokens": self.settings.openrouter_max_tokens,
            },
            timeout=self.settings.chat_timeout_s,
        )
        r.raise_for_status()
        body = r.json()
        choices = body.get("choices") or []
        if not choices:
            log.warning(
                "openrouter returned no choices for %r (body keys: %s)",
                query, list(body.keys()),
            )
            return ""
        return ((choices[0].get("message") or {}).get("content") or "")

    # ----- streaming -----

    def stream(
        self, query: str, results: list[dict[str, Any]], answer_lang: str,
        *, provider: str | None = None, model: str | None = None,
    ) -> Iterator[str]:
        """Streaming: yield answer text deltas as the model produces them.

        For reasoning models, `<think>...</think>` content is filtered out —
        the UI sees only the post-think answer tokens.
        """
        if not results:
            yield self._no_context(answer_lang)
            return
        prov, mdl = self._resolve(provider, model)
        raw_stream = (
            self._stream_openrouter(query, results, answer_lang, mdl)
            if prov == "openrouter"
            else self._stream_ollama(query, results, answer_lang, mdl)
        )
        flt = _ThinkFilter()
        for delta in raw_stream:
            out = flt.feed(delta)
            if out:
                yield out
        tail = flt.flush()
        if tail:
            yield tail

    def _stream_ollama(
        self, query: str, results: list[dict[str, Any]], answer_lang: str,
        model: str,
    ) -> Iterator[str]:
        with self._session.post(
            f"{self.settings.ollama_url}/api/chat",
            json=self._ollama_body(query, results, answer_lang, model,
                                   stream=True),
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

    def _stream_openrouter(
        self, query: str, results: list[dict[str, Any]], answer_lang: str,
        model: str,
    ) -> Iterator[str]:
        with self._session.post(
            self._openrouter_endpoint("/chat/completions"),
            headers=self._openrouter_headers(),
            json={
                "model": model,
                "messages": self._messages(query, results, answer_lang),
                "stream": True,
                "temperature": self.settings.chat_temperature,
                "max_tokens": self.settings.openrouter_max_tokens,
            },
            timeout=self.settings.chat_timeout_s,
            stream=True,
        ) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                # OpenAI-compatible SSE: lines start with "data: " (or are
                # ":..." comments / heartbeats — skip those).
                if raw.startswith(":"):
                    continue
                if raw.startswith("data:"):
                    payload = raw[5:].strip()
                else:
                    payload = raw.strip()
                if not payload or payload == "[DONE]":
                    if payload == "[DONE]":
                        break
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    log.warning("openrouter stream: skipped malformed sse line")
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta:
                    yield delta
                if choices[0].get("finish_reason"):
                    break
