"""Optional query expansion (HyDE) for the dense retrieval arm.

bge-m3 is a strong multilingual retriever, but a short or vague query
("anger ke baare mein", "what about the mind?") gives it little to match.
HyDE — Hypothetical Document Embeddings — closes that gap: the chat model
writes a short *hypothetical* discourse passage answering the query, and we
embed that alongside the query. The hypothetical text looks like the
transcripts being searched, so the blended vector lands closer to the right
chunks.

Cost: one extra chat-model call (~1-3 s on Qwen 7B). It is therefore opt-in
(`expand_query=true` on the request); default retrieval latency is unchanged.

The generated text is never shown to the user and never cited — it only
steers the embedding. The final answer is still grounded in, and cites, real
retrieved transcript passages.

The prompt builder is a pure function (unit-tested); the network call lives
in `QueryExpander`.
"""
from __future__ import annotations

import logging

import requests

from rag_api.config import Settings
from rag_api.lang import detect_language, language_label

log = logging.getLogger("rag_api.expand")

_HYDE_SYSTEM = (
    "You generate a short hypothetical passage that helps a search engine "
    "find relevant transcript excerpts. You do not answer the user directly "
    "and you do not add commentary."
)


def build_hyde_prompt(query: str, lang_label: str) -> str:
    """Prompt asking the chat model for one corpus-like hypothetical passage.

    Pure function — unit-tested. `lang_label` is a human-readable language
    name (see `rag_api.lang.language_label`).
    """
    return (
        f"Write a short hypothetical passage (2 to 4 sentences) in {lang_label} "
        "that could plausibly appear in a spiritual discourse (satsang / "
        "pravachan) by a Hindu teacher, and that directly answers the question "
        "below.\n"
        "Write ONLY the passage itself — no preamble, no headings, no "
        "markdown, no quotation marks, no citations.\n\n"
        f"Question: {query}"
    )


class QueryExpander:
    """Generates HyDE hypothetical passages via the Ollama chat model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = requests.Session()

    def hypothetical(self, query: str) -> str:
        """Return one hypothetical passage for `query`, or '' if generation
        fails.

        Failure is deliberately non-fatal: query expansion is an enhancement,
        so an unreachable / slow chat model degrades to query-only retrieval
        rather than failing the request.
        """
        lang = detect_language(query)
        prompt = build_hyde_prompt(query, language_label(lang))
        try:
            r = self._session.post(
                f"{self.settings.ollama_url}/api/chat",
                json={
                    "model": self.settings.chat_model,
                    "messages": [
                        {"role": "system", "content": _HYDE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    # A hypothetical passage is short — cap context and output
                    # so expansion stays cheap relative to answer generation.
                    "options": {
                        "temperature": self.settings.chat_temperature,
                        "num_ctx": 2048,
                        "num_predict": 256,
                    },
                },
                timeout=self.settings.chat_timeout_s,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("HyDE expansion failed for %r: %s — query-only", query, e)
            return ""
        content = ((r.json().get("message") or {}).get("content") or "").strip()
        if not content:
            log.warning("HyDE expansion returned empty content for %r", query)
        return content
