"""Conversational follow-up rewriting (Stage 4.3).

A follow-up like "उसके बाद क्या हुआ" / "what about his early life" is
meaningless to the retriever on its own — the entity/topic it refers to lives
in the previous turn. This module rewrites such a context-dependent query into
a standalone one *before retrieval*, using the stored conversation history
(`conversation_history` — this is its first consumer).

Deterministic-first, like the rest of Stage 4: the router only tags a query
`followup` when history is present AND it has an anaphoric / continuation shape,
so the one chat round-trip here fires only for genuine follow-ups, never on a
standalone question that merely happens to arrive mid-conversation.

`think=False` on the call — chat_model (qwen3.5:9b) is a thinking model and
would otherwise wrap the rewrite in a reasoning chain (same gotcha fixed in
expand.py / pageindex.py). The prompt builder is a pure function (unit-tested);
the network call lives in `FollowupRewriter`. On any failure the raw query is
returned unchanged, so a slow/unreachable model degrades to raw-query retrieval
rather than failing the request.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from rag_api.config import Settings

log = logging.getLogger("rag_api.followup")

_REWRITE_SYSTEM = (
    "You rewrite a follow-up question into a single standalone search query "
    "using the conversation so far. You resolve pronouns and implicit "
    "references to the entity or topic they point at. You never answer the "
    "question and you never add commentary."
)

# Keep the history window small — recent turns carry the referent; older turns
# only add noise and tokens.
_MAX_TURNS = 3
_MAX_ANSWER_CHARS = 400


def _turn_text(turn: dict[str, Any]) -> tuple[str, str]:
    """Pull (question, answer) from a history turn, tolerating both the
    dashboard's {question, answer} shape and a {role, content} message shape."""
    q = str(turn.get("question") or turn.get("query") or "").strip()
    a = str(turn.get("answer") or "").strip()
    if not q and turn.get("role") == "user":
        q = str(turn.get("content") or "").strip()
    if not a and turn.get("role") == "assistant":
        a = str(turn.get("content") or "").strip()
    return q, a


def build_rewrite_prompt(query: str, history: list[dict[str, Any]]) -> str:
    """Prompt asking the chat model for one standalone rewrite. Pure function.

    Only the last `_MAX_TURNS` turns are included; assistant answers are
    truncated so the referent survives without bloating the prompt.
    """
    lines: list[str] = []
    for turn in history[-_MAX_TURNS:]:
        q, a = _turn_text(turn)
        if q:
            lines.append(f"User: {q}")
        if a:
            lines.append(f"Assistant: {a[:_MAX_ANSWER_CHARS]}")
    convo = "\n".join(lines) if lines else "(no prior turns)"
    return (
        "Conversation so far:\n"
        f"{convo}\n\n"
        f"Follow-up question: {query}\n\n"
        "Rewrite the follow-up as ONE standalone search query in the same "
        "language, naming the entity or topic it refers to. Output ONLY the "
        "rewritten query — no preamble, no quotes, no explanation."
    )


class FollowupRewriter:
    """Rewrites a context-dependent follow-up into a standalone query."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = requests.Session()

    def rewrite(self, query: str, history: list[dict[str, Any]] | None) -> str:
        """Return a standalone rewrite of `query`, or `query` unchanged when
        there is no history or the call fails/returns empty."""
        if not history:
            return query
        prompt = build_rewrite_prompt(query, history)
        try:
            r = self._session.post(
                f"{self.settings.ollama_url}/api/chat",
                json={
                    "model": self.settings.chat_model,
                    "messages": [
                        {"role": "system", "content": _REWRITE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    # See module docstring — thinking-model pollution guard.
                    "think": False,
                    "options": {
                        "temperature": 0.0,
                        "num_ctx": 2048,
                        "num_predict": 128,
                    },
                },
                timeout=self.settings.chat_timeout_s,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("follow-up rewrite failed for %r: %s — raw query", query, e)
            return query
        content = ((r.json().get("message") or {}).get("content") or "").strip()
        if not content:
            log.warning("follow-up rewrite returned empty for %r — raw query", query)
            return query
        # Models sometimes wrap the rewrite in quotes despite the instruction.
        return content.strip().strip('"').strip("'").strip()
