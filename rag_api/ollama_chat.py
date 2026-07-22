"""One shared Ollama `/api/chat` text call for the query-side helpers.

HyDE expansion (`expand.py`) and follow-up rewriting (`followup.py`) both want the
same thing: one short, non-streaming completion from the chat model with
`think=False` — the thinking-model pollution guard (a reasoning chain otherwise
leaks into the passage/rewrite; same gotcha fixed in `pageindex.py`). The request
body, the non-fatal error degradation, and the reply unwrap lived in both modules
verbatim; they live here once instead.

Failure is deliberately non-fatal — these are retrieval enhancements, so an
unreachable/slow chat model returns ``""`` and the caller degrades to raw-query
retrieval rather than failing the request.
"""
from __future__ import annotations

import logging

import requests


def chat_text(
    session: requests.Session,
    ollama_url: str,
    model: str,
    system: str,
    user: str,
    *,
    timeout: float,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    think: bool = False,
    log: logging.Logger,
    label: str,
    subject: str,
) -> str:
    """Return one chat completion's text, or ``""`` on failure/empty content.

    ``label`` names the operation and ``subject`` the query, only for the log
    line (e.g. ``label="HyDE expansion"``, ``subject=query``).
    """
    try:
        r = session.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": think,
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                },
            },
            timeout=timeout,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("%s failed for %r: %s — raw query", label, subject, e)
        return ""
    content = ((r.json().get("message") or {}).get("content") or "").strip()
    if not content:
        log.warning("%s returned empty content for %r", label, subject)
    return content
