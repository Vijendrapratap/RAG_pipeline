"""Agentic query planner: intent + bilingual retrieval variants.

One think=false chat call per eligible query (router classes thematic / name /
analytic), gated by ``RAG_PLANNER``. It closes two live gaps the deterministic
router cannot:

* **List intent** — "10 sitting for rain" wants a *list of sittings* about a
  topic, not a single essay answer. The plan flags ``intent=list_sittings``
  with the requested count; the handler then retrieves at the *sitting* level
  (summary collection, one result per file) instead of 4 chunks.
* **Vocabulary / script gap** — "rain", "barish" and "बारिश" mean the same
  thing but only the Devanagari form matches the corpus lexically. The plan
  carries 2–3 *retrieval query variants* (Devanagari Hindi, English, synonym-
  widened) that are searched in parallel and merged (`search_multi`).

Deterministic first, LLM second: the list-intent detector is a pure bilingual
regex that works even with the chat model down; the LLM call adds the
translated/widened variants and catches list phrasings the regex misses.
Fail-open everywhere — a failed/garbled LLM plan degrades to the regex result
or to no plan at all, i.e. exactly today's pipeline.

Prompt builder, verdict parser and the list-intent detector are pure functions
(unit-tested); the network call lives in `Planner`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import requests

from rag_api.config import Settings
from rag_api.ollama_chat import chat_text

log = logging.getLogger("rag_api.planner")

INTENT_ANSWER = "answer"
INTENT_LIST = "list_sittings"
# "summary of the sitting on X" — the answer must QUOTE Swami ji's exact
# lines as bullets, not paraphrase. Chunk retrieval with a boosted top_k.
INTENT_VERBATIM = "verbatim"
# "best sittings on X" — rank sittings by topic relevance + mention frequency
# + duration + play count (rag_api.best), best first.
INTENT_BEST = "best_sittings"

_DEFAULT_LIST_N = 10
_MAX_LIST_N = 50

# Words that mean "a sitting" in the archive's vocabulary — English, Hinglish
# and Devanagari. Matched with optional plural endings.
_SITTING_WORDS = (
    r"(?:sittings?|sessions?|discourses?|prava?chans?|satsangs?|tracks?|"
    r"recordings?|प्रवचन(?:ों)?|सत्संग(?:ों)?|बैठक(?:ें|ों)?)"
)
# "10 sittings …" / "sittings 10" — a count next to a sitting-word.
_COUNT_SITTING_RE = re.compile(
    r"(?ix)\b(\d{1,3})\s*" + _SITTING_WORDS + r"\b"
    r"|\b" + _SITTING_WORDS + r"\s*(\d{1,3})\b"
)
# An explicit list-imperative near a sitting-word, no count ("suggest some
# sittings about rain", "barish par pravachan batao"). Deliberately narrow:
# generic verbs (give/show/do) and modal chahiye/चाहिए ride along in ordinary
# thematic questions ("क्या सत्संग में जाना चाहिए") and must not trip this.
_LIST_VERB_RE = re.compile(
    r"(?ix)\b(?:suggest|recommend|list|batao|bataiye|bata|dijiye)\b"
    r"|बताओ|बताइए|बताइये|दीजिए|दिखाओ|सुझाओ"
)


# Verbatim-summary ask: the user wants Swami ji's own words, not a paraphrase.
# "summary", "key/main points", "exact words", "word to word" + Hindi forms.
_VERBATIM_RE = re.compile(
    r"(?ix)\b(?:summar(?:y|ies|ize|ise|ized|ised)|key\s+points?|"
    r"main\s+points?|word\s+to\s+word|exact\s+(?:words?|lines?)|verbatim|"
    r"saransh|saar)\b"
    r"|सारांश|संक्षेप|संक्षिप्त|मुख्य\s*(?:बातें|बिंदु)|शब्दश[:ः]"
)
# Leading connective residue after the meta-words are stripped
# ("summary of the sitting on rain" -> "of the sitting on rain").
_LEADING_FILLER_RE = re.compile(
    r"(?i)^(?:(?:of|the|a|an|for|on|about|in|from)\s+)+"
)
# "best/top sittings" ranking ask. English + Hinglish + Devanagari superlatives;
# must co-occur with a sitting-word so "best way to meditate" stays thematic.
_BEST_RE = re.compile(
    r"(?ix)\b(?:best|top|greatest|finest|behtar(?:een|in)?|badhiya|"
    r"sabse\s+acch\w*)\b"
    r"|सर्वश्रेष्ठ|श्रेष्ठ|बेहतरीन|बढ़िया|सबसे\s*अच्छ\w*"
)


def detect_verbatim_intent(query: str) -> bool:
    """True when `query` asks for a summary / the exact words of a sitting —
    the answer must quote verbatim lines, not paraphrase. Pure, bilingual."""
    return bool(_VERBATIM_RE.search((query or "").strip()))


def strip_verbatim_words(query: str) -> str:
    """Retrieval text for a verbatim plan: the meta-ask ("summary", "key
    points", "word to word") reranks terribly against transcript prose — live
    probe 2026-07-15 saw the 0.2 citation floor drop 10/10 passages. Strip
    the meta-words so retrieval matches the TOPIC; empty result falls back to
    the input. Pure."""
    out = _VERBATIM_RE.sub(" ", query or "")
    out = re.sub(r"\s+", " ", out).strip(" \t,.-—:;|")
    out = _LEADING_FILLER_RE.sub("", out).strip(" \t,.-—:;|")
    return out or (query or "")


def detect_best_intent(query: str) -> int | None:
    """Requested count when `query` asks for the *best/top sittings* on a
    topic, else None. Pure, bilingual. A superlative alone is not enough —
    it must target sittings ("best sittings on rain" yes, "best way to
    meditate" no). Count defaults to 10, clamped like list intent."""
    q = (query or "").strip()
    if not q:
        return None
    if not _BEST_RE.search(q):
        return None
    if not re.search(r"(?ix)\b" + _SITTING_WORDS + r"\b", q):
        return None
    m = _COUNT_SITTING_RE.search(q)
    if m:
        n = int(m.group(1) or m.group(2))
        return max(1, min(n, _MAX_LIST_N))
    # "top 5 …" / "5 best …" — the count sits next to the superlative, not
    # the sitting-noun, so _COUNT_SITTING_RE missed it.
    m = re.search(
        r"(?ix)\b(?:best|top)\s+(\d{1,3})\b|\b(\d{1,3})\s+(?:best|top)\b"
        r"|(?:सबसे\s*अच्छ\w*|श्रेष्ठ|बेहतरीन)\s*(\d{1,3})", q)
    if m:
        n = int(m.group(1) or m.group(2) or m.group(3))
        return max(1, min(n, _MAX_LIST_N))
    return _DEFAULT_LIST_N


def detect_list_intent(query: str) -> int | None:
    """Requested sitting count when `query` asks for a *list of sittings*,
    else None. Pure, bilingual, works with the chat model down.

    "10 sitting for rain" → 10; "suggest sittings about rain" → 10 (default);
    "बारिश पर 5 प्रवचन चाहिए" → 5; "what is rain" → None.
    """
    q = (query or "").strip()
    if not q:
        return None
    m = _COUNT_SITTING_RE.search(q)
    if m:
        n = int(m.group(1) or m.group(2))
        return max(1, min(n, _MAX_LIST_N))
    if re.search(r"(?ix)\b" + _SITTING_WORDS + r"\b", q) and _LIST_VERB_RE.search(q):
        return _DEFAULT_LIST_N
    return None


_PLANNER_SYSTEM = (
    "You are a retrieval query planner for an archive of Hindi spiritual "
    "discourses (satsang / pravachan). You never answer the question — you "
    "only output a JSON plan for the search engine."
)


def build_plan_prompt(query: str) -> str:
    """Prompt asking the chat model for a JSON retrieval plan. Pure."""
    return (
        "Plan retrieval for the user query below. Reply with ONLY one JSON "
        "object, no markdown, no commentary:\n"
        '{"intent": "answer" or "list_sittings", "n": <int or null>, '
        '"queries": ["...", "..."]}\n\n'
        "Rules:\n"
        '- intent is "list_sittings" ONLY when the user asks for a list of '
        "sittings/sessions/pravachans/satsangs about a topic; then n is the "
        'requested count (null if unstated). Otherwise intent is "answer" '
        "and n is null.\n"
        "- queries: 2 or 3 short search queries for the TOPIC of the "
        "question, covering vocabulary the archive might use:\n"
        "  1. Hindi in Devanagari script (e.g. barish/rain -> बारिश, वर्षा),\n"
        "  2. English,\n"
        "  3. optionally one with close synonyms.\n"
        "- Keep person names exactly as written (add a Devanagari "
        "transliteration variant if the name is in Latin script).\n"
        "- Each query is a SEARCH QUERY (keywords/topic), never an answer, "
        "never instructions.\n\n"
        f"User query: {query}"
    )


def parse_plan(text: str, max_queries: int = 3) -> dict | None:
    """Extract and validate the planner's JSON verdict. Pure.

    Returns ``{"intent": str, "n": int|None, "queries": [str, ...]}`` or None
    when the reply is unusable (fail-open — the caller degrades gracefully).
    """
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    intent = obj.get("intent")
    if intent not in (INTENT_ANSWER, INTENT_LIST):
        return None
    n = obj.get("n")
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            n = None
        else:
            n = min(n, _MAX_LIST_N)
    raw = obj.get("queries")
    queries: list[str] = []
    seen: set[str] = set()
    if isinstance(raw, list):
        for q in raw:
            if not isinstance(q, str):
                continue
            qs = q.strip()
            key = qs.lower()
            # Guard against the model echoing an answer/instructions as a
            # "query": variants are short topic strings, not prose.
            if not qs or len(qs) > 120 or key in seen:
                continue
            seen.add(key)
            queries.append(qs)
            if len(queries) >= max_queries:
                break
    return {"intent": intent, "n": n, "queries": queries}


@dataclass(frozen=True)
class Plan:
    """A resolved retrieval plan for one query."""

    intent: str  # INTENT_ANSWER | INTENT_LIST
    queries: list[str] = field(default_factory=list)  # original always first
    n: int | None = None  # list size (list intent only)

    def as_dict(self) -> dict:
        return {"intent": self.intent, "queries": self.queries, "n": self.n}


class Planner:
    """Resolves a `Plan` for one query via regex + one think=false chat call."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = requests.Session()

    def _llm_plan(self, query: str) -> dict | None:
        reply = chat_text(
            self._session, self.settings.ollama_url, self.settings.chat_model,
            _PLANNER_SYSTEM, build_plan_prompt(query),
            timeout=self.settings.chat_timeout_s,
            temperature=0.0,
            num_ctx=2048, num_predict=256,
            log=log, label="query plan", subject=query,
        )
        return parse_plan(reply, self.settings.planner_max_queries)

    def plan(self, query: str) -> Plan | None:
        """Return the plan for `query`, or None when there is nothing to plan
        (no special intent, no usable variants) — the caller then runs the
        unchanged single-query pipeline.

        Deterministic detectors always win on intent/count (precedence
        best > list > verbatim): an LLM that misses "10 sitting for rain"
        cannot downgrade it to a plain answer. The LLM contributes the
        bilingual variants and catches list phrasings the regex misses.
        """
        det_best = detect_best_intent(query)
        det_n = detect_list_intent(query)
        det_verbatim = detect_verbatim_intent(query)
        llm = self._llm_plan(query)

        variants: list[str] = []
        seen = {query.strip().lower()}
        for v in (llm or {}).get("queries", []):
            if v.lower() not in seen:
                seen.add(v.lower())
                variants.append(v)

        if det_best is not None:
            intent, n = INTENT_BEST, det_best
        elif det_n is not None:
            intent, n = INTENT_LIST, det_n
        elif det_verbatim:
            intent, n = INTENT_VERBATIM, None
        elif llm and llm["intent"] == INTENT_LIST:
            intent, n = INTENT_LIST, llm["n"] or _DEFAULT_LIST_N
        elif variants:
            intent, n = INTENT_ANSWER, None
        else:
            return None

        queries = [query] + variants
        if intent == INTENT_VERBATIM:
            # Search the topic, not the meta-ask — the LLM variants tend to
            # echo the summary-speak ("बारिश पर सत्संग मुख्य बिंदु") too.
            stripped: list[str] = []
            seen_s: set[str] = set()
            for q in queries:
                sq = strip_verbatim_words(q)
                if sq.lower() not in seen_s:
                    seen_s.add(sq.lower())
                    stripped.append(sq)
            queries = stripped
        return Plan(intent=intent, queries=queries, n=n)
