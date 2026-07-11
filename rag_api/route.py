"""Deterministic query router (Stage 4.1).

Classifies each query into one of five shapes and maps it to per-class
retrieval settings, so a located quote, a person lookup, a thematic question,
an analytic count and a conversational follow-up each get the retrieval
behaviour they want instead of one-size-fits-all.

    name     — a lookup ABOUT a specific named person (discourse subject).
    quote    — a pasted verbatim Devanagari passage the user wants LOCATED.
    thematic — a conceptual / abstract question (the default fallthrough).
    analytic — an explicit count / list / enumerate intent.
    followup — a context-dependent continuation; only WITH conversation history.

Deterministic first — no I/O, no LLM. `classify` reuses
`query_parse.detect_quote` for the quote shape and cheap regex / lexical rules
for the rest; an optional LLM fallback (a later commit) fires only when this
core is ambiguous, never on the common path.

Per-class settings are STARTING values from docs/stage4_kickoff.md §4.1.c —
tune by A/B, do not treat as final. The two dominant, robust levers here are
(1) `include_catalog=off` for every retrieval class (the live `/api/query`
default is on, which the Stage 3 baseline proved regresses topic MRR
0.487→0.328) and (2) a per-class `bm25_weight`. Per-class candidate budgets
are deliberately omitted: the Stage 3 grid showed retrieval flat across that
knob (the entity ceiling is a sparse-chunk gap, not a budget gap).

`name` detection is a structural heuristic, not a vocab match: the enriched
`people_named` field is mostly romanized and covers 1/8 of the golden
Devanagari subjects, so a people-vocab lookup would miss most of them (see
docs/stage4_kickoff.md §4.1.b caution). It catches seeded recurring subjects,
name-suffix / honorific shapes, and multi-token proper spans before a cue; it
knowingly under-recalls bare single-word devotee names (नीलिमा). That gap is a
retrieval concern, not a routing blocker — every retrieval class routes catalog
off, so a name↔thematic slip still captures the headline win.

Pure and unit-tested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rag_api.config import Settings
from rag_api.query_parse import detect_quote

QueryClass = Literal["name", "quote", "thematic", "analytic", "followup"]

NAME: QueryClass = "name"
QUOTE: QueryClass = "quote"
THEMATIC: QueryClass = "thematic"
ANALYTIC: QueryClass = "analytic"
FOLLOWUP: QueryClass = "followup"

# --- analytic: explicit count / list / enumerate intent -------------------
# English + romanized/Devanagari counting verbs, anchored to intent words so a
# thematic "how does the mind work" or "कैसे करें" is NOT analytic.
_ANALYTIC_RE = re.compile(
    r"(?ix)"
    r"\b(?:how\s+many|how\s+much|how\s+often|number\s+of|count(?:\s+the|\s+of)?|"
    r"list\s+(?:all|the|out)|which\s+(?:speakers?|discourses?|transcripts?|"
    r"sittings?|files?|prava?chan\w*)|most\s+often|kitne|kitni)\b"
    r"|कितने|कितनी|कितनी\s+बार|कौन\s*कौन|किन\s*किन|गिन(?:ती|ो|कर|कर\s)"
)

# --- followup: anaphora / continuation shape (guarded by history) ---------
_FOLLOWUP_RE = re.compile(
    r"(?ix)"
    r"^(?:and|so|but|then|what\s+about|how\s+about|what\s+else|tell\s+me\s+more)\b"
    r"|\b(?:his|her|their|its|that|those|these|the\s+same|the\s+above|previous)\b"
    r"|उसके|उसकी|उसका|उन्हीं|उन्हें|इसके|इसकी|इसका|उनके|उनकी|उनका|वही|उसी|इसी"
    r"|uske|uski|uska|iske|iski|iska|unke|unki|unka|phir"
)

# --- name: person-lookup cues + shape -------------------------------------
# "about X" / "mention of X" / "the tale/story of X" — a discourse ABOUT a
# named subject. NOT query_parse.detect_signals (that matches catalog singer
# names, Latin — not discourse subjects). ज़िक्र with an optional intervening
# कोई ("का कोई ज़िक्र") is one cue.
_NAME_CUE_RE = re.compile(
    r"के\s*बारे\s*में|की\s*कथा|की\s*कहानी|का\s*(?:कोई\s*)?ज़िक्र|"
    r"का\s*(?:कोई\s*)?जिक्र|नाम\s*(?:के|का)|(?ix:\bwho\s+(?:is|was|were)\b|"
    r"\btell\s+me\s+about\b|\bstory\s+of\b|\btale\s+of\b)"
)
# Devanagari name-suffixes (surname / epithet endings). Modest concept-collision
# risk (देव=deity) accepted for a starting rule; multi-syllable famous subjects
# whose ending is ambiguous (…नंद) live in the seed set instead.
_NAME_SUFFIXES = ("सिंह", "देव", "दास", "नाथ", "तीर्थ", "प्रसाद")
# Honorifics that mark the *teacher*, not a looked-up subject — a bare "जी"
# after one of these is not a name signal (स्वामी जी is on nearly every query).
_TEACHER_HONORIFICS = frozenset({
    "स्वामी", "गुरु", "गुरुजी", "गुरुदेव", "सद्गुरु", "सतगुरु",
    "ऋषि", "बाबा", "महाराज",
})
# Well-known recurring discourse subjects — stable across the archive. Devotee
# names are deliberately absent (they belong to a future people-vocab, not a
# hardcode). Romanized forms included for cross-script queries.
_SEED_PEOPLE = frozenset({
    "विवेकानंद", "रामतीर्थ", "रामकृष्ण", "नामदेव", "नानक", "कबीर", "मीरा",
    "मीराबाई", "बुद्ध", "कृष्ण", "अर्जुन", "प्रह्लाद", "ध्रुव", "गोरखनाथ",
    "रैदास", "तुलसीदास", "सूरदास", "शुकदेव", "नरसी",
    "vivekananda", "rama tirtha", "ramtirth", "ramakrishna", "namdev",
    "nanak", "kabir", "meera", "mira", "buddha", "krishna", "arjuna",
})
# Abstract-concept words that must never read as a name-shaped subject even in
# a multi-token span before a cue.
_CONCEPT_STOP = frozenset({
    "क्रोध", "ध्यान", "प्रेम", "भक्ति", "मन", "समर्पण", "वैराग्य", "अहंकार",
    "मोक्ष", "माया", "आत्मा", "परमात्मा", "शांति", "करुणा", "सेवा", "साधना",
    "ज्ञान", "कर्म", "योग", "मृत्यु", "जीवन", "सत्य", "आनंद", "श्रद्धा",
})
# Devanagari particles / question words that never form part of a name span.
_SPAN_STOP = frozenset({
    "के", "का", "की", "को", "में", "से", "पर", "और", "या", "है", "हैं",
    "कोई", "किस", "किन", "क्या", "कौन", "कहाँ", "कहां", "कब", "कैसे", "क्यों",
    "बारे", "ज़िक्र", "जिक्र", "कथा", "कहानी", "नाम", "बात", "जी", "ने", "स्वामी",
}) | _TEACHER_HONORIFICS

_DEVANAGARI_TOKEN_RE = re.compile(r"[ऀ-ॿ]+")
_CAP_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _has_seed_person(query: str) -> bool:
    toks = _DEVANAGARI_TOKEN_RE.findall(query)
    if any(t in _SEED_PEOPLE for t in toks):
        return True
    low = _norm(query)
    return any(p in low for p in _SEED_PEOPLE if p.isascii())


def _has_name_shape(query: str) -> bool:
    """A Devanagari token that looks like a personal name — a name-suffix
    surname, or a `<name> जी` honorific whose name is not the teacher."""
    toks = _DEVANAGARI_TOKEN_RE.findall(query)
    for i, tok in enumerate(toks):
        if tok in _CONCEPT_STOP or tok in _SPAN_STOP:
            continue
        if len(tok) >= 3 and tok.endswith(_NAME_SUFFIXES):
            return True
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if nxt == "जी" and tok not in _TEACHER_HONORIFICS:
            return True
    return False


def _has_multiword_subject_before_cue(query: str) -> bool:
    """Two+ consecutive name-eligible Devanagari tokens immediately before a
    name cue — a proper span like "पुरन सिंह" / "चेतन विश्वास"."""
    m = _NAME_CUE_RE.search(query)
    if not m:
        return False
    before = query[: m.start()]
    toks = _DEVANAGARI_TOKEN_RE.findall(before)
    run = 0
    for tok in reversed(toks):
        if tok in _SPAN_STOP or tok in _CONCEPT_STOP or len(tok) < 2:
            break
        run += 1
        if run >= 2:
            return True
    return False


def _looks_like_name(query: str) -> bool:
    if _has_seed_person(query) or _has_name_shape(query):
        return True
    if _NAME_CUE_RE.search(query) and _has_multiword_subject_before_cue(query):
        return True
    # English "who is/was <Capitalized Name>" — a person lookup.
    if re.search(r"(?i)\bwho\s+(?:is|was|were)\b", query) and _CAP_NAME_RE.search(query):
        return True
    return False


def _quote_class(query: str) -> tuple[bool, str]:
    """The quote shape: a pasted verbatim passage. `detect_quote` supplies the
    long-Devanagari test; the name-cue guard rejects long *questions about* the
    corpus ("... के बारे में स्वामी जी ने कहाँ बताया") that trip the length
    threshold but are lookups, not pastes. Returns (is_quote, cleaned_query)."""
    qd = detect_quote(query)
    if not qd.is_quote:
        return False, query
    if _NAME_CUE_RE.search(query):
        return False, query
    return True, qd.query


def classify(query: str, history_present: bool = False) -> QueryClass:
    """Deterministic class for `query`. Pure — no I/O, no LLM.

    Precedence: analytic (explicit count intent) → quote (verbatim paste) →
    followup (anaphora, only with history) → name (person lookup) → thematic
    (default fallthrough). Analytic outranks name so "पुरन सिंह का ज़िक्र कितनी
    बार" is a count, not a lookup.
    """
    q = (query or "").strip()
    if not q:
        return THEMATIC
    if _ANALYTIC_RE.search(q):
        return ANALYTIC
    is_quote, _ = _quote_class(q)
    if is_quote:
        return QUOTE
    if history_present and _FOLLOWUP_RE.search(q):
        return FOLLOWUP
    if _looks_like_name(q):
        return NAME
    return THEMATIC


@dataclass(frozen=True)
class RouteDecision:
    """A routed query: its class plus the per-class retrieval settings.

    `find_quote` and `query` mirror what `_prepare` needs: when the class is
    quote, `find_quote` is True and `query` is the tail-stripped passage;
    otherwise `find_quote` is False and `query` is unchanged. `expand_query`
    is the HyDE recommendation — a later item (4.2) decides whether to act on
    it; the router only records it.
    """

    query_class: QueryClass
    query: str
    find_quote: bool
    bm25_weight: float | None
    include_catalog: bool
    expand_query: bool
    reason: str


def route(
    query: str, settings: Settings, *, history_present: bool = False,
) -> RouteDecision:
    """Classify `query` and resolve its per-class retrieval settings.

    The weight values come from `settings` (so `quote_bm25_weight` etc. stay
    configurable); `include_catalog` is off for every class (the headline
    lever). `bm25_weight=None` means "use the pipeline default".
    """
    cls = classify(query, history_present=history_present)
    is_quote, cleaned = _quote_class(query) if cls == QUOTE else (False, query)

    if cls == QUOTE:
        return RouteDecision(
            QUOTE, cleaned, True, settings.quote_bm25_weight, False, False,
            "verbatim Devanagari passage",
        )
    if cls == NAME:
        return RouteDecision(
            NAME, query, False, settings.name_bm25_weight, False, False,
            "person-lookup shape",
        )
    if cls == ANALYTIC:
        # Analytic queries still need retrieval context in /api/query; route
        # them lexically with catalog off. Dispatching to the Postgres
        # analytics endpoints is a separate integration (those endpoints exist).
        return RouteDecision(
            ANALYTIC, query, False, None, False, False,
            "count / list intent",
        )
    if cls == FOLLOWUP:
        # A follow-up inherits its underlying class after a rewrite (4.3);
        # until then, retrieve like the thematic default with catalog off.
        return RouteDecision(
            FOLLOWUP, query, False, None, False, True,
            "conversational follow-up",
        )
    return RouteDecision(
        THEMATIC, query, False, None, False, True,
        "thematic (default)",
    )
