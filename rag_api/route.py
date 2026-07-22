"""Deterministic query router (Stage 4.1).

Classifies each query into one of six shapes and maps it to per-class
retrieval settings, so a located quote, a person lookup, a thematic question,
an analytic count and a conversational follow-up each get the retrieval
behaviour they want instead of one-size-fits-all.

    chitchat — a greeting / social pleasantry / meta question; NOT a corpus
               query. Short-circuits BEFORE retrieval (retrieve=False) to a
               fixed reply. Fires first: the reranker scores a bare "hi" at
               ~0.55, so an abstention floor cannot catch these — they have to
               be gated pre-retrieval or the system answers a greeting with
               doctrine.
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

QueryClass = Literal["chitchat", "name", "quote", "thematic", "analytic", "followup"]

CHITCHAT: QueryClass = "chitchat"
NAME: QueryClass = "name"
QUOTE: QueryClass = "quote"
THEMATIC: QueryClass = "thematic"
ANALYTIC: QueryClass = "analytic"
FOLLOWUP: QueryClass = "followup"

# --- chitchat: greeting / social pleasantry / meta — NOT a corpus query ----
# Matched only as (near) the WHOLE short query, so a substantive question that
# merely contains a greeting word ("how are thoughts stilled?") is unaffected:
# a trailing honorific is stripped, punctuation/emoji dropped, then the residue
# must equal a known pleasantry (<=6 words). Bilingual (Latin + Devanagari).
_CHITCHAT_EXACT = frozenset({
    "hi", "hii", "hello", "helo", "hey", "heya", "hiya", "yo",
    "hi there", "hello there", "hello ji", "hi ji", "namaste ji",
    "namaste", "namaskar", "namaskaar", "pranam", "pranaam",
    "ram ram", "radhe radhe", "sat sri akal", "jai gurudev", "jai guru",
    "good morning", "good afternoon", "good evening", "good night", "gm", "gn",
    "how are you", "how are you doing", "how r u", "how are u", "how do you do",
    "hows it going", "how s it going", "how is it going", "whats up", "what s up",
    "wassup", "sup",
    "kaise ho", "kaise hain", "kaise hain aap", "kaise ho aap", "kaisa hai",
    "kya haal", "kya haal hai", "kya haal chaal", "kya chal raha hai",
    "thanks", "thank you", "thankyou", "thx", "ty", "thank u",
    "dhanyavaad", "dhanyawad", "shukriya",
    "bye", "goodbye", "good bye", "alvida", "tata", "ok bye", "see you", "see ya",
    "ok", "okay", "cool", "nice", "great",
    "who are you", "what are you", "what is this", "what is this app",
    "what can you do", "what do you do", "what can i ask", "how does this work",
    "are you a bot", "are you ai", "are you real", "help", "help me",
    "tum kaun ho", "aap kaun ho", "tum kya kar sakte ho", "aap kya kar sakte ho",
    "नमस्ते", "नमस्कार", "प्रणाम", "राम राम", "राधे राधे", "सत श्री अकाल",
    "हैलो", "हेलो", "कैसे हो", "कैसे हैं", "कैसे हैं आप", "क्या हाल",
    "क्या हाल है", "क्या चल रहा है", "धन्यवाद", "शुक्रिया", "अलविदा",
    "तुम कौन हो", "आप कौन हो",
})
# Elongations not worth enumerating (hiii, hellooo, heyyy, yooo).
_CHITCHAT_ELONGATION_RE = re.compile(r"(?i)^(?:h+i+|h+e+l+o+|h+e+y+|yo+)$")
_TRAILING_HONORIFIC_RE = re.compile(
    r"[\s,]*(?:ji|जी|please|plz|kripya|कृपया)\s*$", re.IGNORECASE
)

# Composed-chitchat vocabulary: exact-match alone misses combined or reordered
# pleasantries ("hi hello what you can do"). A short query is chitchat when
# EVERY token is one of these non-content words AND at least one anchor token
# addresses the assistant (greeting / thanks / you-word / app-word). Content
# words (rain, ध्यान, kabir…) are never in this set, so a corpus query cannot
# be swallowed; "it/that/more" are deliberately absent so follow-up anaphora
# ("what about that") still reaches the followup class.
_CHITCHAT_ANCHORS = frozenset({
    "hi", "hii", "hello", "helo", "hey", "heya", "hiya", "yo", "namaste",
    "namaskar", "namaskaar", "pranam", "pranaam", "wassup", "sup",
    "bye", "goodbye", "tata", "alvida",
    "thanks", "thank", "thankyou", "thx", "ty",
    "dhanyavaad", "dhanyawad", "shukriya", "gm", "gn",
    "you", "u", "your", "yourself", "tum", "aap",
    "app", "dashboard", "bot", "ai", "assistant", "chatbot", "help", "madad",
    "नमस्ते", "नमस्कार", "प्रणाम", "हैलो", "हेलो", "धन्यवाद", "शुक्रिया",
    "अलविदा", "आप", "तुम", "मदद",
})
_CHITCHAT_VOCAB = _CHITCHAT_ANCHORS | frozenset({
    "good", "morning", "afternoon", "evening", "night", "ok", "okay",
    "what", "who", "how", "why", "are", "is", "am", "do", "does", "did",
    "can", "could", "would", "should", "tell", "say", "ask", "know",
    "i", "me", "my", "we", "a", "an", "the", "this", "there", "here",
    "for", "of", "and",
    "about", "things", "stuff", "question", "questions", "answer", "answers",
    "name", "please", "plz", "so", "much", "ji",
    "kya", "kar", "karo", "sakte", "sakta", "sakti", "ho", "hain", "hai",
    "kaun", "kaise", "mujhe", "batao", "bata", "kripya",
    "क्या", "कर", "करो", "सकते", "सकती", "सकता", "हो", "हैं", "है",
    "कौन", "कैसे", "मुझे", "बताओ", "जी", "कृपया",
})

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

# --- romanized (Hinglish) name cues + shape --------------------------------
# The live gap: "puran singh ke baare mein" classified thematic because every
# name cue above is Devanagari. Same structural idea, Latin script: a person
# cue ("ke baare mein" / "ki katha" / "ka zikr" / "kaun tha") preceded by a
# 2+ token run of name-eligible words. Spelling variants (bare/baare, me/mein,
# zikr/jikr) are folded into the regex, not enumerated.
_ROMAN_NAME_CUE_RE = re.compile(
    r"(?ix)"
    r"\bke\s+baa?re\s+me(?:i?n)?\b|\bki\s+katha\b|\bki\s+kaha?ni\b|"
    r"\bka\s+(?:koi\s+)?[zj]ikr\b|\bkaun\s+th(?:a|e|i)\b"
)
# Romanized mirrors of _SPAN_STOP / _CONCEPT_STOP / _TEACHER_HONORIFICS: words
# that never form part of a personal-name span.
_ROMAN_SPAN_STOP = frozenset({
    "ke", "ka", "ki", "ko", "mein", "me", "se", "par", "aur", "ya", "hai",
    "hain", "koi", "kis", "kin", "kya", "kaun", "kahan", "kab", "kaise",
    "kyon", "kyu", "kyun", "bare", "baare", "zikr", "jikr", "katha",
    "kahani", "kahaani", "naam", "baat", "ji", "ne", "tha", "the", "thi",
    "swami", "guru", "guruji", "gurudev", "sadguru", "satguru", "baba",
    "maharaj", "rishi",
    "about", "what", "who", "did", "say", "said", "tell", "does", "the",
})
_ROMAN_CONCEPT_STOP = frozenset({
    "dhyan", "dhyaan", "prem", "bhakti", "krodh", "gussa", "man", "samarpan",
    "vairagya", "ahankar", "moksha", "maya", "atma", "aatma", "paramatma",
    "shanti", "karuna", "seva", "sadhana", "gyan", "gyaan", "karma", "yog",
    "yoga", "mrityu", "jeevan", "satya", "anand", "shraddha", "meditation",
    "love", "anger", "devotion", "surrender", "death", "life", "truth",
})
_LATIN_TOKEN_RE = re.compile(r"[a-z]+")


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


def _has_romanized_subject_before_cue(query: str) -> bool:
    """Two+ consecutive name-eligible Latin tokens immediately before a
    romanized name cue — "puran singh ke baare mein". The Latin analogue of
    `_has_multiword_subject_before_cue` (Hinglish is lowercase in practice, so
    capitalization carries no signal here)."""
    low = _norm(query)
    m = _ROMAN_NAME_CUE_RE.search(low)
    if not m:
        return False
    toks = _LATIN_TOKEN_RE.findall(low[: m.start()])
    run = 0
    for tok in reversed(toks):
        if tok in _ROMAN_SPAN_STOP or tok in _ROMAN_CONCEPT_STOP or len(tok) < 2:
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
    if _has_romanized_subject_before_cue(query):
        return True
    # English "who is/was <Capitalized Name>" — a person lookup.
    if re.search(r"(?i)\bwho\s+(?:is|was|were)\b", query) and _CAP_NAME_RE.search(query):
        return True
    return False


def _normalize_chitchat(query: str) -> str:
    """Lowercase, strip a trailing honorific, drop punctuation/emoji (keep
    Devanagari + word chars), collapse whitespace — so "Namaste 🙏" and
    "namaste, ji!" both reduce to "namaste".

    The Devanagari block is excluded from the punctuation strip explicitly:
    Python's \\w does NOT match combining vowel marks (matras, category Mn),
    so a bare ``[^\\w\\s]`` strip shreds "कैसे" into "क स". The danda (। ॥)
    sits inside that block, so it is dropped first."""
    q = _TRAILING_HONORIFIC_RE.sub("", (query or "").strip().lower())
    q = re.sub(r"[।॥]", " ", q)
    q = re.sub(r"[^\w\sऀ-ॿ]", " ", q, flags=re.UNICODE)
    return re.sub(r"\s+", " ", q).strip()


def is_chitchat(query: str) -> bool:
    """True when the query is a greeting / social pleasantry / meta question
    rather than a corpus lookup. Pure; matched against the whole short residue
    so a real question containing a greeting word is not swallowed.

    Two rules: the exact-phrase set, then the composed rule — every token in
    the chitchat vocabulary with at least one assistant-addressed anchor
    ("hi hello what you can do"). One content word defeats the composed rule.
    """
    n = _normalize_chitchat(query)
    if not n:
        return False
    toks = n.split()
    if len(toks) > 8:
        return False
    if n in _CHITCHAT_EXACT or bool(_CHITCHAT_ELONGATION_RE.match(n)):
        return True
    return (
        all(t in _CHITCHAT_VOCAB or _CHITCHAT_ELONGATION_RE.match(t) for t in toks)
        and any(t in _CHITCHAT_ANCHORS for t in toks)
    )


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


def _classify(query: str, history_present: bool) -> tuple[QueryClass, str]:
    """Class plus the tail-stripped passage (only meaningful for QUOTE).

    Precedence: chitchat (greeting / meta — never retrieves) → analytic
    (explicit count intent) → quote (verbatim paste) → followup (anaphora, only
    with history) → name (person lookup) → thematic (default fallthrough).
    Chitchat is first because it must never reach the retriever; analytic
    outranks name so "पुरन सिंह का ज़िक्र कितनी बार" is a count, not a lookup.
    Returning the cleaned quote here means `route` never re-runs `detect_quote`.
    """
    q = (query or "").strip()
    if not q:
        return THEMATIC, ""
    if is_chitchat(q):
        return CHITCHAT, ""
    if _ANALYTIC_RE.search(q):
        return ANALYTIC, ""
    is_quote, cleaned = _quote_class(q)
    if is_quote:
        return QUOTE, cleaned
    if history_present and _FOLLOWUP_RE.search(q):
        return FOLLOWUP, ""
    if _looks_like_name(q):
        return NAME, ""
    return THEMATIC, ""


def classify(query: str, history_present: bool = False) -> QueryClass:
    """Deterministic class for `query`. Pure — no I/O, no LLM."""
    return _classify(query, history_present)[0]


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
    # False only for `chitchat`: the handler answers with a fixed reply and
    # never calls the retriever or the synthesiser.
    retrieve: bool = True


# Per-class tuning for the non-quote classes: (expand_query, reason). Every one
# of these has include_catalog=False (the headline lever) and find_quote=False —
# only QUOTE differs — so neither is stored here; bm25_weight is the name class's
# lone override.
#   analytic  — still needs retrieval context in /api/query (catalog off);
#               dispatching to the Postgres analytics endpoints is separate.
#   followup  — inherits its underlying class after a 4.3 rewrite; until then it
#               retrieves like the thematic default.
_TUNING: dict[QueryClass, tuple[bool, str]] = {
    NAME: (False, "person-lookup shape"),
    ANALYTIC: (False, "count / list intent"),
    FOLLOWUP: (True, "conversational follow-up"),
    THEMATIC: (True, "thematic (default)"),
}


def route(
    query: str, settings: Settings, *, history_present: bool = False,
) -> RouteDecision:
    """Classify `query` and resolve its per-class retrieval settings.

    The weight values come from `settings` (so `quote_bm25_weight` etc. stay
    configurable); `include_catalog` is off for every class (the headline
    lever). `bm25_weight=None` means "use the pipeline default".
    """
    cls, cleaned = _classify(query, history_present)
    if cls == CHITCHAT:
        return RouteDecision(
            CHITCHAT, query, False, None, False, False,
            "greeting / small talk", retrieve=False,
        )
    if cls == QUOTE:
        return RouteDecision(
            QUOTE, cleaned, True, settings.quote_bm25_weight, False, False,
            "verbatim Devanagari passage",
        )
    expand_query, reason = _TUNING[cls]
    bm25_weight = settings.name_bm25_weight if cls == NAME else None
    return RouteDecision(
        cls, query, False, bm25_weight, False, expand_query, reason,
    )
