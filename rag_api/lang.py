"""Language detection for bilingual (Hindi / English) query handling.

The transcript corpus is Hindi (Swami ji's discourses); queries arrive in
either Hindi (Devanagari) or English. We detect the query's script so the
answer can be written back in the same language.

Pure, dependency-free, unit-testable.
"""
from __future__ import annotations

LANG_HINDI = "hindi"
LANG_ENGLISH = "english"


def _is_devanagari(ch: str) -> bool:
    """True for the Devanagari Unicode block U+0900–U+097F (Hindi + Sanskrit,
    including vowel-sign matras and the danda)."""
    return "ऀ" <= ch <= "ॿ"


def _is_latin_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def detect_language(text: str, threshold: float = 0.20) -> str:
    """Return ``'hindi'`` or ``'english'`` for a query string.

    Counts Devanagari vs Latin letters; if Devanagari is at least
    ``threshold`` of the letters the query is Hindi. A low threshold is
    deliberate — a romanised query peppered with one Hindi word
    ("barsat के बारे में") is still a Hindi query. Text with no letters at
    all (digits, punctuation) defaults to English.
    """
    deva = sum(1 for c in text if _is_devanagari(c))
    latin = sum(1 for c in text if _is_latin_letter(c))
    total = deva + latin
    if total == 0:
        return LANG_ENGLISH
    return LANG_HINDI if (deva / total) >= threshold else LANG_ENGLISH


def language_label(lang: str) -> str:
    """Human-readable label used inside LLM prompts."""
    return "Hindi (Devanagari script)" if lang == LANG_HINDI else "English"


def resolve_language(requested: str, query: str) -> str:
    """Resolve the answer language.

    ``requested`` is the API's ``answer_language`` field: ``'hindi'`` or
    ``'english'`` force that language; anything else (``'auto'``) detects
    from the query text.
    """
    if requested in (LANG_HINDI, LANG_ENGLISH):
        return requested
    return detect_language(query)
