"""Unit tests for rag_api.lang — bilingual query language detection."""
from __future__ import annotations

from rag_api.lang import (
    LANG_ENGLISH,
    LANG_HINDI,
    detect_language,
    language_label,
    resolve_language,
)


def test_pure_hindi_query():
    assert detect_language("स्वामी जी ने कर्म योग के बारे में क्या कहा") == LANG_HINDI


def test_pure_english_query():
    assert detect_language("what did swami ji say about karma yoga") == LANG_ENGLISH


def test_mixed_query_with_devanagari_is_hindi():
    # A romanised query carrying Hindi words still counts as Hindi.
    assert detect_language("barsat के बारे में बताइए") == LANG_HINDI


def test_query_with_no_letters_defaults_english():
    assert detect_language("12345 ?!.") == LANG_ENGLISH
    assert detect_language("") == LANG_ENGLISH


def test_threshold_keeps_mostly_english_as_english():
    # One stray Devanagari char in a long English query stays English.
    assert detect_language("explain the concept of self inquiry क") == LANG_ENGLISH


def test_resolve_language_explicit_overrides_detection():
    assert resolve_language("english", "हिंदी में लिखा प्रश्न") == LANG_ENGLISH
    assert resolve_language("hindi", "an english question") == LANG_HINDI


def test_resolve_language_auto_detects():
    assert resolve_language("auto", "योग क्या है") == LANG_HINDI
    assert resolve_language("auto", "what is yoga") == LANG_ENGLISH


def test_language_label():
    assert "Devanagari" in language_label(LANG_HINDI)
    assert language_label(LANG_ENGLISH) == "English"
