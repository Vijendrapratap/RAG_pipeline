"""Unit tests for rag_api.route — the deterministic query router (Stage 4.1).

Pure classification + per-class settings; no network, no Postgres.
"""
from __future__ import annotations

from rag_api.config import Settings
from rag_api.route import (
    ANALYTIC, CHITCHAT, FOLLOWUP, NAME, QUOTE, THEMATIC, classify, is_chitchat,
    route,
)

# A long, mostly-Devanagari verbatim passage (no name cue) — the quote shape.
QUOTE_PASSAGE = (
    "में, कोई परवाह नहीं, उसे कुछ मालूम नहीं, दिवानीगी है, शिकर है, "
    "वो अपने महबूब की गली का दिवाना है"
)


# ---- chitchat --------------------------------------------------------------

def test_chitchat_greetings_classify_as_chitchat():
    for q in ["hi", "Hello!", "hey", "namaste 🙏", "Namaste ji",
              "how are you", "how are you?", "कैसे हैं आप", "नमस्ते",
              "thanks ji", "धन्यवाद", "who are you", "what can you do",
              "tum kaun ho"]:
        assert classify(q) == CHITCHAT, q


def test_chitchat_elongations_match():
    for q in ["hiii", "Hellooo", "heyyyy", "yoo"]:
        assert is_chitchat(q), q


def test_chitchat_never_swallows_real_questions():
    # Queries that CONTAIN a pleasantry word but are corpus lookups.
    for q in [
        "how are thoughts stilled in meditation",
        "hi swami ji ne dhyan ke baare mein kya kaha",
        "what does swami ji say about gratitude",
        "ध्यान कैसे करना चाहिए स्वामी जी",
    ]:
        assert classify(q) != CHITCHAT, q


def test_chitchat_composed_greeting_meta_combinations():
    # Combined / reordered pleasantries the exact set misses (2026-07-15):
    # every token non-content + one assistant-addressed anchor.
    for q in [
        "hi hello what you can do",
        "Hello can you help me",
        "hi what can you do for me",
        "are you there",
        "hii hello ji",
        "क्या आप मदद कर सकते हैं",
    ]:
        assert classify(q) == CHITCHAT, q


def test_chitchat_composed_rule_requires_content_free_query():
    # One content word — or a missing assistant anchor — defeats the composed
    # rule; follow-up anaphora ("that", "more") stays out of the vocabulary so
    # continuations still reach the followup class.
    for q in [
        "what do the sittings say about rain",
        "can you tell me about kabir",
        "what is dhyan",
        "hi swami ji ne kya kaha",
        "what about that",
        "tell me more",
    ]:
        assert classify(q) != CHITCHAT, q


def test_chitchat_route_disables_retrieval():
    s = get_settings()
    d = route("hi", s)
    assert d.query_class == CHITCHAT
    assert d.retrieve is False
    assert d.find_quote is False
    # Every retrieval class keeps retrieve=True.
    assert route("पुरन सिंह के बारे में क्या कहा", s).retrieve is True
    assert route(QUOTE_PASSAGE, s).retrieve is True


# ---- quote ---------------------------------------------------------------

def test_quote_passage_classifies_as_quote():
    assert classify(QUOTE_PASSAGE) == QUOTE


def test_quote_route_strips_tail_and_sets_find_quote():
    d = route(QUOTE_PASSAGE + " ye kab kaha", get_settings())
    assert d.query_class == QUOTE
    assert d.find_quote is True
    # The romanized meta-question tail is stripped from the retrieval query.
    assert "kab kaha" not in d.query
    assert d.include_catalog is False


def test_long_devanagari_question_is_not_a_quote():
    # 12+ Devanagari words trips detect_quote's length test, but a name cue
    # ("के बारे में") marks it a lookup, not a paste — must NOT be quote.
    q = "स्वामी विवेकानंद के बारे में स्वामी जी ने विस्तार से कहाँ बताया"
    assert classify(q) == NAME


# ---- name ----------------------------------------------------------------

def test_name_multiword_subject_with_cue():
    assert classify("पुरन सिंह के बारे में स्वामी जी ने क्या कहा है") == NAME


def test_name_honorific_subject():
    assert classify("आशीष जी के बारे में स्वामी जी ने क्या कहा है") == NAME


def test_name_suffix_subject():
    assert classify("नामदेव की कथा स्वामी जी ने कहाँ सुनाई है") == NAME


def test_name_seed_person_romanized():
    assert classify("who was Swami Rama Tirtha and what did he teach") == NAME


def test_name_romanized_subject_with_cue():
    # The live gap this rule closes: romanized (Hinglish) person lookups used
    # to fall through to thematic because every cue was Devanagari-only.
    assert classify("puran singh ke baare mein") == NAME
    assert classify("swami ji ne puran singh ke bare me kya kaha") == NAME
    assert classify("chetan vishwas ki katha") == NAME


def test_name_romanized_concept_stays_thematic():
    # Structurally identical, but the subject is a (romanized) concept.
    assert classify("dhyan ke baare mein kya kaha") == THEMATIC
    assert classify("prem aur bhakti ke bare me") == THEMATIC


def test_swami_ji_honorific_alone_is_not_a_name_signal():
    # स्वामी जी is on nearly every query; a bare thematic question with it must
    # still fall through to thematic, not name.
    assert classify("ध्यान कैसे करना चाहिए स्वामी जी") == THEMATIC


# ---- thematic ------------------------------------------------------------

def test_concept_about_is_thematic_not_name():
    # Structurally identical to a name query but the subject is a concept.
    assert classify("क्रोध के बारे में स्वामी जी ने क्या कहा है") == THEMATIC


def test_plain_conceptual_question_is_thematic():
    assert classify("प्रेम और भक्ति में क्या अंतर है") == THEMATIC


def test_empty_query_is_thematic():
    assert classify("") == THEMATIC
    assert classify("   ") == THEMATIC


# ---- analytic ------------------------------------------------------------

def test_analytic_how_many():
    assert classify("how many discourses mention Nanak") == ANALYTIC


def test_analytic_devanagari_count():
    assert classify("नामदेव का ज़िक्र कितने प्रवचनों में है") == ANALYTIC


def test_analytic_outranks_name():
    # A count intent wins even when a name cue ("का ज़िक्र") is present.
    assert classify("स्वामी जी ने कितनी बार पुरन सिंह का ज़िक्र किया है") == ANALYTIC


# ---- followup ------------------------------------------------------------

def test_followup_needs_history():
    q = "what about his early life"
    assert classify(q, history_present=True) == FOLLOWUP
    # Same text, no history to resolve against -> falls through to default.
    assert classify(q, history_present=False) == THEMATIC


def test_followup_devanagari_anaphora():
    assert classify("उसके बाद क्या हुआ", history_present=True) == FOLLOWUP


# ---- per-class settings --------------------------------------------------

def get_settings() -> Settings:
    return Settings()


def test_every_retrieval_class_routes_catalog_off():
    s = get_settings()
    for q, hist in [
        ("पुरन सिंह के बारे में क्या कहा", False),
        ("क्रोध के बारे में क्या कहा", False),
        (QUOTE_PASSAGE, False),
        ("how many mention Nanak", False),
        ("उसके बाद क्या हुआ", True),
    ]:
        assert route(q, s, history_present=hist).include_catalog is False


def test_name_class_uses_name_bm25_weight():
    s = get_settings()
    d = route("पुरन सिंह के बारे में क्या कहा", s)
    assert d.query_class == NAME
    assert d.bm25_weight == s.name_bm25_weight


def test_quote_class_uses_quote_bm25_weight():
    s = get_settings()
    d = route(QUOTE_PASSAGE, s)
    assert d.query_class == QUOTE
    assert d.bm25_weight == s.quote_bm25_weight


def test_thematic_recommends_hyde_expansion():
    # The router records a HyDE recommendation for thematic; 4.2 decides to act.
    s = get_settings()
    assert route("प्रेम और भक्ति में क्या अंतर है", s).expand_query is True
    # A precise name lookup must not recommend HyDE (noisy for exact lookups).
    assert route("पुरन सिंह के बारे में क्या कहा", s).expand_query is False
