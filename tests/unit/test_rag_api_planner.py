"""Unit tests for rag_api.planner — the agentic query planner.

Pure functions (detector, prompt, parser, merge) plus the Planner fail-open
contract with the chat call monkeypatched. No network.
"""
from __future__ import annotations

import json

import rag_api.planner as planner_mod
from rag_api.config import Settings
from rag_api.planner import (
    INTENT_ANSWER,
    INTENT_BEST,
    INTENT_LIST,
    INTENT_VERBATIM,
    Plan,
    Planner,
    build_plan_prompt,
    detect_best_intent,
    detect_list_intent,
    detect_verbatim_intent,
    parse_plan,
)
from rag_api.retrieval import merge_result_lists


# ---- detect_list_intent (deterministic, works with the LLM down) ----------

def test_list_intent_count_plus_sitting_word():
    assert detect_list_intent("10 sitting for rain") == 10
    assert detect_list_intent("give me 5 sittings about barish") == 5
    assert detect_list_intent("बारिश पर 5 प्रवचन चाहिए") == 5
    assert detect_list_intent("3 satsangs on gratitude") == 3


def test_list_intent_imperative_without_count_defaults_to_10():
    assert detect_list_intent("suggest sittings about rain") == 10
    assert detect_list_intent("barish par pravachan batao") == 10
    assert detect_list_intent("बारिश पर प्रवचन बताओ") == 10


def test_list_intent_count_is_clamped():
    assert detect_list_intent("500 sittings about rain") == 50


def test_list_intent_ignores_ordinary_questions():
    # Thematic / modal / meta questions must NOT read as list requests.
    for q in [
        "what is rain",
        "बारिश के बारे में स्वामी जी ने क्या कहा",
        "क्या सत्संग में जाना चाहिए",       # modal चाहिए is not a list verb
        "what do the sittings say about rain",  # generic verb, no imperative
        "how are thoughts stilled in meditation",
        "",
    ]:
        assert detect_list_intent(q) is None, q


# ---- detect_verbatim_intent (2026-07-15: word-for-word summary asks) -------

def test_verbatim_intent_positive():
    for q in [
        "summary of the sitting on rain",
        "summarize the pravachan about dhyan",
        "give me the key points of this sitting",
        "what did he say, word to word",
        "बारिश वाले प्रवचन का सारांश दो",
        "मुख्य बातें बताइए",
    ]:
        assert detect_verbatim_intent(q), q


def test_strip_verbatim_words_leaves_the_topic():
    # Retrieval must match the TOPIC, not the meta-ask — the live 2026-07-15
    # probe saw the citation floor drop 10/10 passages on the raw wording.
    from rag_api.planner import strip_verbatim_words
    assert strip_verbatim_words(
        "summary of the sitting on rain, key points word to word"
    ) == "sitting on rain"
    assert strip_verbatim_words("बारिश पर सत्संग मुख्य बिंदु") == "बारिश पर सत्संग"
    assert strip_verbatim_words("वर्षा प्रवचन संक्षेप") == "वर्षा प्रवचन"
    # A query that is ONLY meta-words falls back to itself.
    assert strip_verbatim_words("summary key points") == "summary key points"


def test_plan_verbatim_queries_are_topic_stripped(monkeypatch):
    _patch_reply(monkeypatch, json.dumps({
        "intent": "answer", "n": None,
        "queries": ["बारिश पर सत्संग मुख्य बिंदु", "rain key points"],
    }))
    p = _mk_planner().plan("summary of the sitting on rain")
    assert p.intent == INTENT_VERBATIM
    assert p.queries == ["sitting on rain", "बारिश पर सत्संग", "rain"]


def test_verbatim_intent_negative():
    for q in [
        "what did swami ji say about rain",
        "10 sitting for rain",
        "sansaar me dukh kyon hai",  # "saar" inside a word must not trip \bsaar\b
        "बारिश के बारे में क्या कहा",
        "",
    ]:
        assert not detect_verbatim_intent(q), q


# ---- detect_best_intent (2026-07-15: ranked best-sittings asks) ------------

def test_best_intent_superlative_plus_sitting_word():
    assert detect_best_intent("best sittings on rain") == 10
    assert detect_best_intent("top 5 sittings about rain") == 5
    assert detect_best_intent("5 best sittings for monsoon") == 5
    assert detect_best_intent("which sitting is best for rain") == 10
    assert detect_best_intent("सबसे अच्छे प्रवचन बारिश पर") == 10
    assert detect_best_intent("top 500 sittings on rain") == 50  # clamped


def test_best_intent_needs_both_superlative_and_sitting_word():
    for q in [
        "best way to meditate",          # superlative, no sitting word
        "10 sitting for rain",           # sitting word, no superlative
        "what is the best",              # neither
        "बारिश पर प्रवचन बताओ",           # list ask, not a ranking ask
        "",
    ]:
        assert detect_best_intent(q) is None, q


# ---- build_plan_prompt -----------------------------------------------------

def test_plan_prompt_carries_query_and_demands_json():
    p = build_plan_prompt("10 sitting for rain")
    assert "10 sitting for rain" in p
    assert "JSON" in p
    assert "list_sittings" in p


# ---- parse_plan ------------------------------------------------------------

def test_parse_plan_valid():
    text = json.dumps({
        "intent": "list_sittings", "n": 10,
        "queries": ["बारिश वर्षा", "rain in discourses"],
    })
    got = parse_plan(text)
    assert got == {
        "intent": INTENT_LIST, "n": 10,
        "queries": ["बारिश वर्षा", "rain in discourses"],
    }


def test_parse_plan_extracts_json_from_prose():
    text = 'Sure! Here is the plan:\n{"intent": "answer", "n": null, "queries": ["rain"]}'
    got = parse_plan(text)
    assert got is not None and got["intent"] == INTENT_ANSWER
    assert got["queries"] == ["rain"]


def test_parse_plan_rejects_garbage():
    for bad in ["", "no json here", "[1, 2, 3]", '{"intent": "essay"}',
                '{"queries": ["x"]}']:
        assert parse_plan(bad) is None, bad


def test_parse_plan_sanitizes_n():
    assert parse_plan('{"intent": "list_sittings", "n": true, "queries": []}')["n"] is None
    assert parse_plan('{"intent": "list_sittings", "n": -3, "queries": []}')["n"] is None
    assert parse_plan('{"intent": "list_sittings", "n": 500, "queries": []}')["n"] == 50


def test_parse_plan_query_hygiene():
    # Non-strings skipped, dupes collapsed, prose-length "queries" dropped,
    # list capped at max_queries.
    text = json.dumps({"intent": "answer", "n": None, "queries": [
        "rain", "Rain", 7, "x" * 200, "बारिश", "varsha", "monsoon",
    ]})
    got = parse_plan(text, max_queries=3)
    assert got["queries"] == ["rain", "बारिश", "varsha"]


# ---- Planner.plan (LLM monkeypatched; fail-open contract) ------------------

def _mk_planner() -> Planner:
    return Planner(Settings())


def _patch_reply(monkeypatch, reply: str) -> None:
    monkeypatch.setattr(
        planner_mod, "chat_text",
        lambda *a, **k: reply,
    )


def test_plan_llm_variants_answer_intent(monkeypatch):
    _patch_reply(monkeypatch, json.dumps({
        "intent": "answer", "n": None, "queries": ["बारिश", "rain varsha"],
    }))
    p = _mk_planner().plan("barish ke baare mein kya kaha")
    assert p is not None
    assert p.intent == INTENT_ANSWER and p.n is None
    # Original query always leads; variants follow.
    assert p.queries[0] == "barish ke baare mein kya kaha"
    assert "बारिश" in p.queries and "rain varsha" in p.queries


def test_plan_deterministic_list_intent_survives_llm_garbage(monkeypatch):
    _patch_reply(monkeypatch, "I could not decide, sorry!")
    p = _mk_planner().plan("10 sitting for rain")
    assert p == Plan(intent=INTENT_LIST, queries=["10 sitting for rain"], n=10)


def test_plan_deterministic_intent_outranks_llm_answer(monkeypatch):
    # The LLM downgrades to "answer" but the regex saw an explicit count+word.
    _patch_reply(monkeypatch, json.dumps({
        "intent": "answer", "n": None, "queries": ["बारिश"],
    }))
    p = _mk_planner().plan("10 sitting for rain")
    assert p.intent == INTENT_LIST and p.n == 10
    assert p.queries == ["10 sitting for rain", "बारिश"]


def test_plan_llm_list_intent_without_regex(monkeypatch):
    # Phrasings the regex misses still get list intent from the LLM.
    _patch_reply(monkeypatch, json.dumps({
        "intent": "list_sittings", "n": 7, "queries": ["बारिश", "rain"],
    }))
    p = _mk_planner().plan("i want recordings where swamiji talks of rain")
    assert p.intent == INTENT_LIST and p.n == 7


def test_plan_nothing_to_plan_returns_none(monkeypatch):
    # No list intent, no usable variants -> None -> unchanged pipeline.
    _patch_reply(monkeypatch, "")
    assert _mk_planner().plan("ध्यान कैसे करें") is None


def test_plan_best_intent_outranks_list_and_llm(monkeypatch):
    # "top 5 sittings" trips BOTH the count+sitting list regex and the best
    # detector — best wins; an LLM "answer" verdict cannot downgrade it.
    _patch_reply(monkeypatch, json.dumps({
        "intent": "answer", "n": None, "queries": ["बारिश", "rain"],
    }))
    p = _mk_planner().plan("top 5 sittings about rain")
    assert p.intent == INTENT_BEST and p.n == 5
    assert p.queries == ["top 5 sittings about rain", "बारिश", "rain"]


def test_plan_verbatim_intent_survives_llm_garbage(monkeypatch):
    # Queries come out topic-stripped even with the LLM down.
    _patch_reply(monkeypatch, "cannot help, sorry")
    p = _mk_planner().plan("summary of the sitting on rain")
    assert p == Plan(intent=INTENT_VERBATIM, queries=["sitting on rain"], n=None)


def test_plan_verbatim_intent_keeps_llm_variants(monkeypatch):
    _patch_reply(monkeypatch, json.dumps({
        "intent": "answer", "n": None, "queries": ["बारिश", "rain"],
    }))
    p = _mk_planner().plan("summary of the sitting on rain")
    assert p.intent == INTENT_VERBATIM and p.n is None
    assert p.queries == ["sitting on rain", "बारिश", "rain"]


def test_plan_list_intent_outranks_verbatim(monkeypatch):
    # An explicit count+sitting ask wins over a summary word in the same query.
    _patch_reply(monkeypatch, "")
    p = _mk_planner().plan("list 5 sittings with a summary of rain")
    assert p.intent == INTENT_LIST and p.n == 5


def test_plan_dedupes_variant_equal_to_original(monkeypatch):
    _patch_reply(monkeypatch, json.dumps({
        "intent": "answer", "n": None,
        "queries": ["Barish ke baare mein kya kaha", "बारिश"],
    }))
    p = _mk_planner().plan("barish ke baare mein kya kaha")
    assert p.queries == ["barish ke baare mein kya kaha", "बारिश"]


# ---- merge_result_lists (multi-query merge) --------------------------------

def _r(cid, score, **extra):
    return {"chunk_id": cid, "score": score, "text": "t", **extra}


def test_merge_dedupes_by_chunk_id_keeping_best_score():
    a = [_r("c1", 0.9), _r("c2", 0.5)]
    b = [_r("c2", 0.8), _r("c3", 0.7)]
    merged = merge_result_lists([a, b], top_k=10)
    assert [(r["chunk_id"], r["score"]) for r in merged] == [
        ("c1", 0.9), ("c2", 0.8), ("c3", 0.7)]


def test_merge_sorts_and_caps():
    a = [_r("c1", 0.2)]
    b = [_r("c2", 0.9), _r("c3", 0.4)]
    merged = merge_result_lists([a, b], top_k=2)
    assert [r["chunk_id"] for r in merged] == ["c2", "c3"]


def test_merge_summary_results_key_on_source_file():
    a = [{"chunk_id": None, "source_file": "f1.json", "score": 0.6}]
    b = [{"chunk_id": None, "source_file": "f1.json", "score": 0.9},
         {"chunk_id": None, "source_file": "f2.json", "score": 0.3}]
    merged = merge_result_lists([a, b], top_k=10)
    assert [(r["source_file"], r["score"]) for r in merged] == [
        ("f1.json", 0.9), ("f2.json", 0.3)]


def test_merge_handles_empty_lists():
    assert merge_result_lists([], top_k=5) == []
    assert merge_result_lists([[], []], top_k=5) == []
