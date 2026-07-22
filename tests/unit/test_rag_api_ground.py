"""Unit tests for rag_api.ground — the post-synthesis groundedness check.

Pure functions (splitter / prompt / verdict parse / apply) plus the
GroundChecker fail-open contract, with the chat call monkeypatched — no
network.
"""
from __future__ import annotations

import rag_api.ground as ground
from rag_api.config import Settings
from rag_api.ground import (
    GroundChecker, apply_verdict, build_ground_prompt, parse_verdict,
    split_sentences,
)

RESULTS = [{"source_file": "a.json", "text": "पुरन सिंह सेवा करते थे", "score": 0.9}]


# ---- split_sentences -------------------------------------------------------

def test_split_sentences_bilingual():
    text = "पुरन सिंह सेवा करते थे। He served with devotion [1]. Was it so?"
    assert split_sentences(text) == [
        "पुरन सिंह सेवा करते थे।",
        "He served with devotion [1].",
        "Was it so?",
    ]


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


# ---- build_ground_prompt ---------------------------------------------------

def test_ground_prompt_numbers_sentences_and_embeds_context():
    p = build_ground_prompt(["First.", "Second."], "[1] METADATA: ...")
    assert "S1: First." in p
    assert "S2: Second." in p
    assert "[1] METADATA: ..." in p
    assert '"unsupported"' in p


# ---- parse_verdict ---------------------------------------------------------

def test_parse_verdict_happy_path():
    assert parse_verdict('{"unsupported": [2]}', 3) == [2]
    assert parse_verdict('{"unsupported": []}', 3) == []


def test_parse_verdict_tolerates_prose_wrapper():
    assert parse_verdict('Here: {"unsupported": [1, 3]} done', 3) == [1, 3]


def test_parse_verdict_rejects_garbage():
    assert parse_verdict("not json at all", 3) is None
    assert parse_verdict('{"supported": [1]}', 3) is None
    assert parse_verdict('{"unsupported": "2"}', 3) is None
    # Out-of-range and boolean entries invalidate the verdict.
    assert parse_verdict('{"unsupported": [4]}', 3) is None
    assert parse_verdict('{"unsupported": [0]}', 3) is None
    assert parse_verdict('{"unsupported": [true]}', 3) is None


def test_parse_verdict_dedupes():
    assert parse_verdict('{"unsupported": [2, 2, 1]}', 3) == [2, 1]


# ---- apply_verdict ---------------------------------------------------------

def test_apply_verdict_drops_flagged_sentences():
    out = apply_verdict(["Keep one.", "Drop this.", "Keep two."], [2])
    assert out == "Keep one. Keep two."


# ---- GroundChecker.check — fail-open contract ------------------------------

def _checker(monkeypatch, reply: str | None) -> GroundChecker:
    c = GroundChecker(Settings())
    monkeypatch.setattr(
        ground, "chat_text",
        lambda *a, **k: reply if reply is not None else "",
    )
    return c


def test_check_drops_unsupported_sentence(monkeypatch):
    c = _checker(monkeypatch, '{"unsupported": [2]}')
    answer, dropped = c.check("Grounded claim [1]. Invented claim.", RESULTS)
    assert answer == "Grounded claim [1]."
    assert dropped == 1


def test_check_keeps_fully_grounded_answer(monkeypatch):
    c = _checker(monkeypatch, '{"unsupported": []}')
    answer, dropped = c.check("Grounded claim [1]. Also grounded [1].", RESULTS)
    assert dropped == 0
    assert answer == "Grounded claim [1]. Also grounded [1]."


def test_check_fails_open_on_empty_reply(monkeypatch):
    c = _checker(monkeypatch, "")
    original = "Claim one. Claim two."
    assert c.check(original, RESULTS) == (original, 0)


def test_check_fails_open_on_unusable_verdict(monkeypatch):
    c = _checker(monkeypatch, "the model rambled instead of JSON")
    original = "Claim one. Claim two."
    assert c.check(original, RESULTS) == (original, 0)


def test_check_never_empties_the_answer(monkeypatch):
    # A verdict rejecting EVERY sentence reads as judge failure — keep answer.
    c = _checker(monkeypatch, '{"unsupported": [1, 2]}')
    original = "Claim one. Claim two."
    assert c.check(original, RESULTS) == (original, 0)


def test_check_skips_when_no_results(monkeypatch):
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        return '{"unsupported": []}'

    c = GroundChecker(Settings())
    monkeypatch.setattr(ground, "chat_text", _boom)
    assert c.check("An answer.", []) == ("An answer.", 0)
    assert called["n"] == 0
