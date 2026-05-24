"""Unit tests for rag_api.expand — the HyDE prompt builder.

The network call (QueryExpander.hypothetical) needs a live chat model; this
covers the pure prompt builder.
"""
from __future__ import annotations

from rag_api.expand import build_hyde_prompt


def test_hyde_prompt_includes_query_and_language():
    p = build_hyde_prompt("what is karma yoga", "English")
    assert "what is karma yoga" in p
    assert "English" in p


def test_hyde_prompt_carries_hindi_query_and_label():
    p = build_hyde_prompt("क्रोध क्या है", "Hindi (Devanagari script)")
    assert "क्रोध क्या है" in p
    assert "Hindi (Devanagari script)" in p


def test_hyde_prompt_demands_passage_only():
    # The model must produce a retrieval aid, not a user-facing answer.
    p = build_hyde_prompt("anything", "English")
    assert "ONLY" in p
    assert "passage" in p
