"""Unit tests for rag_api.expand — the HyDE prompt builder and request body.

The network call (QueryExpander.hypothetical) needs a live chat model; the body
test below stubs the session so no model is required.
"""
from __future__ import annotations

from rag_api.config import Settings
from rag_api.expand import QueryExpander, build_hyde_prompt


class _FakeResp:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"message": {"content": self._content}}


class _CapturingSession:
    """Records the JSON body of the last POST instead of hitting the network."""

    def __init__(self, content: str = "a hypothetical passage") -> None:
        self.last_json: dict | None = None
        self._content = content

    def post(self, url: str, json: dict | None = None, timeout=None) -> _FakeResp:
        self.last_json = json
        return _FakeResp(self._content)


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


def test_hyde_request_body_carries_think_false():
    # qwen3.5:9b is a thinking model; without think=False the reasoning chain
    # pollutes the hypothetical passage and degrades the embedding HyDE sharpens.
    exp = QueryExpander(Settings())
    exp._session = _CapturingSession()
    out = exp.hypothetical("क्रोध क्या है")
    assert out == "a hypothetical passage"
    assert exp._session.last_json is not None
    assert exp._session.last_json.get("think") is False
    assert exp._session.last_json.get("stream") is False
