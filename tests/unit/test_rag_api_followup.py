"""Unit tests for rag_api.followup — conversational follow-up rewriting.

The network call is stubbed; the prompt builder is pure.
"""
from __future__ import annotations

from rag_api.config import Settings
from rag_api.followup import FollowupRewriter, build_rewrite_prompt

HISTORY = [
    {"question": "पुरन सिंह के बारे में बताओ", "answer": "स्वामी जी ने पुरन सिंह की भक्ति का ज़िक्र किया।"},
]


class _FakeResp:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"message": {"content": self._content}}


class _CapturingSession:
    def __init__(self, content: str) -> None:
        self.last_json: dict | None = None
        self._content = content

    def post(self, url: str, json: dict | None = None, timeout=None) -> _FakeResp:
        self.last_json = json
        return _FakeResp(self._content)


def test_rewrite_prompt_includes_history_and_followup():
    p = build_rewrite_prompt("उसके बाद क्या हुआ", HISTORY)
    assert "पुरन सिंह" in p            # the referent from history
    assert "उसके बाद क्या हुआ" in p     # the follow-up itself
    assert "standalone" in p


def test_rewrite_prompt_windows_last_turns():
    long_history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(6)]
    p = build_rewrite_prompt("and then?", long_history)
    assert "q5" in p and "q3" in p      # last 3 turns kept
    assert "q0" not in p                # older turns dropped


def test_rewrite_returns_raw_query_when_no_history():
    r = FollowupRewriter(Settings())
    assert r.rewrite("उसके बाद क्या हुआ", []) == "उसके बाद क्या हुआ"
    assert r.rewrite("uske baad", None) == "uske baad"


def test_rewrite_body_carries_think_false():
    r = FollowupRewriter(Settings())
    r._session = _CapturingSession("पुरन सिंह का ज़िक्र और कहाँ है")
    out = r.rewrite("उसके बाद क्या हुआ", HISTORY)
    assert out == "पुरन सिंह का ज़िक्र और कहाँ है"
    assert r._session.last_json is not None
    assert r._session.last_json.get("think") is False


def test_rewrite_strips_wrapping_quotes():
    r = FollowupRewriter(Settings())
    r._session = _CapturingSession('"पुरन सिंह का ज़िक्र"')
    assert r.rewrite("उसके बाद", HISTORY) == "पुरन सिंह का ज़िक्र"
