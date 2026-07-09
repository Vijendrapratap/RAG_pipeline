"""Unit tests for Phase 13 content-tag helpers.

Covers the pure-function module (tag_schema.py) and the small pure parts
of enrich_content_tags.py. The Ollama/Postgres/Qdrant integration is out
of scope for unit tests — the smoke/integration story lives in the
manual end-to-end path described in PRD §6 Phase 13 acceptance criteria.
"""
from __future__ import annotations

import json

import pytest

from ingestion import enrich_content_tags as enrich
from ingestion.utils.tag_schema import (
    ALLOWED_EVENT_TYPES,
    ALLOWED_LANGUAGES,
    REQUIRED_KEYS,
    build_mapreduce_chunk_prompt,
    build_prompt,
    parse_model_json,
    validate_tags,
)


# ---- build_prompt -------------------------------------------------------


def test_build_prompt_includes_all_schema_keys() -> None:
    prompt = build_prompt("hello world")
    for k in REQUIRED_KEYS:
        assert f'"{k}"' in prompt, f"prompt missing key: {k}"


def test_build_prompt_appends_transcript_last() -> None:
    transcript = "UNIQUE_MARKER_12345"
    prompt = build_prompt(transcript)
    # Transcript must appear AFTER the schema and rules so it stays in the
    # model's most-recent attention window for long inputs.
    schema_pos = prompt.find("event_type")
    transcript_pos = prompt.find(transcript)
    assert schema_pos > 0
    assert transcript_pos > schema_pos


def test_build_prompt_demands_devanagari_for_hindi_summary() -> None:
    prompt = build_prompt("x")
    assert "Devanagari" in prompt
    assert "WHOLE transcript" in prompt


def test_build_prompt_demands_verbatim_clues() -> None:
    prompt = build_prompt("x")
    assert "verbatim" in prompt.lower()


def test_build_mapreduce_chunk_prompt_returns_json_schema() -> None:
    p = build_mapreduce_chunk_prompt("fragment text")
    assert '"summary"' in p
    assert '"people"' in p
    assert '"places"' in p
    assert '"scriptures"' in p
    assert "fragment text" in p


# ---- validate_tags ------------------------------------------------------


def _valid_tags() -> dict:
    return {
        "event_type": "discourse",
        "primary_language": "hindi",
        "topics": ["karma-yoga", "self-inquiry", "guru-bhakti"],
        "people_named": ["Anush"],
        "places_named": ["Noida"],
        "scriptures_referenced": ["Bhagavad Gita"],
        "timing_clues": ["आज सुबह"],
        "location_clues": ["यह नोएडा शिविर है"],
        "summary_hindi": "गुरु जी ने कर्म योग पर प्रवचन दिया।",
        "summary_english": "Guruji gave a discourse on karma yoga.",
    }


def test_validate_accepts_well_formed_object() -> None:
    ok, errs = validate_tags(_valid_tags())
    assert ok, errs
    assert errs == []


def test_validate_rejects_non_dict() -> None:
    ok, errs = validate_tags(["not", "a", "dict"])
    assert not ok
    assert any("not a dict" in e for e in errs)


def test_validate_rejects_missing_keys() -> None:
    tags = _valid_tags()
    del tags["summary_hindi"]
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("missing required keys" in e for e in errs)
    assert any("summary_hindi" in e for e in errs)


def test_validate_rejects_extra_keys() -> None:
    tags = _valid_tags()
    tags["unexpected"] = "value"
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("unexpected keys" in e for e in errs)


def test_validate_rejects_unknown_event_type() -> None:
    tags = _valid_tags()
    tags["event_type"] = "lecture"  # not in enum
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("event_type" in e and "lecture" in e for e in errs)


def test_validate_rejects_unknown_language() -> None:
    tags = _valid_tags()
    tags["primary_language"] = "english"  # not in enum
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("primary_language" in e for e in errs)


def test_validate_rejects_empty_summaries() -> None:
    tags = _valid_tags()
    tags["summary_hindi"] = "   "
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("summary_hindi" in e and "non-empty" in e for e in errs)


def test_validate_rejects_array_with_non_strings() -> None:
    tags = _valid_tags()
    tags["topics"] = ["karma-yoga", 42, "self-inquiry"]
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("topics" in e for e in errs)


def test_validate_accepts_empty_arrays() -> None:
    tags = _valid_tags()
    tags["scriptures_referenced"] = []
    tags["timing_clues"] = []
    tags["location_clues"] = []
    ok, errs = validate_tags(tags)
    assert ok, errs


def test_allowed_enums_match_doc_contract() -> None:
    # Sanity check: the enums the validator enforces match what we tell
    # the model in the prompt. Catches accidental drift if someone edits
    # one and forgets the other.
    assert ALLOWED_EVENT_TYPES == frozenset({
        "satsang", "bhajan", "meditation", "qa",
        "discourse", "mixed", "unknown",
    })
    assert ALLOWED_LANGUAGES == frozenset({"hindi", "sanskrit", "mixed"})


# ---- parse_model_json ---------------------------------------------------


def test_parse_clean_json() -> None:
    raw = json.dumps(_valid_tags())
    obj, err = parse_model_json(raw)
    assert err is None
    assert obj is not None
    assert obj["event_type"] == "discourse"


def test_parse_strips_markdown_fences() -> None:
    raw = "```json\n" + json.dumps(_valid_tags()) + "\n```"
    obj, err = parse_model_json(raw)
    assert err is None, err
    assert obj is not None
    assert obj["primary_language"] == "hindi"


def test_parse_strips_bare_fences() -> None:
    raw = "```\n" + json.dumps(_valid_tags()) + "\n```"
    obj, err = parse_model_json(raw)
    assert err is None, err
    assert obj is not None


def test_parse_rejects_empty() -> None:
    obj, err = parse_model_json("")
    assert obj is None
    assert err == "empty response"


def test_parse_rejects_garbage() -> None:
    obj, err = parse_model_json("this is not JSON at all")
    assert obj is None
    assert err is not None
    assert "json decode failed" in err


def test_parse_rejects_non_object() -> None:
    obj, err = parse_model_json('["array", "at", "top"]')
    assert obj is None
    assert err is not None
    assert "not a JSON object" in err


# ---- end-to-end pure pipeline (parse + validate) ------------------------


def test_full_pipeline_clean_model_output() -> None:
    """A realistic clean model output should parse and validate."""
    raw = json.dumps(_valid_tags())
    obj, perr = parse_model_json(raw)
    assert perr is None
    ok, errs = validate_tags(obj)
    assert ok, errs


def test_full_pipeline_dead_letters_bad_json() -> None:
    """Bad JSON → parse returns (None, reason). Validator never sees it."""
    raw = '{"event_type": "discourse", "topics": [unfinished'
    obj, perr = parse_model_json(raw)
    assert obj is None
    assert perr is not None


def test_full_pipeline_dead_letters_validation_failure() -> None:
    """Parseable JSON but bad schema → validate returns (False, errs)."""
    bad = _valid_tags()
    bad["event_type"] = "definitely_not_in_enum"
    raw = json.dumps(bad)
    obj, perr = parse_model_json(raw)
    assert perr is None  # JSON is fine
    ok, errs = validate_tags(obj)
    assert not ok
    assert errs  # at least one validation error


# ---- 2.0 durability: dead-letter filenames ------------------------------


def test_dead_letter_long_paths_do_not_collide(tmp_path, monkeypatch) -> None:
    """Two source_files sharing a >100-char prefix must not overwrite each
    other's forensic artifact (the old [:120] slug truncation did)."""
    monkeypatch.setattr(enrich, "DEAD_LETTER_DIR", tmp_path)
    prefix = "Live Masters 2010/" + "A" * 150
    a = prefix + "/one.json"
    b = prefix + "/two.json"
    dst_a = enrich.dead_letter(a, "raw-a", "reason-a")
    dst_b = enrich.dead_letter(b, "raw-b", "reason-b")
    assert dst_a != dst_b
    assert dst_a.exists() and dst_b.exists()
    assert "reason-a" in dst_a.read_text(encoding="utf-8")
    assert "reason-b" in dst_b.read_text(encoding="utf-8")


# ---- 2.0 durability: process_one_file ordering & failure handling -------


class _FakeStore:
    """Records write_tags / load_transcript calls without touching Postgres."""

    def __init__(self, transcript: str = "a short transcript body", calls=None) -> None:
        self._transcript = transcript
        self.write_calls: list[tuple] = []
        self._order = calls  # optional shared call-order log

    def load_transcript(self, source_file: str) -> str:
        return self._transcript

    def write_tags(self, source_file: str, tags: dict, model: str) -> None:
        self.write_calls.append((source_file, model))
        if self._order is not None:
            self._order.append("commit")


def test_process_one_file_dead_letters_escaped_transport_exception(tmp_path, monkeypatch) -> None:
    """A transport error that escapes the retry decorator must produce a
    dead-letter artifact and leave the file untagged (write_tags not called)."""
    monkeypatch.setattr(enrich, "DEAD_LETTER_DIR", tmp_path)

    def _boom(transcript, model):
        raise enrich.requests.ConnectionError("qdrant/ollama down")

    monkeypatch.setattr(enrich, "tag_transcript_single_pass", _boom)
    store = _FakeStore()

    ok, reason = enrich.process_one_file(
        "Some/File.json", store, qclient=object(), model="qwen3.5:9b",
        max_tokens_single_pass=28_000, dry_run=False,
    )

    assert ok is False
    assert "ConnectionError" in reason
    assert store.write_calls == []  # never marked tagged
    artifacts = list((tmp_path / "tag_failures").glob("*.txt"))
    assert len(artifacts) == 1
    assert "tagging call raised" in artifacts[0].read_text(encoding="utf-8")


def test_process_one_file_leaves_untagged_when_propagation_fails(tmp_path, monkeypatch) -> None:
    """Propagate-then-commit: if Qdrant set_payload fails, tagged_at must NOT
    be committed, so resume (WHERE tagged_at IS NULL) retries the file."""
    monkeypatch.setattr(enrich, "DEAD_LETTER_DIR", tmp_path)
    monkeypatch.setattr(
        enrich, "tag_transcript_single_pass",
        lambda transcript, model: (_valid_tags(), "{}", None),
    )

    def _boom(qclient, source_file, tags):
        raise RuntimeError("qdrant restarted mid-run")

    monkeypatch.setattr(enrich, "qdrant_set_payload_for_file", _boom)
    store = _FakeStore()

    ok, reason = enrich.process_one_file(
        "Some/File.json", store, qclient=object(), model="qwen3.5:9b",
        max_tokens_single_pass=28_000, dry_run=False,
    )

    assert ok is False
    assert "qdrant propagation failed" in reason
    assert store.write_calls == []  # crucial: NOT marked tagged


def test_process_one_file_propagates_before_committing(tmp_path, monkeypatch) -> None:
    """Happy path: propagation must run BEFORE the Postgres tagged_at commit."""
    monkeypatch.setattr(enrich, "DEAD_LETTER_DIR", tmp_path)
    order: list[str] = []
    monkeypatch.setattr(
        enrich, "tag_transcript_single_pass",
        lambda transcript, model: (_valid_tags(), "{}", None),
    )
    monkeypatch.setattr(
        enrich, "qdrant_set_payload_for_file",
        lambda qclient, source_file, tags: order.append("propagate"),
    )
    store = _FakeStore(calls=order)

    ok, reason = enrich.process_one_file(
        "Some/File.json", store, qclient=object(), model="qwen3.5:9b",
        max_tokens_single_pass=28_000, dry_run=False,
    )

    assert ok is True and reason is None
    assert order == ["propagate", "commit"]
    assert store.write_calls == [("Some/File.json", "qwen3.5:9b")]


# ---- 2.1 unblock: think=false + installed model default -----------------


def test_ollama_generate_json_sends_think_false(monkeypatch) -> None:
    """qwen3.5:9b is a thinking model. With format=json and thinking ON, the
    output routes into `thinking` and `response` is EMPTY → every file would
    dead-letter as 'empty response'. The tagging call MUST send think=false."""
    captured: dict = {}

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"response": '{"ok": true}'}

    def _fake_post(url, json, timeout):  # noqa: A002 - matches requests.post kwarg
        captured["url"] = url
        captured["body"] = json
        return _Resp()

    monkeypatch.setattr(enrich.requests, "post", _fake_post)
    out = enrich.ollama_generate_json("some prompt", "qwen3.5:9b")

    assert out == '{"ok": true}'
    assert captured["body"]["think"] is False
    assert captured["body"]["format"] == "json"
    assert captured["body"]["stream"] is False


def test_ollama_generate_json_bounds_generation(monkeypatch) -> None:
    """num_predict + repeat_penalty must be sent so a runaway repetition loop
    can't generate until the HTTP read times out (burned ~15 min/file in calib)."""
    captured: dict = {}

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"response": "{}"}

    def _fake_post(url, json, timeout):  # noqa: A002
        captured["body"] = json
        return _Resp()

    monkeypatch.setattr(enrich.requests, "post", _fake_post)
    enrich.ollama_generate_json("p", "qwen3.5:9b")

    opts = captured["body"]["options"]
    assert opts["num_predict"] == enrich.NUM_PREDICT
    assert opts["repeat_penalty"] == enrich.REPEAT_PENALTY


def test_read_timeout_is_not_retried(monkeypatch) -> None:
    """A read timeout means the model is stuck — retrying the identical prompt
    just times out again. Must fail after exactly ONE attempt, not max_tries."""
    calls = {"n": 0}

    def _timeout_post(url, json, timeout):  # noqa: A002
        calls["n"] += 1
        raise enrich.requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(enrich.requests, "post", _timeout_post)
    with pytest.raises(RuntimeError, match="stuck generating"):
        enrich.ollama_generate_json("prompt", "qwen3.5:9b")
    assert calls["n"] == 1  # NOT retried 3x


def test_qdrant_set_payload_uses_points_kwarg_with_filter() -> None:
    """Regression: qdrant-client 1.18 has NO `points_selector` kwarg — the
    selector arg is `points` (a Filter is accepted). The old name raised
    TypeError on every propagation, so with 2.0's propagate-then-commit no file
    could ever be marked tagged."""
    captured: dict = {}

    class _FakeQ:
        def set_payload(self, **kwargs):
            captured.update(kwargs)

    tags = {"event_type": "discourse", "people_named": ["arjun"], "topics": []}
    enrich.qdrant_set_payload_for_file(_FakeQ(), "Some/File.json", tags)

    assert "points_selector" not in captured
    assert isinstance(captured.get("points"), enrich.Filter)
    assert captured["payload"]["event_type"] == "discourse"
    assert captured["payload"]["people_named"] == ["arjun"]
    assert "topics" not in captured["payload"]  # empty lists dropped


def test_tag_model_default_is_installed_model(monkeypatch) -> None:
    """Guard against regressing to the uninstalled qwen2.5:7b default that made
    every enrichment run dead-letter with an empty/absent-model response."""
    import importlib

    monkeypatch.delenv("TAG_MODEL", raising=False)
    reloaded = importlib.reload(enrich)
    try:
        assert reloaded.TAG_MODEL == "qwen3.5:9b"
    finally:
        importlib.reload(enrich)  # restore module state for later tests
