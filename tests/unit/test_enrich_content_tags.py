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
    ARRAY_KEYS,
    ARRAY_MAX_ITEMS,
    BOUNDED_ITEM_MAX_LENGTH,
    BOUNDED_MAX_ITEMS,
    BOUNDED_SUMMARY_MAX_LENGTH,
    ITEM_MAX_LENGTH,
    REQUIRED_KEYS,
    SUMMARY_MAX_LENGTH,
    TAG_FORMAT_SCHEMA,
    TAG_FORMAT_SCHEMA_BOUNDED,
    build_mapreduce_chunk_prompt,
    build_prompt,
    dedupe_arrays,
    dedupe_list,
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
    """qwen3.5:9b is a thinking model. With a JSON format and thinking ON, the
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
    out = enrich.ollama_generate_json("some prompt", "qwen3.5:9b", TAG_FORMAT_SCHEMA)

    assert out == '{"ok": true}'
    assert captured["body"]["think"] is False
    assert captured["body"]["stream"] is False


def test_ollama_generate_json_sends_grammar_not_bare_json(monkeypatch) -> None:
    """Regression: `format: "json"` lets the model emit unbounded arrays — 658
    items on one bhajan — which overruns num_predict and truncates the JSON
    mid-string. A schema in `format` compiles to a decoding grammar, so maxItems
    is structurally unviolable. The string "json" must never be sent again."""
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
    enrich.ollama_generate_json("p", "qwen3.5:9b", TAG_FORMAT_SCHEMA)

    fmt = captured["body"]["format"]
    assert fmt != "json"
    assert isinstance(fmt, dict)
    for key in ARRAY_KEYS:
        assert fmt["properties"][key]["maxItems"] == ARRAY_MAX_ITEMS[key]
    assert set(fmt["properties"]["event_type"]["enum"]) == ALLOWED_EVENT_TYPES
    assert set(fmt["properties"]["primary_language"]["enum"]) == ALLOWED_LANGUAGES


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
    enrich.ollama_generate_json("p", "qwen3.5:9b", TAG_FORMAT_SCHEMA)

    opts = captured["body"]["options"]
    assert opts["num_predict"] == enrich.NUM_PREDICT
    assert opts["repeat_penalty"] == enrich.REPEAT_PENALTY


def test_num_predict_clears_worst_observed_response() -> None:
    """The worst complete tag response measured on the corpus was 2,860 output
    tokens. At the old 2048 it was cut mid-string and the file dead-lettered."""
    assert enrich.NUM_PREDICT >= 2_860


def test_single_pass_budget_fits_in_context() -> None:
    """transcript + build_prompt overhead + NUM_PREDICT must fit inside NUM_CTX,
    or Ollama silently truncates the prompt (prompt_eval_count pins at num_ctx)."""
    overhead = enrich._estimate_tokens(build_prompt(""))
    assert (
        enrich.DEFAULT_MAX_TOKENS_SINGLE_PASS + overhead + enrich.NUM_PREDICT
        <= enrich.NUM_CTX
    )


def test_chars_per_token_matches_devanagari_density() -> None:
    """Measured against Ollama's prompt_eval_count on six corpus files: 1.68
    chars/token. The old Latin-derived 4 made a 55k-token prompt look like 23k,
    so map-reduce never fired and long files were silently truncated."""
    assert 1.5 <= enrich.CHARS_PER_TOKEN <= 2.0
    # A 92k-char transcript (the corpus maximum) must route to map-reduce.
    assert enrich._estimate_tokens("x" * 92_000) > enrich.DEFAULT_MAX_TOKENS_SINGLE_PASS


def test_read_timeout_is_not_retried(monkeypatch) -> None:
    """A read timeout means the model is stuck — retrying the identical prompt
    just times out again. Must fail after exactly ONE attempt, not max_tries."""
    calls = {"n": 0}

    def _timeout_post(url, json, timeout):  # noqa: A002
        calls["n"] += 1
        raise enrich.requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(enrich.requests, "post", _timeout_post)
    with pytest.raises(RuntimeError, match="stuck generating"):
        enrich.ollama_generate_json("prompt", "qwen3.5:9b", TAG_FORMAT_SCHEMA)
    assert calls["n"] == 1  # NOT retried 3x


# ---- 2.3 correctness: dedupe + cardinality ------------------------------


def test_dedupe_list_is_order_preserving_and_case_insensitive() -> None:
    out = dedupe_list(["Guru Nanak", "guru nanak", "Kabir", "GURU NANAK", "Kabir"])
    assert out == ["Guru Nanak", "Kabir"]  # first surface form wins


def test_dedupe_list_collapses_internal_whitespace() -> None:
    assert dedupe_list(["Guru   Nanak", "Guru Nanak"]) == ["Guru Nanak"]


def test_dedupe_list_drops_empties_and_non_strings() -> None:
    assert dedupe_list(["a", "", "   ", None, 42, "A"]) == ["a"]  # type: ignore[list-item]


def test_dedupe_arrays_kills_the_runaway() -> None:
    """The observed worst case: 658 items, 11 distinct, 'ram' x647. The grammar
    caps length at 200; dedupe must reduce that to the 11 real names."""
    tags = _valid_tags()
    tags["people_named"] = (["ram"] * 190) + [f"name{i}" for i in range(10)]
    out = dedupe_arrays(tags)
    assert out["people_named"] == ["ram"] + [f"name{i}" for i in range(10)]


def test_dedupe_arrays_caps_at_max_items() -> None:
    tags = _valid_tags()
    tags["people_named"] = [f"person{i}" for i in range(500)]
    out = dedupe_arrays(tags)
    assert len(out["people_named"]) == ARRAY_MAX_ITEMS["people_named"]


def test_dedupe_arrays_does_not_clip_a_legitimate_roll_call() -> None:
    """One Panchkula satsang names 122 distinct people. A 15-item cap would have
    thrown 107 real names away; the 200-item cap must leave them alone."""
    tags = _valid_tags()
    tags["people_named"] = [f"rishi kumar {i}" for i in range(122)]
    out = dedupe_arrays(tags)
    assert len(out["people_named"]) == 122


def test_dedupe_arrays_leaves_strings_untouched() -> None:
    tags = _valid_tags()
    out = dedupe_arrays(tags)
    assert out["summary_hindi"] == tags["summary_hindi"]
    assert out["event_type"] == "discourse"


def test_validate_rejects_array_over_cap() -> None:
    """Belt to the grammar's braces: if format= ever reverts to "json", an
    unbounded array must fail loudly, not reach Postgres and Qdrant."""
    tags = _valid_tags()
    tags["people_named"] = ["ram"] * (ARRAY_MAX_ITEMS["people_named"] + 1)
    ok, errs = validate_tags(tags)
    assert not ok
    assert any("exceeds cap" in e and "people_named" in e for e in errs)


def test_single_pass_dedupes_before_returning(monkeypatch) -> None:
    """dedupe must happen inside the tagging path, so both the Postgres write
    and the Qdrant payload fan-out see clean arrays."""
    dirty = _valid_tags()
    dirty["people_named"] = ["Ram", "ram", "RAM", "Kabir"]
    monkeypatch.setattr(
        enrich, "ollama_generate_json",
        lambda prompt, model, fmt: json.dumps(dirty),
    )
    tags, _raw, err = enrich.tag_transcript_single_pass("transcript", "qwen3.5:9b")
    assert err is None
    assert tags["people_named"] == ["Ram", "Kabir"]


# ---- 2.3 correctness: string-length runaway + bounded fallback ----------


def test_schema_caps_every_string_length() -> None:
    """maxItems bounds how MANY items; on a chant loop the model instead repeats
    inside ONE string ('नमो नमो …' x 14k chars) and overruns num_predict just the
    same. Every string in the grammar must carry maxLength."""
    props = TAG_FORMAT_SCHEMA["properties"]
    for key in ARRAY_KEYS:
        assert props[key]["items"]["maxLength"] == ITEM_MAX_LENGTH[key]
    assert props["summary_hindi"]["maxLength"] == SUMMARY_MAX_LENGTH
    assert props["summary_english"]["maxLength"] == SUMMARY_MAX_LENGTH


def test_bounded_schema_worst_case_fits_num_predict() -> None:
    """The whole point of the fallback: its maximal serialised output must fit
    inside NUM_PREDICT, so a degenerate file yields a record, not a dead-letter."""
    worst_chars = sum(
        BOUNDED_MAX_ITEMS[k] * BOUNDED_ITEM_MAX_LENGTH[k] for k in ARRAY_KEYS
    ) + 2 * BOUNDED_SUMMARY_MAX_LENGTH
    worst_tokens = worst_chars / enrich.CHARS_PER_TOKEN
    assert worst_tokens < enrich.NUM_PREDICT, f"{worst_tokens:.0f} tokens >= NUM_PREDICT"


def test_bounded_schema_is_strictly_tighter_than_primary() -> None:
    """validate_tags checks against the primary caps, so bounded output must be a
    subset — otherwise a fallback result could fail validation."""
    for key in ARRAY_KEYS:
        assert BOUNDED_MAX_ITEMS[key] <= ARRAY_MAX_ITEMS[key]
        assert BOUNDED_ITEM_MAX_LENGTH[key] <= ITEM_MAX_LENGTH[key]
    assert BOUNDED_SUMMARY_MAX_LENGTH <= SUMMARY_MAX_LENGTH


def test_single_pass_falls_back_to_bounded_on_truncation(monkeypatch) -> None:
    """A grammar-constrained response can only fail to parse by hitting
    num_predict. Retry once with the bounded schema rather than dead-lettering."""
    seen: list = []
    good = json.dumps(_valid_tags())

    def _gen(prompt, model, fmt):
        seen.append(fmt)
        # first call truncates mid-string, second (bounded) succeeds
        return '{"event_type": "bhajan", "people_named": ["ram' if len(seen) == 1 else good

    monkeypatch.setattr(enrich, "ollama_generate_json", _gen)
    tags, _raw, err = enrich.tag_transcript_single_pass("chant loop", "qwen3.5:9b")

    assert err is None
    assert tags["event_type"] == "discourse"
    assert seen == [TAG_FORMAT_SCHEMA, TAG_FORMAT_SCHEMA_BOUNDED]


def test_single_pass_does_not_retry_when_primary_parses(monkeypatch) -> None:
    """The fallback costs a whole extra generation. Healthy files must not pay."""
    seen: list = []

    def _gen(prompt, model, fmt):
        seen.append(fmt)
        return json.dumps(_valid_tags())

    monkeypatch.setattr(enrich, "ollama_generate_json", _gen)
    enrich.tag_transcript_single_pass("healthy transcript", "qwen3.5:9b")
    assert seen == [TAG_FORMAT_SCHEMA]


def test_single_pass_gives_up_after_one_fallback(monkeypatch) -> None:
    """Exactly one retry. A file the bounded schema cannot tag must dead-letter
    with the reason, not spin."""
    seen: list = []

    def _gen(prompt, model, fmt):
        seen.append(fmt)
        return '{"truncated'

    monkeypatch.setattr(enrich, "ollama_generate_json", _gen)
    tags, _raw, err = enrich.tag_transcript_single_pass("hopeless", "qwen3.5:9b")

    assert tags is None
    assert "bounded-schema retry also failed" in err
    assert len(seen) == 2


def test_validate_failure_does_not_trigger_fallback(monkeypatch) -> None:
    """A validation error (bad enum) is not a truncation — retrying the identical
    prompt would just reproduce it. Only parse failures fall back."""
    seen: list = []
    bad = _valid_tags()
    bad["event_type"] = "announcement"

    def _gen(prompt, model, fmt):
        seen.append(fmt)
        return json.dumps(bad)

    monkeypatch.setattr(enrich, "ollama_generate_json", _gen)
    tags, _raw, err = enrich.tag_transcript_single_pass("x", "qwen3.5:9b")

    assert tags is None
    assert "announcement" in err
    assert seen == [TAG_FORMAT_SCHEMA]  # no retry


# ---- 2.3 correctness: map-reduce actually keeps what it finds ------------


def test_mapreduce_unions_fragment_entities_into_result(monkeypatch) -> None:
    """The reduce pass only sees fragment summaries, so a name mentioned once in
    a 90k-char transcript would be summarised away. Fragment entities must be
    merged back in — long transcripts are where the roll-calls live."""
    calls: list[dict] = []

    def _fake_gen(prompt, model, fmt):
        calls.append(fmt)
        return json.dumps({
            "summary": "a fragment", "people": ["Aashish ji", "aashish ji"],
            "places": ["Dagshai"], "scriptures": [],
        })

    reduced = _valid_tags()
    reduced["people_named"] = ["Swami ji"]
    reduced["places_named"] = []

    monkeypatch.setattr(enrich, "ollama_generate_json", _fake_gen)
    monkeypatch.setattr(
        enrich, "tag_transcript_single_pass",
        lambda transcript, model: (dict(reduced), "{}", None),
    )

    tags, _raw, err = enrich.tag_transcript_mapreduce(
        "x" * 25_000, "qwen3.5:9b", max_chars_per_chunk=10_000
    )
    assert err is None
    assert len(calls) == 3  # 3 fragments, each mapped
    assert all(f is enrich.MAPREDUCE_FRAGMENT_SCHEMA for f in calls)
    # reduce-phase name kept, fragment names unioned in, duplicates collapsed
    assert tags["people_named"] == ["Swami ji", "Aashish ji"]
    assert tags["places_named"] == ["Dagshai"]


def test_mapreduce_uses_fragment_schema_not_tag_schema(monkeypatch) -> None:
    """The map phase returns {summary, people, places, scriptures}, a different
    shape than the tag object. Sending TAG_FORMAT_SCHEMA would force the grammar
    to emit the wrong keys and every fragment would fall back to raw text."""
    seen: list = []
    monkeypatch.setattr(
        enrich, "ollama_generate_json",
        lambda prompt, model, fmt: seen.append(fmt) or json.dumps(
            {"summary": "s", "people": [], "places": [], "scriptures": []}
        ),
    )
    monkeypatch.setattr(
        enrich, "tag_transcript_single_pass",
        lambda transcript, model: (_valid_tags(), "{}", None),
    )
    enrich.tag_transcript_mapreduce("y" * 5_000, "qwen3.5:9b")
    assert seen == [enrich.MAPREDUCE_FRAGMENT_SCHEMA]
    assert "summary" in seen[0]["properties"]
    assert "event_type" not in seen[0]["properties"]


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
