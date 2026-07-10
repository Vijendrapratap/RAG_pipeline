"""Tag-schema helpers for the Phase 13 content-tagging pass.

Pure functions: prompt builder + validator. No network, no DB. The
enrichment script wires these into a Qwen call and writes the result.
Keeping this module isolated lets us tune the prompt without touching the
network/DB plumbing — and tests stay fast.
"""
from __future__ import annotations

import json
from typing import Any

# The fixed schema the model must return. Keys, types, and enums are
# enforced post-hoc by validate_tags(). Any change here must be matched in
# the Postgres column list and the search_transcripts filter args.
TAG_SCHEMA: dict[str, str] = {
    "event_type": "one of [satsang, bhajan, meditation, qa, discourse, mixed, unknown]",
    "primary_language": "one of [hindi, sanskrit, mixed]",
    "topics": "list of 3-5 short topic tags (lowercase, hyphenated)",
    "people_named": "list of names Guruji mentions (may contain ASR errors)",
    "places_named": "list of places Guruji mentions",
    "scriptures_referenced": "list of scriptures, texts, or sutras cited",
    "timing_clues": "list of verbatim quotes hinting at when this was recorded",
    "location_clues": "list of verbatim quotes hinting at where this was recorded",
    "summary_hindi": "2-3 sentence summary in Hindi (Devanagari script)",
    "summary_english": "2-3 sentence summary in English",
}

ALLOWED_EVENT_TYPES: frozenset[str] = frozenset(
    {"satsang", "bhajan", "meditation", "qa", "discourse", "mixed", "unknown"}
)
ALLOWED_LANGUAGES: frozenset[str] = frozenset({"hindi", "sanskrit", "mixed"})

REQUIRED_KEYS: tuple[str, ...] = tuple(TAG_SCHEMA.keys())
ARRAY_KEYS: frozenset[str] = frozenset({
    "topics", "people_named", "places_named", "scriptures_referenced",
    "timing_clues", "location_clues",
})
STRING_KEYS: frozenset[str] = frozenset({
    "event_type", "primary_language", "summary_hindi", "summary_english",
})

# Hard per-array caps, enforced by the decoding grammar (see build_format_schema).
# Sized to the real corpus rather than to what a tidy response looks like: one
# Panchkula satsang legitimately names 122 distinct people in a roll-call, so a
# people_named cap of 15 would silently truncate real data. The cap exists to
# bound a *runaway*, not to shape a good answer.
ARRAY_MAX_ITEMS: dict[str, int] = {
    "topics": 8,
    "people_named": 200,
    "places_named": 60,
    "scriptures_referenced": 60,
    "timing_clues": 6,
    "location_clues": 6,
}
# Per-string caps. maxItems alone is not enough: on a chant loop the model stops
# repeating across array *items* and starts repeating inside a single *string*
# ("नमो नमो नमो …" for 14k chars), which overruns num_predict and truncates the
# JSON just the same. maxLength is grammar-enforced (measured: summary_hindi came
# back at exactly 1200 chars).
ITEM_MAX_LENGTH: dict[str, int] = {
    "topics": 40,
    "people_named": 60,
    "places_named": 60,
    "scriptures_referenced": 80,
    "timing_clues": 120,
    "location_clues": 120,
}
SUMMARY_MAX_LENGTH = 1200

# The fallback used when the model mirrors a degenerate transcript and fills the
# whole output budget with padding. Worst-case serialised output is
#   5*40 + 15*40 + 10*40 + 10*60 + 4*100 + 4*100 + 2*600 = 3,800 chars
# which at the measured 1.68 chars/token is ~2,260 tokens — provably inside
# NUM_PREDICT. A degenerate file therefore yields a small valid record instead of
# a dead-letter. Real roll-calls never trigger it (they parse on the first try).
BOUNDED_MAX_ITEMS: dict[str, int] = {
    "topics": 5, "people_named": 15, "places_named": 10,
    "scriptures_referenced": 10, "timing_clues": 4, "location_clues": 4,
}
BOUNDED_ITEM_MAX_LENGTH: dict[str, int] = {
    "topics": 40, "people_named": 40, "places_named": 40,
    "scriptures_referenced": 60, "timing_clues": 100, "location_clues": 100,
}
BOUNDED_SUMMARY_MAX_LENGTH = 600

assert frozenset(ARRAY_MAX_ITEMS) == ARRAY_KEYS, "ARRAY_MAX_ITEMS must cover ARRAY_KEYS"
assert frozenset(ITEM_MAX_LENGTH) == ARRAY_KEYS, "ITEM_MAX_LENGTH must cover ARRAY_KEYS"


def build_format_schema(
    max_items: dict[str, int] | None = None,
    item_max_length: dict[str, int] | None = None,
    summary_max_length: int = SUMMARY_MAX_LENGTH,
) -> dict[str, Any]:
    """The JSON Schema handed to Ollama's `format` parameter.

    Ollama compiles this into a GBNF decoding grammar, so `maxItems`, `maxLength`
    and the two enums become *structural* constraints the model physically cannot
    violate — unlike the same rules stated in the prompt, which a model mirroring
    a repetitive transcript will happily ignore until num_predict cuts the JSON
    mid-string and the file dead-letters.

    `uniqueItems` is deliberately absent: llama.cpp's grammar compiler ignores it
    (measured — byte-identical output, same token count, with and without).
    Duplicates are stripped afterwards by dedupe_arrays().
    """
    items = max_items or ARRAY_MAX_ITEMS
    lens = item_max_length or ITEM_MAX_LENGTH
    return {
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "enum": sorted(ALLOWED_EVENT_TYPES)},
            "primary_language": {"type": "string", "enum": sorted(ALLOWED_LANGUAGES)},
            **{
                k: {
                    "type": "array",
                    "maxItems": items[k],
                    "items": {"type": "string", "maxLength": lens[k]},
                }
                for k in items
            },
            "summary_hindi": {"type": "string", "maxLength": summary_max_length},
            "summary_english": {"type": "string", "maxLength": summary_max_length},
        },
        "required": list(REQUIRED_KEYS),
    }


TAG_FORMAT_SCHEMA: dict[str, Any] = build_format_schema()
TAG_FORMAT_SCHEMA_BOUNDED: dict[str, Any] = build_format_schema(
    BOUNDED_MAX_ITEMS, BOUNDED_ITEM_MAX_LENGTH, BOUNDED_SUMMARY_MAX_LENGTH
)

# The map phase of map-reduce returns a different shape than the tag object.
# It gets its own grammar for the same reason: fragments of a roll-call bhajan
# are exactly the input that makes an unconstrained array run away.
MAPREDUCE_FRAGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "people": {"type": "array", "items": {"type": "string"}, "maxItems": 60},
        "places": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
        "scriptures": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
    },
    "required": ["summary", "people", "places", "scriptures"],
}


def dedupe_list(items: list[str]) -> list[str]:
    """Order-preserving, case- and whitespace-insensitive dedupe.

    First surface form of each name wins, so 'Guru Nanak' survives and a later
    'guru nanak' does not. Internal whitespace is collapsed on the kept value.
    """
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if not isinstance(x, str):
            continue
        key = " ".join(x.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(" ".join(x.split()))
    return out


def dedupe_arrays(tags: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `tags` with every array field deduped and capped.

    The grammar bounds array *length*, not array *content*: given a transcript
    that repeats one name 600 times the model still emits it up to maxItems
    times. This is the deterministic half of that fix.
    """
    out = dict(tags)
    for k in ARRAY_KEYS:
        v = out.get(k)
        if isinstance(v, list):
            out[k] = dedupe_list(v)[: ARRAY_MAX_ITEMS[k]]
    return out


def _caps_sentence() -> str:
    return ", ".join(f"{k} {n}" for k, n in ARRAY_MAX_ITEMS.items())


def build_prompt(transcript: str) -> str:
    """Build the Qwen prompt. Schema-strict; demands valid JSON output.

    The model is told (a) what each field means, (b) that arrays may be
    empty, (c) that summaries must reflect the *entire* transcript, and
    (d) that timing/location clues must be verbatim quotes — not
    paraphrases. The transcript itself is appended last so long contexts
    don't push the schema out of the model's recency window.
    """
    schema_lines = "\n".join(f'  "{k}": {v}' for k, v in TAG_SCHEMA.items())
    return (
        "You are a metadata extractor for spiritual-discourse transcripts.\n"
        "Read the transcript below and return a SINGLE JSON object with "
        "exactly these keys (no extra keys, no markdown fences):\n\n"
        "{\n"
        f"{schema_lines}\n"
        "}\n\n"
        "Rules:\n"
        "- summary_hindi MUST be in Devanagari script and cover the WHOLE transcript.\n"
        "- summary_english MUST cover the WHOLE transcript.\n"
        "- timing_clues and location_clues MUST be verbatim quotes from the transcript "
        "(copy the exact words). If none, use [].\n"
        "- topics MUST be 3-5 short lowercase hyphenated tags (e.g. 'karma-yoga', "
        "'self-inquiry').\n"
        "- event_type MUST be one of: satsang, bhajan, meditation, qa, discourse, "
        "mixed, unknown. Use 'unknown' if genuinely unclear.\n"
        "- primary_language MUST be one of: hindi, sanskrit, mixed.\n"
        "- Arrays may be empty ([]) if nothing applies.\n"
        # Refrain-heavy bhajans and roll-calls make the model mirror the input's
        # repetition, emitting one array element hundreds of times. Stating the
        # rule here is advisory only — the model ignores it on exactly the inputs
        # that need it (measured: 136 of 163 failing arrays blew a 15-item prompt
        # cap). The load-bearing enforcement is the grammar in TAG_FORMAT_SCHEMA
        # plus dedupe_arrays(); this sentence only nudges the easy cases.
        "- Every array MUST contain only UNIQUE values. NEVER repeat a value "
        "within an array, and never pad an array with duplicates to make it "
        f"longer. Maximum items: {_caps_sentence()}.\n"
        "- Return ONLY the JSON object. No prose before or after.\n\n"
        "Transcript:\n"
        "---\n"
        f"{transcript}\n"
        "---\n"
    )


def build_mapreduce_chunk_prompt(chunk_text: str) -> str:
    """For the map phase of map-reduce on very long files.

    Each chunk gets a 2-sentence summary + local tag extraction. The
    reduce phase then runs build_prompt() on the concatenated summaries.
    """
    return (
        "Summarize this transcript fragment in 2 sentences (English only) and "
        "extract any people, places, or scriptures named. Return a JSON object:\n"
        '{"summary": "...", "people": [...], "places": [...], "scriptures": [...]}\n\n'
        "Fragment:\n"
        "---\n"
        f"{chunk_text}\n"
        "---\n"
    )


def validate_tags(obj: Any) -> tuple[bool, list[str]]:
    """Return (is_valid, errors). Empty errors list iff is_valid.

    Checks: type is dict, all required keys present, each key has the
    right value type, event_type and primary_language are in their enum
    sets, summaries are non-empty strings.
    """
    errors: list[str] = []

    if not isinstance(obj, dict):
        return False, [f"top-level value is not a dict (got {type(obj).__name__})"]

    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        errors.append(f"missing required keys: {missing}")

    extra = [k for k in obj.keys() if k not in REQUIRED_KEYS]
    if extra:
        errors.append(f"unexpected keys: {extra}")

    for k in ARRAY_KEYS:
        v = obj.get(k)
        if v is None:
            continue
        if not isinstance(v, list):
            errors.append(f"{k!r} must be a list (got {type(v).__name__})")
            continue
        if not all(isinstance(x, str) for x in v):
            errors.append(f"{k!r} must be a list of strings")
        # Belt to the grammar's braces: if the caller ever drops TAG_FORMAT_SCHEMA
        # and reverts to format="json", a runaway array fails loudly here instead
        # of being written to Postgres and fanned out to every chunk payload.
        if len(v) > ARRAY_MAX_ITEMS[k]:
            errors.append(f"{k!r} has {len(v)} items, exceeds cap {ARRAY_MAX_ITEMS[k]}")

    for k in STRING_KEYS:
        v = obj.get(k)
        if v is None:
            continue
        if not isinstance(v, str):
            errors.append(f"{k!r} must be a string (got {type(v).__name__})")

    et = obj.get("event_type")
    if isinstance(et, str) and et not in ALLOWED_EVENT_TYPES:
        errors.append(f"event_type {et!r} not in {sorted(ALLOWED_EVENT_TYPES)}")

    pl = obj.get("primary_language")
    if isinstance(pl, str) and pl not in ALLOWED_LANGUAGES:
        errors.append(f"primary_language {pl!r} not in {sorted(ALLOWED_LANGUAGES)}")

    for k in ("summary_hindi", "summary_english"):
        v = obj.get(k)
        if isinstance(v, str) and not v.strip():
            errors.append(f"{k!r} must be non-empty")

    return (len(errors) == 0), errors


def parse_model_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the model's response into a dict. Returns (obj, error).

    Tolerates: markdown ``` fences the model adds anyway, leading/trailing
    whitespace. Does NOT tolerate: missing braces, truncated JSON. On
    failure returns (None, reason) so caller can dead-letter the raw
    response intact.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty response"
    text = raw.strip()
    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first line (```json or ```) and last line if it's ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"json decode failed: {e}"
    if not isinstance(obj, dict):
        return None, f"top-level value is not a JSON object (got {type(obj).__name__})"
    return obj, None
