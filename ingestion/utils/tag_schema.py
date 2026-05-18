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
