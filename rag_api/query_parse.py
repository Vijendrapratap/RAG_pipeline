"""Deterministic filter extraction from natural-language queries.

A query like "karma yoga discourses from monsoon 2015 at noida" carries
filterable signals — a year, a season, a place. Pulling them out and turning
them into metadata filters sharpens retrieval: the dense / BM25 arms then only
score chunks from the right files, instead of ranking the whole corpus and
hoping the metadata-irrelevant noise sorts itself.

No LLM here — extraction is regex + vocabulary matching, so it is fast, free,
and reproducible. Two confidence tiers:

  * **strong** — bare years and ISO dates. Unambiguous; auto-applied.
  * **soft**   — a season / place / topic-tag word found in the query. A word
    like "winter" is as likely to be the *subject* of the question as a filter
    on it, so soft signals are surfaced as suggestions, never auto-applied.

Every detection is reported back to the caller (and the UI), so a wrong guess
is visible and removable rather than a silent quality regression.

Pure, dependency-free, unit-tested.
"""
from __future__ import annotations

import datetime
import re
from typing import Any, NamedTuple

CONFIDENCE_STRONG = "strong"
CONFIDENCE_SOFT = "soft"

# Devanagari Unicode block — used to recognise a pasted Hindi verbatim quote.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
# A trailing run of romanized / Latin text appended after a Devanagari quote,
# e.g. "... ye kab kaha and kya bola gaya tha". Latin letters, digits, spaces
# and light punctuation only — no Devanagari.
_LATIN_TAIL_RE = re.compile(r"[\sA-Za-z0-9?.,!;:'\"()\-–—]+$")

# A bare 4-digit year, not embedded in a longer number.
_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
# An ISO calendar date.
_ISO_DATE_RE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_WS_RE = re.compile(r"\s+")

# Vocab key (as returned by rag_api.db.get_filter_options) -> filter field.
_VOCAB_FIELDS: tuple[tuple[str, str], ...] = (
    ("seasons", "season"),
    ("locations", "location"),
    ("event_ids", "event_id"),
    ("track_types", "track_type"),
    ("event_types", "event_type"),
    ("primary_languages", "primary_language"),
    ("topics", "topics"),
    ("scriptures_referenced", "scriptures_referenced"),
    ("performers", "performers"),
)
# Filter fields that accept a list (OR within the field).
_LIST_FIELDS = frozenset({"track_type", "topics", "scriptures_referenced", "performers"})
# Vocab values shorter than this are too ambiguous to match on.
_MIN_VOCAB_LEN = 3


def _normalise(text: str) -> str:
    """Lowercase and flatten separators so a vocab value matches regardless of
    hyphen / underscore styling ("karma-yoga" == "karma yoga"). Devanagari is
    unaffected by lowercasing."""
    t = text.lower().replace("-", " ").replace("_", " ").replace("/", " ")
    return _WS_RE.sub(" ", t).strip()


def _valid_iso(token: str) -> bool:
    try:
        datetime.date.fromisoformat(token)
        return True
    except ValueError:
        return False


def detect_signals(
    query: str, vocab: dict[str, list[Any]] | None = None
) -> list[dict[str, Any]]:
    """Find filterable signals in `query`.

    Returns a list of detection dicts — ``{field, value, matched, confidence}``
    — one per signal found. `vocab` is the filter-option dict from
    `rag_api.db.get_filter_options`; an empty / missing vocab still yields the
    date signals (those need no vocabulary). Pure: no I/O.
    """
    vocab = vocab or {}
    detections: list[dict[str, Any]] = []

    # ---- strong: dates --------------------------------------------------
    iso = sorted({m.group(0) for m in _ISO_DATE_RE.finditer(query)
                  if _valid_iso(m.group(0))})
    # Blank ISO dates before scanning bare years, so 2010-01-07 is not also
    # read as the year 2010.
    yearless = _ISO_DATE_RE.sub(" ", query)
    years = sorted({m.group(0) for m in _YEAR_RE.finditer(yearless)})

    if iso:
        detections.append({
            "field": "date_range", "value": [iso[0], iso[-1]],
            "matched": ", ".join(iso), "confidence": CONFIDENCE_STRONG,
        })
    elif years:
        detections.append({
            "field": "date_range",
            "value": [f"{years[0]}-01-01", f"{years[-1]}-12-31"],
            "matched": ", ".join(years), "confidence": CONFIDENCE_STRONG,
        })

    # ---- soft: vocabulary matches --------------------------------------
    padded = f" {_normalise(query)} "
    seen: set[tuple[str, str]] = set()
    for vocab_key, field in _VOCAB_FIELDS:
        for value in vocab.get(vocab_key) or []:
            if value is None:
                continue
            norm = _normalise(str(value))
            if len(norm) < _MIN_VOCAB_LEN or f" {norm} " not in padded:
                continue
            key = (field, str(value))
            if key in seen:
                continue
            seen.add(key)
            detections.append({
                "field": field, "value": value, "matched": norm,
                "confidence": CONFIDENCE_SOFT,
            })
    return detections


def signals_to_filters(
    detections: list[dict[str, Any]], include_soft: bool = False
) -> dict[str, Any]:
    """Collapse detections into a filter dict.

    Strong signals always count; soft ones only when `include_soft` is set.
    List-valued fields accumulate; scalar fields take the last detection.
    """
    filters: dict[str, Any] = {}
    for d in detections:
        if d["confidence"] == CONFIDENCE_SOFT and not include_soft:
            continue
        field, value = d["field"], d["value"]
        if field == "date_range":
            filters["date_range"] = tuple(value)
        elif field in _LIST_FIELDS:
            bucket = filters.setdefault(field, [])
            if value not in bucket:
                bucket.append(value)
        else:
            filters[field] = value
    return filters


def merge_filters(
    explicit: dict[str, Any] | None, auto: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge user-set and auto-extracted filters.

    A field the user set explicitly always wins; auto-extracted filters only
    fill gaps. Returns a lean dict — only fields with a non-None value.
    """
    merged = {k: v for k, v in (explicit or {}).items() if v is not None}
    for k, v in (auto or {}).items():
        if merged.get(k) is None:
            merged[k] = v
    return merged


class QuoteDetection(NamedTuple):
    """Result of `detect_quote`.

    is_quote      — route this query to lexical quote search.
    query         — the query to retrieve with (trailing romanized tail removed
                    when `is_quote`, else the original verbatim).
    stripped_tail — the removed romanized meta-question, for transparency ("").
    """
    is_quote: bool
    query: str
    stripped_tail: str


def _last_devanagari_index(s: str) -> int:
    for i in range(len(s) - 1, -1, -1):
        if "ऀ" <= s[i] <= "ॿ":
            return i
    return -1


def detect_quote(query: str) -> QuoteDetection:
    """Decide whether a query is a pasted Devanagari verbatim passage the user
    wants *located* (→ lexical quote search), and strip any trailing romanized
    meta-question.

    Users naturally paste a remembered line plus a question — e.g.
    ``"<long Hindi quote> ye kab kaha and kya bola gaya tha"`` — into the main
    box. Run as a normal semantic query that fails: the English/romanized tail
    dilutes the dense vector and the right chunk ranks far down. Detected as a
    quote and cleaned, the same text resolves to the exact sitting.

    Conservative by design — only a long, mostly-Devanagari passage qualifies,
    so ordinary (short) Hindi questions keep the normal semantic path. A false
    positive degrades gracefully: quote search is hybrid with a high BM25 weight,
    so the dense arm still contributes. Pure: no I/O.
    """
    raw = (query or "").strip()
    if not raw:
        return QuoteDetection(False, query, "")

    # Strip a romanized tail that follows the last Devanagari character.
    cleaned, tail = raw, ""
    last = _last_devanagari_index(raw)
    if last != -1:
        candidate = raw[last + 1:]
        if candidate.strip() and _LATIN_TAIL_RE.fullmatch(candidate):
            cleaned = raw[: last + 1].strip()
            tail = candidate.strip()

    words = cleaned.split()
    letters = [c for c in cleaned if c.isalpha()]
    deva = _DEVANAGARI_RE.findall(cleaned)
    deva_ratio = (len(deva) / len(letters)) if letters else 0.0

    long_passage = len(words) >= 12 and len(cleaned) >= 60
    is_quote = deva_ratio >= 0.6 and (long_passage or (bool(tail) and len(words) >= 8))

    if not is_quote:
        return QuoteDetection(False, query, "")
    return QuoteDetection(True, cleaned, tail)


