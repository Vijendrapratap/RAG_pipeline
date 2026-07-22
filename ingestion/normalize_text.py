"""Deterministic Hindi text normalization (PRD Phase 17).

Repairs the damage WhisperX and the upstream Qwen 3.5 cleaner leave in
`<stem>.cleaned.txt` *before* it reaches the chunker. Pure functions: no IO, no
network, no model. `load_rules()` is the one function that touches disk.

Scope is deliberately narrow. A rule is allowed to rewrite the transcript only
where the surface form has **no valid reading** — an intra-word script mix like
`मEDITATION`, a Devanagari misspelling of a loanword, a YouTube artifact, an ASR
repetition loop. Everything else is bridged at query time by `data/synonyms.json`
and never touched here, because the speaker really does say `mind` and `watch`
in English (43.9% of `watch`'s Latin occurrences sit inside English sentences)
and rewriting those would corrupt real speech to fix a BM25 problem that a
synonym table solves without risk.

Two rules that looked obvious were measured against the real Tantivy `default`
tokenizer and dropped: it splits on any non-`is_alphanumeric` char, so a glued
danda (`आया।दूसरा`) and an em-dash gloss (`जागरण—enlightened`) already index as
two tokens each. Only `मEDITATION` — Devanagari and Latin letters are *both*
alphanumeric — fuses into one unreachable token. That is the defect worth fixing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A run of the SAME token repeated this many times or more is an ASR loop.
# Below 6 the corpus is dominated by legitimate reduplication: 71,240 length-2
# runs, 3,648 length-3 (`बहुत बहुत बहुत`, `ॐ ॐ ॐ`, `नमो नमो नमो`). Hand-checked
# on 196 runs >= 6: none lost retrievable meaning when collapsed.
LOOP_MIN_RUN = 6

# What a collapsed loop is reduced to. 3 >= the longest legitimate consecutive
# repeat in this corpus, so an emphatic triad or a chant fragment survives intact
# and the "this was repetitive" signal is preserved for the chunker's ratio guard.
LOOP_KEEP = 3

# Stripped before an exact-token lookup and restored after. A glued danda must not
# hide `मEDITATION।` from the token map — but the danda itself is left in place,
# because Tantivy already tokenizes across it.
_EDGE_PUNCT = ",।॥.!?;:\"'()[]{}—–-"

# The Latin parentheticals the cleaner invented: `(Master)`, `(Third Eye)`,
# `(Antarmukhi)`. 4,010 across 305 files. Latin-only inside the parens, so a
# Devanagari aside such as `(यानि भीतर)` is never touched.
_LATIN_GLOSS = re.compile(r"\s*\(\s*([A-Za-z][A-Za-z0-9 '\-/&.]*)\s*\)")

# A token is script-mixed iff it carries both a Devanagari and a Latin letter.
# No valid Hindi/Sanskrit orthography does this, so every hit is corruption.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

_WS = re.compile(r"[ \t]{2,}")
# `[ \t]` not `\s`: a danda opening a line must not be pulled onto the previous one.
_ORPHAN_PUNCT = re.compile(r"[ \t]+([,।॥])")
_DANGLING_COMMA = re.compile(r",([ \t]*,)+")


@dataclass(frozen=True)
class Rules:
    """Rule tables. Every map is exact **whole-token**, never substring.

    `डांसिंग` contains `सिंग`, `पोटेंशियल` contains `टेंशन`, `डिस्लाइक` contains
    `लाइक`. A substring pass over any of these tables corrupts real words.
    """

    loanword_tokens: dict[str, str] = field(default_factory=dict)
    native_tokens: dict[str, str] = field(default_factory=dict)
    spelling_scatter: dict[str, str] = field(default_factory=dict)
    hallucination_tokens: tuple[str, ...] = ()
    # Verb inflections the hallucination token may absorb. The match may not cross
    # `और` or any word absent from this list: 159 segments glue `सब्सक्राइब` to
    # genuine meditation vocabulary, and a greedy match eats `कर दो` / `करते हैं`,
    # which are ordinary Hindi.
    hallucination_tail: tuple[str, ...] = ()
    loop_min_run: int = LOOP_MIN_RUN
    loop_keep: int = LOOP_KEEP


@dataclass
class NormalizeResult:
    text: str
    glosses: list[str]
    counts: dict[str, int]


def load_rules(path: str | Path) -> Rules:
    """Read `data/normalize_rules.json`. The only IO in this module."""
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return Rules(
        loanword_tokens=raw.get("loanword_tokens", {}),
        native_tokens=raw.get("native_tokens", {}),
        spelling_scatter=raw.get("spelling_scatter", {}),
        hallucination_tokens=tuple(raw.get("hallucination_tokens", ())),
        hallucination_tail=tuple(raw.get("hallucination_tail", ())),
        loop_min_run=int(raw.get("loop_min_run", LOOP_MIN_RUN)),
        loop_keep=int(raw.get("loop_keep", LOOP_KEEP)),
    )


def is_script_mixed(token: str) -> bool:
    """True iff `token` carries both Devanagari and Latin letters."""
    bare = token.strip(_EDGE_PUNCT)
    return bool(_DEVANAGARI.search(bare) and _LATIN.search(bare))


def _split_edges(token: str) -> tuple[str, str, str]:
    """Return (leading punct, bare token, trailing punct)."""
    bare = token.strip(_EDGE_PUNCT)
    if not bare:
        return "", token, ""
    start = token.index(bare)
    return token[:start], bare, token[start + len(bare):]


def map_tokens(text: str, table: dict[str, str]) -> tuple[str, int]:
    """Exact whole-token replacement, preserving surrounding punctuation.

    Returns (text, n_replacements). Whitespace between tokens is not preserved
    verbatim — runs of spaces collapse to one — which is harmless for retrieval
    and keeps the round-trip simple.
    """
    if not table:
        return text, 0
    n = 0
    out: list[str] = []
    for line in text.split("\n"):
        toks: list[str] = []
        for tok in line.split():
            lead, bare, trail = _split_edges(tok)
            repl = table.get(bare)
            if repl is None:
                toks.append(tok)
                continue
            n += 1
            toks.append(f"{lead}{repl}{trail}")
        out.append(" ".join(toks))
    return "\n".join(out), n


def strip_latin_glosses(text: str) -> tuple[str, list[str]]:
    """Remove the cleaner's invented `(Master)` parentheticals; return them.

    Only Latin-alphabet contents match, so a genuine Devanagari aside survives.
    The captured strings feed `data/aliases.json` for query expansion.
    """
    found: list[str] = []

    def _take(m: re.Match[str]) -> str:
        found.append(m.group(1).strip())
        return ""

    return _LATIN_GLOSS.sub(_take, text), found


def excise_hallucination(text: str, rules: Rules) -> tuple[str, int]:
    """Delete the `सब्सक्राइब करते हैं` YouTube artifact, token by token.

    Never deletes a line: 159 segments fuse the artifact to real meditation
    vocabulary. The tail may only absorb the listed inflections, so `और गहरा` and
    a standalone `कर दो` are untouched.
    """
    if not rules.hallucination_tokens:
        return text, 0
    head = "|".join(re.escape(t) for t in rules.hallucination_tokens)
    tail = "|".join(re.escape(t) for t in rules.hallucination_tail) or r"(?!)"
    pat = re.compile(rf"(?:{head})(?:[\s,]+(?:{tail})){{0,8}}[\s,]*")
    text, n = pat.subn(" ", text)
    return _tidy(text), n


def collapse_loops(text: str, min_run: int = LOOP_MIN_RUN,
                   keep: int = LOOP_KEEP) -> tuple[str, int]:
    """Collapse a run of >= `min_run` identical tokens down to `keep` copies.

    Tokens compare equal after stripping edge punctuation, so the comma-joined ASR
    form `ओ, ओ, ओ, …` and the bare `ओ ओ ओ …` are one signature.

    `min_run` below 6 or `keep` below 3 corrupts legitimate reduplication and is
    rejected, not clamped — a caller asking for it has misunderstood the rule.
    """
    if min_run < LOOP_MIN_RUN or keep < LOOP_KEEP:
        raise ValueError(
            f"loop collapse refuses min_run={min_run} keep={keep}: below "
            f"{LOOP_MIN_RUN}/{LOOP_KEEP} the corpus is dominated by legitimate "
            "reduplication (71,240 length-2 runs, 3,648 length-3)"
        )
    collapsed = 0
    out_lines: list[str] = []
    for line in text.split("\n"):
        toks = line.split()
        out: list[str] = []
        i = 0
        while i < len(toks):
            bare = toks[i].strip(_EDGE_PUNCT)
            j = i + 1
            while j < len(toks) and bare and toks[j].strip(_EDGE_PUNCT) == bare:
                j += 1
            run = j - i
            if run >= min_run:
                out.extend(toks[i:i + keep])
                collapsed += 1
            else:
                out.extend(toks[i:j])
            i = j
        out_lines.append(" ".join(out))
    return "\n".join(out_lines), collapsed


def _tidy(text: str) -> str:
    text = _DANGLING_COMMA.sub(",", text)
    text = _ORPHAN_PUNCT.sub(r"\1", text)
    text = _WS.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n"))


def normalize(text: str, rules: Rules) -> NormalizeResult:
    """Apply every rule, in the one order that is correct.

    Hallucination excision runs *before* loop collapse so that a
    `सब्सक्राइब सब्सक्राइब …` run is deleted outright rather than collapsed to
    three surviving copies. The token maps run before both, so a corrupt token
    inside a loop is repaired and then counted as one run.
    """
    counts: dict[str, int] = {}
    text, counts["loanword_tokens"] = map_tokens(text, rules.loanword_tokens)
    text, counts["native_tokens"] = map_tokens(text, rules.native_tokens)
    text, counts["spelling_scatter"] = map_tokens(text, rules.spelling_scatter)
    text, glosses = strip_latin_glosses(text)
    counts["glosses"] = len(glosses)
    text, counts["hallucination"] = excise_hallucination(text, rules)
    text, counts["loops"] = collapse_loops(text, rules.loop_min_run, rules.loop_keep)
    return NormalizeResult(text=_tidy(text), glosses=glosses, counts=counts)
