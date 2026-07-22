"""Corpus-mined entity gazetteer (PRD Phase 17, Stage 2).

The corpus is its own gazetteer. A proper name misheard by WhisperX is spelled
correctly *somewhere else* in the same corpus, usually more often than it is
spelled wrong: `पूरन सिंह` occurs 42x in 14 files, `पूर्ण सिंग` 3x, `वारेपूर्ण` 1x.
Clustering the surface forms of one name by a lossy phonetic key and taking the
majority vote recovers the canonical spelling **with no model**.

This module never rewrites anything on its own. `mine()` scans the corpus,
clusters, majority-votes, and writes `data/gazetteer_review.md` — one proposed
rewrite per line, with evidence counts and a verbatim sample — then **stops**.
The human strikes or edits lines. `compile_review()` turns the *approved* file
into `data/gazetteer.json`. Only then does `apply_gazetteer()` touch text, and
only for phrases that survived the review.

Two guardrails, both learned from measurement and both enforced here:

* **Word boundaries are mandatory.** `सिंग` has 527 corpus hits, but `डांसिंग`
  (*dancing*) contains it. Every rewrite is a whole **token sequence**, matched on
  whitespace-delimited tokens, so no rule can fire inside a longer word.
* **Never a single common word.** `पूर्ण` alone means "complete". Only multi-token
  phrases are ever mined or applied; a lone token is never a gazetteer entry.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("gazetteer")

# --- character classes -------------------------------------------------------

# Stripped off a token before it is compared or keyed. Mirrors normalize_text.
_EDGE_PUNCT = ",।॥.!?;:\"'()[]{}—–-*"

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"[0-9०-९]")

# The cleaner's invented Latin parentheticals. When one trails a Devanagari
# phrase (`वारेपूर्ण सिंग (Vare Pun Singh)`) the cleaner has told us that phrase is
# a named entity and handed us a romanization — a high-precision candidate signal.
_LATIN_GLOSS = re.compile(r"\(\s*([A-Za-z][A-Za-z0-9 '\-/&.]*)\s*\)")

# Consonant -> phonetic class. Vowels, matras, halant and visarga carry no class
# and are dropped: the vowel is the least stable part of an ASR-misheard name.
# Folds that model real confusions — retroflex<->dental (ण/न, ट/त), aspirated<->
# plain (ग/घ, क/ख), the three sibilants (श/ष/स) — collapse to one class. `ह` is
# dropped as a weak/elided coda. `म` is kept distinct from the other nasals to
# avoid over-merging (`मन`/`मोहन` should not fold into every -n- name).
_CLASS: dict[str, str] = {}
for _chars, _code in [
    ("कखगघ", "K"),
    ("चछजझ", "C"),
    ("टठडढतथदध", "T"),
    ("ङञणन", "N"),
    ("म", "M"),
    ("पफबभ", "P"),
    ("यय़", "Y"),
    ("रड़ढ़", "R"),
    ("लळ", "L"),
    ("व", "W"),
    ("शषस", "S"),
]:
    for _c in _chars:
        _CLASS[_c] = _code
# `ह`, avagraha, anusvara/chandrabindu and visarga handled specially in skeleton().

_NASAL_SIGNS = "ंँ"  # anusvara, chandrabindu -> nasal 'N'


def is_devanagari_word(token: str) -> bool:
    """True iff `token` (edges stripped) is a pure Devanagari word: at least one
    Devanagari letter, no Latin, no digit."""
    bare = token.strip(_EDGE_PUNCT)
    if not bare:
        return False
    if _LATIN.search(bare) or _DIGIT.search(bare):
        return False
    return bool(_DEVANAGARI.search(bare))


def _tokens(line: str) -> list[str]:
    """Whitespace split, edges stripped, keeping only pure-Devanagari words."""
    out: list[str] = []
    for raw in line.split():
        bare = raw.strip(_EDGE_PUNCT)
        if is_devanagari_word(bare):
            out.append(bare)
    return out


def skeleton(token: str) -> str:
    """Lossy phonetic key of one Devanagari token: a consonant-class string.

    `सिंग` and `सिंह` both key to ``SN``; `पूर्ण` and `पूरन` both to ``PRN``. Vowels
    and matras are dropped, consonants folded to place/manner classes, `ह` treated
    as a weak coda, and a trailing velar right after a nasal (the ``-ng``/``-nh``
    Singh coda) is dropped so `सिंग` (SNK) collapses onto `सिंह` (SN).
    """
    bare = unicodedata.normalize("NFD", token.strip(_EDGE_PUNCT))
    codes: list[str] = []
    for ch in bare:
        if ch == "़":  # nukta combining mark -> ignore (क़ == क for keying)
            continue
        if ch in _NASAL_SIGNS:
            codes.append("N")
            continue
        if ch == "ह" or ch == "ः":  # ha (weak coda) / visarga -> drop
            continue
        cls = _CLASS.get(ch)
        if cls is not None:
            codes.append(cls)
        # vowels, matras, halant, avagraha, anything else -> dropped
    # Trailing velar immediately after a nasal is the unstable Singh-type coda.
    if len(codes) >= 2 and codes[-1] == "K" and codes[-2] == "N":
        codes.pop()
    return "".join(codes)


def phrase_skeleton(tokens: list[str]) -> str:
    """Space-joined per-token skeletons — the cluster key for a phrase."""
    return " ".join(skeleton(t) for t in tokens)


# --- entity candidacy --------------------------------------------------------

# A bigram/trigram is an entity candidate only when a **surname** ends it. The
# token(s) before a surname are a given name, not grammar — that is what keeps the
# candidate set clean. Head-honorific anchoring (`गुरु X`, `बाबा X`) was tried and
# abandoned: after a title the next token is almost always a particle or verb
# (`गुरु के`, `गुरु कहा`), not a name, so it drowns the file in non-entities.
# Match is on the exact surface (not skeleton) so `सुन` (listen, keys to SN) is
# never mistaken for the `सिंह`/`सिंग` tail; known misspellings are listed out.
#
# Deliberately excluded from the tail set, each a common noun/particle in this
# devotional corpus that a surname test would drown in: `देव`/`नाथ`/`राम`/`जी`
# (deity, "lord", greeting, honorific), and — confirmed against a full mine —
# `शरण` ("refuge": चरण शरण, मोहे शरण), `राय` ("opinion": दो राय), `प्रसाद` ("the
# food offering": भोजन प्रसाद). Those three yielded 13 phrases and zero real names.
# The kept tails still double as words (`लाल` "red", `चंद` "few", `कुमार` "prince"),
# so a handful of common-noun lines survive for the human to strike — the review
# gate's job — but each also carries genuine recurring names (हनुमंत लाल, दूनी चंद).
ANCHOR_TAILS: frozenset[str] = frozenset({
    "सिंह", "सिंग", "सिंघ",
    "दास", "कुमार", "कुँवर", "कुंवर",
    "लाल", "चंद", "चन्द", "चंद्र",
    "नंद", "नन्द", "दयाल",
})

# The name slot before a surname must carry an actual name. If any non-anchor
# token of a candidate is a function word / ultra-common verb / honorific, the
# phrase is grammar (`एक सिंह`, `है दास`), not an entity, and is rejected. Kept
# deliberately broad on the closed class; a missed real name is safer than noise.
STOPWORDS: frozenset[str] = frozenset("""
के का की को में से पर है हैं था थे थी हुआ हुई हुए हो होता होती होते होना और या कि तो
ना नहीं भी ही एक यह वह ये वो जो जिस जिसे जिन जब तब अब कभी सदा यहाँ वहाँ कहाँ क्या कौन
कैसे क्यों इस उस इसे उसे इन उन इसका इसकी इसके उसका उसकी उसके हम तुम आप मैं मुझे तुझे
हमें तुम्हें उन्हें अपना अपनी अपने मेरा मेरी मेरे तेरा तेरी तेरे सब सभी कुछ कोई किसी
जैसे जैसा जैसी तक बहुत बड़ा बड़ी बड़े छोटा गया गई गए दिया दी दे देना देते लिया ली लेना
रहा रही रहे रहता रहती कर करो करा करना करते करती किया कहा कहो कहना कहते कहती आया आई आए
आगे पीछे ऊपर नीचे बीच अंदर बाहर पास दूर साथ बिना लिए द्वारा वाला वाली वाले गुरु बाबा जी
हाँ अरे ओ ऐसा ऐसी ऐसे फिर सा सी से जी श्री संत सन्त स्वामी भगत राजा महाराज पंडित पण्डित
मन देव नाथ राम हे वाह वाहि कुरु नमः नमो ॐ ओम प्रभु मा माँ बात दिन साल तुम्हारा हमारा
""".split())

MIN_TOKENS = 2
MAX_TOKENS = 3
MIN_NAME_LEN = 2  # a given-name token shorter than this is an ASR fragment


@dataclass
class Surface:
    """One observed spelling of a phrase."""
    text: str
    count: int = 0
    files: set[str] = field(default_factory=set)
    sample: str = ""
    romans: set[str] = field(default_factory=set)


def _is_name_token(tok: str) -> bool:
    """A token fit to fill a given-name slot: not a function word, not a fragment."""
    return tok not in STOPWORDS and tok not in ANCHOR_TAILS and len(tok) >= MIN_NAME_LEN


def _entity_ngrams(tokens: list[str]) -> list[tuple[str, ...]]:
    """Yield the 2- and 3-token windows ending in a surname whose name slot is a
    real name (no function word, no fragment)."""
    grams: list[tuple[str, ...]] = []
    n = len(tokens)
    for size in range(MIN_TOKENS, MAX_TOKENS + 1):
        for i in range(n - size + 1):
            win = tuple(tokens[i:i + size])
            if win[-1] not in ANCHOR_TAILS:
                continue
            name_slot = win[:-1]
            if all(_is_name_token(t) for t in name_slot):
                grams.append(win)
    return grams


def _context(line: str, phrase: str) -> str:
    """A verbatim window around the first occurrence of `phrase` in `line`."""
    line = line.strip()
    idx = line.find(phrase)
    if idx < 0:
        return line[:80]
    lo = max(0, idx - 30)
    hi = min(len(line), idx + len(phrase) + 30)
    pre = "…" if lo > 0 else ""
    post = "…" if hi < len(line) else ""
    return f"{pre}{line[lo:hi]}{post}"


def _gloss_after(line: str, phrase: str) -> str:
    """The Latin gloss the cleaner attached directly after `phrase`, or ""."""
    idx = line.find(phrase)
    if idx < 0:
        return ""
    tail = line[idx + len(phrase):]
    m = re.match(r"\s*" + _LATIN_GLOSS.pattern, tail)
    return m.group(1).strip() if m else ""


# --- mining ------------------------------------------------------------------

@dataclass
class Proposal:
    variant: str
    canonical: str
    variant_count: int
    canonical_count: int
    variant_files: int
    canonical_files: int
    sample: str
    romans: tuple[str, ...]


@dataclass
class MineConfig:
    min_canonical: int = 5     # canonical must be attested at least this often
    dominance: float = 4.0     # canonical_count >= dominance * variant_count
    glob: str = "*.cleaned.txt"


def _collect(corpus: Path, cfg: MineConfig) -> dict[str, dict[str, Surface]]:
    """Scan the corpus, returning ``skeleton -> {surface_text: Surface}``."""
    clusters: dict[str, dict[str, Surface]] = defaultdict(dict)
    n_files = 0
    files = sorted(corpus.rglob(cfg.glob))
    if not files:
        raise FileNotFoundError(f"no {cfg.glob} under {corpus}")
    for path in files:
        n_files += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            log.warning("skip %s: %s: %s", path.name, type(e).__name__, e)
            continue
        for line in text.split("\n"):
            toks = _tokens(line)
            if len(toks) < MIN_TOKENS:
                continue
            for win in _entity_ngrams(toks):
                phrase = " ".join(win)
                key = phrase_skeleton(list(win))
                if not key.replace(" ", ""):
                    continue
                bucket = clusters[key]
                surf = bucket.get(phrase)
                if surf is None:
                    surf = Surface(text=phrase)
                    bucket[phrase] = surf
                surf.count += 1
                surf.files.add(path.name)
                if not surf.sample:
                    surf.sample = _context(line, phrase)
                # Capture the cleaner's romanization only when it immediately
                # follows *this* phrase (`वारेपूर्ण सिंग (Vare Pun Singh)`), not
                # every gloss on the line.
                roman = _gloss_after(line, phrase)
                if roman:
                    surf.romans.add(roman)
        if n_files % 1000 == 0:
            log.info("scanned %d files (%d clusters)", n_files, len(clusters))
    log.info("scan complete: %d files, %d clusters", n_files, len(clusters))
    return clusters


def _propose(clusters: dict[str, dict[str, Surface]], cfg: MineConfig) -> list[Proposal]:
    """Turn multi-surface clusters into majority-vote rewrite proposals."""
    proposals: list[Proposal] = []
    for _key, bucket in clusters.items():
        if len(bucket) < 2:
            continue
        ranked = sorted(bucket.values(), key=lambda s: (s.count, s.text), reverse=True)
        canon = ranked[0]
        if canon.count < cfg.min_canonical:
            continue
        for variant in ranked[1:]:
            if variant.text == canon.text:
                continue
            if canon.count < cfg.dominance * variant.count:
                continue  # not dominant enough -> ambiguous, do not propose
            romans = tuple(sorted(canon.romans | variant.romans))
            proposals.append(Proposal(
                variant=variant.text,
                canonical=canon.text,
                variant_count=variant.count,
                canonical_count=canon.count,
                variant_files=len(variant.files),
                canonical_files=len(canon.files),
                sample=variant.sample or canon.sample,
                romans=romans,
            ))
    # Most-confident first: biggest canonical, then biggest gap.
    proposals.sort(key=lambda p: (p.canonical_count, p.canonical_count - p.variant_count),
                   reverse=True)
    return proposals


REVIEW_HEADER = """\
# Gazetteer review — Phase 17, Stage 2

**This file is a proposal, not a change.** Nothing below has been applied. Each
line proposes rewriting a rare misspelling of a proper name to its corpus-majority
spelling, recovered by phonetic clustering + majority vote. No model was used.

**How to review** (this is the hard human gate — PRD Phase 17):
- **Keep** a line to approve the rewrite.
- **Reject** a line by deleting it, prefixing it with `#`, or wrapping it in `~~`.
- **Correct** a line by editing the text after `→` (e.g. supply the right spelling
  the corpus never got right). `compile` reads whatever survives.

Then run `python -m ingestion.gazetteer compile` to generate `data/gazetteer.json`
from the surviving lines. Only multi-token phrases are ever rewritten, and only on
exact whole-token boundaries, so `डांसिंग` can never be touched by a `सिंग` rule and
a lone `पूर्ण` (\"complete\") is never rewritten.

Format:  `variant → canonical   [variant_count vs canonical_count]   (roman)  "context"`

---

"""


def render_review(proposals: list[Proposal]) -> str:
    lines = [REVIEW_HEADER]
    if not proposals:
        lines.append("_No proposals cleared the dominance + attestation gate._\n")
        return "".join(lines)
    for p in proposals:
        roman = f"  ({'; '.join(p.romans)})" if p.romans else ""
        lines.append(
            f"{p.variant} → {p.canonical}   "
            f"[{p.variant_count} vs {p.canonical_count}]"
            f"{roman}   \"{p.sample}\"\n"
        )
    return "".join(lines)


def mine(corpus: Path, out: Path, cfg: MineConfig | None = None) -> list[Proposal]:
    """Scan `corpus`, write the review file to `out`, and return the proposals.

    Does **not** write `data/gazetteer.json`; that is `compile_review`'s job, run
    only after a human has approved this file.
    """
    cfg = cfg or MineConfig()
    clusters = _collect(corpus, cfg)
    proposals = _propose(clusters, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_review(proposals), encoding="utf-8")
    log.info("wrote %d proposals -> %s (STOP: awaiting human sign-off)",
             len(proposals), out)
    return proposals


# --- compile & apply ---------------------------------------------------------

# A surviving review line: `variant → canonical   [..]  ...`. Struck (`~~`),
# commented (`#`) and blank lines are ignored.
_REVIEW_LINE = re.compile(r"^\s*(?!#)(?!~~)(.+?)\s+→\s+(.+?)\s+\[\d+\s+vs\s+\d+\]")


def compile_review(review: Path) -> dict[str, str]:
    """Parse the approved review file into a ``{variant: canonical}`` map.

    Only lines matching the review grammar and not struck/commented are read, so
    an operator can reject a proposal by deleting or `#`-prefixing its line.
    """
    mapping: dict[str, str] = {}
    for raw in review.read_text(encoding="utf-8").split("\n"):
        m = _REVIEW_LINE.match(raw)
        if not m:
            continue
        variant, canonical = m.group(1).strip(), m.group(2).strip()
        if len(variant.split()) < MIN_TOKENS:
            log.warning("skip single-token variant %r (min %d tokens)", variant, MIN_TOKENS)
            continue
        if variant == canonical:
            continue
        mapping[variant] = canonical
    return mapping


def write_gazetteer_json(mapping: dict[str, str], out: Path) -> None:
    payload = {
        "_meta": {
            "source": "data/gazetteer_review.md (human-approved)",
            "note": "Whole-token-sequence rewrites only; word-boundary anchored.",
            "count": len(mapping),
        },
        "entities": mapping,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_gazetteer(path: Path) -> dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(raw.get("entities", {}))


def apply_gazetteer(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Rewrite approved entity phrases, matching whole token sequences only.

    Boundary-safe by construction: matching is over whitespace-split tokens (with
    edge punctuation preserved), so a phrase like `पूर्ण सिंग` can only replace two
    *whole* adjacent tokens — never a substring of `डांसिंग`, and never a lone
    `पूर्ण`. Returns (text, n_replacements).
    """
    if not mapping:
        return text, 0
    # Longest phrases first so a trigram entry wins over a contained bigram.
    entries = sorted(
        ((k.split(), v) for k, v in mapping.items()),
        key=lambda kv: len(kv[0]), reverse=True,
    )
    n = 0
    out_lines: list[str] = []
    for line in text.split("\n"):
        raw_toks = line.split()
        bare = [t.strip(_EDGE_PUNCT) for t in raw_toks]
        i = 0
        out: list[str] = []
        while i < len(raw_toks):
            matched = False
            for phrase_toks, canonical in entries:
                w = len(phrase_toks)
                if bare[i:i + w] == phrase_toks:
                    lead = raw_toks[i][:len(raw_toks[i]) - len(raw_toks[i].lstrip(_EDGE_PUNCT))]
                    last = raw_toks[i + w - 1]
                    trail = last[len(last.rstrip(_EDGE_PUNCT)):]
                    out.append(f"{lead}{canonical}{trail}")
                    i += w
                    n += 1
                    matched = True
                    break
            if not matched:
                out.append(raw_toks[i])
                i += 1
        out_lines.append(" ".join(out))
    return "\n".join(out_lines), n


# --- cli ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Corpus-mined entity gazetteer (Phase 17).")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="Scan corpus, emit review file, STOP.")
    m.add_argument("--corpus", required=True, help="Root of *.cleaned.txt transcripts.")
    m.add_argument("--out", default="data/gazetteer_review.md")
    m.add_argument("--min-canonical", type=int, default=5)
    m.add_argument("--dominance", type=float, default=4.0)

    c = sub.add_parser("compile", help="Approved review file -> data/gazetteer.json.")
    c.add_argument("--review", default="data/gazetteer_review.md")
    c.add_argument("--out", default="data/gazetteer.json")

    args = p.parse_args(argv)

    if args.cmd == "mine":
        cfg = MineConfig(min_canonical=args.min_canonical, dominance=args.dominance)
        proposals = mine(Path(args.corpus), Path(args.out), cfg)
        print(f"\n{len(proposals)} proposals written to {args.out}. "
              "Review it, then run `gazetteer compile`. Nothing applied.")
        return 0

    if args.cmd == "compile":
        review = Path(args.review)
        if not review.exists():
            log.error("review file not found: %s (run `mine` first)", review)
            return 2
        mapping = compile_review(review)
        write_gazetteer_json(mapping, Path(args.out))
        print(f"\ncompiled {len(mapping)} approved entities -> {args.out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
