"""Best-sitting ranking (plan intent ``best_sittings``).

"Which are the best sittings on rain?" is a ranking question, not plain
retrieval. The criteria (user request, 2026-07-15):

* **Topic word frequency** — how often the topic's words occur in the
  sitting, with synonym / cross-script bridging (rain = barish = बारिश).
  The planner's bilingual variants supply the bridge; Postgres full-text
  counts per file supply the frequency.
* **Length** — longer sittings promoted (``file_meta.duration_sec``).
* **Popularity** — how often the sitting has been played
  (``play_events``, fed by the dashboard's track player).

Candidates come from the summary collection (``search_summaries_multi`` over
the plan variants — semantic relevance, honours detected filters), then
Postgres adds per-file stats and a pure composite re-ranks:

    0.50·semantic + 0.25·mentions + 0.15·duration + 0.10·plays

Counts are log-scaled, every component min-max normalised across the
candidate pool. The weights are deliberate constants, not env knobs — tune
them with eval evidence, not configuration.

Fail-open: Postgres being down (or the ``play_events`` migration not yet
applied) zeroes the affected stats with a log line; the semantic order
survives. The scoring core is pure and unit-tested.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

try:
    import psycopg2  # type: ignore[import-not-found]
    _PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover
    psycopg2 = None
    _PSYCOPG2_AVAILABLE = False

from rag_api.analytics import _FTS_MATCH
from rag_api.config import Settings
from rag_api.synthesis import filter_min_score

log = logging.getLogger("rag_api.best")

_W_SEMANTIC = 0.50
_W_MENTIONS = 0.25
_W_DURATION = 0.15
_W_PLAYS = 0.10

# Words that never carry the TOPIC of a best-sittings ask: ranking meta,
# sitting-nouns, function words — English, Hinglish, Devanagari. Everything
# else in the plan variants is counted as a topic term.
_TERM_STOP = frozenset({
    "best", "top", "greatest", "finest", "badhiya", "behtareen", "sabse",
    "accha", "acche", "achha", "achhe",
    "sitting", "sittings", "session", "sessions", "satsang", "satsangs",
    "pravachan", "pravachans", "discourse", "discourses", "track", "tracks",
    "recording", "recordings",
    "the", "for", "and", "about", "with", "from", "what", "which", "when",
    "where", "who", "how", "swami", "swamiji", "guruji",
    "par", "ke", "ka", "ki", "mein", "aur",
    "प्रवचन", "सत्संग", "बैठक", "पर", "के", "का", "की", "में", "से", "और",
    "सबसे", "अच्छा", "अच्छे", "श्रेष्ठ", "सर्वश्रेष्ठ", "बेहतरीन", "बढ़िया",
    "स्वामी", "जी",
})
_TOKEN_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def stat_terms(queries: list[str], max_terms: int = 8) -> list[str]:
    """Distinct topic words from the plan variants, for per-word FTS counts.

    `plainto_tsquery` ANDs every word of a phrase, so counting whole variants
    ("rain satsang sessions") matches ~nothing — live probe 2026-07-15 came
    back all-zero. "Most frequent word" means per WORD: tokenize the variants,
    drop ranking-meta / sitting-nouns / function words and digits, dedupe,
    cap. Pure — unit-tested.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for q in queries:
        for tok in _TOKEN_RE.findall((q or "").lower()):
            if tok.isdigit() or tok in _TERM_STOP or tok in seen:
                continue
            # Short Latin tokens are function-word noise; Devanagari words
            # are dense enough that two characters can be a real topic word.
            if len(tok) < 3 and not _DEVANAGARI_RE.search(tok):
                continue
            seen.add(tok)
            terms.append(tok)
            if len(terms) >= max_terms:
                return terms
    return terms


def _normalize(values: list[float]) -> list[float]:
    """Min-max to [0,1]; a constant column contributes 0 to everyone (it
    cannot discriminate, so it must not shift the ranking)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def score_candidates(
    candidates: list[dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Re-rank summary candidates by the composite score. Pure — unit-tested.

    Each returned result is a copy whose ``score`` is the composite ([0,1])
    and whose ``metadata`` gains ``mention_count`` / ``duration_min`` /
    ``play_count`` / ``semantic_score`` — the answer prompt and the UI cards
    read the ranking rationale from there.
    """
    if not candidates:
        return []
    sem: list[float] = []
    men: list[float] = []
    dur: list[float] = []
    ply: list[float] = []
    enriched: list[dict[str, Any]] = []
    for c in candidates:
        st = stats.get(str(c.get("source_file") or ""), {})
        mentions = int(st.get("mention_count") or 0)
        duration_sec = float(st.get("duration_sec") or 0.0)
        plays = int(st.get("play_count") or 0)
        semantic = float(c.get("score") or 0.0)
        sem.append(semantic)
        men.append(math.log1p(mentions))
        dur.append(math.log1p(duration_sec / 60.0))
        ply.append(math.log1p(plays))
        r = dict(c)
        meta = dict(r.get("metadata") or {})
        meta["semantic_score"] = round(semantic, 4)
        if mentions:
            meta["mention_count"] = mentions
        if duration_sec:
            meta["duration_min"] = round(duration_sec / 60.0, 1)
        if plays:
            meta["play_count"] = plays
        r["metadata"] = meta
        enriched.append(r)
    sem_n, men_n = _normalize(sem), _normalize(men)
    dur_n, ply_n = _normalize(dur), _normalize(ply)
    for r, s, m, d, p in zip(enriched, sem_n, men_n, dur_n, ply_n):
        r["score"] = round(
            _W_SEMANTIC * s + _W_MENTIONS * m + _W_DURATION * d + _W_PLAYS * p,
            4,
        )
    enriched.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    return enriched[:top_k]


class BestSittings:
    """Ranks sittings for the ``best_sittings`` plan intent.

    Semantic candidates come from the caller (retriever); this class owns the
    Postgres stat lookups and the composite re-rank. One connection per call —
    this is not a hot path.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pg_dsn = settings.pg_dsn
        self.statement_timeout_ms = 10_000

    def _stats(
        self, files: list[str], terms: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Per-file mention/duration/play stats for the candidate pool.

        Autocommit + per-query guards, like ``rag_api.db``: one failing stat
        (e.g. ``play_events`` migration not applied yet) degrades to zeros for
        that stat only and never poisons the others. Returns {} when Postgres
        itself is unreachable (logged) — the ranking then rides on semantics.

        `terms` are single topic words (see `stat_terms`); each is counted
        independently and summed, so synonym/script variants (rain + बारिश +
        वर्षा) accumulate into one mention figure.
        """
        out: dict[str, dict[str, Any]] = {f: {} for f in files}
        if not files:
            return out
        if not _PSYCOPG2_AVAILABLE:
            log.warning("psycopg2 unavailable — best-sitting stats skipped")
            return out
        try:
            conn = psycopg2.connect(self.pg_dsn, connect_timeout=5)
        except Exception as e:  # noqa: BLE001 - fail-open, semantic rank survives
            log.warning("best-sitting stats unavailable (connect): %s", e)
            return out
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"SET statement_timeout = {int(self.statement_timeout_ms)}"
                    )
                except psycopg2.Error:  # pragma: no cover - defensive
                    pass
                for term in terms:
                    try:
                        cur.execute(
                            f"SELECT source_file, COUNT(*) FROM chunk_meta "
                            f"WHERE {_FTS_MATCH} AND source_file = ANY(%s) "
                            f"GROUP BY source_file",
                            (term, files),
                        )
                        for sf, n in cur.fetchall():
                            st = out.setdefault(sf, {})
                            st["mention_count"] = (
                                st.get("mention_count", 0) + int(n)
                            )
                    except psycopg2.Error as e:
                        log.warning("mention count failed for %r: %s", term, e)
                try:
                    cur.execute(
                        "SELECT source_file, duration_sec FROM file_meta "
                        "WHERE source_file = ANY(%s)",
                        (files,),
                    )
                    for sf, d in cur.fetchall():
                        if d is not None:
                            out.setdefault(sf, {})["duration_sec"] = float(d)
                except psycopg2.Error as e:
                    log.warning("duration lookup failed: %s", e)
                try:
                    cur.execute(
                        "SELECT source_file, COUNT(*) FROM play_events "
                        "WHERE source_file = ANY(%s) GROUP BY source_file",
                        (files,),
                    )
                    for sf, n in cur.fetchall():
                        out.setdefault(sf, {})["play_count"] = int(n)
                except psycopg2.Error as e:
                    log.info("play counts unavailable (migration 004?): %s", e)
        finally:
            conn.close()
        return out

    def rank(
        self,
        retriever: Any,
        queries: list[str],
        filters: dict[str, Any] | None,
        top_k: int,
        pool: int = 20,
    ) -> list[dict[str, Any]]:
        """Top `top_k` sittings for the plan variants, best first.

        `pool` summary candidates are fetched (so a low-semantic but
        high-frequency sitting can still climb), the citation floor is applied
        to their SEMANTIC scores here (the composite that replaces them is a
        rank, not a relevance — the /api/query floor must not re-filter it),
        then stats attach and the composite re-ranks. Retrieval errors
        propagate — the caller maps them to 502 exactly like every other
        retrieval path.
        """
        candidates = retriever.search_summaries_multi(
            queries, filters, top_k=max(pool, top_k),
        )
        candidates = filter_min_score(
            candidates, self.settings.cite_min_score, self.settings.allow_abstain,
        )
        files = [
            str(c.get("source_file")) for c in candidates
            if c.get("source_file")
        ]
        stats = self._stats(files, stat_terms(queries))
        return score_candidates(candidates, stats, top_k)
