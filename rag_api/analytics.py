"""Postgres-backed corpus analytics for the dashboard.

Ported from `open_webui_functions/analytics.py` (the Open WebUI function
tool) into the dashboard API. Two changes from the original:

  1. **Hindi-correct full-text search.** The corpus is Hindi. The original
     queries used `to_tsvector('english', …)`, which applies English stemming
     and stopwords to Devanagari text — producing wrong lexemes and missed
     matches. These queries use the `'simple'` config (lowercase + tokenise,
     no stemming, no stopwords), which is correct for Hindi. The GIN index in
     `infra/postgres/analytics_schema.sql` is changed to match; an existing
     database must run `infra/postgres/migrations/001_hindi_fts.sql` to
     rebuild the index, or these queries fall back to a sequential scan.
  2. Methods return structured dicts, not preformatted strings — the API
     serves JSON and the React dashboard renders it.

psycopg2 is imported under a guard so the module imports even where the
driver is absent (test envs); calls raise clearly if it is missing.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    import psycopg2  # type: ignore[import-not-found]
    _PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover
    psycopg2 = None
    _PSYCOPG2_AVAILABLE = False

log = logging.getLogger("rag_api.analytics")

# 'simple' tokenises + lowercases without English stemming or stopwords —
# correct for Devanagari. The GIN index MUST use the same config or the query
# cannot use it (Postgres matches index expressions textually).
FTS_CONFIG = "simple"
# Full-text match predicate. The %s placeholder binds the search term.
_FTS_MATCH = (
    f"to_tsvector('{FTS_CONFIG}', text) @@ plainto_tsquery('{FTS_CONFIG}', %s)"
)


def shape_speaker_counts(rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Shape (speaker, count) rows into result dicts. Pure — unit-tested."""
    return [{"speaker": s, "chunk_count": int(n)} for s, n in rows]


def shape_transcript_counts(rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Shape (source_file, count) rows into result dicts. Pure — unit-tested."""
    return [{"source_file": f, "chunk_count": int(n)} for f, n in rows]


class Analytics:
    """Postgres analytics queries for the dashboard. One instance per app.

    Each call opens and closes its own connection — analytics is low-traffic
    (a dashboard panel, not the hot retrieval path), so a pool is not worth
    the complexity.
    """

    def __init__(self, pg_dsn: str, statement_timeout_ms: int = 10_000) -> None:
        self.pg_dsn = pg_dsn
        self.statement_timeout_ms = statement_timeout_ms

    def _query(self, sql: str, args: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        if not _PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not installed — analytics unavailable")
        conn = psycopg2.connect(self.pg_dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SET statement_timeout = {int(self.statement_timeout_ms)}"
                )
                cur.execute(sql, args)
                return cur.fetchall()
        finally:
            conn.close()

    def count_mentions(
        self, term: str, speaker: str | None = None
    ) -> dict[str, Any]:
        """Count chunks whose text matches `term` (full-text search).

        Optionally restrict to chunks that feature `speaker`. Answers
        "how many times did we talk about X" questions.
        """
        if speaker:
            rows = self._query(
                f"SELECT COUNT(*) FROM chunk_meta "
                f"WHERE {_FTS_MATCH} AND %s = ANY(speakers)",
                (term, speaker),
            )
        else:
            rows = self._query(
                f"SELECT COUNT(*) FROM chunk_meta WHERE {_FTS_MATCH}",
                (term,),
            )
        n = int(rows[0][0]) if rows else 0
        return {"term": term, "speaker": speaker, "chunk_count": n}

    def top_speakers(self, term: str, limit: int = 10) -> dict[str, Any]:
        """Rank speakers by how many `term`-matching chunks they appear in.

        Answers "who talks most about X" questions.
        """
        rows = self._query(
            f"SELECT speaker, COUNT(*) AS n FROM ("
            f"  SELECT UNNEST(speakers) AS speaker FROM chunk_meta "
            f"  WHERE {_FTS_MATCH}"
            f") s GROUP BY speaker ORDER BY n DESC LIMIT %s",
            (term, int(limit)),
        )
        return {"term": term, "speakers": shape_speaker_counts(rows)}

    def list_transcripts(self, term: str, limit: int = 20) -> dict[str, Any]:
        """List source files containing `term`, ranked by match count.

        Answers "which transcripts mention X" questions.
        """
        rows = self._query(
            f"SELECT source_file, COUNT(*) AS n FROM chunk_meta "
            f"WHERE {_FTS_MATCH} "
            f"GROUP BY source_file ORDER BY n DESC LIMIT %s",
            (term, int(limit)),
        )
        return {"term": term, "transcripts": shape_transcript_counts(rows)}
