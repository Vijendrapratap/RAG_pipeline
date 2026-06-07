"""Postgres reads for the dashboard API.

Two things live here:
  * `get_filter_options` — the distinct file_meta values that populate the
    dashboard's filter dropdowns.
  * `VocabCache` — a TTL cache of those same values, used on the hot path by
    query-time filter extraction (`rag_api.query_parse`).

Analytics queries (count_mentions etc.) live in `rag_api.analytics`. psycopg2
is imported under a guard so the module imports even where the driver is
absent (e.g. a test environment) — the functions raise clearly if called
without it.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

try:
    import psycopg2  # type: ignore[import-not-found]
    _PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover
    psycopg2 = None
    _PSYCOPG2_AVAILABLE = False

log = logging.getLogger("rag_api.db")

# Scalar columns on file_meta whose distinct values become filter options.
_SCALAR_FACETS: tuple[tuple[str, str], ...] = (
    ("seasons", "season"),
    ("locations", "location"),
    ("event_ids", "event_id"),
    ("track_types", "track_type"),
    ("event_types", "event_type"),
    ("primary_languages", "primary_language"),
)
# Array columns — distinct values come from UNNEST.
_ARRAY_FACETS: tuple[tuple[str, str], ...] = (
    ("topics", "topics"),
    ("scriptures_referenced", "scriptures_referenced"),
)


def _distinct_performers(cur: Any) -> list[str]:
    """Distinct performer names from catalog_sitting. Returns [] when the
    catalog table is absent (DB predates Phase 14) — never raises."""
    try:
        cur.execute(
            "SELECT DISTINCT v FROM catalog_sitting, UNNEST(performers) AS v "
            "WHERE performers IS NOT NULL ORDER BY v"
        )
        return [r[0] for r in cur.fetchall()]
    except psycopg2.Error:
        # Roll back the aborted statement so the surrounding txn can continue.
        cur.connection.rollback()
        log.info("catalog_sitting absent — performers facet empty")
        return []


def get_filter_options(pg_dsn: str, statement_timeout_ms: int = 10_000) -> dict[str, list[Any]]:
    """Return distinct filter values from file_meta, keyed for the dashboard.

    Column names are hard-coded constants (never user input), so the f-string
    interpolation below is safe.
    """
    if not _PSYCOPG2_AVAILABLE:
        raise RuntimeError("psycopg2 not installed — cannot read filter options")

    conn = psycopg2.connect(pg_dsn, connect_timeout=5)
    try:
        out: dict[str, list[Any]] = {}
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
            for key, col in _SCALAR_FACETS:
                cur.execute(
                    f"SELECT DISTINCT {col} FROM file_meta "
                    f"WHERE {col} IS NOT NULL ORDER BY {col}"
                )
                out[key] = [r[0] for r in cur.fetchall()]
            for key, col in _ARRAY_FACETS:
                cur.execute(
                    f"SELECT DISTINCT v FROM file_meta, UNNEST({col}) AS v "
                    f"WHERE {col} IS NOT NULL ORDER BY v"
                )
                out[key] = [r[0] for r in cur.fetchall()]
            # Phase 14: performers come from the curated catalog, not file_meta.
            # Guarded — a DB predating the catalog tables simply yields none.
            out["performers"] = _distinct_performers(cur)
        return out
    finally:
        conn.close()


class VocabCache:
    """TTL cache of the file_meta filter vocabulary.

    Query-time filter extraction (`rag_api.query_parse.detect_signals`) needs
    the set of known seasons / places / topic tags on every request. Hitting
    Postgres each time would be wasteful, so values are cached for `ttl_s`.

    Resilient by design: a refresh that fails (DB down, schema missing) logs
    and keeps serving the last good snapshot — empty `{}` if there has never
    been one. Thread-safe: FastAPI runs sync endpoints in a threadpool.
    """

    def __init__(self, ttl_s: float = 600.0) -> None:
        self._ttl = ttl_s
        self._lock = threading.Lock()
        self._value: dict[str, list[Any]] = {}
        self._fetched_at = 0.0
        self.ok = False

    def get(self, pg_dsn: str) -> dict[str, list[Any]]:
        """Return the cached vocabulary, refreshing it from Postgres if the
        snapshot is older than the TTL (or has never been fetched)."""
        now = time.monotonic()
        with self._lock:
            fresh = self.ok and (now - self._fetched_at) < self._ttl
            if fresh:
                return self._value
        try:
            options = get_filter_options(pg_dsn)
        except Exception as e:  # noqa: BLE001 - keep serving last snapshot
            log.warning(
                "vocab refresh failed: %s — using last snapshot (%d facets)",
                e, len(self._value),
            )
            return self._value
        with self._lock:
            self._value = options
            self._fetched_at = now
            self.ok = True
        return options
