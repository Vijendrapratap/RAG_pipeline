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


def _safe_distinct(cur: Any, key: str, sql: str) -> list[str]:
    """Run one facet's DISTINCT query, returning [] on any error instead of
    raising. The connection is autocommit, so a failure here (missing table or
    column on an older DB, a transient) is isolated to this one facet and never
    blanks the others. Never raises."""
    try:
        cur.execute(sql)
        return [r[0] for r in cur.fetchall()]
    except psycopg2.Error as e:
        log.info("filter facet %r unavailable: %s", key, str(e).strip())
        return []


def get_filter_options(pg_dsn: str, statement_timeout_ms: int = 10_000) -> dict[str, list[Any]]:
    """Return distinct filter values from file_meta, keyed for the dashboard.

    Column names are hard-coded constants (never user input), so the f-string
    interpolation below is safe.
    """
    if not _PSYCOPG2_AVAILABLE:
        raise RuntimeError("psycopg2 not installed — cannot read filter options")

    conn = psycopg2.connect(pg_dsn, connect_timeout=5)
    # Autocommit so each facet query is independent: one failing query (a
    # transient, a missing column on an old DB, a startup race) must not abort a
    # shared transaction and blank EVERY dropdown. Each facet degrades to [] on
    # its own; the dashboard keeps all the facets that did resolve.
    conn.autocommit = True
    try:
        out: dict[str, list[Any]] = {}
        with conn.cursor() as cur:
            try:
                cur.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
            except psycopg2.Error:  # pragma: no cover - defensive
                pass
            for key, col in _SCALAR_FACETS:
                out[key] = _safe_distinct(
                    cur, key,
                    f"SELECT DISTINCT {col} FROM file_meta "
                    f"WHERE {col} IS NOT NULL ORDER BY {col}",
                )
            for key, col in _ARRAY_FACETS:
                out[key] = _safe_distinct(
                    cur, key,
                    f"SELECT DISTINCT v FROM file_meta, UNNEST({col}) AS v "
                    f"WHERE {col} IS NOT NULL ORDER BY v",
                )
            # Phase 14: performers come from the curated catalog, not file_meta.
            out["performers"] = _safe_distinct(
                cur, "performers",
                "SELECT DISTINCT v FROM catalog_sitting, UNNEST(performers) AS v "
                "WHERE performers IS NOT NULL ORDER BY v",
            )
            # Phase 14: the curated catalog covers the whole 34-year archive,
            # so its columns make the dropdowns rich even before audio is
            # transcribed. Union the catalog's values into the location /
            # season / track_type facets and add a `years` facet — these are
            # the sheet columns the user browses by.
            _union(out, "locations", _safe_distinct(
                cur, "catalog_locations",
                "SELECT DISTINCT location FROM catalog_sitting "
                "WHERE location IS NOT NULL ORDER BY location"))
            _union(out, "seasons", _safe_distinct(
                cur, "catalog_seasons",
                "SELECT DISTINCT season FROM catalog_sitting "
                "WHERE season IS NOT NULL ORDER BY season"))
            _union(out, "track_types", _safe_distinct(
                cur, "catalog_track_types",
                "SELECT DISTINCT track_type FROM catalog_track "
                "WHERE track_type IS NOT NULL ORDER BY track_type"))
            out["years"] = _safe_distinct(
                cur, "years",
                "SELECT DISTINCT camp_year::text FROM catalog_sitting "
                "WHERE year_reliable AND camp_year IS NOT NULL "
                "ORDER BY camp_year::text DESC")
        return out
    finally:
        conn.close()


def _union(out: dict[str, list[Any]], key: str, extra: list[str]) -> None:
    """Merge `extra` into out[key], dedup'd and sorted. Used to fold the
    catalog's facet values into the transcript-derived ones."""
    merged = set(out.get(key) or []) | set(extra)
    out[key] = sorted(v for v in merged if v is not None)


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
