"""Read-only view of the indexed corpus, shaped for the archive map.

`file_meta.source_file` already *is* the hierarchy —
``Dagshai 2002/03 MAR - 2002/DAGSHAI 26 - 29 MAR 2002 HOLI CAMP/26 MAR - 1$ - 7 PM/
02 OM GURUVE NAMAH.json`` — so the tree needs no new table, no new mount, and no
filesystem access. Split the key on '/' and aggregate.

Two endpoints back the view:

  * ``summary()`` — one cheap aggregate. Its ``version`` is the live-refresh key:
    when (n_files, n_chunks, max_ingested_at) changes, the map refetches. Polling
    this every 10 s costs a single indexed aggregate.
  * ``state(prefix, depth)`` — the nodes at most `depth` levels below `prefix`.

`state()` reads all 9,335 rows and aggregates in Python rather than pushing the
grouping into SQL. At this scale the fetch is milliseconds, the result is cached
against `version`, and — the real reason — `build_nodes()` becomes a pure
function that the tests exercise without a database.

**Node states are what Postgres can actually prove**, and no more:

  remembered  chunk_count > 0 and session_date is known — ask about it, it answers,
              and the system knows when it was said
  written     chunk_count > 0, no session_date — searchable, but undated
  failed      a degenerate identity key with no folder path (transcribe_local.py's
              flat default out-dir put 3 of these in the index)

The `dark` (audio exists, never transcribed) and `heard` (isolated, not
transcribed) states need the filesystem, which this container cannot see. They
arrive with the desktop overlay. The legend must not imply otherwise: this map
shows the 9,335 tracks that are indexed, not the 27,806 that exist on disk.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Iterable, Literal

try:
    import psycopg2  # type: ignore[import-not-found]
    _PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover
    psycopg2 = None
    _PSYCOPG2_AVAILABLE = False

log = logging.getLogger("rag_api.corpus")

CState = Literal["remembered", "written", "failed"]

MAX_DEPTH = 4
MAX_PREFIX_LEN = 512

# (source_file, chunk_count, duration_sec, session_date)
Row = tuple[str, int, float | None, Any]


def track_state(source_file: str, chunk_count: int, session_date: Any) -> CState:
    """The state of one track. A key with no '/' has no place in the archive —
    it cannot be located, so it is broken regardless of how many chunks it has."""
    if "/" not in source_file:
        return "failed"
    if chunk_count > 0 and session_date is not None:
        return "remembered"
    return "written"


def _blank(path: str, name: str, depth: int, is_leaf: bool) -> dict[str, Any]:
    return {
        "path": path, "name": name, "depth": depth, "is_leaf": is_leaf,
        "n_files": 0, "n_chunks": 0, "duration_sec": 0.0,
        "remembered": 0, "written": 0, "failed": 0,
        "session_date": None,
    }


def build_nodes(rows: Iterable[Row], prefix: str = "", depth: int = 1) -> list[dict[str, Any]]:
    """Aggregate `rows` into the nodes at most `depth` levels below `prefix`.

    Pure. `prefix` is either "" (the root) or a path ending in '/'. A node is a
    leaf when it is a track file rather than a folder — which is decided by
    whether anything follows it in the key, never by its depth. (Directory depth
    in this archive ranges 1..4: Dagshai inserts a month bucket that Live Masters
    has no equivalent of. Reading level from position is the bug that cost this
    project 52% of its session dates.)
    """
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    base_depth = prefix.count("/") if prefix else 0

    out: dict[str, dict[str, Any]] = {}
    for source_file, chunk_count, duration_sec, session_date in rows:
        if prefix and not source_file.startswith(prefix):
            continue
        rest = source_file[len(prefix):]
        if not rest:
            continue
        segments = rest.split("/")
        st = track_state(source_file, chunk_count, session_date)

        # Roll this track into every ancestor within `depth` of the prefix.
        for level in range(1, min(depth, len(segments)) + 1):
            path = prefix + "/".join(segments[:level])
            is_leaf = level == len(segments)
            node = out.setdefault(path, _blank(path, segments[level - 1],
                                               base_depth + level, is_leaf))
            node["n_files"] += 1
            node["n_chunks"] += chunk_count
            node["duration_sec"] += duration_sec or 0.0
            node[st] += 1
            if is_leaf:
                node["session_date"] = (
                    session_date.isoformat() if hasattr(session_date, "isoformat")
                    else session_date
                )

    # Folders before files, then alphabetically — the order a person scans in.
    return sorted(out.values(), key=lambda n: (n["depth"], n["is_leaf"], n["name"]))


def summarize(rows: Iterable[Row]) -> dict[str, Any]:
    """Whole-corpus totals. Pure, so the tests can assert on fixed rows."""
    n_files = n_chunks = 0
    counts = {"remembered": 0, "written": 0, "failed": 0}
    degenerate: list[str] = []
    duration = 0.0
    dates: list[Any] = []
    for source_file, chunk_count, duration_sec, session_date in rows:
        n_files += 1
        n_chunks += chunk_count
        duration += duration_sec or 0.0
        st = track_state(source_file, chunk_count, session_date)
        counts[st] += 1
        if st == "failed":
            degenerate.append(source_file)
        if session_date is not None:
            dates.append(session_date)

    def _iso(d: Any) -> str | None:
        return d.isoformat() if hasattr(d, "isoformat") else d

    return {
        "n_files": n_files,
        "n_chunks": n_chunks,
        "n_remembered": counts["remembered"],
        "n_written": counts["written"],
        "n_failed": counts["failed"],
        "degenerate_files": sorted(degenerate),
        "duration_sec": round(duration, 1),
        "min_session_date": _iso(min(dates)) if dates else None,
        "max_session_date": _iso(max(dates)) if dates else None,
    }


def version_key(n_files: int, n_chunks: int, max_ingested_at: Any) -> str:
    """Cheap change-detector. The map polls this; when it moves, it refetches.

    Chunk count alone is not enough — a re-ingest that replaces a file's chunks
    one-for-one leaves it unchanged. `max_ingested_at` catches that.
    """
    raw = f"{n_files}|{n_chunks}|{max_ingested_at}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class CorpusReader:
    """Postgres reads for the archive map. One instance per app.

    Low traffic (one poll every 10 s per open tab), so each call opens its own
    connection — same reasoning as `rag_api.analytics.Analytics`.
    """

    def __init__(self, pg_dsn: str, statement_timeout_ms: int = 10_000) -> None:
        self.pg_dsn = pg_dsn
        self.statement_timeout_ms = statement_timeout_ms
        self._cached_version: str | None = None
        self._cached_rows: list[Row] | None = None

    def _query(self, sql: str, args: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        if not _PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not installed — corpus map unavailable")
        conn = psycopg2.connect(self.pg_dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {int(self.statement_timeout_ms)}")
                cur.execute(sql, args)
                return cur.fetchall()
        finally:
            conn.close()

    def _version(self) -> tuple[str, int, int, Any]:
        rows = self._query(
            "SELECT count(*), coalesce(sum(chunk_count), 0), max(ingested_at) "
            "FROM file_meta"
        )
        n_files, n_chunks, max_ingested = rows[0]
        n_files, n_chunks = int(n_files), int(n_chunks)
        return version_key(n_files, n_chunks, max_ingested), n_files, n_chunks, max_ingested

    def _rows(self, version: str) -> list[Row]:
        """All of file_meta, memoized against `version`. 9,335 rows ~= 1 MB."""
        if self._cached_version == version and self._cached_rows is not None:
            return self._cached_rows
        rows = [
            (str(sf), int(cc or 0), float(d) if d is not None else None, sd)
            for sf, cc, d, sd in self._query(
                "SELECT source_file, chunk_count, duration_sec, session_date "
                "FROM file_meta ORDER BY source_file"
            )
        ]
        self._cached_version, self._cached_rows = version, rows
        return rows

    def has_track(self, source_file: str) -> bool:
        """Is this exact key an indexed track?

        The allowlist for `/api/track/*`. A caller cannot name a file that is
        not a row, so there is no path to traverse — the primary-key lookup is
        the security boundary, not a filter on top of one. Parameterized, and
        deliberately not served from `_rows()`'s cache: a track ingested since
        the last poll must be playable immediately.
        """
        return bool(self._query(
            "SELECT 1 FROM file_meta WHERE source_file = %s LIMIT 1", (source_file,)
        ))

    def summary(self) -> dict[str, Any]:
        version, _, _, max_ingested = self._version()
        out = summarize(self._rows(version))
        out["version"] = version
        out["max_ingested_at"] = (
            max_ingested.isoformat() if hasattr(max_ingested, "isoformat")
            else max_ingested
        )
        return out

    def state(self, prefix: str = "", depth: int = 1) -> dict[str, Any]:
        """Nodes at most `depth` levels below `prefix`.

        `prefix` never reaches SQL — the rows are already in memory and filtered
        in Python — so folder names containing '%', '_' or '$' cannot be read as
        LIKE metacharacters, and there is nothing to inject into.
        """
        depth = max(1, min(int(depth), MAX_DEPTH))
        prefix = (prefix or "")[:MAX_PREFIX_LEN]
        version, _, _, _ = self._version()
        children = build_nodes(self._rows(version), prefix, depth)
        return {"version": version, "prefix": prefix, "depth": depth,
                "children": children}
