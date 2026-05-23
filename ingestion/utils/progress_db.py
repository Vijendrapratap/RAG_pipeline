"""SQLite progress tracking for resumable bulk ingestion.

Per PRD §6 Phase 4 deliverable 3. Schema (exact PRD spec):

    CREATE TABLE IF NOT EXISTS ingest_status (
        file        TEXT PRIMARY KEY,
        status      TEXT NOT NULL,   -- 'ok' | 'failed' | 'skipped' | 'in_progress'
        n_chunks    INTEGER DEFAULT 0,
        reason      TEXT,            -- failure reason if status='failed'
        attempts    INTEGER DEFAULT 0,
        last_ts     REAL DEFAULT (strftime('%s', 'now'))
    );
    CREATE INDEX IF NOT EXISTS idx_status ON ingest_status(status);
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_status (
    file        TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    n_chunks    INTEGER DEFAULT 0,
    reason      TEXT,
    attempts    INTEGER DEFAULT 0,
    last_ts     REAL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_status ON ingest_status(status);
"""


class ProgressDB:
    def __init__(self, path: str | Path = "./ingest_progress.sqlite") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ProgressDB":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_status(self, file: str) -> str | None:
        row = self._conn.execute(
            "SELECT status FROM ingest_status WHERE file = ?", (file,)
        ).fetchone()
        return row[0] if row else None

    def mark_ok(self, file: str, n: int) -> None:
        self._conn.execute(
            """INSERT INTO ingest_status (file, status, n_chunks, reason, attempts, last_ts)
                 VALUES (?, 'ok', ?, NULL, COALESCE(
                   (SELECT attempts FROM ingest_status WHERE file = ?), 0) + 1,
                   strftime('%s', 'now'))
               ON CONFLICT(file) DO UPDATE SET
                   status='ok', n_chunks=excluded.n_chunks, reason=NULL,
                   attempts=ingest_status.attempts + 1,
                   last_ts=strftime('%s', 'now')""",
            (file, n, file),
        )
        self._conn.commit()

    def mark_failed(self, file: str, reason: str) -> None:
        self._conn.execute(
            """INSERT INTO ingest_status (file, status, n_chunks, reason, attempts, last_ts)
                 VALUES (?, 'failed', 0, ?, COALESCE(
                   (SELECT attempts FROM ingest_status WHERE file = ?), 0) + 1,
                   strftime('%s', 'now'))
               ON CONFLICT(file) DO UPDATE SET
                   status='failed', reason=excluded.reason,
                   attempts=ingest_status.attempts + 1,
                   last_ts=strftime('%s', 'now')""",
            (file, reason, file),
        )
        self._conn.commit()

    def mark_in_progress(self, file: str) -> None:
        self._conn.execute(
            """INSERT INTO ingest_status (file, status, attempts, last_ts)
                 VALUES (?, 'in_progress', COALESCE(
                   (SELECT attempts FROM ingest_status WHERE file = ?), 0) + 1,
                   strftime('%s', 'now'))
               ON CONFLICT(file) DO UPDATE SET
                   status='in_progress',
                   attempts=ingest_status.attempts + 1,
                   last_ts=strftime('%s', 'now')""",
            (file, file),
        )
        self._conn.commit()

    def reset_to_pending(self, file: str) -> None:
        """Used by retry_dead_letter to clear a 'failed' marker so the file
        becomes eligible for re-ingest."""
        self._conn.execute("DELETE FROM ingest_status WHERE file = ?", (file,))
        self._conn.commit()

    def list_failed(self) -> list[tuple[str, str | None]]:
        return [
            (r[0], r[1])
            for r in self._conn.execute(
                "SELECT file, reason FROM ingest_status WHERE status = 'failed'"
            ).fetchall()
        ]

    def stats(self) -> dict[str, int]:
        out = {"ok": 0, "failed": 0, "skipped": 0, "in_progress": 0}
        for status, n in self._conn.execute(
            "SELECT status, COUNT(*) FROM ingest_status GROUP BY status"
        ).fetchall():
            out[status] = n
        return out
