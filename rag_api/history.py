"""Conversation history persistence for the dashboard.

One row per completed Q&A turn. The dashboard's "Recent" sidebar reads
summaries (id + title + created_at); opening a row fetches the full record
to re-render the saved answer and its citations.

The schema lives in ``infra/postgres/migrations/002_conversations.sql``. The
table is intentionally single-tenant (no ``user_id``) because the dashboard
auth is a single shared password — every signed-in user can see every entry.
Add a column here and a filter in the API the day per-user auth lands.

psycopg2 is imported under a guard so the module imports cleanly in test
environments where the driver isn't available; calls raise clearly if it is
missing, matching the pattern in ``rag_api.analytics``.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

try:
    import psycopg2  # type: ignore[import-not-found]
    from psycopg2.extras import Json, RealDictCursor  # type: ignore[import-not-found]
    _PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover
    psycopg2 = None
    Json = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]
    _PSYCOPG2_AVAILABLE = False

log = logging.getLogger("rag_api.history")

# Cap the auto-derived title at this many characters. Longer questions still
# save in full to `question`; only the sidebar label is truncated.
TITLE_MAX_LEN = 80


def derive_title(question: str) -> str:
    """First line of the question, trimmed to TITLE_MAX_LEN with an ellipsis."""
    first_line = question.strip().splitlines()[0] if question.strip() else ""
    if len(first_line) <= TITLE_MAX_LEN:
        return first_line or "Untitled"
    return first_line[: TITLE_MAX_LEN - 1].rstrip() + "…"


class History:
    """Postgres CRUD for the conversation_history table.

    One short-lived connection per call — history is dashboard-rate traffic,
    not the retrieval hot path, so a pool would be over-engineering. Each
    method raises on driver / connectivity failures; the API layer maps those
    to 502 responses without swallowing the cause.
    """

    def __init__(self, pg_dsn: str, statement_timeout_ms: int = 10_000) -> None:
        self.pg_dsn = pg_dsn
        self.statement_timeout_ms = statement_timeout_ms

    def _connect(self):
        if not _PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not installed — history unavailable")
        return psycopg2.connect(self.pg_dsn, connect_timeout=5)

    def _set_timeout(self, cur) -> None:
        cur.execute(f"SET statement_timeout = {int(self.statement_timeout_ms)}")

    # ---- writes ---------------------------------------------------------

    def save(self, record: dict[str, Any]) -> dict[str, Any]:
        """Insert one conversation. Returns the row, including id + created_at.

        `record` carries the fields the dashboard knows after a turn:
          question, answer, mode, scope, top_k, find_quote, expanded,
          answer_language, filters, applied_filters, detected_filters,
          citations. Missing optional fields fall back to sensible defaults.
        """
        new_id = uuid.uuid4()
        title = derive_title(record.get("question", ""))
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                self._set_timeout(cur)
                cur.execute(
                    """
                    INSERT INTO conversation_history (
                        id, title, question, answer, mode, scope,
                        answer_language, find_quote, expanded, top_k,
                        filters, applied_filters, detected_filters, citations
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    RETURNING id, title, created_at, question, answer, mode,
                              scope, answer_language, find_quote, expanded,
                              top_k, filters, applied_filters,
                              detected_filters, citations
                    """,
                    (
                        str(new_id),
                        title,
                        record.get("question", ""),
                        record.get("answer", ""),
                        record.get("mode", "answer"),
                        record.get("scope", "chunks"),
                        record.get("answer_language"),
                        bool(record.get("find_quote", False)),
                        bool(record.get("expanded", False)),
                        int(record.get("top_k", 8)),
                        Json(record.get("filters") or {}),
                        Json(record.get("applied_filters") or {}),
                        Json(record.get("detected_filters") or []),
                        Json(record.get("citations") or []),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        finally:
            conn.close()
        return _serialize(row)

    def delete(self, conversation_id: str) -> bool:
        """Remove one conversation. Returns True if a row was deleted."""
        _validate_uuid(conversation_id)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                self._set_timeout(cur)
                cur.execute(
                    "DELETE FROM conversation_history WHERE id = %s",
                    (conversation_id,),
                )
                deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return bool(deleted)

    # ---- reads ----------------------------------------------------------

    def list_summaries(self, limit: int = 200) -> list[dict[str, Any]]:
        """Sidebar listing — light payload (no answer / citation bodies)."""
        limit = max(1, min(int(limit), 500))
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                self._set_timeout(cur)
                cur.execute(
                    """
                    SELECT id, title, created_at, mode, scope
                    FROM conversation_history
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [_serialize_summary(r) for r in rows]

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        """Full record for the read-only viewer. Returns None if not found."""
        _validate_uuid(conversation_id)
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                self._set_timeout(cur)
                cur.execute(
                    """
                    SELECT id, title, created_at, question, answer, mode,
                           scope, answer_language, find_quote, expanded,
                           top_k, filters, applied_filters,
                           detected_filters, citations
                    FROM conversation_history
                    WHERE id = %s
                    """,
                    (conversation_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        return _serialize(row) if row else None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _validate_uuid(value: str) -> None:
    """Reject malformed ids before they reach the SQL layer."""
    try:
        uuid.UUID(str(value))
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid conversation id: {value!r}") from e


def _serialize_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "title": row["title"],
        "created_at": row["created_at"].isoformat(),
        "mode": row["mode"],
        "scope": row["scope"],
    }


def _serialize(row: dict[str, Any]) -> dict[str, Any]:
    """Full row → JSON-friendly dict. JSONB columns come back as dict/list
    already with psycopg2, so we only normalize the id + timestamp."""
    out = dict(row)
    out["id"] = str(out["id"])
    out["created_at"] = out["created_at"].isoformat()
    # JSONB columns are returned decoded by psycopg2; guard against drivers
    # that return raw strings (some forks do).
    for k in ("filters", "applied_filters", "detected_filters", "citations"):
        v = out.get(k)
        if isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                out[k] = [] if k in ("detected_filters", "citations") else {}
    return out
