"""Conversation history persistence for the dashboard.

One row per completed Q&A turn; rows are grouped into a multi-turn thread by
``conversation_id`` (migration 003). The "Recent" sidebar reads one summary per
thread (id + first-turn title + latest activity); opening a thread fetches all
its turns to re-render the saved conversation and citations, and asking again
appends a new turn to the same ``conversation_id``.

The schema lives in ``infra/postgres/migrations/002_conversations.sql`` plus
``003_conversation_turns.sql`` (the grouping columns). The table is
intentionally single-tenant (no ``user_id``) because the dashboard auth is a
single shared password — every signed-in user can see every entry. Add a column
here and a filter in the API the day per-user auth lands.

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
        """Insert one turn. Returns the row, including id, conversation_id,
        turn_index + created_at.

        `record` carries the fields the dashboard knows after a turn:
          question, answer, mode, scope, top_k, find_quote, expanded,
          answer_language, filters, applied_filters, detected_filters,
          citations, and an optional `conversation_id`. Missing optional
          fields fall back to sensible defaults.

        When `conversation_id` is absent this starts a new thread (a fresh id,
        turn_index 0); when present the turn is appended to that thread
        (turn_index = current max + 1, computed in the same transaction).
        """
        new_id = uuid.uuid4()
        title = derive_title(record.get("question", ""))
        conv_id = record.get("conversation_id")
        if conv_id:
            _validate_uuid(conv_id)
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                self._set_timeout(cur)
                if conv_id:
                    cur.execute(
                        "SELECT COALESCE(MAX(turn_index) + 1, 0) AS next "
                        "FROM conversation_history WHERE conversation_id = %s",
                        (str(conv_id),),
                    )
                    turn_index = cur.fetchone()["next"]
                else:
                    conv_id = uuid.uuid4()
                    turn_index = 0
                cur.execute(
                    """
                    INSERT INTO conversation_history (
                        id, conversation_id, turn_index,
                        title, question, answer, mode, scope,
                        answer_language, find_quote, expanded, top_k,
                        filters, applied_filters, detected_filters, citations
                    ) VALUES (
                        %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    RETURNING id, conversation_id, turn_index, title,
                              created_at, question, answer, mode, scope,
                              answer_language, find_quote, expanded, top_k,
                              filters, applied_filters, detected_filters,
                              citations
                    """,
                    (
                        str(new_id),
                        str(conv_id),
                        turn_index,
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
        """Remove a whole conversation (all its turns). Returns True if any
        row was deleted."""
        _validate_uuid(conversation_id)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                self._set_timeout(cur)
                cur.execute(
                    "DELETE FROM conversation_history WHERE conversation_id = %s",
                    (conversation_id,),
                )
                deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return bool(deleted)

    # ---- reads ----------------------------------------------------------

    def list_summaries(self, limit: int = 200) -> list[dict[str, Any]]:
        """Sidebar listing — one row per conversation, light payload (no answer
        / citation bodies).

        Turns are grouped by `conversation_id`; the label is the first turn's
        title, and the thread sorts by its most-recent activity (so continuing
        an old chat floats it back to the top / into "Today").
        """
        limit = max(1, min(int(limit), 500))
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                self._set_timeout(cur)
                cur.execute(
                    """
                    SELECT
                        conversation_id AS id,
                        (ARRAY_AGG(title ORDER BY turn_index ASC))[1] AS title,
                        (ARRAY_AGG(mode  ORDER BY turn_index ASC))[1] AS mode,
                        (ARRAY_AGG(scope ORDER BY turn_index ASC))[1] AS scope,
                        MAX(created_at) AS created_at,
                        COUNT(*)        AS turn_count
                    FROM conversation_history
                    GROUP BY conversation_id
                    ORDER BY MAX(created_at) DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [_serialize_summary(r) for r in rows]

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        """Full conversation for the viewer — all turns, oldest first. Returns
        None if the conversation has no turns.

        Shape: ``{id, title, created_at, turns: [<turn>, ...]}`` where each turn
        carries the per-turn fields (question, answer, citations, …). `title`
        and `created_at` come from the first turn.
        """
        _validate_uuid(conversation_id)
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                self._set_timeout(cur)
                cur.execute(
                    """
                    SELECT id, conversation_id, turn_index, title, created_at,
                           question, answer, mode, scope, answer_language,
                           find_quote, expanded, top_k, filters,
                           applied_filters, detected_filters, citations
                    FROM conversation_history
                    WHERE conversation_id = %s
                    ORDER BY turn_index ASC
                    """,
                    (conversation_id,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        if not rows:
            return None
        turns = [_serialize(r) for r in rows]
        first = turns[0]
        return {
            "id": str(conversation_id),
            "title": first["title"],
            "created_at": first["created_at"],
            "mode": first["mode"],
            "scope": first["scope"],
            "turns": turns,
        }


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
        "turn_count": int(row.get("turn_count", 1)),
    }


def _serialize(row: dict[str, Any]) -> dict[str, Any]:
    """Full row → JSON-friendly dict. JSONB columns come back as dict/list
    already with psycopg2, so we only normalize the ids + timestamp."""
    out = dict(row)
    out["id"] = str(out["id"])
    if out.get("conversation_id") is not None:
        out["conversation_id"] = str(out["conversation_id"])
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
