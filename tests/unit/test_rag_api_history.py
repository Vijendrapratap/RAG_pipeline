"""Unit tests for rag_api.history — multi-turn conversation persistence.

The Postgres round-trip is faked (a scripted cursor), so these run without a
live database. They pin the behaviour migration 003 added: a new chat mints a
thread id at turn 0, a follow-up appends at max(turn_index)+1, a conversation
reads back as an ordered `turns` list, and deleting removes the whole thread.
The pure serialization helpers are tested directly.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from rag_api.history import (
    History,
    _serialize,
    _serialize_summary,
    _validate_uuid,
    derive_title,
)

FIXED_TS = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)

# The order save() passes params in, matching the INSERT column list.
_INSERT_ORDER = [
    "id", "conversation_id", "turn_index", "title", "question", "answer",
    "mode", "scope", "answer_language", "find_quote", "expanded", "top_k",
    "filters", "applied_filters", "detected_filters", "citations",
]


def _unwrap(v):
    """psycopg2 wraps JSONB params in Json(...); the driver returns them decoded
    on read, so the fake mirrors that."""
    return getattr(v, "adapted", v)


class _FakeCursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self.conn = conn
        self._sql = ""
        self._params: tuple | None = None
        self.rowcount = 0

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def execute(self, sql: str, params=None) -> None:
        self._sql, self._params = sql, params
        self.conn.executed.append((sql, params))
        if sql.strip().startswith("DELETE"):
            self.rowcount = self.conn.rowcount

    def fetchone(self):
        if "MAX(turn_index)" in self._sql:
            return {"next": self.conn.next_turn}
        if "RETURNING" in self._sql:
            row = {k: _unwrap(v) for k, v in zip(_INSERT_ORDER, self._params)}
            row["created_at"] = FIXED_TS
            return row
        return None

    def fetchall(self):
        return self.conn.fetchall_rows


class _FakeConn:
    def __init__(self, next_turn=0, fetchall_rows=None, rowcount=0) -> None:
        self.next_turn = next_turn
        self.fetchall_rows = fetchall_rows or []
        self.rowcount = rowcount
        self.executed: list[tuple] = []

    def cursor(self, cursor_factory=None) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


def _history(conn: _FakeConn) -> History:
    h = History("postgresql://fake/db")
    h._connect = lambda: conn  # type: ignore[method-assign]
    return h


def _turn_row(conv: str, idx: int, title: str, q: str, a: str) -> dict:
    """A read row shaped like get()'s SELECT returns."""
    return {
        "id": uuid.uuid4(),
        "conversation_id": uuid.UUID(conv),
        "turn_index": idx,
        "title": title,
        "created_at": FIXED_TS,
        "question": q,
        "answer": a,
        "mode": "answer",
        "scope": "chunks",
        "answer_language": "hindi",
        "find_quote": False,
        "expanded": False,
        "top_k": 4,
        "filters": {},
        "applied_filters": {},
        "detected_filters": [],
        "citations": [],
    }


# --- save: new thread vs append ------------------------------------------


def test_save_without_conversation_id_starts_a_new_thread_at_turn_zero():
    conn = _FakeConn()
    row = _history(conn).save({"question": "Q1", "answer": "A1"})
    assert row["turn_index"] == 0
    uuid.UUID(row["conversation_id"])                 # a real minted id
    assert row["id"] != row["conversation_id"]        # turn id ≠ thread id
    # The new-thread path must NOT query for a previous max turn.
    assert not any("MAX(turn_index)" in s for s, _ in conn.executed)


def test_save_with_conversation_id_appends_at_next_turn_index():
    conv = str(uuid.uuid4())
    conn = _FakeConn(next_turn=3)
    row = _history(conn).save({"question": "Q2", "answer": "A2",
                               "conversation_id": conv})
    assert row["conversation_id"] == conv             # same thread
    assert row["turn_index"] == 3                      # max + 1
    assert any("MAX(turn_index)" in s for s, _ in conn.executed)


def test_save_rejects_a_malformed_conversation_id_before_touching_the_db():
    conn = _FakeConn()
    with pytest.raises(ValueError):
        _history(conn).save({"question": "Q", "conversation_id": "not-a-uuid"})
    assert conn.executed == []


# --- get: whole thread, ordered ------------------------------------------


def test_get_assembles_all_turns_oldest_first():
    conv = str(uuid.uuid4())
    conn = _FakeConn(fetchall_rows=[
        _turn_row(conv, 0, "First question", "Q0", "A0"),
        _turn_row(conv, 1, "Follow-up", "Q1", "A1"),
    ])
    rec = _history(conn).get(conv)
    assert rec is not None
    assert rec["id"] == conv
    assert rec["title"] == "First question"           # first turn's title
    assert [t["turn_index"] for t in rec["turns"]] == [0, 1]
    assert rec["turns"][1]["answer"] == "A1"


def test_get_returns_none_for_a_conversation_with_no_turns():
    conn = _FakeConn(fetchall_rows=[])
    assert _history(conn).get(str(uuid.uuid4())) is None


# --- delete: whole thread -------------------------------------------------


def test_delete_removes_the_whole_thread_by_conversation_id():
    conn = _FakeConn(rowcount=2)
    assert _history(conn).delete(str(uuid.uuid4())) is True
    delete_sql = [s for s, _ in conn.executed if s.strip().startswith("DELETE")]
    assert delete_sql and "conversation_id" in delete_sql[0]


def test_delete_returns_false_when_nothing_matched():
    conn = _FakeConn(rowcount=0)
    assert _history(conn).delete(str(uuid.uuid4())) is False


# --- pure helpers ---------------------------------------------------------


def test_derive_title_takes_the_first_line_and_truncates():
    assert derive_title("hello\nworld") == "hello"
    assert derive_title("") == "Untitled"
    long = "x" * 200
    out = derive_title(long)
    assert len(out) == 80 and out.endswith("…")


def test_validate_uuid_rejects_garbage():
    _validate_uuid(str(uuid.uuid4()))                 # no raise
    with pytest.raises(ValueError):
        _validate_uuid("nope")


def test_serialize_stringifies_both_ids_and_decodes_json_strings():
    out = _serialize({
        "id": uuid.uuid4(),
        "conversation_id": uuid.uuid4(),
        "created_at": FIXED_TS,
        "filters": '{"speaker": "X"}',               # a driver that returns raw
        "citations": [],
    })
    assert isinstance(out["id"], str) and isinstance(out["conversation_id"], str)
    assert out["created_at"] == FIXED_TS.isoformat()
    assert out["filters"] == {"speaker": "X"}         # decoded, not a string


def test_serialize_summary_carries_turn_count():
    out = _serialize_summary({
        "id": uuid.uuid4(),
        "title": "T",
        "created_at": FIXED_TS,
        "mode": "answer",
        "scope": "chunks",
        "turn_count": 3,
    })
    assert out["turn_count"] == 3
    assert isinstance(out["id"], str)
