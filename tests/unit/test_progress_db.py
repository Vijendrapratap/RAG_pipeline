"""Unit tests for ingestion.utils.progress_db."""
from __future__ import annotations

import pytest

from ingestion.utils.progress_db import ProgressDB


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "progress.sqlite"
    d = ProgressDB(p)
    yield d
    d.close()


def test_get_status_unknown_returns_none(db):
    assert db.get_status("nope.json") is None


def test_mark_ok_then_status_ok(db):
    db.mark_ok("a.chunks.json", 42)
    assert db.get_status("a.chunks.json") == "ok"


def test_mark_failed_then_status_failed_with_reason(db):
    db.mark_failed("b.chunks.json", "JSONDecodeError: line 3")
    assert db.get_status("b.chunks.json") == "failed"
    failed = db.list_failed()
    assert any(f == "b.chunks.json" and "JSONDecodeError" in (r or "")
               for f, r in failed)


def test_mark_in_progress_visible(db):
    db.mark_in_progress("c.chunks.json")
    assert db.get_status("c.chunks.json") == "in_progress"


def test_attempts_increment_across_marks(db):
    f = "d.chunks.json"
    db.mark_in_progress(f)
    db.mark_failed(f, "x")
    db.mark_in_progress(f)
    db.mark_ok(f, 5)
    stats = db.stats()
    assert stats["ok"] == 1
    # 4 transitions; counter starts at 0 and increments each call.
    row = db._conn.execute(
        "SELECT attempts FROM ingest_status WHERE file = ?", (f,)
    ).fetchone()
    assert row[0] == 4


def test_reset_to_pending_removes_entry(db):
    db.mark_failed("x.json", "oops")
    db.reset_to_pending("x.json")
    assert db.get_status("x.json") is None


def test_stats_aggregates(db):
    db.mark_ok("a", 1)
    db.mark_ok("b", 2)
    db.mark_failed("c", "r")
    db.mark_in_progress("d")
    s = db.stats()
    assert s["ok"] == 2
    assert s["failed"] == 1
    assert s["in_progress"] == 1
    assert s["skipped"] == 0


def test_persists_across_reopens(tmp_path):
    p = tmp_path / "x.sqlite"
    d1 = ProgressDB(p)
    d1.mark_ok("persisted.json", 7)
    d1.close()
    d2 = ProgressDB(p)
    try:
        assert d2.get_status("persisted.json") == "ok"
    finally:
        d2.close()
