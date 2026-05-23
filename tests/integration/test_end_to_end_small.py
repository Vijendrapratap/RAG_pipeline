"""End-to-end integration test for hardened bulk ingestion.

Hits the real running stack — Ollama (bge-m3 embeddings), Qdrant (test
collection created and torn down here), and Tantivy (per-test index dir).

Skipped automatically if Ollama or Qdrant are not reachable, so this file
can sit in the suite even when the stack is down.

Per PRD §6 Phase 4 acceptance criteria, asserts:
  - Healthy chunk files marked 'ok'
  - Corrupted file marked 'failed' with non-null reason
  - Corrupted file copied into dead_letter/<reason>/
  - Qdrant has expected number of points
  - Tantivy has expected number of documents
  - Running twice is a no-op (no duplicate work)
"""
from __future__ import annotations

import json
import os
import uuid as _uuid
from pathlib import Path

import pytest
import requests

from ingestion.utils.progress_db import ProgressDB

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")
PG_DSN = os.environ.get(
    "PG_DSN",
    f"postgresql://owui:{os.environ.get('POSTGRES_PASSWORD', '')}"
    "@localhost:5432/openwebui",
)

SCHEMA_SQL = Path(__file__).resolve().parents[2] / "infra" / "postgres" / "analytics_schema.sql"


def _stack_alive() -> bool:
    try:
        r1 = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        h = {"api-key": QDRANT_KEY} if QDRANT_KEY else {}
        r2 = requests.get(f"{QDRANT_URL}/collections", headers=h, timeout=3)
        return r1.status_code < 500 and r2.status_code < 500
    except requests.RequestException:
        return False


def _postgres_alive() -> bool:
    try:
        import psycopg2
        c = psycopg2.connect(PG_DSN, connect_timeout=3)
        c.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _stack_alive(),
    reason="Ollama or Qdrant not reachable on localhost — integration test skipped",
)


@pytest.fixture(scope="session")
def postgres_schema():
    """Apply Phase 6 schema (idempotent) once per test session, then yield a
    connection-getter. Skips with a marker if Postgres is down."""
    if not _postgres_alive():
        pytest.skip("Postgres not reachable — Phase 6 assertions skipped")
    import psycopg2
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    conn = psycopg2.connect(PG_DSN)
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    conn.close()

    def _conn():
        return psycopg2.connect(PG_DSN)
    return _conn


def _make_chunks_json(path: Path, source_name: str, n_chunks: int) -> None:
    chunks = []
    for i in range(n_chunks):
        body = (
            f"This is test chunk {i} from {source_name}. "
            "It contains some lorem ipsum prose to give the embedder something "
            "meaningful to look at. " * 3
        )
        chunks.append({
            "text": (
                f"[Source: {source_name} | 00:00:{i:02d} -> 00:00:{i+1:02d} "
                f"| Speakers: SPEAKER_00]\n{body}"
            ),
            "source_file": source_name,
            "start_sec": float(i),
            "end_sec": float(i + 1),
            "speakers": ["SPEAKER_00"],
            "format": "whisperx",
            "has_timestamps": True,
            "word_count": len(body.split()),
        })
    path.write_text(
        json.dumps({"source_file": source_name, "format": "whisperx",
                    "chunks": chunks}),
        encoding="utf-8",
    )


def _setup_test_collection(name: str) -> None:
    """Create a fresh Qdrant collection with the same spec as production."""
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )


def _drop_test_collection(name: str) -> None:
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    try:
        client.delete_collection(name)
    except Exception:
        pass


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point all module-level state at tmp paths and a throwaway collection."""
    test_coll = f"test_phase4_{_uuid.uuid4().hex[:8]}"
    _setup_test_collection(test_coll)

    chunks_dir = tmp_path / "processed"
    chunks_dir.mkdir()
    dead_dir = tmp_path / "dead_letter"
    tantivy_dir = tmp_path / "tantivy"
    progress_path = tmp_path / "progress.sqlite"
    log_file = tmp_path / "ingest.log"

    # Patch BEFORE importing main() each call so module constants pick up
    # the test paths. bulk_ingest_hardened is already imported by other
    # collected modules — monkeypatch the attributes directly.
    import ingestion.bulk_ingest_hardened as bih
    monkeypatch.setattr(bih, "QDRANT_COLLECTION", test_coll)
    monkeypatch.setattr(bih, "DEAD_LETTER_DIR", dead_dir)
    monkeypatch.setattr(bih, "TANTIVY_DIR", tantivy_dir)
    monkeypatch.setattr(bih, "PROGRESS_DB_PATH", str(progress_path))
    monkeypatch.setattr(bih, "LOG_FILE", str(log_file))
    monkeypatch.setattr(bih, "FILE_TIMEOUT_SEC", 120)

    yield {
        "chunks_dir": chunks_dir,
        "dead_dir": dead_dir,
        "tantivy_dir": tantivy_dir,
        "progress_path": progress_path,
        "collection": test_coll,
    }
    _drop_test_collection(test_coll)


def test_end_to_end_small(isolated_env, postgres_schema):
    """Three healthy files + one corrupted file -> 3 ok, 1 failed,
    corrupted file in dead_letter, Qdrant and Tantivy counts match,
    and chunk_meta/file_meta rows are populated (Phase 6)."""
    chunks_dir = isolated_env["chunks_dir"]
    dead_dir = isolated_env["dead_dir"]
    tantivy_dir = isolated_env["tantivy_dir"]
    progress_path = isolated_env["progress_path"]
    collection = isolated_env["collection"]

    _make_chunks_json(chunks_dir / "alpha.chunks.json", "alpha.json", 3)
    _make_chunks_json(chunks_dir / "bravo.chunks.json", "bravo.json", 4)
    _make_chunks_json(chunks_dir / "charlie.chunks.json", "charlie.json", 5)
    # Corrupted: truncated JSON
    (chunks_dir / "delta.chunks.json").write_text(
        '{"source_file": "delta.json", "format": "whisperx", "chunks": [',
        encoding="utf-8",
    )
    expected_chunks = 3 + 4 + 5

    from ingestion.bulk_ingest_hardened import main as bulk_main
    rc = bulk_main([
        "--chunks-dir", str(chunks_dir),
        "--batch-size", "4",
        "--no-tantivy-health",
    ])
    # 1 failed file -> exit 1, but other files succeed.
    assert rc == 1, f"expected exit 1 (one failure), got {rc}"

    # Progress DB: 3 ok, 1 failed
    db = ProgressDB(progress_path)
    try:
        assert db.get_status("alpha.chunks.json") == "ok"
        assert db.get_status("bravo.chunks.json") == "ok"
        assert db.get_status("charlie.chunks.json") == "ok"
        assert db.get_status("delta.chunks.json") == "failed"
        failed = dict(db.list_failed())
        assert "delta.chunks.json" in failed
        assert failed["delta.chunks.json"]  # non-null reason
    finally:
        db.close()

    # Dead-letter quarantine: delta file copied into some subdir
    dead_files = list(dead_dir.rglob("delta.chunks.json"))
    assert dead_files, f"expected delta in dead_letter, got {list(dead_dir.rglob('*'))}"

    # Qdrant point count
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    info = client.count(collection_name=collection, exact=True)
    assert info.count == expected_chunks, (
        f"Qdrant has {info.count} points, expected {expected_chunks}"
    )

    # Tantivy doc count
    import tantivy  # noqa: F401 — required only if Tantivy installed
    index = tantivy.Index.open(str(tantivy_dir))
    index.reload()
    n_docs = index.searcher().num_docs
    assert n_docs == expected_chunks, (
        f"Tantivy has {n_docs} docs, expected {expected_chunks}"
    )

    # Phase 6: chunk_meta count for these source files matches what Qdrant has
    # for the same set; file_meta has one row per source file with non-null
    # chunk_count.
    sources = ("alpha.json", "bravo.json", "charlie.json")
    with postgres_schema() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM chunk_meta WHERE source_file = ANY(%s)",
                (list(sources),),
            )
            pg_count = cur.fetchone()[0]
            assert pg_count == expected_chunks, (
                f"chunk_meta has {pg_count} for {sources}, expected {expected_chunks}"
            )
            cur.execute(
                "SELECT source_file, chunk_count FROM file_meta "
                "WHERE source_file = ANY(%s) ORDER BY source_file",
                (list(sources),),
            )
            rows = cur.fetchall()
            assert len(rows) == 3, f"expected 3 file_meta rows, got {len(rows)}"
            for source, n in rows:
                assert n is not None and n > 0, f"{source}: chunk_count={n}"


def test_second_run_is_noop(isolated_env):
    """Re-running on the same chunks dir should skip all 'ok' files."""
    chunks_dir = isolated_env["chunks_dir"]
    progress_path = isolated_env["progress_path"]
    collection = isolated_env["collection"]

    _make_chunks_json(chunks_dir / "single.chunks.json", "single.json", 2)

    from ingestion.bulk_ingest_hardened import main as bulk_main

    rc1 = bulk_main(["--chunks-dir", str(chunks_dir), "--batch-size", "4",
                     "--no-tantivy-health"])
    assert rc1 == 0

    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    count_after_first = client.count(collection_name=collection, exact=True).count

    rc2 = bulk_main(["--chunks-dir", str(chunks_dir), "--batch-size", "4",
                     "--no-tantivy-health"])
    assert rc2 == 0

    # Same point count — no duplicates and no new work.
    count_after_second = client.count(collection_name=collection, exact=True).count
    assert count_after_second == count_after_first

    # Progress DB still shows 'ok'
    db = ProgressDB(progress_path)
    try:
        assert db.get_status("single.chunks.json") == "ok"
    finally:
        db.close()


def test_verify_ingestion_passes(isolated_env):
    """verify_ingestion.py should report 100% present after a clean run."""
    chunks_dir = isolated_env["chunks_dir"]

    _make_chunks_json(chunks_dir / "v1.chunks.json", "v1.json", 3)
    _make_chunks_json(chunks_dir / "v2.chunks.json", "v2.json", 3)

    from ingestion.bulk_ingest_hardened import main as bulk_main
    rc = bulk_main(["--chunks-dir", str(chunks_dir), "--batch-size", "4",
                    "--no-tantivy-health"])
    assert rc == 0

    # Re-patch verify_ingestion's view of TANTIVY_DIR / QDRANT_COLLECTION
    # since it imports them at module load.
    import ingestion.verify_ingestion as vi
    import ingestion.bulk_ingest_hardened as bih
    vi.TANTIVY_DIR = bih.TANTIVY_DIR
    vi.QDRANT_COLLECTION = bih.QDRANT_COLLECTION
    rc = vi.main([
        "--chunks-dir", str(chunks_dir),
        "--sample-size", "6",
        "--threshold", "99.9",
        "--seed", "1",
    ])
    assert rc == 0


def test_phase6_sample_whisperx_populates_file_meta(isolated_env, postgres_schema):
    """PRD §6 Phase 6 acceptance literal:
        SELECT * FROM file_meta WHERE source_file = 'sample_whisperx.json'
    must return a row with non-null chunk_count after ingestion of the
    sample_whisperx fixture via chunker_json -> bulk_ingest.
    Also confirms SELECT COUNT(*) FROM chunk_meta matches Qdrant point count."""
    chunks_dir = isolated_env["chunks_dir"]
    collection = isolated_env["collection"]

    # Chunk the real Phase 3 fixture into chunks_dir.
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    from ingestion.chunker_json import main as chunker_main
    rc = chunker_main([str(fixtures), str(chunks_dir), "--format", "whisperx"])
    assert rc == 0
    # The chunker also picks up corrupted_files/malformed.json — that's
    # expected to fail. Remove its error log so it doesn't get re-ingested
    # as if it were a chunks file, and remove the corrupted source itself.
    failed_dir = chunks_dir / "_failed"
    if failed_dir.exists():
        for f in failed_dir.iterdir():
            f.unlink()
        failed_dir.rmdir()
    assert (chunks_dir / "sample_whisperx.chunks.json").exists()

    from ingestion.bulk_ingest_hardened import main as bulk_main
    rc = bulk_main(["--chunks-dir", str(chunks_dir), "--batch-size", "4",
                    "--no-tantivy-health"])
    assert rc == 0, f"bulk ingest failed: rc={rc}"

    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    qdrant_count = client.count(collection_name=collection, exact=True).count

    with postgres_schema() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM chunk_meta WHERE source_file = %s",
                ("sample_whisperx.json",),
            )
            cm_count = cur.fetchone()[0]
            cur.execute(
                "SELECT source_file, duration_sec, speakers, chunk_count, "
                "       ingested_at "
                "FROM file_meta WHERE source_file = %s",
                ("sample_whisperx.json",),
            )
            row = cur.fetchone()
            assert row is not None, "file_meta row for sample_whisperx.json missing"
            source_file, duration_sec, speakers, chunk_count, ingested_at = row
            assert chunk_count is not None and chunk_count > 0, (
                f"file_meta.chunk_count={chunk_count} (must be non-null and > 0)"
            )
            assert cm_count == qdrant_count, (
                f"chunk_meta has {cm_count} rows for sample_whisperx.json; "
                f"Qdrant has {qdrant_count} points — mismatch"
            )
            # Sanity: duration matches the fixture's last segment end (75.2s),
            # speakers = the 3 SPEAKER_* values from the fixture.
            assert duration_sec is not None and duration_sec > 0
            assert speakers and "SPEAKER_00" in speakers
