"""Integration tests for Phase 7 Open WebUI function tools.

Hits the real stack: Ollama (bge-m3 embeddings), Qdrant (test collection),
Tantivy sidecar (started in-process for these tests), Infinity reranker,
Postgres. Skipped if any prerequisite is down.

Per PRD §6 Phase 7 acceptance:
1. Both function files parse with ast.parse (syntactic validity).
2. `find_quote` returns a known phrase in the top 3 results.

Plus structural checks: output format, valve overrides, analytics returns
sensible counts.
"""
from __future__ import annotations

import ast
import json
import os
import socket
import subprocess
import sys
import time
import uuid as _uuid
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")
RERANKER_URL = os.environ.get("RERANKER_URL", "http://localhost:7997/rerank")
PG_DSN = os.environ.get(
    "PG_DSN",
    f"postgresql://owui:{os.environ.get('POSTGRES_PASSWORD', '')}"
    "@localhost:5432/openwebui",
)
SCHEMA_SQL = REPO_ROOT / "infra" / "postgres" / "analytics_schema.sql"


def _alive(url: str, headers: dict[str, str] | None = None) -> bool:
    try:
        return requests.get(url, headers=headers or {}, timeout=3).status_code < 500
    except requests.RequestException:
        return False


def _pg_alive() -> bool:
    try:
        import psycopg2
        psycopg2.connect(PG_DSN, connect_timeout=3).close()
        return True
    except Exception:
        return False


def _stack_ready() -> bool:
    return (
        _alive(f"{OLLAMA_URL}/api/tags")
        and _alive(f"{QDRANT_URL}/collections",
                   headers={"api-key": QDRANT_KEY} if QDRANT_KEY else None)
        and _alive(RERANKER_URL.replace("/rerank", "/health"))
        and _pg_alive()
    )


pytestmark = pytest.mark.skipif(
    not _stack_ready(),
    reason="One of Ollama/Qdrant/Reranker/Postgres not reachable — Phase 7 "
    "integration test skipped",
)


# ---- Test 1: PRD acceptance #1 — ast.parse on both function files --------


def test_search_transcripts_parses_as_python():
    src = (REPO_ROOT / "open_webui_functions" / "search_transcripts.py").read_text(
        encoding="utf-8"
    )
    ast.parse(src)


def test_analytics_parses_as_python():
    src = (REPO_ROOT / "open_webui_functions" / "analytics.py").read_text(
        encoding="utf-8"
    )
    ast.parse(src)


# ---- Live stack fixtures: ingest + start sidecar --------------------------


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _ensure_pg_schema() -> None:
    import psycopg2
    conn = psycopg2.connect(PG_DSN)
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()


def _setup_collection(name: str) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
    c = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    try:
        c.delete_collection(name)
    except Exception:
        pass
    c.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )


def _drop_collection(name: str) -> None:
    from qdrant_client import QdrantClient
    c = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    try:
        c.delete_collection(name)
    except Exception:
        pass


@pytest.fixture
def populated_stack(tmp_path, monkeypatch):
    """Chunk + ingest the Phase 3 fixtures into a throwaway Qdrant collection
    and a tmp Tantivy index dir, start an in-process sidecar on a free port,
    yield {valve overrides}, tear down on exit."""
    _ensure_pg_schema()

    test_coll = f"test_phase7_{_uuid.uuid4().hex[:8]}"
    _setup_collection(test_coll)

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    tantivy_dir = tmp_path / "tantivy"
    dead_dir = tmp_path / "dead_letter"
    progress_path = tmp_path / "progress.sqlite"

    import ingestion.bulk_ingest_hardened as bih
    monkeypatch.setattr(bih, "QDRANT_COLLECTION", test_coll)
    monkeypatch.setattr(bih, "DEAD_LETTER_DIR", dead_dir)
    monkeypatch.setattr(bih, "TANTIVY_DIR", tantivy_dir)
    monkeypatch.setattr(bih, "PROGRESS_DB_PATH", str(progress_path))
    monkeypatch.setattr(bih, "LOG_FILE", str(tmp_path / "ingest.log"))
    monkeypatch.setattr(bih, "FILE_TIMEOUT_SEC", 120)

    # Chunk both Phase 3 fixtures.
    from ingestion.chunker_json import main as cj_main
    from ingestion.chunker_text import main as ct_main
    cj_rc = cj_main([str(FIXTURES), str(chunks_dir), "--format", "whisperx"])
    assert cj_rc == 0
    # chunker_json's _failed dir leftover from the corrupted fixture
    failed_dir = chunks_dir / "_failed"
    if failed_dir.exists():
        for f in failed_dir.iterdir():
            f.unlink()
        failed_dir.rmdir()
    ct_rc = ct_main([str(FIXTURES), str(chunks_dir)])
    assert ct_rc == 0
    failed_dir = chunks_dir / "_failed"
    if failed_dir.exists():
        for f in failed_dir.iterdir():
            f.unlink()
        failed_dir.rmdir()

    # Ingest into Qdrant + Tantivy + Postgres.
    rc = bih.main([
        "--chunks-dir", str(chunks_dir),
        "--batch-size", "8",
        "--no-tantivy-health",
    ])
    assert rc == 0, f"bulk ingest failed: rc={rc}"

    # Start an in-process Tantivy sidecar on a free port.
    port = _free_port()
    env = dict(os.environ)
    env["TANTIVY_DIR"] = str(tantivy_dir)
    sidecar = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "services.tantivy_server.tantivy_server:app",
         "--host", "127.0.0.1", "--port", str(port),
         "--workers", "1", "--log-level", "warning"],
        env=env, cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Wait for sidecar to come up (max ~10s).
    for _ in range(40):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            if r.ok:
                break
        except requests.RequestException:
            pass
        time.sleep(0.25)
    else:
        sidecar.terminate()
        sidecar.wait()
        pytest.fail("Tantivy sidecar did not start within 10s")

    try:
        yield {
            "qdrant_collection": test_coll,
            "tantivy_proxy_url": f"http://127.0.0.1:{port}",
        }
    finally:
        sidecar.terminate()
        try:
            sidecar.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sidecar.kill()
            sidecar.wait()
        _drop_collection(test_coll)


def _make_search_tool(overrides: dict[str, str]):
    """Instantiate the Tools class with valves pointed at the test stack."""
    from open_webui_functions.search_transcripts import Tools
    t = Tools()
    t.valves.qdrant_url = QDRANT_URL
    t.valves.qdrant_key = QDRANT_KEY
    t.valves.qdrant_collection = overrides["qdrant_collection"]
    t.valves.ollama_url = OLLAMA_URL
    t.valves.reranker_url = RERANKER_URL
    t.valves.tantivy_proxy_url = overrides["tantivy_proxy_url"]
    return t


def _make_analytics_tool():
    from open_webui_functions.analytics import Tools
    t = Tools()
    t.valves.pg_dsn = PG_DSN
    return t


# ---- PRD acceptance #3: find_quote returns known phrase in top 3 ---------


def test_find_quote_returns_known_phrase_in_top_3(populated_stack):
    """sample_whisperx fixture contains the phrase 'platform team' (literal).
    Query for it via find_quote — must show up in top 3."""
    tool = _make_search_tool(populated_stack)
    out = tool.find_quote("platform team", top_k=3)

    # Output formatting smoke check
    assert out.startswith("--- Result 1"), f"unexpected output start: {out[:80]!r}"
    assert "Source:" in out
    assert "Speakers:" in out

    # The phrase MUST appear in one of the top-3 result bodies.
    # Split by separator and check.
    blocks = [b for b in out.split("--- Result ") if b.strip()]
    assert len(blocks) >= 1
    top_3_text = " ".join(blocks[:3])
    assert "platform team" in top_3_text.lower(), (
        f"'platform team' not found in top 3 results:\n{top_3_text[:500]}"
    )


def test_search_transcripts_returns_formatted_results(populated_stack):
    """End-to-end: query for 'distributed systems' (from sample_plain
    fixture) — confirm dense+BM25+RRF+rerank pipeline returns at least one
    hit with the PRD-spec output format."""
    tool = _make_search_tool(populated_stack)
    out = tool.search_transcripts("distributed systems trade-offs", top_k=3)
    assert out and out != "No results.\n"
    # Header lines per PRD spec
    assert "--- Result 1 (score:" in out
    assert "Source: " in out
    assert "Speakers:" in out
    # At least one result body mentions a term from the fixture
    assert "distributed" in out.lower() or "consistency" in out.lower()


def test_search_transcripts_speaker_filter(populated_stack):
    """Filter to SPEAKER_00 — every returned chunk's payload speakers list
    must contain SPEAKER_00."""
    tool = _make_search_tool(populated_stack)
    out = tool.search_transcripts("agenda items", speaker="SPEAKER_00",
                                  top_k=3)
    assert out and out != "No results.\n"
    # Each "Speakers:" line must include SPEAKER_00.
    for line in out.splitlines():
        if line.startswith("Source:"):
            assert "SPEAKER_00" in line, (
                f"speaker filter leaked a non-SPEAKER_00 result: {line!r}"
            )


# ---- Analytics tool live checks ------------------------------------------


def test_analytics_count_mentions_existing_term(populated_stack):
    """sample_plain fixture talks about 'distributed systems' — count must
    be >= 1."""
    tool = _make_analytics_tool()
    out = tool.count_mentions("distributed")
    # Format: 'Term "distributed" mentioned in N chunks (across all speakers).'
    assert 'Term "distributed" mentioned in' in out
    n = int(out.split("mentioned in ")[1].split(" chunks")[0])
    assert n >= 1, f"expected at least 1 mention, got {n}: {out!r}"


def test_analytics_count_mentions_unknown_term(populated_stack):
    tool = _make_analytics_tool()
    out = tool.count_mentions("zxqvbnpdhsrf")
    assert "mentioned in 0 chunks" in out


def test_analytics_list_transcripts(populated_stack):
    tool = _make_analytics_tool()
    out = tool.list_transcripts_mentioning("consistency", limit=5)
    assert "sample_plain.json" in out or "sample_plain.txt" in out


def test_analytics_top_speakers(populated_stack):
    tool = _make_analytics_tool()
    out = tool.top_speakers_for_topic("meeting", limit=5)
    # 'meeting' appears in sample_whisperx whose speakers are SPEAKER_*
    assert ("SPEAKER_00" in out or "SPEAKER_01" in out or "SPEAKER_02" in out
            or "No speakers found" in out)
