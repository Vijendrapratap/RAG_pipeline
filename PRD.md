# PRD: Local-Only Production RAG Stack for 5TB Whisper Transcripts

**Project codename:** `transcript-rag`
**Owner:** [you]
**Implementer:** Claude Code (via terminal)
**Status:** Ready for implementation
**Target environment:** Single host with NVIDIA RTX 5070 (12 GB VRAM), 64–128 GB RAM, Linux (Ubuntu 22.04+ recommended)

---

## 0. How to use this PRD with Claude Code

This document is the **single source of truth** for Claude Code. Follow these rules:

1. **Place this file at the repo root** as `PRD.md`. Also place `CLAUDE.md` (Section 17) next to it.
2. **Open Claude Code in the repo root** with `claude` and start with the prompt in Section 18.
3. **Execute one phase at a time.** Do not skip ahead. Each phase has explicit *Deliverables* and *Acceptance Criteria* — Claude Code must satisfy both before moving on.
4. **Commit between phases.** After each phase passes acceptance, commit with the suggested message.
5. **No external paid APIs allowed.** Every component in this stack is open-source and self-hosted. If Claude Code suggests OpenAI / Anthropic / Cohere / Voyage / Pinecone / Weaviate Cloud / etc. — reject it.
6. **When in doubt, refer back to this PRD.** Claude Code should re-read it at the start of each phase.

---

## 1. Executive Summary

Build a **strictly local, open-source RAG system** over ~5 TB of Whisper-generated transcripts. The system retrieves and answers four query categories, in priority order:

1. **Quote finding** — "who said X" / "find when X was said" (highest priority)
2. **Single-transcript Q&A** — questions about one transcript
3. **Cross-corpus topic summary** — themes across many transcripts
4. **Analytics** — counts, top speakers, mentions over time

The user interacts via **Open WebUI**. Models call **custom function tools** (not Open WebUI's built-in KB) to query a hybrid retrieval pipeline: **Qdrant (dense) + Tantivy (BM25) + RRF fusion + bge-reranker-v2-m3**.

Realistic accuracy ceiling (weighted across the four query types): **85–88%**. We do not promise higher.

---

## 2. Goals and Non-Goals

### Goals
- ✅ 100% local execution. No data leaves the host.
- ✅ Hybrid retrieval with reranking, tuned for quote-finding.
- ✅ Handle 250–400M chunks (5 TB raw text).
- ✅ Resumable, fault-tolerant bulk ingestion with dead-letter quarantine.
- ✅ Open WebUI front-end with custom tool-calling models.
- ✅ Postgres-backed analytics for aggregation queries.
- ✅ Deployable via `docker compose up -d` + a few Python scripts.

### Non-Goals
- ❌ Paid APIs of any kind (OpenAI, Cohere, Anthropic, etc.).
- ❌ Multi-host clustering (single host only).
- ❌ Real-time ingestion (batch only — re-runs are fine).
- ❌ Tika or Docling content extractors (Whisper text is parsed directly).
- ❌ Fine-tuning models (use off-the-shelf bge-m3, qwen, deepseek).
- ❌ Zero-error ingestion guarantee. We promise *detected, logged, isolated, resumable* errors — not their absence.

---

## 3. Locked Technical Decisions

These are **not up for discussion** during implementation. Claude Code should treat them as fixed.

| Layer | Choice | Why |
|---|---|---|
| Vector DB | **Qdrant** (int8 scalar quantization) | Single binary, fast filters, ~4× storage savings |
| Embeddings | **bge-m3** via Ollama (1024-dim) | Multilingual, 8k context, hybrid-friendly, free |
| Reranker | **bge-reranker-v2-m3** via **Infinity** HTTP server (GPU) | Strong rerank quality, runs locally on 5070 |
| Chat LLM (primary) | **Qwen 2.5 7B Instruct Q4_K_M** via Ollama | Best general tool-calling 7B model, fits VRAM |
| Chat LLM (synthesis) | **DeepSeek-R1-Distill-Qwen-7B Q4_K_M** via Ollama | Better for #3 cross-corpus synthesis |
| BM25 | **Tantivy** (Rust, Python bindings) | Fast, durable, single-file index, no JVM |
| App DB | **Postgres 16** | Open WebUI state + precomputed analytics |
| Cache | **Redis 7-alpine** | Open WebUI sessions |
| UI | **Open WebUI** (latest) | Native function-calling, local-first |
| Retrieval logic | **Custom Open WebUI Function tool** | Bypasses built-in KB pipeline |
| Orchestration | **Docker Compose** | One-command bring-up |
| Ingestion language | **Python 3.11+** | Ecosystem fit |

---

## 4. Architecture

```
┌─────────────────────────── RTX 5070 host (Ubuntu 22.04+) ──────────────────────────┐
│                                                                                     │
│   Open WebUI ◀──── browser                                                          │
│        │                                                                            │
│        │ (function tool calls: search_transcripts, find_quote, count_mentions, …)   │
│        ▼                                                                            │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│   │  Ollama  │   │  Qdrant  │   │ Tantivy  │   │ Infinity │   │ Postgres │         │
│   │ embed +  │   │ vectors  │   │ BM25 idx │   │ reranker │   │ analytics│         │
│   │ chat LLM │   │ + meta   │   │ (sidecar)│   │  (HTTP)  │   │ + app DB │         │
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘         │
│                                                                                     │
│   Redis (cache)                                                                     │
│                                                                                     │
│   Host-side Python workers:                                                         │
│     - chunker_text.py / chunker_json.py                                             │
│     - bulk_ingest_hardened.py   ← writes to Qdrant + Tantivy + Postgres             │
│     - tantivy_server.py         ← exposes BM25 over HTTP on :8765                   │
│     - verify_ingestion.py       ← post-ingest sampling validation                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### VRAM budget (12 GB)

| Component | VRAM | Loaded when |
|---|---|---|
| bge-m3 embedding | ~1.2 GB | Always |
| bge-reranker-v2-m3 (Infinity) | ~1.2 GB | Always (pinned) |
| Qwen 2.5 7B Q4_K_M | ~5.0 GB | When chatting (auto-unloaded after 10 min) |
| KV cache @ 16k context | ~2.5 GB | During chat |
| **Total during query** | **~10 GB** | Within budget |
| **Total during ingestion-only** | **~2.4 GB** | Chat model unloaded |

---

## 5. Repository Structure

Claude Code must create this exact structure:

```
transcript-rag/
├── PRD.md                              ← this file
├── CLAUDE.md                           ← Claude Code context (Section 17)
├── README.md                           ← human-facing quickstart
├── .env.example                        ← template; actual .env gitignored
├── .gitignore
├── docker-compose.yml
├── docker-compose.override.yml         ← optional local tweaks (gitignored)
│
├── infra/
│   ├── postgres/
│   │   └── analytics_schema.sql
│   └── qdrant/
│       └── qdrant_setup.py
│
├── ingestion/
│   ├── __init__.py
│   ├── chunker_json.py                 ← for whisperX/whisper JSON
│   ├── chunker_text.py                 ← for plain .txt
│   ├── bulk_ingest_hardened.py         ← the main ingestion script
│   ├── verify_ingestion.py             ← sampling validation
│   ├── retry_dead_letter.py            ← targeted re-run of failed files
│   └── utils/
│       ├── __init__.py
│       ├── retries.py                  ← exponential backoff decorator
│       ├── encoding.py                 ← utf-8/latin-1 fallback
│       ├── progress_db.py              ← SQLite progress tracker
│       └── health.py                   ← startup health checks
│
├── services/
│   └── tantivy_server/
│       ├── tantivy_server.py
│       └── requirements.txt
│
├── open_webui_functions/
│   ├── search_transcripts.py           ← main retrieval tool
│   └── analytics.py                    ← Postgres analytics tool
│
├── scripts/
│   ├── 00_health_check.sh
│   ├── 01_pull_models.sh
│   ├── 02_init_qdrant.sh
│   ├── 03_init_postgres.sh
│   ├── 04_run_chunker.sh
│   ├── 05_run_ingestion.sh
│   ├── 06_verify.sh
│   └── 99_nuke_everything.sh           ← danger: full reset for dev
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker_text.py
│   │   ├── test_chunker_json.py
│   │   ├── test_retries.py
│   │   ├── test_encoding.py
│   │   └── test_progress_db.py
│   ├── integration/
│   │   ├── test_qdrant_flow.py
│   │   ├── test_tantivy_flow.py
│   │   └── test_end_to_end_small.py    ← runs against a 100MB sample
│   └── fixtures/
│       ├── sample_whisperx.json
│       ├── sample_plain.txt
│       └── corrupted_files/            ← intentionally broken samples
│
├── eval/
│   ├── golden_queries.yaml             ← 30 hand-curated Q/A pairs
│   └── run_eval.py                     ← measures hit@k, MRR
│
├── docs/
│   ├── runbook.md
│   ├── troubleshooting.md
│   └── architecture.md
│
└── pyproject.toml                      ← Python deps via uv or pip
```

---

## 6. Implementation Phases

Each phase has **Objective**, **Deliverables**, **Implementation notes**, **Acceptance criteria**, and **Commit message**. Claude Code completes one phase fully before starting the next.

---

### Phase 0 — Project scaffolding

**Objective:** Lay out the empty repo structure exactly as Section 5, plus baseline config files.

**Deliverables:**
- All directories from Section 5 created (with `.gitkeep` where empty).
- `.gitignore` covering `data/`, `.env`, `__pycache__/`, `*.pyc`, `.venv/`, `ingest.log`, `ingest_progress.sqlite`, `dead_letter/`, `node_modules/`.
- `.env.example` with all variables listed (Section 9), no real secrets.
- `pyproject.toml` with dependencies pinned to known-good major versions:
  - `qdrant-client>=1.9`, `tantivy>=0.21`, `requests>=2.31`, `fastapi>=0.110`, `uvicorn[standard]>=0.27`, `psycopg2-binary>=2.9`, `pydantic>=2.6`, `tenacity>=8.2`, `psutil>=5.9`, `pytest>=8.0`, `pytest-asyncio>=0.23`.
- `README.md` skeleton with a "Quickstart" section pointing to `scripts/`.

**Acceptance criteria:**
- `tree -L 2 -a` matches Section 5.
- `python -c "import qdrant_client, tantivy, requests, fastapi, psycopg2"` succeeds in a fresh venv after `pip install -e .`.

**Commit:** `chore: project scaffolding (Phase 0)`

---

### Phase 1 — Docker Compose infrastructure

**Objective:** A working `docker compose up -d` brings up Ollama, Qdrant, Infinity reranker, Postgres, Redis, and Open WebUI with healthy containers.

**Deliverables:**

`docker-compose.yml` — use this exact spec (do not deviate unless documented):

```yaml
version: '3.9'

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    restart: unless-stopped
    depends_on: [ollama, qdrant, postgres, redis, reranker]
    ports: ["8080:8080"]
    volumes:
      - ./data/open-webui:/app/backend/data
    environment:
      WEBUI_SECRET_KEY: "${WEBUI_SECRET_KEY}"
      ENV: prod
      DATABASE_URL: "postgresql://owui:${POSTGRES_PASSWORD}@postgres:5432/openwebui"
      REDIS_URL: "redis://redis:6379/0"
      ENABLE_BASE_MODELS_CACHE: "True"
      ENABLE_REALTIME_CHAT_SAVE: "False"
      OLLAMA_BASE_URL: "http://ollama:11434"
      VECTOR_DB: "qdrant"
      QDRANT_URI: "http://qdrant:6333"
      QDRANT_API_KEY: "${QDRANT_API_KEY}"
      RAG_EMBEDDING_ENGINE: "ollama"
      RAG_EMBEDDING_MODEL: "bge-m3"
      RAG_OLLAMA_BASE_URL: "http://ollama:11434"
      RAG_EMBEDDING_BATCH_SIZE: "32"
      ENABLE_RAG_HYBRID_SEARCH: "true"
      RAG_RERANKING_ENGINE: "external"
      RAG_EXTERNAL_RERANKER_URL: "http://reranker:7997/rerank"
      RAG_RERANKING_MODEL: "BAAI/bge-reranker-v2-m3"
      RAG_TOP_K: "40"
      RAG_TOP_K_RERANKER: "8"
      RAG_HYBRID_BM25_WEIGHT: "0.65"
      RAG_TEXT_SPLITTER: "token"
      CHUNK_SIZE: "600"
      CHUNK_OVERLAP: "150"
      CHUNK_MIN_SIZE_TARGET: "350"
      RAG_SYSTEM_CONTEXT: "True"
      RAG_FILE_MAX_SIZE: "1000"
      RAG_FILE_MAX_COUNT: "100"

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    ports: ["11434:11434"]
    volumes: [./data/ollama:/root/.ollama]
    environment:
      OLLAMA_KEEP_ALIVE: "10m"
      OLLAMA_NUM_PARALLEL: "2"
      OLLAMA_MAX_LOADED_MODELS: "2"
      OLLAMA_FLASH_ATTENTION: "1"
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]

  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    restart: unless-stopped
    ports: ["6333:6333", "6334:6334"]
    volumes: [./data/qdrant:/qdrant/storage]
    environment:
      QDRANT__SERVICE__API_KEY: "${QDRANT_API_KEY}"
      QDRANT__STORAGE__OPTIMIZERS__DEFAULT_SEGMENT_NUMBER: "4"

  reranker:
    image: michaelf34/infinity:latest
    container_name: reranker
    restart: unless-stopped
    ports: ["7997:7997"]
    command: >
      v2
      --model-id BAAI/bge-reranker-v2-m3
      --port 7997
      --engine torch
      --batch-size 32
    volumes: [./data/infinity-cache:/app/.cache]
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]

  postgres:
    image: postgres:16
    container_name: postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: owui
      POSTGRES_PASSWORD: "${POSTGRES_PASSWORD}"
      POSTGRES_DB: openwebui
    volumes: [./data/postgres:/var/lib/postgresql/data]
    ports: ["5432:5432"]

  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    command: redis-server --maxclients 10000 --timeout 1800
    volumes: [./data/redis:/data]
```

`scripts/00_health_check.sh` — pings every service endpoint and reports status:
- `curl http://localhost:8080/health` (Open WebUI)
- `curl http://localhost:11434/api/tags` (Ollama)
- `curl -H "api-key: $QDRANT_API_KEY" http://localhost:6333/collections` (Qdrant)
- `curl http://localhost:7997/health` (Infinity)
- `pg_isready -h localhost -p 5432 -U owui` (Postgres)
- `redis-cli -h localhost ping` (Redis)

**Implementation notes:**
- The host must have NVIDIA Container Toolkit installed (`nvidia-ctk`). Document this in `README.md`.
- Allow `Q DRANT_API_KEY` to be empty in dev; require it in prod.
- The `data/` directory tree is created automatically by Docker on first run.

**Acceptance criteria:**
- `docker compose up -d` exits successfully.
- `docker compose ps` shows all 6 services as `running (healthy)` (or `running` if no healthcheck) after 60 seconds.
- `bash scripts/00_health_check.sh` returns ✅ for every service.
- `nvidia-smi` shows both `ollama` and `reranker` processes (after a model is loaded).

**Commit:** `feat: docker compose infra with Ollama, Qdrant, Infinity, Postgres, Redis, Open WebUI (Phase 1)`

---

### Phase 2 — Model pulls and Qdrant collection setup

**Objective:** All required models downloaded; Qdrant collection created with the correct config.

**Deliverables:**

`scripts/01_pull_models.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
docker exec ollama ollama pull bge-m3
docker exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M
docker exec ollama ollama pull deepseek-r1:7b-qwen-distill-q4_K_M
echo "✅ All models pulled. Ollama list:"
docker exec ollama ollama list
```

`infra/qdrant/qdrant_setup.py` — creates the `transcripts` collection with int8 quantization, HNSW tuning, payload indexes:

```python
"""Create the Qdrant 'transcripts' collection. Idempotent — safe to re-run."""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, ScalarQuantization, ScalarQuantizationConfig,
    ScalarType, HnswConfigDiff, PayloadSchemaType,
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "transcripts")

def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        print(f"Collection {COLLECTION!r} already exists — skipping creation.")
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8, always_ram=True,
                )
            ),
            hnsw_config=HnswConfigDiff(m=32, ef_construct=256),
            on_disk_payload=True,
        )
        print(f"✅ Created collection {COLLECTION!r}")

    # Idempotent payload indexes
    for field, schema in [
        ("source_file", PayloadSchemaType.KEYWORD),
        ("speakers",    PayloadSchemaType.KEYWORD),
        ("start_sec",   PayloadSchemaType.FLOAT),
        ("date",        PayloadSchemaType.DATETIME),
    ]:
        try:
            client.create_payload_index(COLLECTION, field, schema)
            print(f"✅ Payload index: {field}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"⏭️  Payload index {field} already exists")
            else:
                raise

if __name__ == "__main__":
    main()
```

`scripts/02_init_qdrant.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
export $(grep -v '^#' .env | xargs)
python infra/qdrant/qdrant_setup.py
```

**Acceptance criteria:**
- `bash scripts/01_pull_models.sh` lists all three models in `ollama list`.
- `bash scripts/02_init_qdrant.sh` exits 0.
- Re-running both scripts is idempotent (no errors on second run).
- `curl -H "api-key: $QDRANT_API_KEY" http://localhost:6333/collections/transcripts` returns collection info with `vector_size: 1024` and quantization config present.

**Commit:** `feat: pull models, init Qdrant collection with int8 + payload indexes (Phase 2)`

---

### Phase 3 — Chunking pipeline

**Objective:** Two chunkers (JSON path and plain-text path) producing `.chunks.json` files ready for ingestion.

**Deliverables:**

`ingestion/chunker_json.py` — implement per Section 5a of the original spec. Key requirements:
- Supports two input formats: `whisperx` (with diarization) and `whisper` (without).
- `TARGET_TOKENS=450`, `MAX_TOKENS=700`, `MIN_TOKENS=200`.
- Header prepended to each chunk: `[Source: <file> | HH:MM:SS → HH:MM:SS | Speakers: <list>]`.
- Output: one `<stem>.chunks.json` per input.
- CLI: `python -m ingestion.chunker_json INPUT_DIR OUTPUT_DIR --format whisperx`

`ingestion/chunker_text.py` — implement per Section 5b. Key requirements:
- Sentence splitting via regex `(?<=[.!?।॥])\s+` (Latin `.!?` + Devanagari danda `।`/`॥`, so Hindi prose splits).
- `TARGET_TOKENS=450`, `MAX_TOKENS=700`, `OVERLAP_SENTENCES=2`.
- Header: `[Source: <file> | Approx position: sentences X-Y | NOTE: plain-text source, timestamps unavailable]`.
- Metadata includes `has_timestamps: false`.

**Both chunkers must:**
- Accept input from stdin if no `INPUT_DIR` given (for streaming use).
- Wrap each file's processing in `try/except` and log failures to `<OUTPUT_DIR>/_failed/<file>.error.txt`.
- Skip files >500 MB with a warning (configurable cap).
- Print a summary at the end: total files, total chunks, failures.

**Tests** (`tests/unit/test_chunker_*.py`):
- Empty input → 0 chunks, no crash.
- Single short utterance → 1 chunk.
- Single very long monologue → multiple chunks with overlap.
- Mixed-language Unicode text → preserved exactly.
- Speaker change mid-utterance → chunk boundary respected.
- Malformed JSON → caught, file marked failed, pipeline continues.

**Acceptance criteria:**
- All unit tests pass: `pytest tests/unit/test_chunker_*.py -v`.
- Running chunker on `tests/fixtures/sample_whisperx.json` produces a valid `.chunks.json` with non-zero chunks.
- Running chunker on `tests/fixtures/sample_plain.txt` produces a valid `.chunks.json` with `has_timestamps: false` in metadata.

**Commit:** `feat: JSON + plain-text chunkers with full test coverage (Phase 3)`

---

### Phase 4 — Hardened bulk ingestion

**Objective:** Production-quality ingestion that survives 8–14 days of continuous operation over 5 TB.

**Deliverables:**

`ingestion/utils/retries.py` — `@retry_with_backoff(max_tries=5, base=1.0, max_delay=60)` decorator using `tenacity`. Retries on `requests.RequestException`, `qdrant_client.exceptions.UnexpectedResponse`, `ConnectionError`, `TimeoutError`. Does NOT retry on `ValueError` or `KeyError` (programming bugs).

`ingestion/utils/encoding.py` — `read_text_robust(path: Path) -> str` that tries `utf-8`, `utf-8-sig`, `latin-1` in order. Returns first that decodes without `UnicodeDecodeError`. Raises if all fail.

`ingestion/utils/progress_db.py` — SQLite wrapper with this schema:
```sql
CREATE TABLE IF NOT EXISTS ingest_status (
    file        TEXT PRIMARY KEY,
    status      TEXT NOT NULL,            -- 'ok' | 'failed' | 'skipped' | 'in_progress'
    n_chunks    INTEGER DEFAULT 0,
    reason      TEXT,                     -- failure reason if status='failed'
    attempts    INTEGER DEFAULT 0,
    last_ts     REAL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_status ON ingest_status(status);
```
Functions: `get_status(file)`, `mark_ok(file, n)`, `mark_failed(file, reason)`, `mark_in_progress(file)`, `list_failed()`, `stats()`.

`ingestion/utils/health.py` — `check_all() -> dict` that pings Ollama, Qdrant, Tantivy server. Returns `{service: bool}`. Used at ingestion startup with fail-fast.

`ingestion/bulk_ingest_hardened.py` — the main script. Must implement all of:

1. **Health check at startup.** If any service is down, exit with code 2 and a clear message.
2. **SQLite progress DB** at `./ingest_progress.sqlite`. Skip files marked `ok`. Retry files marked `failed` only if `--retry-failed` flag passed.
3. **Per-file try/except.** A failure on one file never kills the pipeline.
4. **Per-file hard timeout** of 30 minutes (via `signal.SIGALRM` on Linux). Timed-out files marked `failed` with reason `timeout`.
5. **File-size cap of 500 MB** per `.chunks.json`. Oversize → quarantined to `dead_letter/oversize/`, marked `failed`.
6. **Encoding recovery** via `read_text_robust`.
7. **Retry decorator** on `embed()`, `qdrant.upsert()`, every HTTP call.
8. **NaN/Inf embedding detection.** Before upserting, check `not any(math.isnan(x) or math.isinf(x) for x in vec)`. Bad → mark failed, skip.
9. **Periodic Tantivy commits** every 50 files (the writer is destructive on crash; commit often).
10. **Graceful SIGTERM / SIGINT handling.** On signal: commit pending Tantivy work, close DB, log "interrupted at file X", exit 130.
11. **Memory monitoring.** Every 10 files, log `psutil.virtual_memory().percent`. Warn at 85%, abort gracefully at 95%.
12. **Dead-letter quarantine.** Failed files copied to `dead_letter/<reason>/<filename>` for manual review.
13. **Stable chunk IDs** via UUIDv5 (namespace `6ba7b810-9dad-11d1-80b4-00c04fd430c8`, key `source|index|sha256(text)[:16]`).
14. **Postgres `chunk_meta` insert** alongside Qdrant upsert (batch INSERT … ON CONFLICT DO NOTHING).
15. **Structured logging** to `ingest.log` (rotating, 100 MB max, 10 backups) AND stdout. Format: `ISO8601 | LEVEL | file | event | detail`.
16. **Progress line every N files** (default 50): `[12340/2000000] file.json: +47 chunks (total 4.2M, 38.5/s, RAM 47%)`.

CLI:
```
python -m ingestion.bulk_ingest_hardened \
    --chunks-dir /data/processed \
    --batch-size 32 \
    [--retry-failed] \
    [--dry-run] \
    [--max-files N]
```

`ingestion/retry_dead_letter.py` — separate script to re-run files from `dead_letter/` after manual fixes. Resets their status to `pending` and invokes the main ingester.

`ingestion/verify_ingestion.py` — post-ingest sampling validation:
- Sample 1,000 random chunks from source `.chunks.json` files.
- For each, query Qdrant by `chunk_id` and confirm vector + payload exist.
- For each, query Tantivy by `chunk_id` and confirm document exists.
- Report: `Sampled N chunks. Qdrant: X% present. Tantivy: Y% present.`
- Exit code 0 if both ≥ 99.9%, else exit 1.

**Acceptance criteria:**
- Unit tests pass for `retries`, `encoding`, `progress_db`.
- Integration test `tests/integration/test_end_to_end_small.py` runs the full pipeline on `tests/fixtures/` (≤ 10 files) end-to-end, including one intentionally corrupted file in `tests/fixtures/corrupted_files/`. Asserts:
  - Healthy files all marked `ok`.
  - Corrupted file marked `failed` with non-null `reason`.
  - Corrupted file present in `dead_letter/`.
  - Qdrant has expected number of points.
  - Tantivy has expected number of documents.
- Running the integration test twice in a row: second run is a no-op (skips all `ok` files).
- `verify_ingestion.py` passes on the test data.

**Commit:** `feat: hardened bulk ingestion with retries, dead-letter, NaN detection, signal handling (Phase 4)`

---

### Phase 5 — Tantivy BM25 sidecar

**Objective:** Standalone FastAPI service exposing Tantivy search over HTTP on port 8765.

**Deliverables:**

`services/tantivy_server/tantivy_server.py`:
```python
import os
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
from contextlib import asynccontextmanager
import tantivy

TANTIVY_DIR = Path(os.environ.get("TANTIVY_DIR", "./data/tantivy"))

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not TANTIVY_DIR.exists():
        raise RuntimeError(f"Tantivy index dir not found: {TANTIVY_DIR}")
    state["index"] = tantivy.Index.open(str(TANTIVY_DIR))
    state["index"].reload()
    state["searcher"] = state["index"].searcher()
    yield
    state.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"ok": True, "docs": state["searcher"].num_docs}

@app.get("/search")
def search(q: str = Query(...), k: int = Query(40, ge=1, le=200)):
    try:
        idx = state["index"]
        parser = idx.parse_query(q, ["text"])
        hits = state["searcher"].search(parser, limit=k).hits
        out = []
        for score, doc_addr in hits:
            doc = state["searcher"].doc(doc_addr)
            out.append({
                "chunk_id": doc["chunk_id"][0],
                "text": doc["text"][0],
                "source_file": doc["source_file"][0],
                "score": float(score),
            })
        return {"hits": out, "count": len(out)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
def reload_idx():
    state["index"].reload()
    state["searcher"] = state["index"].searcher()
    return {"ok": True, "docs": state["searcher"].num_docs}
```

`services/tantivy_server/requirements.txt`:
```
fastapi>=0.110
uvicorn[standard]>=0.27
tantivy>=0.21
```

`scripts/run_tantivy_server.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec uvicorn services.tantivy_server.tantivy_server:app \
    --host 0.0.0.0 --port 8765 \
    --workers 1 --log-level info
```

Add a `systemd` unit template in `docs/runbook.md` for production hosting.

**Acceptance criteria:**
- After ingestion of fixtures, `curl 'http://localhost:8765/health'` returns `{"ok": true, "docs": N}` with N > 0.
- `curl 'http://localhost:8765/search?q=test&k=5'` returns hits.
- `curl -X POST 'http://localhost:8765/reload'` succeeds.

**Commit:** `feat: Tantivy BM25 HTTP sidecar with /search, /health, /reload (Phase 5)`

---

### Phase 6 — Postgres analytics schema and population

**Objective:** `chunk_meta` and `file_meta` tables populated during ingestion; ready for analytics queries.

**Deliverables:**

`infra/postgres/analytics_schema.sql` — exactly per Section 9 of original spec, plus:
```sql
CREATE TABLE IF NOT EXISTS chunk_meta (
    chunk_id     UUID PRIMARY KEY,
    source_file  TEXT NOT NULL,
    speakers     TEXT[] NOT NULL DEFAULT '{}',
    start_sec    FLOAT,
    end_sec      FLOAT,
    word_count   INTEGER,
    text         TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_speakers ON chunk_meta USING GIN (speakers);
CREATE INDEX IF NOT EXISTS idx_chunk_source ON chunk_meta(source_file);
CREATE INDEX IF NOT EXISTS idx_chunk_text_fts ON chunk_meta USING GIN (to_tsvector('english', text));

CREATE TABLE IF NOT EXISTS file_meta (
    source_file   TEXT PRIMARY KEY,
    duration_sec  FLOAT,
    speakers      TEXT[],
    chunk_count   INTEGER,
    ingested_at   TIMESTAMP DEFAULT NOW()
);
```

`scripts/03_init_postgres.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
export $(grep -v '^#' .env | xargs)
psql "postgresql://owui:${POSTGRES_PASSWORD}@localhost:5432/openwebui" \
    -f infra/postgres/analytics_schema.sql
```

Update `bulk_ingest_hardened.py` to write to `chunk_meta` (batch INSERT) and update `file_meta` at the end of each file.

**Acceptance criteria:**
- `scripts/03_init_postgres.sh` is idempotent.
- After running the integration test, `SELECT COUNT(*) FROM chunk_meta` matches Qdrant point count.
- `SELECT * FROM file_meta WHERE source_file = 'sample_whisperx.json'` returns a row with non-null `chunk_count`.

**Commit:** `feat: Postgres analytics schema and ingestion writes (Phase 6)`

---

### Phase 7 — Open WebUI function tools

**Objective:** Two function tools (search, analytics) installable in Open WebUI, callable by chat models.

**Deliverables:**

`open_webui_functions/search_transcripts.py` — implement per Section 7 of the original spec. Two public methods:
- `search_transcripts(query, speaker=None, source_file=None, top_k=8)` — hybrid retrieval pipeline.
- `find_quote(partial_quote, top_k=5)` — same flow with `bm25_weight=0.85`.

Internal helpers: `_embed`, `_dense`, `_bm25`, `_rrf` (weighted reciprocal rank fusion), `_rerank`.

Valves (configurable in UI):
```
qdrant_url:           http://qdrant:6333
qdrant_key:           (from env)
qdrant_collection:    transcripts
ollama_url:           http://ollama:11434
embed_model:          bge-m3
reranker_url:         http://reranker:7997/rerank
reranker_model:       BAAI/bge-reranker-v2-m3
tantivy_proxy_url:    http://host.docker.internal:8765
candidates_per_source: 40
final_top_k:          8
bm25_weight:          0.65
```

Output format must include for each result:
- `--- Result N (score: X.XXX) ---`
- `Source: <file> | <start>s → <end>s | Speakers: A, B`
- chunk text body
- blank line separator

`open_webui_functions/analytics.py` — per Section 9 of original spec. Methods:
- `count_mentions(term, speaker=None)` — full-text search count.
- `top_speakers_for_topic(term, limit=10)` — speaker aggregation.
- `list_transcripts_mentioning(term, limit=20)` — list source files.

Add docstring examples that show the LLM when to use each tool.

`docs/install_functions.md` — step-by-step instructions for the human:
1. Open WebUI → Admin Panel → Functions → "+ New Function"
2. Paste the file contents into the editor
3. Save and toggle on
4. Repeat for analytics tool
5. Configure valves (set `qdrant_key` from `.env`)

**Acceptance criteria:**
- Both function files pass `python -c "import ast; ast.parse(open(F).read())"` (syntactically valid).
- Manual test: install in Open WebUI, attach to a Qwen model, send "search for X" — model invokes tool, results render with correct formatting.
- `find_quote` with a known-existing phrase returns it in top 3 results.

**Commit:** `feat: Open WebUI function tools for search + analytics (Phase 7)`

---

### Phase 8 — Model configuration in Open WebUI

**Objective:** Two configured models (Qwen primary, DeepSeek synthesis) with system prompt + tools attached.

**Deliverables:**

`docs/model_config.md` documenting the exact UI steps:

For each model (Workspace → Models → Edit):

| Setting | Qwen 2.5 7B | DeepSeek-R1 7B |
|---|---|---|
| Base model | `qwen2.5:7b-instruct-q4_K_M` | `deepseek-r1:7b-qwen-distill-q4_K_M` |
| Context Length | 16384 | 16384 |
| Function Calling | native | native |
| Builtin Tools | none | none |
| Attached Tools | search_transcripts, analytics | search_transcripts, analytics |

System prompt (both):
```
You are a transcript research assistant. You have these tools available:

1. search_transcripts — for finding content, topics, and quotes by meaning
2. find_quote — when the user is hunting for a specific phrase or exact words
3. count_mentions — for "how many times" / "how often" queries
4. top_speakers_for_topic — for "who talks most about X" queries
5. list_transcripts_mentioning — for "which transcripts mention X" queries

Rules:
- ALWAYS call a tool before answering questions about transcript content.
- For quote-finding queries ("who said X", "find when X was said"), use find_quote.
- For "how often" / "how many" / "which speaker most" questions, use the analytics tools.
- For everything else, use search_transcripts.
- Cite source file and timestamp for every claim. If timestamps are unavailable
  (plain-text source), say so explicitly.
- Never invent quotes. If the retrieved chunks don't contain the answer, say so plainly.
- Do not speculate beyond what the retrieved chunks support.
```

**Acceptance criteria:**
- Both models appear in the Open WebUI model dropdown.
- Sending "find the moment someone said 'we have to focus'" to the Qwen model invokes `find_quote`.
- Sending "how many times is X mentioned" invokes `count_mentions`.
- Responses include source + timestamp citations.

**Commit:** `docs: model configuration runbook (Phase 8)`

---

### Phase 9 — Eval harness

**Objective:** A reproducible eval that measures retrieval quality before any production deployment.

**Deliverables:**

`eval/golden_queries.yaml` — 30 hand-curated query/answer pairs across all 4 query types:
- 12 quote-finding queries (priority #1)
- 8 single-transcript Q&A
- 5 cross-corpus topic
- 5 analytics

Schema:
```yaml
- id: q001
  type: quote
  query: "find when someone said 'we have to focus'"
  expected_source_file: "meeting_2024_03_15.json"
  expected_chunk_contains: "we have to focus"
  expected_speaker: "Alice"
- id: q012
  type: analytics
  query: "how many times is 'roadmap' mentioned"
  expected_count_min: 40
  expected_count_max: 80
```

`eval/run_eval.py` — runs every query through the retrieval pipeline (bypassing the LLM, directly calling the search tool's internal methods) and reports:
- Hit@1, Hit@5, Hit@10 per query type
- Mean Reciprocal Rank (MRR)
- Per-query pass/fail with reason
- JSON report saved to `eval/results/<timestamp>.json`

The eval should be runnable as `python -m eval.run_eval --queries eval/golden_queries.yaml`.

**Acceptance criteria:**
- Eval runs without error.
- On the fixture-only test set, Hit@5 ≥ 80% for quote-finding type.
- Report file is well-formed JSON with per-query results.

**Commit:** `feat: eval harness with golden queries + hit@k metrics (Phase 9)`

---

### Phase 10 — Pre-flight validation on 100 GB slice

**Objective:** Validate the entire pipeline on a 2% slice before committing to the full 14-day run.

**Deliverables:**

`scripts/preflight.sh`:
```bash
#!/usr/bin/env bash
# Selects a random 100GB slice of raw transcripts, runs the full pipeline,
# reports timing + failure stats.
set -euo pipefail

SLICE_DIR="${SLICE_DIR:-/data/preflight_slice}"
RAW_DIR="${RAW_DIR:-/data/raw-transcripts}"
PROCESSED_DIR="${PROCESSED_DIR:-/data/processed_preflight}"

# 1. Random 100GB sample (assumes du-sortable file listing)
python scripts/select_random_slice.py \
    --src "$RAW_DIR" --dst "$SLICE_DIR" --target-gb 100

# 2. Chunk
python -m ingestion.chunker_text "$SLICE_DIR" "$PROCESSED_DIR"

# 3. Ingest
python -m ingestion.bulk_ingest_hardened \
    --chunks-dir "$PROCESSED_DIR" --batch-size 32

# 4. Verify
python -m ingestion.verify_ingestion

# 5. Eval
python -m eval.run_eval --queries eval/golden_queries.yaml
```

`scripts/select_random_slice.py` — random sampling utility (uses `os.walk` + reservoir sampling on file sizes).

**Acceptance criteria:**
- Preflight completes within 48 hours on a 100GB slice.
- Failure rate ≤ 5% (i.e., ≥ 95% of files in `status='ok'`).
- All failed files have actionable reasons in `dead_letter/`.
- Verify step passes with ≥99.9% present in both Qdrant and Tantivy.
- Eval Hit@5 ≥ 80% for quote-finding.

**Commit:** `feat: preflight script for 100GB slice validation (Phase 10)`

---

### Phase 11 — Documentation and runbook

**Objective:** A human can take over operating this system after Claude Code is gone.

**Deliverables:**

`docs/runbook.md` covering:
- Daily ops: log review, dead-letter triage, disk monitoring
- How to resume a halted ingestion
- How to reload Tantivy after additional ingestion
- How to back up Qdrant (snapshot API), Tantivy (rsync the dir), Postgres (`pg_dump`)
- How to rotate logs
- How to update models (Ollama `pull`, restart Ollama container)
- How to add a new function tool to Open WebUI

`docs/troubleshooting.md` covering common failure modes:
- "Ollama OOM during embedding" → reduce batch size, set `OLLAMA_NUM_PARALLEL=1`
- "Qdrant returns 503" → check disk space, check segment merges in logs
- "Tantivy lock error" → only one writer allowed; kill stale process
- "Reranker timeout" → check Infinity GPU memory; restart container
- "Open WebUI doesn't see function tool" → confirm toggled on, re-attach to model

`docs/architecture.md` — the diagram from Section 4 plus a paragraph per component explaining its role and failure modes.

`README.md` — final version with Quickstart, links to phases, hardware checklist.

**Acceptance criteria:**
- A new developer can clone the repo, follow `README.md`, and reach "first successful query" within 2 hours (assuming pre-pulled models).

**Commit:** `docs: runbook, troubleshooting, architecture (Phase 11)`

---

### Phase 12 — Path-based metadata extraction

**Objective:** Extract structured metadata from the source-audio folder
hierarchy (collection, year, event, location, session date/time, track type)
and propagate it through chunks → Qdrant payloads → Postgres rows → retrieval
filters. Enables queries like *"what did Swami ji say on a monsoon day about
barsat?"* and *"show me discourses from the Noida camp"* — temporal,
locational, and track-type filtering on top of the existing hybrid retrieval.

**Trigger:** post-Phase-11 user request after corpus inspection revealed that
every audio file's full path encodes rich metadata (e.g.
`Live Masters 2010\01 NOIDA 7 - 10 JAN 2010\7 JAN - 1$ - 6 PM\04 PRAVACHAN`).
A path parser captures all of it with zero ML overhead.

**Locked design decisions:**

| Decision | Choice |
|---|---|
| Season mapping | IMD-standard 4-season — winter (Jan–Feb), summer (Mar–May), monsoon (Jun–Sep), post-monsoon (Oct–Dec) |
| Primary speaker | Constant `"Swami ji"` (renameable in `path_parser.py`); single voice across the corpus |
| Default track-type filter on retrieval | `{discourse, address}` — i.e. PRAVACHAN + SAMBODHAN — for teaching queries; bhajan/music/meditation/invocation searchable but excluded unless the LLM passes them |
| Parse failures | Logged to `path_parse_failures.log`; ingestion proceeds with whatever partial fields parsed (degrade > drop) |
| Track-type vocab | `PRAVACHAN→discourse`, `SAMBODHAN→address`, `MEDITATION→meditation`, `OM GURUVE NAMAH→invocation`, `ENTRY MUSIC`/`RETURN MUSIC→music`, default `→bhajan` |

**Deliverables:**

`ingestion/utils/path_parser.py` — pure function module:
- `parse_path(path: Path, base_dir: Path | None = None) -> PathMetadata`
- `PathMetadata` dataclass with fields: `collection`, `year`, `event_seq`,
  `event_id`, `location`, `event_start`, `event_end`, `session_date`,
  `session_seq`, `session_time`, `track_no`, `track_title`, `track_type`,
  `season`, `primary_speaker`, `parse_warnings: list[str]`
- `season_for(date: date) -> str` helper
- `track_type_for(title: str) -> str` lookup with `bhajan` default
- `PRIMARY_SPEAKER: str = "Swami ji"` module constant
- Graceful per-level parsing: a malformed event folder still yields a
  valid `PathMetadata` with `event_id=None` and a warning appended

`tests/unit/test_path_parser.py` — covers:
- The exact example path from the trigger
- Each track-type vocab token + bhajan default
- Each season boundary month
- Malformed inputs at every level (missing year, missing date, etc.)
- Idempotent re-parsing of the same path

`ingestion/chunker_text.py` + `chunker_json.py`:
- Accept optional `--base-dir` CLI arg (or env `RAW_TRANSCRIPTS_BASE_DIR`)
- For each input file, call `parse_path()` and embed the parsed fields
  into every chunk's metadata + into the chunk header line (so the
  embedding model itself sees `[Event: NOIDA 7 JAN 2010 | Track: PRAVACHAN]`)
- Backwards-compatible: when `base_dir` is absent, parser yields all-None
  metadata and the chunkers behave exactly as before

`ingestion/bulk_ingest_hardened.py`:
- Pass parsed metadata through `_coerce_payload()` into Qdrant payload
- Insert into Postgres `chunk_meta` and `file_meta` with the new columns

`infra/qdrant/qdrant_setup.py`:
- Add idempotent payload indexes: `session_date` (DATETIME), `track_type`
  (KEYWORD), `location` (KEYWORD), `event_id` (KEYWORD), `season` (KEYWORD)

`infra/postgres/analytics_schema.sql`:
- `ALTER TABLE chunk_meta ADD COLUMN IF NOT EXISTS session_date DATE`,
  `track_type TEXT`, `track_title TEXT`, `location TEXT`, `event_id TEXT`,
  `season TEXT` + GIN/BTREE indexes (idempotent)
- Same for `file_meta` where appropriate

`open_webui_functions/search_transcripts.py`:
- Add filter args to `search_transcripts()`:
  `date_range: tuple[str, str] | None`, `season: str | None`,
  `track_type: list[str] | str | None`, `location: str | None`,
  `event_id: str | None`
- Update docstring with examples for each new arg so the LLM learns
  when to use them
- Default behavior unchanged when no new args passed

`docs/best_practices.md` (new) — companion doc covering:
- Chunking principles for retrieval accuracy
- Embedding-side practices (header enrichment, multilingual care)
- Query-side practices (HyDE, multi-query, filter extraction)
- ASR-side practices (Whisper prompt-bias for proper names → addresses
  the unresolved "Anush" problem from the trigger)
- QC + eval discipline (golden-query expansion, spot-check sampling)

**Acceptance criteria:**
- `pytest tests/unit/test_path_parser.py -v` green.
- Full existing unit suite still green (no regressions).
- `parse_path()` on the example path returns:
  `track_type='discourse'`, `season='winter'`, `session_date=2010-01-07`,
  `location='NOIDA'`, `event_id='01 NOIDA 7 - 10 JAN 2010'`,
  `primary_speaker='Swami ji'`.
- `qdrant_setup.py` and `analytics_schema.sql` are idempotent on second run.
- `search_transcripts()` correctly applies each new filter arg
  (covered by extension to `tests/integration/test_function_tools.py`).
- `docs/best_practices.md` exists and is internally consistent with
  the PRD's locked technical decisions (§3).

**Commit:** `feat: path-based metadata extraction with filter-enabled retrieval (Phase 12)`

---

### Phase 13 — Content-based tagging (per-file LLM enrichment)

**Objective:** After ingest, run each file's full transcript through the
local Qwen 2.5 7B (Ollama) to extract a fixed-schema content tag set —
`event_type`, `primary_language`, `topics`, `people_named`, `places_named`,
`scriptures_referenced`, `timing_clues`, `location_clues`, `summary_hindi`,
`summary_english`. These complement Phase 12's path-based metadata
(recording-time facts) with content-based facts (what the audio is
*about*) and unlock filters like "satsangs that mention the Bhagavad Gita"
or "discourses where Guruji talks about a specific person."

**Locked design decisions:**

| Decision | Choice |
|---|---|
| Tagging model | Qwen 2.5 7B (q4_K_M) via Ollama — already deployed, multilingual, fast. `tag_model` column records exact tag for audit. |
| Granularity | Per source file. One tag set per `file_meta.source_file` row. Chunks inherit via Qdrant `set_payload` filtered by `source_file`. |
| Stage | Separate enrichment pass *after* ingest. Resumable: `WHERE tagged_at IS NULL`. Failure in tagging never blocks ingest or search. |
| Long-file strategy | Single-pass with `num_ctx=32768` when transcript ≤ `--max-tokens-single-pass` (default 28000). Beyond that, map-reduce on existing `chunk_meta` rows: per-chunk mini-summaries → final tag pass on the concatenated summaries. Summaries always reflect full content. |
| JSON contract | Ollama `format: "json"` + post-hoc validation (right keys, right types, `event_type` in enum, summaries non-empty). Bad output → dead-letter with raw model response. |
| Path-metadata relationship | Additive only. Phase 12 columns (`session_date`, `season`, `track_type`, `location`, `event_id`) untouched. Content tags live alongside. |

**Schema additions** (`infra/postgres/analytics_schema.sql`, all idempotent):

```sql
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS event_type            TEXT;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS primary_language      TEXT;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS topics                TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS people_named          TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS places_named          TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS scriptures_referenced TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS timing_clues          TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS location_clues        TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS summary_hindi         TEXT;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS summary_english       TEXT;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS tagged_at             TIMESTAMP;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS tag_model             TEXT;
```
Plus GIN indexes on the array columns and B-tree on `event_type`, `primary_language`, `tagged_at`.

**Deliverables:**

`ingestion/enrich_content_tags.py` — new resumable script:
- Selects `source_file` from `file_meta` WHERE `tagged_at IS NULL`
- Reconstructs the full transcript by concatenating `chunk_meta.text` for
  that file (ordered by `start_sec`)
- Routes through single-pass or map-reduce by token-budget check
- Calls Ollama `/api/generate` with Qwen 2.5 7B, `format: "json"`,
  `num_ctx=32768`
- Validates the JSON contract; failures go to `dead_letter/tag_*/`
- Writes tags to `file_meta`; then `qdrant_client.set_payload(...,
  payload_selector=Filter(must=[FieldCondition(key="source_file",
  match=MatchValue(value=src))]))` to propagate to all chunks of that file
- CLI: `--limit N`, `--retry-failed`, `--max-tokens-single-pass`,
  `--model`, `--dry-run`

`ingestion/utils/tag_schema.py` — pure helpers:
- `TAG_SCHEMA` constant (the JSON shape Qwen must return)
- `build_prompt(transcript: str) -> str` — schema-strict prompt builder
- `validate_tags(obj: dict) -> tuple[bool, list[str]]` — checks keys, types, enum

`ingestion/chunker_text.py` + `chunker_json.py`:
- Add `--skip-existing` flag (default ON) so files whose `.chunks.json`
  already exists are skipped. Tiny fix to make re-running the chunker
  on a growing input directory cheap.

`infra/qdrant/qdrant_setup.py`:
- Add idempotent payload indexes for `event_type` (KEYWORD),
  `primary_language` (KEYWORD), `topics` (KEYWORD),
  `people_named` (KEYWORD), `places_named` (KEYWORD),
  `scriptures_referenced` (KEYWORD)

`open_webui_functions/search_transcripts.py`:
- Add filter args to `search_transcripts()`:
  `event_type: str | None`, `primary_language: str | None`,
  `topics: list[str] | None`, `people_named: list[str] | None`,
  `scriptures_referenced: list[str] | None`
- Update docstring with concrete examples (the LLM reads this to learn when to use them)
- All array filters use Qdrant `MatchAny` semantics (logical OR within filter)

`scripts/06_enrich_tags.sh` — wrapper with sane defaults (reads `.env`,
exits non-zero on failure, supports `--limit N`).

`tests/unit/test_enrich_content_tags.py`:
- Prompt builder includes all schema keys and the input transcript
- Validator accepts a well-formed sample and rejects: missing keys, wrong
  types, unknown `event_type`, empty summaries
- Idempotency: re-running on a file already marked `tagged_at IS NOT NULL`
  is a no-op
- Bad model JSON → file moved to dead-letter, not silently dropped

`docs/best_practices.md` — append new section:
- When path metadata vs content tags is the right filter
- Prompt-tuning tips (verbatim quote enforcement, Hindi script integrity)
- Cost expectations (~20–60s per file on GPU; CPU is impractical)

**Acceptance criteria:**
- `pytest tests/unit/test_enrich_content_tags.py -v` green.
- Full existing unit suite still green (no regressions).
- `analytics_schema.sql` and `qdrant_setup.py` remain idempotent on
  second run.
- Running the chunker twice over the same input dir: second run skips
  all already-chunked files (visible in log totals).
- Running `enrich_content_tags.py` twice: second run finds 0 untagged
  files and exits cleanly.
- One end-to-end smoke test: ingest a small file → run enrichment → confirm
  file_meta row has populated `summary_hindi`/`summary_english`/tags and
  `set_payload` on Qdrant succeeded for that file's chunks.
- `search_transcripts()` with `topics=["karma"]` returns only chunks
  whose `source_file` has `karma` in its `topics` array.

**Commit:** `feat: per-file content tagging via Qwen 2.5 7B (Phase 13)`

---

### Phase 16 — The Archive Map (slice 1: read-only)

**Objective:** Make the archive visible. Today there is no way to see what the
corpus contains, no way to see which recordings the system can actually recall,
and no way to notice when something is wrong. That cost real things: 5,420
isolated files sat untranscribed for four days with nobody noticing, and three
tracks are in Qdrant right now with corrupted identity keys that nothing
surfaces.

This slice draws every indexed recording as a point of light on a radial map of
the archive's own folder hierarchy, colours it by what Postgres can prove about
it, and refreshes itself while the pipeline runs.

**Overrides PRD §15** ("a separate frontend is out of scope") for the same
reason Phase E did: by opening a phase, not by ignoring the section. §15 remains
correct about *unbounded* frontend work.

**Locked design decisions:**

| Decision | Choice |
|---|---|
| Data source | `file_meta.source_file` alone. It already *is* the hierarchy (`Collection/Group/Sitting/Track.json`). No new table, no new mount, no filesystem access. |
| Layout | Radial hierarchy, relaxed on **angle only**. Radius encodes tree depth exactly. A force simulation is rejected: every edge here is containment, so it would simulate away a tree we already know and return a hairball. |
| Ring radii | Sized by demand (`2πR ≥ Σ(2r + pad)`), not a fixed gap per level. Depth 3 holds 1,748 sittings; a fixed gap gives each 1.2 units of arc for a dot needing 7. |
| Renderer | Canvas 2D. SVG's per-element DOM cost dominates above ~2,000 nodes. Glow is a pre-rendered sprite, never `ctx.shadowBlur`. |
| Hover picking | Uniform spatial grid, `O(1)`. A quadtree is unnecessary at this density. |
| Frontend deps | **Zero new ones.** `dependencies` stays exactly `react`, `react-dom`. |
| Live refresh | Poll `/api/corpus/summary` every 10 s, only while the tab is visible. On a `version` change, refetch the skeleton plus open clusters — never the whole tree. |
| Control plane | **None in this slice.** Read-only. Safe to build while the WhisperX and isolation runs are in flight. |
| Phase 15 | **Stays closed.** The map needs no entity resolution; every edge it draws is containment. |
| Dark theme | Scoped to `.brain-view[data-theme="dark"]`. The rest of the app stays parchment. App-wide dark mode is separate, honest work. |

**Node states — exactly what Postgres can prove, and no more:**

| State | Rule | Meaning shown to the operator |
|---|---|---|
| `remembered` | `chunk_count > 0 AND session_date IS NOT NULL` | Searchable, and we know the day it was recorded. |
| `written` | `chunk_count > 0 AND session_date IS NULL` | Searchable — but no date could be read from its folder. |
| `failed` | `source_file NOT LIKE '%/%'` | Indexed under a name with no place in the archive. |

`tagged_at` is **zero for all 9,335 rows** (Phase 13 has never been run against
this data), so it cannot carry a colour. The `dark` (never transcribed) and
`heard` (isolated, not transcribed) states need the filesystem, which the
`rag-api` container cannot see; they arrive with the desktop overlay. **The
legend must say that this map shows only what has been indexed** — otherwise a
nearly-full disc reads as a nearly-finished archive.

**Deliverables:**

`rag_api/corpus.py` — `CorpusReader` over `file_meta`, mirroring
`rag_api/analytics.py`. `build_nodes()` and `summarize()` are pure functions over
rows, so the tests need no database. `version_key()` hashes
`(n_files, n_chunks, max_ingested_at)`.

`rag_api/app.py` — `GET /api/corpus/summary` and `GET /api/corpus/state`, both
`Depends(require_auth)`, both declared **above** the `app.mount("/", StaticFiles)`
call that otherwise swallows every route after it.

`frontend/src/viz/{radialTree,relax,grid,draw}.ts`, `frontend/src/corpus.ts`,
`frontend/src/components/{BrainView,BrainLegend,CorpusDetail}.tsx`, additions to
`api.ts` / `App.tsx` / `Sidebar.tsx`, and a scoped dark-token block plus the
project's first `prefers-reduced-motion` query in `styles.css`.

`frontend/scripts/check-viz-invariants.cjs` + `npm run check:viz` — invariant
checks against the compiled layout modules. No test runner is added; `tsc` is
already present.

`tests/unit/test_rag_api_corpus.py`.

**Acceptance criteria:**
- `GET /api/corpus/summary` → `n_files == 9335`, `n_chunks == 24567`
  (cross-checked against `ingest_status`, a different table), `n_failed == 3`,
  and the three degenerate keys named.
- Drilling `Dagshai 2002/` → `03 MAR - 2002/` → `DAGSHAI 26 - 29 MAR 2002 HOLI
  CAMP/` → `26 MAR - 1$ - 7 PM/` reveals `02 OM GURUVE NAMAH.json` as a
  `remembered` node with `n_chunks > 0` and a session date.
- `curl localhost:8081/api/corpus/summary` with no password → **401**.
- `npm run check:viz` green, including *relaxation never moves a node radially*
  — if it could, a recording would drift into the ring where camps live and the
  picture would start lying.
- Frame budget under 16 ms at ≥ 1,900 visible nodes, measured via
  `window.__brainPerf`, not eyeballed. *(Restated by slice 1c: with progressive
  disclosure the map no longer draws 1,900 nodes unless you open the archive's
  largest group. See that slice's budgets.)*
- `frontend/package.json` `dependencies` contains **exactly** `react`,
  `react-dom`.
- `styles.css` contains a `prefers-reduced-motion` block, and honouring it stops
  the drift while leaving the map fully rendered and usable.
- Stopping `rag-api` mid-session: the map holds its last good picture, shows a
  reconnecting indicator, and recovers. It does **not** flash empty.
- `pytest tests/unit -q` all green.

**Commit:** `feat: read-only archive map (Phase 16, slice 1)`

---

### Phase 16 — slice 1b: navigation, stillness, and the recording itself

Still read-only. Still a plain browser. The one new capability is *reading two
files the archive already points at*.

**Objective.** Make the map navigable, make it stop moving, and let a click on a
recording play its isolated vocals beside its transcript.

**Locked decisions**

| Decision | Why |
|---|---|
| Clicking a folder flies the camera into it | Clicking `Dagshai 2001` did nothing at all: `canExpand()` gated on *"are the children un-fetched"* (`depth >= 3`) and was also being used to answer *"can this be opened"*. Two questions, one predicate. |
| The map is **still** when nothing is happening | `drift()` rotated every ring every frame, forcing a redraw and a grid rebuild 60×/s forever. It told the operator the archive was busy while it slept. Stillness is information. |
| `file_meta` **is** the allowlist | `/api/track/*` resolves a path only after the key matches a row. You cannot name a file that is not a row, so traversal is structurally impossible, not filtered. |
| Audio is ticket-gated, not header-gated | `<audio src>` sends no custom headers, and the dashboard password must never ride in a query string. A 5-minute HMAC ticket, bound to one `source_file`, keyed by a per-process random secret. |
| `ISOLATED_DIR` = `GuruAudio/Output`, not a `.ckpt` folder | **Measured:** the 3 tracks `transcribe_local.py` handled were isolated by the *second* model. Rooted at one model's folder, their audio is unreachable while the files sit right there. |
| The WAV path is **read**, not guessed | Each `raw.json` records `audio_file` — the path the transcriber actually opened. |

**Deliverables**

- `rag_api/tracks.py` — path inverse, HMAC ticket, `slim()` (118 KB → ~20 KB by
  dropping the per-word timings; they stay on disk).
- `GET /api/track/transcript` (auth) and `GET /api/track/audio` (ticket), both
  declared **above** the `app.mount("/", StaticFiles(...))`.
- `GET /api/health` gains `transcripts_mounted` / `isolated_mounted`. Docker
  silently creates an empty directory for a missing bind source; existence
  proves nothing, so health probes for *content*.
- Two `:ro` bind mounts, in `docker-compose.yml` **and** repeated inside the
  override's `volumes: !override` block, which replaces the base list wholesale.
- `frontend/src/components/TrackPanel.tsx`, breadcrumb trail, `Esc` to go up.
- `viz/relax.ts` loses `drift()`; `viz/draw.ts` gains focus-subtree dimming.

**Acceptance criteria** — all run and shown

- `GET /api/health` → `transcripts_mounted: true`, `isolated_mounted: true`, and
  `docker exec rag-api ls /app/data/isolated` lists **both** `.ckpt` trees.
- Transcript for `…/02 OM GURUVE NAMAH.json` → `duration 1064.802`, 41 segments,
  8,359 chars of cleaned text, an `audio_url`.
- No password → **401**. `source_file=../../../etc/passwd` → **404**.
- Audio: no ticket → **403**; a ticket minted for a *different* track → **403**;
  a forged signature → **403**.
- Audio with a valid ticket: `206 Partial Content`,
  `content-range: bytes 0-1023/375663274`, `content-type: audio/wav`. Seeking
  200 MB in → 206. Header reads `PCM / 2ch / 44100 Hz / 32-bit`.
  Throughput **90 MB/s** against the 353 KB/s playback needs.
- A degenerate track (isolated by the second model) serves audio: **206**.
- `pytest tests/unit -q` → **430 passed**, including a parity test that walks the
  real tree and asserts `transcript_path(qualified_source(p)) == p` for all
  **9,335** keys, and a second that resolves every one of their `audio_file`s.
  No sampling.
- `npm run check:viz` → 12 invariant checks, including *"relaxation never moves a
  node radially"* and *"relax.ts exports no `drift`"*. *(36 after slice 1c.)*
- `dependencies` is still exactly `react, react-dom`.

**Not verifiable without a browser, and therefore not claimed:** the isolated
WAVs are **32-bit integer PCM**. Chromium decodes `pcm_s32le`; Firefox's WAV
decoder handles 8/16/24-bit int and 32-bit float and gives up here. Electron is
Chromium, so the desktop app is safe; a Firefox tab may show a dead player. The
panel says so rather than failing silently.

**Commit:** `feat: drill-in navigation, a still map, and the recording itself (Phase 16, slice 1b)`

---

### Phase 16 — slice 1c: a map you can navigate, arrange, and read

Still read-only. Still `react` + `react-dom`. Still zero backend changes —
`build_nodes(rows, prefix, 1)` already returns exactly one level of children.

**Objective.** Make the map do what the design promised. Slices 1 and 1b drew
**every node ever fetched, forever**: `nodes` was accumulate-only and `draw()`
culled by viewport frustum alone. `expanded` was never a visibility set — it was
a "children were fetched" ledger whose only visual consumer was a 1.2 px ring.
There was no collapse. Alongside that: give the reader control of node spacing
and of the reading panel's size, and stop the archive totals from vanishing the
moment anything is clicked.

**Locked decisions**

| Decision | Why |
|---|---|
| Visibility is `visibleNodes(nodes, spineOf(focusPath))` | `open` is **derived, never stored**. `focusNode`, `focusUp`, `Esc` and the breadcrumbs already maintain `focusPath`. Two things fall out free: `Esc` *is* collapse, and `selectedPath` can never point at an undrawn node. |
| One branch open at a time | The spine. Lifting it later is one line — make `open` real state, seed it from `spineOf`. Nothing downstream cares. |
| `REVEAL_DEPTH = 1`, `PREFETCH_DEPTH = 2` | First paint is the collections, and nothing else — **one click, one level**, all the way down. At `REVEAL_DEPTH = 2` every group was already on the disc (a solid outer smear of thousands of dots) and clicking a collection was a *visual no-op*, because its children were on screen and its grandchildren needed the **group** opened. The prefetch stays one level ahead of the reveal so that first click does not wait on the network. ~430 KB → ~40 KB, ~1,950 nodes drawn → ~17. |
| `leafCount()` → `weight()` | **Wedge conservation.** A collapsed folder must claim arc for what it *holds*, or a collapsed 1,748-sitting camp claims the wedge of an empty folder and opening it re-angles the whole disc. Since `build_nodes` rolls each track into every ancestor, `weight(collapsed n) == weight(expanded n)`, so expanding moves no angle outside its own subtree. |
| Node movement is free `(x, y)`; the **layout** is not | The law is not "radius is depth" — it is **the system never moves a node radially, only the reader does**. `angle` and `dist` are what the layout chose, and no drag touches them; the reader's displacement lives in `ox`/`oy` and is added at render time (`applyDisplacements`). This is not a second layout engine: a drag is a rigid translation of a subtree, applied *after* the layout has run. It is also *cheaper* than the angular drag it replaces — **0.056 ms** per pointermove at 2,942 drawn nodes, against 0.132 ms, because a translation is two adds per node where a rotation was two trig calls. |
| Relaxation buckets by `(depth, frame)`, and `Placed.fixed` is **deleted** | A rigid translation is an isometry, so every chord *inside* a translated subtree is unchanged — relaxation's chord math (`θ = 2·asin(s / 2R)`) is therefore still exact between two nodes sharing a displacement, and meaningless between two that do not. `frame` is the path of the nearest displaced ancestor-or-self. **Theorem:** a dragged node `D` has `frame(D) = D.path`, and any other node with that frame is a strict descendant, hence deeper — so bucket `(depth(D), D.path) = {D}`, and `relaxAngles` skips buckets of size < 2. Relaxation *cannot* move a node the reader placed; no flag is needed. Measured on three simultaneous drags including a nested one: `1｜C0=1 · 2｜C0/G0=1 · 3｜C0/G0=978 · 2｜C0=11 · 1｜C1=1 · 1｜=15`. The `978` is the payoff — a branch you drag out and then open is as tidy as one you left alone. Bonus: dragging a node out no longer disturbs its old ring-mates, since relaxation only pushes apart and never pulls. |
| A drag clamps to `±MAX_DRAG_COORD = 50,000` | Derived, not chosen. `SpatialGrid.key()` packs a cell index (cell = 48) into `(c + 32768) * 65536`, so a coordinate must stay inside ±1,572,864 or two distant nodes silently share a bucket and hover picks the wrong one. The widest layout this archive can produce — all 55,821 nodes at Spread 420 — reaches 139,566, so 50,000 leaves an order of magnitude of headroom and cannot be reached by hand (at `MIN_SCALE` it is 3,000 px of dragging). It exists mostly to survive a `localStorage` entry that says `1e300`. An infinity clamps to the bound it is heading for; only `NaN` means the origin. |
| Targets are stored **absolute**; `brain.arrange` is bumped to **v2** | A stored delta drifts as the layout changes underneath it; an absolute point is what the reader chose, so a moved node holds its pixel while the Spread slider grows the disc around it. `applyDisplacements` re-derives every `frame` from the ledger on each relayout *and on each pointerup*, which makes the state after a drag identical to the state after a reload. v1 held bare angles; reading one as a point would scatter the map, so the key changes with the shape and the old one is removed on sight. |
| Drag a dot → the dot moves; drag the background → the map pans | No modifier, no mode. A Shift gesture hidden behind a collapsed panel is a feature nobody has; the "Rearrange" toggle is deleted. A press below `DRAG_SLOP_PX = 3` screen pixels is still the click it started as, so a dot is both a button and a handle. Shift now forces a pan, for a press that lands on a dot by accident. |
| Relaxation separates by **chord**, `2·asin(s / 2R)` | `θ = s / R` is its small-angle approximation and it under-separates. At `minRingGap = 60` with 22-unit dots it leaves a **0.98-unit overlap** — and the old invariant check's `- 0.5` tolerance had been silently absorbing exactly this error at the shipped radius. The tolerance is now `1e-6`. |
| `minRingGap` floored at `2 · MAX_NODE_RADIUS = 44`, inside `radialTree` | Floors clamp in the layout, not at the call site, so the checker proves them against every caller — including a slider dragged to zero. It also makes "adjacent rings never touch" true by construction, which nothing checked before. |
| Ring radii solve `Σ 2·asin((ρ + pad/2) / R) ≤ 2π` | The arc form `2πR ≥ Σ(2ρ + pad)` over-promises. `asin` is convex, so the per-node budget bounds every *pair*, which is exactly what relaxation must reach. Reduces to the arc form for large rings; the correction only bites where the old formula broke. |
| Camera compensation is a **pan**, not a rescale | `v ↦ k·v, scale ↦ scale/k` reproduces the whole screen image — making the Spread slider a no-op with extra steps. `anchorPan` keeps *one node* under its pixel and lets the rest move, which is what the reader asked for. |
| The panel resize writes to the DOM, never to state | `.brain-side` renders one `<li>` per transcript segment. A `setState` per pointermove reconciles thousands of them. State commits once, on release. |

**Bugs fixed, found while doing this**

- **`hoverRef` was never re-resolved after a relayout.** `draw.ts` compares
  `Placed` by object identity and `radialTree` reallocates every one, so `keep`
  held a single ghost, `dimmed()` was true for every real node, and the whole map
  dropped to 10% opacity until the next mousemove. Only the 10 s poll triggered
  it. Collapse would have fired it on **every click**.
- **`selectedPath` was in the layout effect's dep array.** Clicking a *recording*
  ran `radialTree` + 24 relaxation passes + a full `SpatialGrid.rebuild` to update
  one `.find()`.
- **`npm run bench:viz` threw.** It imported and called `drift`, deleted in
  `0eba68a`, while `check:viz` asserts `drift` is gone. The two scripts
  contradicted each other and any number resting on the bench was unverified.
- **`expanded` never shrank**, so the poll fanned out one request per folder ever
  opened. The refetch is now the spine: at most one path per level.
- Four `Math.max(64, ...placed.map(...))` argument spreads, which throw
  `RangeError` past ~65k nodes. One pass now yields `byPath`, `extent`, and the
  bounding box.
- The invariant fixture hand-wrote `n_files: 5` on a folder whose only child had
  `n_files: 1` — a shape `build_nodes` cannot produce. It is now generated from a
  track list, rolled up the way the backend rolls up.
- **`relaxAngles` sorted each ring on the raw angle.** `place()` hands out angles
  in `[-π/2, 3π/2]`, so that *happened* to be the circular order — until a
  relaxation pass shoved one across the seam. Then two true neighbours sit at
  opposite ends of the array, nothing pushes them apart, and dots overlap. It
  sorts the wrapped angle now; the regression test measures a **4.02-unit
  overlap** without it.
- **`frameFor` clipped at `MIN_SCALE = 0.06`**, so the "Whole archive" breadcrumb
  could not actually frame the whole archive: anything past an extent of ~6,600
  was silently cropped, and opening the largest camp at Spread 420 reaches 14,575.
  Free `(x, y)` dragging makes it reachable by hand, which is how it was found.
  `MIN_SCALE` is now `0.006`, which frames an extent of 92,000 — enough to cover
  `MAX_DRAG_COORD`. That in turn forced the hover slop to be capped: it is measured
  in *layout* units, so at 0.006 an uncapped `5 / scale` swept 1,521 grid cells on
  every pointermove. `Math.min(5 / scale, 4 · MAX_NODE_RADIUS)` sweeps 49.
- **`applyDisplacements` skipped `applyAngles` on an empty ledger**, so clearing
  the arrangement left every node rendered at its last dragged position until
  something else forced a relayout. Caught by the invariant checker, not by hand.

**Deliverables**

- `frontend/src/viz/camera.ts` — `Viewport`, `toScreen`, `toLayout`, `anchorPan`.
  Split out of `draw.ts`, which touches `document` and so cannot be loaded by the
  checker. Added to both npm scripts.
- `frontend/src/corpus.ts` — `PREFETCH_DEPTH`, `REVEAL_DEPTH`, `MAX_NODE_RADIUS`,
  `spineOf`, `visibleNodes`, `childrenUnfetched` (was `hasHiddenChildren`).
- `frontend/src/viz/radialTree.ts` — `weight()`, `a0`/`a1`/`ox`/`oy`/`frame` on
  `Placed`, `subtreeOf`, `translateSubtree`, `applyDisplacements`, `clampTarget`,
  `MAX_DRAG_COORD`, clamped `LayoutOptions`, chord-exact `seatRadius`.
- `frontend/src/viz/relax.ts` — `asin` separation, `(depth, frame)`-keyed ring
  buckets, always-symmetric correction. **No new exports** (`check:viz` asserts
  `Object.keys(relax) === ["relaxAngles"]`).
- `frontend/src/components/ArchiveHome.tsx` — the totals, collapsible, pinned to
  the top of the panel. `CorpusDetail` is now `memo`'d and returns `null` for a
  null node.
- `BrainView.tsx` — pointer-event migration with capture, free `(x, y)` drag on a
  bare press, two spacing sliders, the `role="separator"` splitter, and three
  `localStorage` preferences (`brain.spacing.v1`, `brain.arrange.v2`,
  `brain.side.v1`), each clamped on read inside a `try`/`catch`.
- `frontend/src/viz/draw.ts` — collections are labelled unconditionally. An
  unlabelled disc of dots is a picture of an archive, not a way into one.
- `frontend/scripts/{check-viz-invariants,bench-viz}.cjs`.

**Acceptance criteria** — all run and shown

- `npm run check:viz` → **36** invariant checks, including *wedge conservation*
  (expanding a node changes no angle outside its subtree, to 1e-9), *chord-exact
  non-overlap to 1e-6 across the whole slider sweep* (`minRingGap ∈ {60,130,420} ×
  padding ∈ {0,2.5,24}`), *adjacent rings never touch*, *the layout stays inside
  `SpatialGrid`'s key space at 27,806 nodes — and still does with four branches
  flung to `±MAX_DRAG_COORD`*, *a drag never changes `dist` or `angle`*, *a rigid
  translation preserves every chord inside the subtree*, *a node the reader placed
  is alone in its `(depth, frame)` bucket and 200 relaxation passes cannot move
  it*, *a moved node survives a relayout at every spacing*, *`clampTarget` bounds a
  hostile `localStorage` value*, *relaxation still separates a ring whose subtree
  was pushed off the seam*, and *the camera compensation is exact*.
- `npm run bench:viz` **runs** (it threw before) and reports, at 27,806 sittings
  with the largest group open (2,942 drawn nodes): expand path **< 100 ms**
  (measured 11.0 ms), node-drag pointermove **< 16 ms** (measured 0.056 ms — it was
  0.132 ms while the drag was angular), hover pick × 1000 < 16 ms. An idle frame
  does zero work.
- `npm run build` clean. `dependencies` is still exactly `react, react-dom`.
- `pytest tests/unit -q` → **430 passed**; `ruff check rag_api/` clean. The
  backend is untouched.
- First paint draws only the collections: `window.__brainPerf.visibleNodes` ≈ 17,
  and one `state?prefix=&depth=2` of ~40 KB in the network panel.
- Drilling `Dagshai 2002/` → `03 MAR - 2002/` → `DAGSHAI 26 - 29 MAR 2002 HOLI
  CAMP/` → `26 MAR - 1$ - 7 PM/` → `02 OM GURUVE NAMAH.json` still reaches the
  transcript and audio, **one level revealed per click**. `Esc` closes one level
  and **no node outside that branch moves**.
- Hovering a dot immediately after clicking one does not darken the map.
- Sweeping Spread 60→420 and Breathing room 0→24: no overlap at any setting, and
  the node at the centre of the screen stays under the same pixel.
- Drag a sitting from the outer ring into the middle of the disc: its 978 tracks
  follow rigidly, nothing outside its subtree moves, and `dist` and `angle` are
  unchanged. Open it and its tracks appear around it, relaxed and non-overlapping.
  Reload — it holds. Sweep Spread 60→420 and it holds its point while the disc
  grows around it. Reset restores everything. A click that jitters under 3 px
  still opens the node.
- Drag the splitter 280↔640 px: the transcript reflows, the canvas re-centres,
  and no `setState` fires per pointermove. Focus it and the arrow keys resize it.
  The value persists, clamped on read. Below 1000 px it becomes a row splitter.
- `docker stop rag-api` for 8 s: the map holds its last picture, shows the
  reconnecting badge, recovers. It does not flash empty.

**Not verifiable without a browser, and therefore not claimed:** the splitter's
`aria-valuenow`/`aria-orientation` announcements, and pinch-zoom on touch —
`touch-action: none` still suppresses native gestures and nothing yet replaces
them. The pointer migration is the prerequisite; the gesture is separate work.

**Commit:** `feat: progressive disclosure, spacing controls, and a resizable panel (Phase 16, slice 1c)`

**Later slices, gated and not opened here:** filesystem overlay (the real five
states, via the desktop app — the only process that can see `D:\Audio Data`);
running a pipeline stage from the console (Electron main owns spawning, because
`scripts/add_transcripts.sh:49` runs `docker compose restart rag-api` and would
kill any progress stream served by `rag-api` itself). When that lands, the picker
chooses **what to process, never where it lands** — `qualified_source` keys off
`base_dir` and `output_wav_path_for` keys off `input_root`, so a new output root
re-keys the whole archive and duplicates the index rather than growing it.
`add_transcripts.sh` already exists to prevent exactly that by hand, and already
takes a narrower folder as `$1`. The performer web stays **blocked on
measurement** — `catalog_track.matched_source_file` is populated for 0 of 22,501
rows, so those edges would connect nothing to nothing.

---

### Phase 17 — Deterministic Hindi text normalization and entity repair

**Objective:** Repair the transcript corpus *before* it is chunked. The
`.cleaned.txt` files produced upstream by the Qwen 3.5 cleaner (in
`D:\Transcription whisperx\Output`) carry four classes of damage that make
retrieval fail, and the chunker faithfully indexes all of it. This phase adds a
pure-Python normalization pass, a corpus-mined entity gazetteer, and fixes the six
chunker/metadata defects that survived adversarial verification. **No LLM, no GPU,
no network.** Two further findings were raised and then **refuted by measurement**
(below); they are recorded so they are not re-raised.

The dominant defect: **31% of files spell the same concept in both scripts**
(`mind`/`माइंड` 25% of files, `meditation`/`मेडिटेशन` 18%). Some tokens are corrupted
*mid-word* — `मEDITATION` (566×), `रishi` (197×), `Kumarों` (16×) — and are
unretrievable by any query in either script. BM25 cannot bridge an orthography
split, and the reranker cannot recover a candidate lexical search never surfaced.

**Locked design decisions:**

| Decision | Choice |
|---|---|
| Method | Deterministic only. Rule tables + a corpus-mined gazetteer. An LLM re-clean (Route B) is **explicitly out of scope** — the upstream cleaner already fabricated 4,010 glosses, amplified a 552-word file to 2,160, produced `ठंडाई` (a cold drink) where context wanted `तन्हाई` (solitude), and left `वारेपूर्ण सिंग` unrepaired after glossing it. |
| Output | `<stem>.normalized.txt` written **alongside** the source. `.cleaned.txt` is never mutated. Revert = delete one glob. Every file stays diffable. |
| Entity truth | The corpus is its own gazetteer. `पूरन सिंह` occurs 42× in 14 files; `पूर्ण सिंग` 3×; `वारेपूर्ण` 1×. Majority vote over phonetic clusters recovers the canonical form with no model. |
| Entity safety | **Human sign-off is a hard gate.** Mining stops and emits `data/gazetteer_review.md`; `data/gazetteer.json` is generated from the approved file only. No name is rewritten on inference. |
| Rewrite safety | Word-boundary anchored, minimum two tokens. `सिंग` has 527 hits but `डांसिंग` (*dancing*) contains it — a naive replace yields `डांसिंह`. `पूर्ण` alone legitimately means "complete"; only the bigram `पूर्ण सिंग` is wrong. |
| Cross-script bridging | **A synonym table, never a source rewrite.** Registered in the Tantivy analyzer at index *and* query time, so it is additive (recall-only) and cannot corrupt text. Rewriting `mind`→`माइंड` would corrupt the verbatim English Guruji actually speaks — 43.9% of `watch` and 47.5% of `seer` sit inside English sentences. Rewrites are reserved for surface forms with **no valid reading** (`मEDITATION`, `रishi`, a glued danda). |
| Loop collapse | **Enabled**, at `N ≥ 6` consecutive identical tokens → `K = 3`. Hand-checked on 196 runs: 0 lost retrievable meaning. `N < 6` or `K < 3` is forbidden — length-2 runs number 71,240 and are legitimate reduplication (`बहुत बहुत`). Phrase loops are a known, documented gap handled by the chunker's distinct-ratio guard. |
| Honorifics | `Swami ji` / `Gurudev` / `Sadguru` / `Guruji` are one person (user-confirmed), but are **never unified in body text** — `हे मेरे गुरुदेव` is a bhajan lyric. Equivalence is recorded in `data/aliases.json` for query expansion only. `Rishi ji` is a distinct speaker and is never aliased to them. |
| Chunking parameters | **Unchanged.** `TARGET_TOKENS=450`, `MAX_TOKENS=700`, and `estimate_tokens`' `1.3` multiplier all stay. Only its backwards docstring is corrected. Re-tuning the multiplier re-chunks the whole corpus for no correctness gain (nothing is truncated: 700 est ≈ 834 real vs bge-m3's 8192) and is a PRD §3-class decision. |
| `track_type` vocabulary | **Head-anchored match**, not substring — substring reclassifies the song `MEDITATION KI MASTIYON MEIN`. Three values added (user-approved): `qa` (a `QUES`/`QUESTION` prefix, 209 tracks), `session` (the substring `SITTING`, 83), and `combined` (≥ 2 distinct vocab types in one title, 108 — a `SAMBODHAN & PRAVACHAN` is genuinely both, and a single-valued field cannot say so). `track_type` stays single-valued; making it a list would touch the Qdrant payload type, `tag_schema.py`, `analytics.py`, `retrieval.py` and the frontend filter, which is outside this phase. |
| Re-ingestion | Non-idempotent (Qdrant point IDs hash chunk text; the Tantivy writer is append-only). Rebuild targets a **freshly created** collection and index, never a live one. |

**Rejected after measurement** (recorded so they are not re-raised):

- *WhisperX word confidence as a suspect-span detector.* Every word in `*.raw.json`
  carries a `score`. Over 44,294 words / 60 files: median `0.200`, 67% below `0.30`,
  0.8% above `0.80`. Compressed alignment noise, not calibrated probability.
- *The `latin-1` fallback in `chunker_text.py:52-62`.* Real as written, but
  **unreachable**: 0/7,413 `.cleaned.txt` and 0/9,335 `.raw.json` fail strict utf-8;
  0 have a BOM. The live path is `chunker_cleaned.py` (via `scripts/add_transcripts.sh:39`),
  which reads strict utf-8 and raises → isolates to `_failed/`. Loud, not silent.
  `chunker_text.read_text` is the legacy plain-`.txt` path; nothing imports it.
- *Danda desegmentation as a source rewrite* (`(\S)([।॥])` → `\1 \2`, 374,024 occurrences
  in 6,885 files). The stated rationale — that `mind।` is a single BM25 token — is **false**.
  Tantivy's `default` tokenizer splits on any non-`is_alphanumeric` character, and `।`
  (U+0964, category Po) is one. Indexing `मैं आया।दूसरा वाक्य।` and querying `दूसरा`
  returns the document. The glued danda is a *sentence-splitting* defect, already fixed at
  `chunker_text.SENTENCE_SPLIT` with a zero-width lookahead, and nothing more. Rewriting
  374,024 sites across 93% of the corpus to fix a non-problem was rejected.
- *Em-dash gloss splitting as a source rewrite* (`(\w)—(\w)` → `\1 \2`, ~420 occ). Refuted
  the same way and for the same reason: U+2014 is category Pd, so `जागरण—enlightened`
  already indexes as both `जागरण` and `enlightened`. No rewrite needed.
- *Devanagari matra conflation.* Checked while refuting the above, because it would have
  outranked every other finding: it does not happen. `आया` does not match `आयी` or `आई`,
  and `सिंग` does not match inside `डांसिंग`. Tantivy indexes Devanagari words whole. The
  substring hazard exists only in **our** rewrite rules, never in BM25.

**Deliverables:**

`ingestion/normalize_text.py` — pure functions, no IO, no network (mirrors the
shape of `ingestion/clean_align.py`, which is verified order-preserving and must
not be modified). Table-driven from `data/normalize_rules.json`.

**The corpus splits into what may be rewritten and what may only be bridged.** A
rewrite is safe only where the surface form has *no valid reading*. Cross-script
canonicalization fails that test: `mind`, `master`, `watch` and `seer` occur inside
verbatim English quotations that Guruji actually spoke — 9.7%, 18.4%, **43.9%** and
**47.5%** of their Latin occurrences respectively (*"you are not a mind, you are a seer
of it"*). Rewriting those to Devanagari would corrupt real speech to fix a BM25 problem.

*Rewritten in `.normalized.txt` (each surface form is invalid Devanagari, so there is
nothing to corrupt). Every rule below was confirmed against the **real Tantivy `default`
tokenizer**, not assumed:*

1. **Intra-word script-mix, loanword class** — exact **whole-token** map.
   `मEDITATION` 1,498 occ / 417 files → `मेडिटेशन`; `सittings` 80 → `सिटिंग्स`;
   `थॉUGHT` 19, `मEDITेशन` 31, `मeditation` 13. Never a substring replace.
   *Verified:* a document containing `मEDITATION` matches **neither** `मेडिटेशन` **nor**
   `meditation` — Devanagari and Latin letters are both `is_alphanumeric`, so the corrupt
   form is one indivisible token, unreachable from either script. This is the defect.
2. **Intra-word script-mix, native class** — a *separate* exact map, because the target is
   a Sanskrit/Hindi word, not a loanword. `रishi` 578 occ / 251 files, `Kumarों` 18/11,
   `सद्गuru` 4/2, `रindo` 23/17, `निभaya` 35/22. **The target spelling requires human
   sign-off:** the corpus contains **zero** standard `ऋषि` and is internally split
   `रिशी` 342 / `रिशि` 207. Corpus-dominant ≠ orthographically standard.
3. **Devanagari-internal spelling scatter** — `{मीडिटेशन, मैडिटेशन, मेंडिटेशन, मेडिटीशन,
   मिडिटेशन, मेडिटिशन}` → `मेडिटेशन` (~11,000 occ / 3,081 files). Devanagari→Devanagari,
   so no English-quotation hazard at all. *Verified:* `मीडिटेशन` matches only itself.
   Highest value per unit of risk in the phase.
4. **Subscribe excision** — `सब्सक्राइब` 5,217 occ / 1,678 files and `सबस्क्राइब` 25/9.
   **Token excision, never line deletion:** 159 segments glue the artifact to genuine
   meditation vocabulary. The match may not cross `और` or any unlisted word.
5. **Loop collapse** — a single punctuation-normalized token repeated **N ≥ 6** times
   consecutively collapses to **K = 3**. Measured: 2,715 runs in 1,530 files (20.6%),
   run lengths 6–13,678, removing ~142,872 loop tokens. **Ships enabled**: a hand-check
   of 196 runs found 0 that lost retrievable meaning at K=3. `N` may never drop below 6
   nor `K` below 3 — lengths 2–5 are legitimate reduplication (71,240 length-2 runs,
   3,648 length-3: `बहुत बहुत बहुत`, `ॐ ॐ ॐ`, `नमो नमो नमो`).

Because a glued danda survives into the token map's input (`मEDITATION।`), the exact-token
matcher strips leading/trailing punctuation before comparing, and restores it after. It
does **not** rewrite the punctuation — see the two rejected rules below.

*Bridged by a synonym table, never rewritten:*

8. **Cross-script concept bridging** — `data/synonyms.json`, a flat `{surface forms} →
   canonical` table registered in the **Tantivy analyzer at index time and query time**.
   Additive: it expands recall and cannot corrupt display text. Canonical direction is
   chosen per concept by frequency *and stability*, not by a blanket script preference:
   `मेडिटेशन` wins for meditation (1.31×), but `awareness` and `spiritual` keep their
   **Latin** canonical because the Devanagari side is 10+ unstable phonetic spellings with
   no dominant form. Dense retrieval already partly bridges this arm; BM25 does not.
9. **Gloss capture** — the 4,010 injected Latin parentheticals across 305 files
   (`(Master)` 98×, `(Antarmukhi)` 73×, `(Third Eye)` 51×) are stripped from indexed text
   and captured to `data/aliases.json` for query expansion.

**Anti-rules — measured, and deliberately NOT written.** Each looks like an obvious
cleanup and would destroy real content. They ship as *tests*, not as rules:

| Surface form | Occ / files | Why it must survive |
|---|---|---|
| `देखते रहना` | 7,928 / 1,413 | Not a "keep watching" outro. It is *the* core meditation instruction. |
| `घंटी` / `घंटा` / `घंटे` | 2,749 / 1,186 | Means "hour(s)" (`चौबीस घंटे`), not the YouTube bell. |
| `लाइक` | 557 legitimate | Means "-worthy" (`जीने लाइक`, `चाइल्ड लाइक`). No `लाइक करें` exists. |
| `धन्यवाद` / `शुक्रिया` / `नमस्कार` | 1,469 / 800 | Genuine thanks and greeting. `देखने के लिए धन्यवाद` = **0**. |
| `चैनल` | 65 / 39 | Real TV channels (`आस्था चैनल`) the program aired on. |
| `वीडियो` / `शेयर` / `कमेंट` | 161 / 149 | Cassette catalog; sharing sorrow; scriptural commentary. |
| bare `आई` | 6,252 | The verb *"came"* (`समझ आई`). Only the bigram `थर्ड आई` means "eye". |
| `ऊर्जा` | 1,412 / 681 | Native Sanskrit "energy", used *alongside* `एनर्जी` as a deliberate gloss pair. Never fold. |
| `झाल`, `ओम् ओम् ओम्` | length ≤ 5 runs | Cymbal onomatopoeia and chant. Only runs ≥ 6 are artifacts. |

Substring matching is banned throughout: `डांसिंग` contains `सिंग`, `पोटेंशियल` contains
`टेंशन`, `डिपॉजिट` contains `पॉजिट`, `सिंसियर` contains `सियर`, `डिस्लाइक` contains `लाइक`.
Every rule is exact-whole-token or explicitly word-boundary-anchored.

**Known coverage gap (documented, not fixed):** phrase-level loops. `ब्लूम लाइक ब्लूम लाइक
…` repeats a *bigram*, so no token is adjacent to itself and the N≥6 single-token rule
cannot see it — it accounts for 4,496 of the 5,089 raw `लाइक` hits in one file. 259 files
have a phrase loop that the collapse rule misses; `chunker_json`'s existing distinct-ratio
guard (`LOOP_UNIQ_RATIO=0.15`) catches them at chunk level. The two filters are
complementary. Do **not** lower `N` to chase phrase loops: it will not catch them and adds
false-positive risk.

`ingestion/gazetteer.py` — mine surface forms → cluster by phonetic key
(Devanagari→ISO transliteration, so `सिंग`≈`सिंह`, `पूर्ण`≈`पूरन`) → majority-vote →
emit `data/gazetteer_review.md` and **stop**. A second invocation compiles the
approved file into `data/gazetteer.json`.

`ingestion/chunker_cleaned.py` — `_cleaned_is_lossy` (line 140) gains an **upper**
bound, `MAX_CLEANED_WORD_RATIO = 1.15`, symmetric with the existing `0.85` low side.
Today only the low side is guarded (`ratio >= 0.85 → False`), so 453 files that
*expand* pass silently. Threshold chosen from the measured distribution over 7,040
raw/cleaned pairs (raw ≥ 50 words): p75 = 1.000, p90 = 1.010, **p95 = 1.088**, then a
long tail to 3.913×. At 1.15 the guard fires on 268 files (3.8%). On a high ratio it
falls back to raw, exactly as the low side does — accepting hallucinated expansion
costs *fabricated* content that pollutes retrieval, which is strictly worse than the
spelling normalization lost by preferring raw. The worst expanders are chant-heavy
tracks (`02 MEDITATION` 552→2,160; `OM GURUVE NAMAH` ×3; `02 MEDITATION (WITHOUT OM)`
4,608→16,961), where raw is itself a repetition loop and `chunker_json` will drop it —
the correct outcome for a track with no retrievable speech.

`ingestion/chunker_json.py` — `_emit_guarded` (line 194) leaks a chunk whose whole
content is one repeated token: a lone `ओ` spanning 16 s survives because `wc=1` is
below the `MIN_LOOP_WORDS = 50` floor. That floor is **deliberate** (line 34-35: it
"keeps a legitimately short, naturally-repetitive chunk from being flagged") and must
not be lowered. Add a separate, narrower guard: drop when `len(set(words)) == 1`.
Measured on disk: **10 of 24,567 chunks (0.041%)**. Three are the meditation syllable
`ओ`; the other seven are single real Hindi words (`बस`, `हर`, `बेटा`, `आमीन।`, `झाल`,
`हिष्ठ`, `गिर`), which this guard also drops. That is intended — a one-word chunk
carrying a full source/timing header is unretrievable noise either way — but it is a
*drop*, so it must be dead-lettered with a reason, never discarded silently. The guard
runs **below** the loop guard, so a 5,000-word `राम` loop still classifies as
`asr_loop` (which `audit_chunks.py` keys on), not `single_token`.

`ingestion/chunker_text.py` — `SENTENCE_SPLIT` (line 37) requires whitespace *after*
the danda, so `।` glued to the next word never splits (117 in an 800-file sample).
Fixing this also reduces how often `chunker_json._subdivide_body` raises. Its
`estimate_tokens` docstring (line 46) claims the `1.3` heuristic is "conservative for
non-Latin scripts"; measured against the real bge-m3 tokenizer it *undercounts*
Devanagari (median 1.2×, tail 1.79×). Correct the docstring; leave the constant.
**Both functions are imported by `catalog/index_catalog.py:37`** — the danda fix
changes catalog chunking too, and needs a catalog test.

`ingestion/chunker_json.py` — `_subdivide_body` (line 146) raises `ValueError` on a
single sentence over `MAX_TOKENS`. Confirmed reachable:
`chunker_cleaned.py:235 → cj.chunk_segments → _emit_guarded → line 207`, and it fires
on **40 of 9,335 files (0.43%)** — a >700-est-token run with no danda — each of which
produces zero chunks and is never indexed. Window-split instead of dead-lettering the
file. (The largest single terminator-free sentence measured is 6,194 est-tokens.)

`ingestion/utils/path_parser.py` (only — see the note on `tag_schema.py` below):

- `track_type_for()` today is an **exact** dict lookup, so `MEDITATION (WITHOUT OM)`
  (258 tracks) falls to `bhajan`. Replace with a **head-anchored** match — a vocab word
  at the start of the title, followed by end-of-string, `(`, `-`, or a bare number —
  plus three carve-outs: a `QUES`/`QUESTION` prefix → `qa`, the substring `SITTING` →
  `session`, and **two or more distinct vocab types present → `combined`** (user-approved).
  Substring matching was rejected: it reclassifies the song `MEDITATION KI MASTIYON MEIN`.

  Measured over all 9,335 tracks with the shipped `track_type_for`. `typed` rises
  4,212 → 5,262 (**+1,050**):

  | type | today | Phase 17 | Δ |
  |---|---|---|---|
  | bhajan | 5,123 | 4,073 | −1,050 |
  | meditation | 1,511 | 1,912 | +401 |
  | invocation | 1,365 | 1,412 | +47 |
  | address | 1,025 | 1,189 | +164 |
  | discourse | 311 | 349 | +38 |
  | qa | 0 | 209 | +209 |
  | combined | 0 | 108 | +108 |
  | session | 0 | 83 | +83 |
  | music | 0 | 0 | 0 |

  **False-positive gate — passed.** All 200 distinct reclassified titles were enumerated.
  Zero songs move: `MEDITATION KI MASTIYON MEIN` (13) and `MERI SHAM MEDITATION MEIN`
  (12) both stay `bhajan`, the first because its tail is a lyric, the second because
  `MEDITATION` is not at the head.

  `combined` = 108 (`SAMBODHAN & PRAVACHAN` 35, `PRAVACHAN & SAMBODHAN` 33,
  `MEDITATION & PRAVACHAN` 9). Note this is **not** the 224 titles containing `&`:
  most of those name a song on one side (`RUHANI GEET & SAMBODHAN`) and carry only one
  vocab type, so they stay `bhajan`. Only multi-*type* titles become `combined`.

  `ENTRY MUSIC` and `RETURN MUSIC` match **zero** tracks today and zero after. Both
  vocabulary keys are dead. They are left in place — removing them is not this phase's
  job — but `music` must not be reported as a live facet.

  ~115 tracks still contain a vocab word and stay `bhajan` (`STANDING MEDITATION` 10,
  `DHYAN MEIN UTRO & MEDITATION` 8, `PRAVACHAN ON SAMVAAD` 3). Accepted: the rule is
  deliberately conservative, and no false positive is worth these.

- `PRIMARY_SPEAKER` stays `"Swami ji"` (the default). Diarization is **0/9,335**
  corpus-wide and all 24,567 chunks read `('Swami ji',)`, doubly caused: `normalize_segments`
  nulls the speaker when `fmt != 'whisperx'`, and `chunker_cleaned.py:56` hardcodes
  `FMT="whisper"`. The facet is live (`retrieval.py:64-65` filters the chunk `speakers`
  array; `/api/analytics/speakers`), so a non-Swami query returns empty. The stamping site is
  `chunker_json._render_chunk` — when diarization is empty it fell back to the module constant
  `PRIMARY_SPEAKER`; it now falls back to `path_meta.primary_speaker`. A new
  `primary_speaker_for(title)` returns `"Rishi ji"` for a title naming **only** him
  (`SAMBODHAN - RISHI JI` etc., **73 tracks**) and keeps `"Swami ji"` when the title names
  both (**38 tracks** — he is genuinely present, the field is single-valued). The **209**
  `QUES` tracks are handled on the `track_type` axis (`qa`), not the speaker axis: a Q&A is
  still Guruji answering. `chunker_text.py`'s plain-text path was made consistent. The
  now-accurate comments replace the old "every transcript is Swami ji's voice" claims.

- Hyphenated-location grammar (`06 SEC-9 PANCHKULA 5 - 6 DEC 2015`): **80 tracks**
  (54 hyphenated-location date-range events + 26 single-date events). Of the 3,365
  files missing `event_id` under current code, the other **3,285** are Dagshai trees
  with no event level — correct behavior, not a bug. Low priority.

**`tag_schema.py` is deliberately NOT touched.** `event_type` (the LLM's judgment of
transcript *content*, `ALLOWED_EVENT_TYPES`) and `track_type` (deterministic, from the
title) are independent fields on independent axes. Adding `session`/`combined`/`address`/
`invocation` to the LLM's allowed outputs would change what Phase 13 enrichment asks the
model to emit and force a re-enrichment — a Phase 13 decision, out of scope here (rule 3).
The `track_type` controlled vocabulary is self-contained: `path_parser.TRACK_TYPES` is the
frozenset of the nine values `track_type_for` can return, used for validation.

**Acceptance criteria:**

- `pytest tests/unit/` passes. Each rule family has a positive test **and a
  false-positive test**: `ओम् ओम् ओम्` survives loop collapse; `डांसिंग` survives the
  `सिंग` rule; bare `पूर्ण` survives the gazetteer; a legitimate Latin passage survives
  loanword mapping; `हे मेरे गुरुदेव` survives honorific handling unchanged.
- Corpus before/after counts printed. These are the **actual dry-run results** of
  `normalize()` over all 7,413 `.cleaned.txt` (47% of files change):

  | Metric | before | after | note |
  |---|---|---|---|
  | `सब्सक्राइब` + `सबस्क्राइब` | 5,244 | **0** | token-excised, no line deleted |
  | single-token runs ≥ 6 | 2,483 | **0** | collapsed to K=3 |
  | `मEDITATION` (whole token) | 1,498 | **0** | exact-token map |
  | mixed-script token occ (all) | 3,234 | **758** | residual is flagged/rare — below |
  | injected Latin glosses | — | **4,061** captured | to `data/aliases.json` |

  The 758 residual mixed-script occurrences are **not** silent failures: they are the
  16 flagged personal-name/unverifiable-target forms (`data/normalize_review.md`) plus a
  539-type tail of ≤ 4-occurrence variants, two of which are morphological (`मEDITATIONों`,
  `इमEDITATION`, 1 occ each) that a whole-token map cannot reach without a bespoke rule
  per inflection. Every one is enumerated in `normalize_review.md`; none is guessed at.
  These supersede the first-pass figures (`मEDITATION` 566, `रishi` 197), which
  under-counted by excluding punctuation-adjacent forms.
- **Anti-rule tests are mandatory and are acceptance criteria, not nice-to-haves.** After
  normalization, `देखते रहना` still occurs 7,928× in 1,413 files; `घंटे` 2,128×; legitimate
  `लाइक` 557×; `चैनल` 65×; `धन्यवाद` 1,159×; `ऊर्जा` 1,412× and is never `एनर्जी`. `ओम् ओम्
  ओम्` (a length-3 run) is unchanged. Bare `आई` is never mapped to `eye`.
- Cross-script bridging is verified as **recall-only**: `.normalized.txt` contains the same
  count of Latin `mind` / `watch` / `seer` / `master` tokens as `.cleaned.txt`. A diff that
  shows any of these rewritten is a **failed** phase.
- `वारेपूर्ण सिंग` → `वाह रे पूरन सिंह` in
  `Live Masters 2010_isolation/16 HOTEL KEYS LUDHIANA 17 - 19 SEP 2010_isolation/18 SEP - 5$ - 1130 AM_isolation/04 NA GARZ MUJHEY HARAM SE.normalized.txt`.
- `data/gazetteer_review.md` exists and was approved by the user before
  `data/gazetteer.json` was generated. Every applied rewrite traces to an approved line.
- Chunker invariants on a re-chunk: no chunk with `len(words) == 1` (baseline 10);
  no chunk outside the new two-sided ratio bounds; **zero files dead-lettered by
  `_subdivide_body`** (baseline 40). Chunks *dropped* by a guard are dead-lettered with
  a reason — `single_token` or `asr_loop` — and the two classifications must not be
  confused: a 5,000-word `राम` loop is `asr_loop`.
- Metadata invariants: `track_type` distribution matches the table above exactly
  (`typed` 4,212 → 5,262); the reclassified distinct titles are printed and contain
  no song. **73** tracks are stamped `Rishi ji`; zero tracks whose title names *only*
  Rishi ji are stamped `Swami ji` (baseline 73 such contradictions → 0). The 38
  dual-speaker tracks keep `Swami ji` by design.
- **Retrieval A/B against a fresh collection** — a fixed query set run on old and new
  indexes, including `meditation` and its twin `मेडिटेशन` (today: disjoint result sets),
  plus `पूरन सिंह`. Recall improvement reported, not asserted.
- A timestamped entry appended to `doc.md`.

**Commit:** `feat(ingest): deterministic Hindi normalization, entity gazetteer, and chunker/metadata repair (Phase 17)`

**Explicitly out of scope:** the LLM re-clean (Route B). Tier-3 errors — a valid word
misheard as another valid word, e.g. `ठंडाई` for `तन्हाई` — survive this phase by choice.
Whether to spend 10–16 h of GPU on them is decided *after* the A/B numbers exist.

---

## 7. End-to-end deployment runbook (the happy path)

After all phases complete, this is the sequence to actually deploy:

```bash
# 1. Clone + configure
git clone <repo> transcript-rag && cd transcript-rag
cp .env.example .env
# Edit .env: WEBUI_SECRET_KEY=$(openssl rand -hex 32)
#           POSTGRES_PASSWORD=$(openssl rand -hex 32)
#           QDRANT_API_KEY=$(openssl rand -hex 32)

# 2. Infrastructure up
docker compose up -d
sleep 60
bash scripts/00_health_check.sh

# 3. Models + Qdrant
bash scripts/01_pull_models.sh
bash scripts/02_init_qdrant.sh
bash scripts/03_init_postgres.sh

# 4. Start Tantivy sidecar (host process; consider systemd in prod)
bash scripts/run_tantivy_server.sh &

# 5. Chunk transcripts (point to your actual data)
python -m ingestion.chunker_text /data/raw-transcripts /data/processed
# OR for JSON: python -m ingestion.chunker_json /data/whisperx-out /data/processed --format whisperx

# 6. Preflight on 100GB slice (mandatory before full run)
bash scripts/preflight.sh

# 7. Review preflight results. If green, go to full run; else fix issues first.

# 8. Full ingestion (run in tmux/screen — will take 8–14 days)
tmux new -s ingest
python -m ingestion.bulk_ingest_hardened --chunks-dir /data/processed
# Ctrl-B D to detach; reattach with `tmux a -t ingest`

# 9. Verify
python -m ingestion.verify_ingestion

# 10. Configure Open WebUI (one-time, via UI):
#     - Admin → Functions → install search_transcripts.py + analytics.py
#     - Workspace → Models → configure Qwen and DeepSeek per docs/model_config.md

# 11. Start using it at http://localhost:8080
```

---

## 8. Capacity and performance expectations

| Resource | Estimate (plain text input) |
|---|---|
| Chunks total | 250–400M |
| Qdrant storage (int8 quantized) | ~1.2 TB |
| Tantivy BM25 index | ~1.2 TB |
| Postgres `chunk_meta` | ~400 GB |
| **Total disk** | **~3 TB** |
| RAM (Qdrant quantized vectors in-RAM) | 24–40 GB |
| **Recommended host RAM** | **64 GB minimum, 128 GB ideal** |
| Ingestion wall-clock | 8–14 days |
| Query latency (dense + BM25 + rerank) | 300–700 ms |

If RAM is constrained, set `always_ram=False` in `qdrant_setup.py` — queries slow to 1–2 s but still functional.

---

## 9. Configuration reference (`.env`)

```bash
# Secrets (generate with `openssl rand -hex 32`)
WEBUI_SECRET_KEY=replace_me
POSTGRES_PASSWORD=replace_me
QDRANT_API_KEY=replace_me

# Service URLs (used by host-side scripts; containers use internal DNS)
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
TANTIVY_URL=http://localhost:8765
RERANKER_URL=http://localhost:7997
POSTGRES_DSN=postgresql://owui:${POSTGRES_PASSWORD}@localhost:5432/openwebui

# Collection / model names
QDRANT_COLLECTION=transcripts
EMBED_MODEL=bge-m3

# Data paths (HOST paths)
RAW_TRANSCRIPTS_DIR=/data/raw-transcripts
PROCESSED_CHUNKS_DIR=/data/processed
TANTIVY_DIR=/data/tantivy
DEAD_LETTER_DIR=./dead_letter

# Ingestion tuning
INGEST_BATCH_SIZE=32
INGEST_PER_FILE_TIMEOUT_SEC=1800
INGEST_MAX_FILE_SIZE_MB=500
```

---

## 10. Realistic accuracy ceilings

This is what Claude Code must NOT promise above:

| Query type | Plain text ceiling | JSON-input ceiling |
|---|---|---|
| Quote finding | 88–92% | 94–97% |
| Single-transcript Q&A | 85–90% | 88–92% |
| Cross-corpus topic summary | 70–80% | 72–82% |
| Analytics | 80–88% | 82–90% |
| **Weighted average** | **~85–88%** | **~88–91%** |

Push past these only via: re-transcribing to JSON, eval-driven tuning loops, HyDE/step-back query rewriting, or larger local models (would need a second GPU).

---

## 11. Failure modes Claude Code must handle, not hide

During ingestion (over 8–14 days):

| Failure | Mitigation in code |
|---|---|
| Malformed text/JSON file | Caught per-file, quarantined to `dead_letter/`, marked `failed` |
| Encoding errors | `read_text_robust` tries utf-8 → utf-8-sig → latin-1 |
| Ollama timeout | `@retry_with_backoff` retries up to 5 times |
| Qdrant connection drop | Same retry decorator |
| Tantivy commit failure | Caught; retry once; if persistent, mark batch failed |
| OOM on giant file | 500 MB size cap; oversize → `dead_letter/oversize/` |
| NaN/Inf embedding | Detected pre-upsert; mark failed |
| Power blip / SIGTERM | Graceful: commit Tantivy, close DB, log, exit 130 |
| RAM creep | psutil monitoring; warn 85%, abort 95% |
| Stuck file | `signal.SIGALRM` hard timeout 30 min/file |

What Claude Code MUST NOT do:
- ❌ Silently catch exceptions without logging.
- ❌ Promise "no errors at 5 TB scale" — this is mathematically false.
- ❌ Skip the verification step (`verify_ingestion.py`).
- ❌ Skip the preflight on 100 GB slice.

---

## 12. Testing strategy

**Unit tests** (fast, run on every commit):
- Chunker boundary cases (empty, huge, mid-sentence)
- Retry decorator (raises after max attempts, succeeds on transient)
- Encoding fallback chain
- Progress DB transitions (pending → in_progress → ok / failed)
- UUIDv5 stability (same input → same ID)

**Integration tests** (medium, run before each commit to main):
- End-to-end on `tests/fixtures/` (~10 files including corrupted ones)
- Idempotency: re-run produces zero new chunks
- Dead-letter quarantine works
- Verify script passes

**Manual tests** (run after Open WebUI tool install):
- Send 5 representative queries per query type, observe model behavior, confirm tool calls and citations.

**Preflight** (run once before full ingestion):
- 100 GB random slice, full pipeline, all metrics green.

---

## 13. Security and privacy

- All data stays on the host. No outbound calls except for Docker image pulls and Ollama model downloads (one-time setup).
- After initial setup, an `iptables OUTPUT DROP` rule (or equivalent) is *recommended* for paranoid deployments — document this in `docs/runbook.md` but do not enable by default.
- `.env` is gitignored. `QDRANT_API_KEY` enforced in prod (validated by `qdrant_setup.py` rejecting empty key when `ENV=prod`).
- Open WebUI bound to `0.0.0.0:8080` — document putting it behind a reverse proxy with auth for any networked deployment.

---

## 14. Open-source license verification

Claude Code must verify each dependency's license before adding it. Acceptable: MIT, Apache-2.0, BSD, MPL-2.0, LGPL. **Not acceptable**: anything with "Commercial license required for production" terms. Specifically green-lit:

| Component | License |
|---|---|
| Qdrant | Apache-2.0 |
| Ollama | MIT |
| bge-m3, bge-reranker-v2-m3 | MIT (BAAI) |
| Qwen 2.5 7B | Apache-2.0 (check Qwen Research License for specifics) |
| DeepSeek-R1-Distill-Qwen-7B | MIT |
| Tantivy | MIT |
| Infinity | MIT |
| Postgres | PostgreSQL License (BSD-like) |
| Redis 7-alpine | Redis Source Available v2 — 7.x branch is dual-licensed; alpine image distribution is fine for self-hosting (no third-party SaaS). Document in `docs/architecture.md`. If you need pure permissive, swap to `valkey:8-alpine` — also documented as a one-line `docker-compose.yml` change. |
| Open WebUI | BSD-3-Clause (with a small trademark notice) |
| FastAPI, Pydantic, requests, tenacity | MIT / BSD / Apache |

---

## 15. What's out of scope

- Web UI custom theming or branding.
- Multi-tenant access control beyond Open WebUI's built-in roles.
- HTTPS termination (do this with a reverse proxy you operate, not in this repo).
- Audio re-transcription pipeline (separate project — if you go this route, see Section 0 of the original spec for the `whisperX` command).
- GPU sharing across multiple users — assumes single-user or small-team usage.
- A separate frontend application — Open WebUI is the frontend.

---

## 16. Definition of Done (project-level)

The project is **DONE** when all of the following are true:

- [ ] All 11 phases passed acceptance.
- [ ] Preflight on 100 GB slice green.
- [ ] Full 5 TB ingestion completed with ≥95% of files in `ok` status.
- [ ] `verify_ingestion.py` reports ≥99.9% present in both Qdrant and Tantivy.
- [ ] Eval Hit@5 ≥ 85% for quote-finding queries on the golden set.
- [ ] A new user can sign up in Open WebUI, ask a quote-finding question, and get a sourced answer with citation in under 2 seconds.
- [ ] `docs/runbook.md` covers all daily operations.
- [ ] Repo has zero TODO comments without a tracking issue.

---

## 17. CLAUDE.md (companion file — also place at repo root)

Create a file named `CLAUDE.md` in the repo root with this content:

```markdown
# Claude Code Context for transcript-rag

## What this project is
A local-only RAG system over 5TB of Whisper transcripts. See PRD.md for full spec.

## Hard rules
1. Read PRD.md before any new phase. It is the single source of truth.
2. Open-source only. No OpenAI/Anthropic/Cohere/Pinecone/Voyage. Verify licenses (PRD §14).
3. Implement one phase at a time. Do not skip ahead.
4. After each phase: run its acceptance criteria. If they fail, fix before moving on.
5. Commit after each green phase with the message specified in PRD.
6. Never silently catch exceptions. Always log with context.
7. Never promise zero errors at scale. Promise "detected, logged, isolated, resumable."
8. Default to the simplest thing that satisfies the acceptance criteria. No premature abstraction.

## Where to start
Run the prompt in PRD §18.

## When stuck
- Re-read the relevant PRD phase, including its Implementation notes.
- Look at the original architecture context provided alongside this PRD (in `docs/_context/` if I drop it there).
- Ask the user before deviating from PRD locked decisions (§3).

## Conventions
- Python 3.11+, type hints required on public functions.
- Line length 100, ruff for lint + format.
- Pytest for tests; fixtures in tests/fixtures/.
- Logs to stdout AND ingest.log (rotating).
- Environment via `.env`; never commit it.
- Bash scripts: `set -euo pipefail` at the top.

## Service ports (all localhost)
- Open WebUI: 8080
- Ollama: 11434
- Qdrant: 6333 (REST), 6334 (gRPC)
- Tantivy sidecar: 8765
- Infinity reranker: 7997
- Postgres: 5432
- Redis: 6379

## File of last resort
If something is ambiguous, prefer PRD.md over your own judgment. If PRD.md is silent, ask.
```

---

## 18. The starter prompt for Claude Code

Run `claude` in the empty repo directory, then paste this:

> I'm building the project described in `PRD.md` (read it now, fully, before responding). My job is to oversee; your job is to implement.
>
> Rules:
> - Implement one phase at a time, starting with Phase 0.
> - For each phase: confirm you've read its section, then implement the deliverables, then run the acceptance criteria, then commit with the suggested message.
> - Do not move to the next phase until I say "go" or "next phase".
> - Use only open-source components — no paid APIs anywhere.
> - When you hit ambiguity, ask me. Do not guess on locked decisions (PRD §3).
>
> Before starting Phase 0:
> 1. Confirm you've read PRD.md (and CLAUDE.md if present).
> 2. List any prerequisites I should verify on my host (Docker, NVIDIA Container Toolkit, Python version, disk space).
> 3. Tell me what your first concrete file/command will be.

After Phase 0 passes, prompt with `next phase` (or `phase 1`, `phase 2`, etc.) to advance.

---

## 19. Appendix: How to operate Claude Code effectively for this build

A few practical tips for this specific project, based on the size and risk profile:

**Be the gating function, not the rubber stamp.** Claude Code is very capable but will sometimes try to optimize or add features beyond the PRD. For each phase, ask:
- "Did you implement exactly what the PRD asked for, no more?"
- "Did the acceptance criteria *actually* pass, or did you say they passed?"
- "Show me the test output."

**Run the tests yourself in a separate terminal.** Don't rely solely on Claude Code's reports of test pass/fail. After a phase claims green, run the tests yourself and confirm.

**Use git aggressively.** Branch per phase (`phase-1-infra`, `phase-2-models`, ...). Squash-merge to `main` only after acceptance passes. This means a bad phase can be reverted without dragging earlier work back.

**The preflight is non-negotiable.** When Claude Code finishes Phase 10, do not let it skip running the actual 100 GB preflight even if it argues it would take too long. The cost of catching issues in preflight is hours; the cost of catching them in production at 5 TB is days.

**For the long ingestion, use a separate tmux session.** Do not run `bulk_ingest_hardened.py` inside Claude Code's session. Launch it yourself, in tmux, with a clear name (`tmux new -s ingest`). Claude Code can monitor logs but should not own the long-running process.

**Keep the dead_letter directory in git LFS or excluded.** It will grow during ingestion. The PRD already gitignores `dead_letter/`; honor that.

**One LLM at a time during ingestion.** While ingestion runs, you should not also be chatting with the local LLMs through Open WebUI — they'll fight for GPU. Ingestion finishes, *then* you query.

---

End of PRD.
