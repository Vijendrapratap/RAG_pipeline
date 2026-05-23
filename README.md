# transcript-rag

Local-only, open-source RAG system over ~5 TB of Whisper-generated
transcripts. Hybrid retrieval (Qdrant dense + Tantivy BM25 +
bge-reranker-v2-m3) with **bilingual (Hindi/English) answer synthesis**, a
custom **FastAPI + React dashboard**, and Postgres-backed analytics.
**Single host, single GPU, no paid APIs.**

Spec: [PRD.md](PRD.md) (the original 13-phase build — ingestion / retrieval
internals). Dashboard track: [DASHBOARD.md](DASHBOARD.md) (Phases A–F that
replaced Open WebUI). Per-phase change log: [CHANGES.md](CHANGES.md).
Upgrading an existing deployment: [UPGRADE.md](UPGRADE.md). AI assistant
context: [CLAUDE.md](CLAUDE.md).

## What it does

You point it at a directory of Whisper-generated transcripts (either
whisperX JSON or plain `.txt`), it embeds and indexes them, then the
dashboard lets you ask questions and get bilingual, citation-grounded
answers over the corpus.

The dashboard speaks to a small FastAPI surface:

- **`POST /api/query`** — full RAG turn: hybrid retrieve + answer (SSE
  stream), with optional HyDE expansion and a deterministic filter
  extractor on the query text.
- **`POST /api/search`** — retrieval only, no LLM. Three scopes:
  `chunks`, `summaries` (one hit per file), `two_stage` (summary search
  picks files, then chunks within them).
- **`GET /api/analytics/*`** — mention counts, top speakers, transcripts
  ranking. Hindi-correct full-text search.
- **`GET /api/filters`** — distinct metadata for the dashboard
  dropdowns.

Citation-grounded answers reference passages as `[N]`, mapped to a
clickable citation list. See [DASHBOARD.md](DASHBOARD.md#api-endpoints)
for the full contract.

## Architecture in one paragraph

Browser → React dashboard → **rag-api** (FastAPI; Tantivy BM25
**in-process**) → Qdrant (1024-d int8 dense vectors, bge-m3) + Postgres
(`chunk_meta`/`file_meta` with GIN indexes) → fused with weighted RRF →
reranked by bge-reranker-v2-m3 (Infinity) → answer synthesised by Ollama
(qwen2.5:7b today; the fine-tuned 26B model later, one-line .env swap).
Five containers in one docker-compose stack on one RTX 5070 (12 GB VRAM).
Diagram + per-component detail: [docs/architecture.md](docs/architecture.md).

## Hardware

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04+ (or WSL2 on Windows) | Ubuntu 22.04+ |
| GPU | NVIDIA RTX 5070 (12 GB VRAM) | same |
| RAM | 64 GB | 128 GB |
| Disk (indices) | ~3 TB | + your source transcripts |
| Docker | Compose v2 | + NVIDIA Container Toolkit |

## Prerequisites

- **Docker Engine + Compose v2.** On Windows, Docker Desktop with WSL2
  integration enabled.
- **NVIDIA Container Toolkit (`nvidia-ctk`).** Required for the `ollama`
  and `reranker` services. Verify:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
  ```
  before bringing up the stack. [Install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **NVIDIA driver visible to WSL2.** `nvidia-smi` must work inside WSL.

## Quickstart

```bash
# 1. Clone + configure
git clone <repo> transcript-rag && cd transcript-rag
cp .env.example .env
# Edit .env, generate secrets:
#   POSTGRES_PASSWORD=$(openssl rand -hex 32)
#   QDRANT_API_KEY=$(openssl rand -hex 32)
#   DASHBOARD_PASSWORD=$(openssl rand -hex 16)   # empty = no auth (dev only)

# 2. Bring the stack up — builds the rag-api image on first run
docker compose up -d --build
bash scripts/00_health_check.sh         # five services should be ✅

# 3. Models + storage (idempotent)
bash scripts/01_pull_models.sh          # bge-m3, qwen2.5:7b
bash scripts/02_init_qdrant.sh          # creates `transcripts` collection
bash scripts/03_init_postgres.sh        # applies infra/postgres/analytics_schema.sql

# 4. Chunk your transcripts
python -m ingestion.chunker_text /data/raw-transcripts /data/processed
# OR for whisperX JSON:
# python -m ingestion.chunker_json /data/whisperx-out /data/processed

# 5. Preflight on a 100 GB slice (mandatory before full corpus run)
bash scripts/preflight.sh
# Acceptance: ≥95% files ok, eval Hit@5 ≥80%.

# 6. Full ingestion (run in tmux — 8–14 days for ~5 TB)
tmux new -s ingest
python -m ingestion.bulk_ingest_hardened --chunks-dir /data/processed
# Ctrl-B D to detach; reattach: tmux a -t ingest

# 7. Verify
python -m ingestion.verify_ingestion --chunks-dir /data/processed

# 8. (After Phase 13 enrichment) build the file-summary index for
#    scope=summaries / scope=two_stage retrieval:
python -m ingestion.build_summary_index

# 9. Open the dashboard
xdg-open http://localhost:8080
```

The dashboard is at `http://localhost:8080`. Sign in with
`DASHBOARD_PASSWORD` (empty = no login screen).

## Upgrading an existing (pre-dashboard) deployment

If you already ran the old Open WebUI–based stack, follow
[UPGRADE.md](UPGRADE.md). The one change you must apply on existing data
is the Hindi-correct Postgres FTS index rebuild
([`infra/postgres/migrations/001_hindi_fts.sql`](infra/postgres/migrations/001_hindi_fts.sql)).
Everything else is additive.

## Frontend (dashboard)

The React dashboard is a sibling project in [`frontend/`](frontend/).
Phase F builds it inside the rag-api image and serves it from `/`, so a
plain `docker compose up -d --build` ships both API and UI together.
For dashboard-only development:

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api → :8080
```

See [frontend/README.md](frontend/README.md).

## Documentation

- [DASHBOARD.md](DASHBOARD.md) — dashboard architecture, API contract,
  per-phase status (A–F). **Start here for the current setup.**
- [CHANGES.md](CHANGES.md) — per-phase change log (every file touched).
- [UPGRADE.md](UPGRADE.md) — operator handoff for existing deployments.
- [PRD.md](PRD.md) — original 13-phase build (ingestion + retrieval
  internals). Still the source of truth for everything pre-dashboard.
- [docs/architecture.md](docs/architecture.md), [docs/runbook.md](docs/runbook.md),
  [docs/troubleshooting.md](docs/troubleshooting.md),
  [docs/user_guide.md](docs/user_guide.md) — operator docs. These were
  written for the Open WebUI era; each carries a Phase F banner noting
  the parts that have changed.

## Tests

```bash
# Unit tests (no live services required)
pytest tests/unit -q

# Integration tests (require docker-compose stack up + Phase 6 schema applied)
pytest tests/integration -q

# Single eval pass (legacy harness — still uses open_webui_functions/)
python -m eval.run_eval --queries eval/golden_queries.yaml

# Frontend type-check + build
cd frontend && npm run build
```

## License notes

All components open-source (MIT / Apache-2.0 / BSD / similar). Per-component
license verification: PRD §14. **No paid APIs anywhere** — a hard
requirement, enforced phase by phase.
