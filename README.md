# transcript-rag

A private, on-premise system that lets you **ask questions in plain language
across a large library of voice transcripts** and get cited answers back.
It runs entirely on your own machine — no data leaves the host, no paid
APIs, no cloud accounts required.

> If you are the operator setting this up for the first time, follow
> [Quick start](#quick-start) below. If you are a developer joining the
> project, jump to [How the code is organised](#how-the-code-is-organised).

## What it does, in plain English

You point the system at a folder of transcripts (whichever Whisper produced —
either JSON or plain `.txt`). It reads them, builds a searchable index, and
opens a small website on your computer where anyone with the dashboard
password can:

- **Ask a question** in Hindi or English and get a written answer with
  numbered citations `[1] [2] [3]` linking back to the exact passages.
- **Search transcripts** by speaker, source file, date, or any combination.
- **See analytics**: how often a term is mentioned, who talks about it most,
  which transcripts cover it.
- **Browse past conversations** in a sidebar (history is saved automatically).

The dashboard is at `http://localhost:8080` once the system is running.

## What's inside

| Piece | Job | Where it runs |
|---|---|---|
| Dashboard (React) | The website users actually click | inside the `rag-api` container at `/` |
| `rag-api` (FastAPI) | Receives questions, finds passages, writes the answer | container, port 8080 |
| Qdrant | Stores the "meaning" of every passage as vectors for semantic search | container, port 6333 |
| Tantivy | Keyword (BM25) search index; runs inside `rag-api` | in-process |
| bge-reranker (Infinity) | Re-ranks the top results so the best one is first | container, port 7997 |
| Ollama | Runs the language model that writes the final answer | container, port 11434 |
| Postgres | Stores chat history, transcript metadata, analytics counters | container, port 5432 |

Five Docker containers on one machine. One NVIDIA GPU (RTX 5070, 12 GB
VRAM) handles the embeddings, reranker, and answer model.

## Hardware you need

| | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04+ (or WSL2 on Windows 11) | Ubuntu 22.04+ |
| GPU | NVIDIA RTX 5070, 12 GB VRAM | same or better |
| RAM | 64 GB | 128 GB |
| Disk | ~3 TB free for indices, plus room for your transcripts | + 50% headroom |
| Software | Docker + NVIDIA Container Toolkit | + Docker Compose v2 |

## Quick start

This walks you from a fresh checkout to a running dashboard. Plan on
about two hours the first time (most of which is models downloading).

```bash
# 1. Get the code and create your settings file
git clone <repo-url> transcript-rag
cd transcript-rag
cp .env.example .env
```

Open `.env` in a text editor and fill in the three passwords at the top:

```text
POSTGRES_PASSWORD=...    # run: openssl rand -hex 32
QDRANT_API_KEY=...       # run: openssl rand -hex 32
DASHBOARD_PASSWORD=...   # any password you'll share with users (or leave empty for no login)
```

```bash
# 2. Start the system. First run builds the dashboard image (~5 minutes).
docker compose up -d --build

# 3. Sanity-check that all five services answered.
bash scripts/00_health_check.sh

# 4. Download the AI models (one-time, ~10 GB).
bash scripts/01_pull_models.sh

# 5. Set up the search indices (idempotent — safe to re-run).
bash scripts/02_init_qdrant.sh
bash scripts/03_init_postgres.sh

# 6. Process your transcripts into searchable chunks.
python -m ingestion.chunker_text /path/to/raw-transcripts /path/to/processed
# OR for whisperX JSON output:
# python -m ingestion.chunker_json /path/to/whisperx-out /path/to/processed

# 7. Run the preflight check on a 100 GB slice before the full ingestion.
bash scripts/preflight.sh
# Acceptance: ≥95% of files OK, eval Hit@5 ≥80%. Fix any failures here
# before committing to the multi-day full run.

# 8. Full ingestion. For ~5 TB of transcripts this takes 8–14 days,
#    so run it in a tmux session and detach.
tmux new -s ingest
python -m ingestion.bulk_ingest_hardened --chunks-dir /path/to/processed
# Ctrl-B then D to detach; reattach later with: tmux a -t ingest

# 9. Verify ingestion finished cleanly.
python -m ingestion.verify_ingestion --chunks-dir /path/to/processed

# 10. (Optional, after content tagging) build the file-level summary index
#     so the dashboard's "summaries" and "two-stage" scopes work.
python -m ingestion.build_summary_index
```

Open `http://localhost:8080` in a browser. Sign in with the
`DASHBOARD_PASSWORD` you set above and ask your first question.

## Day-to-day use

Most users only see the dashboard. The tabs across the top:

- **Search** — the main view. Type a question, pick scope (chunks /
  summaries / two-stage), and read the cited answer that streams back.
- **Analytics** — counts and rankings (mentions of a term, who talks
  about it most, which transcripts cover it).
- **Sidebar** — your past conversations. Click one to reopen it.

For phrasing tips that get noticeably better answers, see
[docs/how_to_ask.md](docs/how_to_ask.md).

## How the code is organised

```
transcript-rag/
├── rag_api/                 The FastAPI backend: receives queries, runs retrieval, writes answers
├── frontend/                The React dashboard (Vite + TypeScript); built into the rag-api image
├── ingestion/               Batch jobs that chunk transcripts and load them into Qdrant + Postgres
├── infra/
│   ├── postgres/            Schema + numbered migrations (001_hindi_fts.sql, 002_conversations.sql)
│   └── qdrant/              Collection bootstrap script
├── eval/                    Golden-query harness for measuring retrieval quality
├── scripts/                 Operator shell scripts (00_health_check.sh, 01_pull_models.sh, …)
├── services/
│   └── rag_api/Dockerfile   Multi-stage build for the dashboard + API container
├── tests/                   pytest suite (unit + integration)
├── docs/                    Operator and developer documentation
├── docker-compose.yml       The 5-container stack
├── .env.example             Template for the configuration file (copy to .env)
└── PRD.md                   The full product spec — the source of truth for design decisions
```

The dashboard and API ship as **one container** (`rag-api`). FastAPI
serves the React build at `/` and the API at `/api/*`. There is no
separate frontend server in production.

## Documentation map

The docs are layered. Pick the one that matches what you need:

| If you want to… | Read |
|---|---|
| Understand what was built and why | [PRD.md](PRD.md) (full spec, source of truth) |
| See the dashboard's API contract | [docs/dashboard.md](docs/dashboard.md) |
| Upgrade an older deployment | [docs/upgrade.md](docs/upgrade.md) |
| See what changed and when | [docs/changes.md](docs/changes.md) |
| Run the system day-to-day | [docs/runbook.md](docs/runbook.md) |
| Diagnose a failure | [docs/troubleshooting.md](docs/troubleshooting.md) |
| See the architecture diagram | [docs/architecture.md](docs/architecture.md) |
| Learn the dashboard as an end user | [docs/user_guide.md](docs/user_guide.md), [docs/how_to_ask.md](docs/how_to_ask.md) |
| Configure Claude Code on this repo | [CLAUDE.md](CLAUDE.md) |

## Running the tests

```bash
# Fast unit tests — no services required
pytest tests/unit -q

# Integration tests — docker compose stack must be up + schema applied
pytest tests/integration -q

# Retrieval quality eval against the live stack
python -m eval.run_eval --queries eval/golden_queries.yaml

# Frontend type-check + production build
cd frontend && npm run build
```

## Licensing

Every component is open-source (MIT / Apache-2.0 / BSD or similar) and
the system runs without any paid APIs. Per-component licence verification
lives in [PRD.md](PRD.md) §14. The one gated exception is
`CHAT_PROVIDER=openrouter`, which lets you A/B-test larger open-weight
models that don't fit the local GPU — when enabled, retrieved passages
leave the host, so leave it off unless that trade-off is acceptable. The
default deployment stays strictly local.
