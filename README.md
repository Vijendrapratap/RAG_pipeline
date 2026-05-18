# transcript-rag

Local-only, open-source RAG system over ~5 TB of Whisper-generated
transcripts. Hybrid retrieval (Qdrant dense + Tantivy BM25 +
bge-reranker-v2-m3) behind Open WebUI function tools, with
Postgres-backed analytics. **Single host, single GPU, no paid APIs.**

Full spec: [PRD.md](PRD.md) (1258 lines, the source of truth). AI
assistant context: [CLAUDE.md](CLAUDE.md). Status: 12/12 phases
complete; full per-phase log in `doc.md` (gitignored).

## What it does

You point it at a directory of Whisper-generated transcripts (either
whisperX JSON or plain `.txt`), it embeds and indexes them, and then a
chat model in Open WebUI can answer questions over the corpus by calling
five function tools:

- **`search_transcripts(query, speaker?, source_file?, top_k?)`** —
  semantic search with optional filters.
- **`find_quote(partial_quote, top_k?)`** — BM25-heavy variant for
  finding exact phrasings.
- **`count_mentions(term, speaker?)`** — how often a term appears.
- **`top_speakers_for_topic(term, limit?)`** — who talks about it most.
- **`list_transcripts_mentioning(term, limit?)`** — which files cover it.

Results are returned with timestamp + speaker + source-file citations.

## Architecture in one paragraph

Open WebUI (chat) → calls tools → tools query Qdrant (1024-d int8 dense
vectors, bge-m3 embeddings) and Tantivy (BM25 sidecar on the host) →
fuse with weighted RRF → rerank top-40 with bge-reranker-v2-m3 hosted by
Infinity → return top-8 to the LLM. Analytics tools hit Postgres
(`chunk_meta` with GIN indexes). Everything runs in one docker-compose
stack on one RTX 5070 (12 GB VRAM). Diagram and per-component detail in
[docs/architecture.md](docs/architecture.md).

## Hardware

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04+ (or WSL2 on Windows) | Ubuntu 22.04+ |
| GPU | NVIDIA RTX 5070 (12 GB VRAM) | same |
| RAM | 64 GB | 128 GB |
| Disk (indices) | ~3 TB | + your source transcripts |
| Docker | Compose v2 | + NVIDIA Container Toolkit |

## Prerequisites

- **Docker Engine + Compose v2.** On Windows, use Docker Desktop with
  WSL2 integration enabled for your Ubuntu distro (Settings → Resources
  → WSL Integration).
- **NVIDIA Container Toolkit (`nvidia-ctk`).** Required for the
  `ollama` and `reranker` services. Verify with
  `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
  before bringing up the stack.
  [Install instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **NVIDIA driver visible to WSL2.** Run `nvidia-smi` inside your WSL
  distro before Phase 1; if it errors, update the host NVIDIA driver
  on Windows so `/dev/dxg` is exposed.

## Quickstart — first query in under 2 hours (assuming pre-pulled models)

```bash
# 1. Clone + configure
git clone <repo> transcript-rag && cd transcript-rag
cp .env.example .env
# Edit .env, generate secrets with:
#   WEBUI_SECRET_KEY=$(openssl rand -hex 32)
#   POSTGRES_PASSWORD=$(openssl rand -hex 32)
#   QDRANT_API_KEY=$(openssl rand -hex 32)

# 2. Infrastructure up
docker compose up -d
sleep 60
bash scripts/00_health_check.sh         # all six services should be ✅

# 3. Models + storage
bash scripts/01_pull_models.sh          # bge-m3, qwen2.5:7b, deepseek-r1:7b
bash scripts/02_init_qdrant.sh          # creates `transcripts` collection
bash scripts/03_init_postgres.sh        # applies infra/postgres/analytics_schema.sql

# 4. Tantivy sidecar (host process — production: use systemd, see runbook)
bash scripts/run_tantivy_server.sh &

# 5. Chunk your transcripts
python -m ingestion.chunker_text /data/raw-transcripts /data/processed
# OR for whisperX JSON:
# python -m ingestion.chunker_json /data/whisperx-out /data/processed

# 6. Preflight on a 100 GB slice (mandatory before full corpus run)
bash scripts/preflight.sh
# Acceptance: ≥95% files ok, eval Hit@5 ≥80%. If these fail, fix before
# committing the multi-day full run.

# 7. Full ingestion (run in tmux — takes 8–14 days for ~5 TB)
tmux new -s ingest
python -m ingestion.bulk_ingest_hardened --chunks-dir /data/processed
# Ctrl-B D to detach; reattach with `tmux a -t ingest`

# 8. Verify
python -m ingestion.verify_ingestion --chunks-dir /data/processed

# 9. One-time Open WebUI setup (in the browser at http://localhost:8080):
#    - Sign in as admin
#    - Install function tools: see docs/install_functions.md
#    - Configure chat models: see docs/model_config.md
```

After step 9, ask the chat model a question like *"find when someone
mentioned the platform team"*. You should see citation badges and chunks
returned from your transcripts.

## Documentation

- [PRD.md](PRD.md) — full product requirements + every locked design
  decision. Read this first when anything is unclear.
- [docs/architecture.md](docs/architecture.md) — diagram, per-component
  role + failure modes, VRAM budget, network surface.
- [docs/runbook.md](docs/runbook.md) — daily ops, log review,
  dead-letter triage, disk monitoring, ingestion resume, backup
  (Qdrant snapshots / Tantivy rsync / Postgres `pg_dump`), model
  updates, adding new tools.
- [docs/troubleshooting.md](docs/troubleshooting.md) — common failure
  modes: Ollama OOM, Qdrant 503, Tantivy lock, reranker timeouts, tools
  not visible.
- [docs/install_functions.md](docs/install_functions.md) — one-time UI
  steps to install `search_transcripts` and `analytics`.
- [docs/model_config.md](docs/model_config.md) — per-model context
  length, system prompt, function-calling settings.

## Build phases

Each phase has a single squash commit on `main`. To re-walk the build
history:

| # | Phase | What it built |
|---|---|---|
| 0 | Scaffolding | repo layout, `.env.example`, `pyproject.toml` |
| 1 | Docker Compose infra | the 6-service `docker-compose.yml` + health check |
| 2 | Model pulls + Qdrant init | bge-m3 + chat models, `transcripts` collection |
| 3 | Chunkers | `chunker_json.py` (whisperX) + `chunker_text.py` |
| 4 | Hardened bulk ingestion | `bulk_ingest_hardened.py` + retries, SIGALRM, dead-letter |
| 5 | Tantivy BM25 sidecar | `services/tantivy_server/` + systemd unit |
| 6 | Postgres analytics schema | `chunk_meta` / `file_meta` + GIN indexes |
| 7 | Open WebUI function tools | `search_transcripts.py` + `analytics.py` |
| 8 | Model config runbook | per-model settings + system prompt |
| 9 | Eval harness | `eval/run_eval.py` + 30 golden queries |
| 10 | Preflight on 100 GB slice | `scripts/preflight.sh` + reservoir sampler |
| 11 | Documentation | this README + the docs/ directory |

PRD §6 has the deliverables and acceptance criteria for every phase;
`git log --oneline` shows the commits.

## Tests

```bash
# Unit tests (no live services required)
pytest tests/unit -q

# Integration tests (require docker-compose stack up + Phase 6 schema applied)
pytest tests/integration -q

# Single eval pass against the live stack
python -m eval.run_eval --queries eval/golden_queries.yaml
```

## License notes

All components are open-source (MIT / Apache-2.0 / BSD / similar).
Per-component license verification documented in PRD §14. **No paid
APIs anywhere** — this is a hard requirement and is enforced phase by
phase.
# Vishwas_RAG-pipeline
