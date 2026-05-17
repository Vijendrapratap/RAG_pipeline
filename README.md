# transcript-rag

Local-only, open-source RAG system over ~5 TB of Whisper-generated transcripts.
Hybrid retrieval (Qdrant dense + Tantivy BM25 + bge-reranker-v2-m3) behind Open
WebUI function tools, with Postgres-backed analytics. Single host, single GPU,
no paid APIs.

Full spec: [PRD.md](PRD.md). AI assistant context: [CLAUDE.md](CLAUDE.md).

## Status

Phase 0 (scaffolding) complete. See PRD §6 for phase progression.

## Quickstart

Per PRD §7 — bring up infrastructure, pull models, init storage, then ingest:

```bash
cp .env.example .env
# Generate secrets: openssl rand -hex 32
# Fill in WEBUI_SECRET_KEY, POSTGRES_PASSWORD, QDRANT_API_KEY

# Infra (Phase 1)
docker compose up -d
bash scripts/00_health_check.sh

# Models + storage (Phase 2, 6)
bash scripts/01_pull_models.sh
bash scripts/02_init_qdrant.sh
bash scripts/03_init_postgres.sh

# Ingestion (Phase 3, 4)
bash scripts/04_run_chunker.sh
bash scripts/05_run_ingestion.sh
bash scripts/06_verify.sh
```

Scripts under `scripts/` are added incrementally per phase.

## Hardware

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04+ (or WSL2 on Windows) | Ubuntu 22.04+ |
| GPU | NVIDIA RTX 5070 (12 GB VRAM) | same |
| RAM | 64 GB | 128 GB |
| Disk (indices) | ~3 TB | + your source transcripts |
| Docker | Compose v2 | + NVIDIA Container Toolkit |

## License notes

All components are open-source (MIT / Apache-2.0 / BSD / similar). License
verification per component is documented in PRD §14.
