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

## Prerequisites

- **Docker Engine + Compose v2.** On Windows, use Docker Desktop with WSL2
  integration enabled for your Ubuntu distro (Settings → Resources → WSL
  Integration).
- **NVIDIA Container Toolkit (`nvidia-ctk`).** Required for the `ollama` and
  `reranker` services, which reserve GPU devices in `docker-compose.yml`. Verify
  with `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
  before bringing up the stack. Install instructions:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- **NVIDIA driver visible to WSL2.** Run `nvidia-smi` inside your WSL distro
  before Phase 1; if it errors, update the host NVIDIA driver on Windows so
  `/dev/dxg` is exposed.

## License notes

All components are open-source (MIT / Apache-2.0 / BSD / similar). License
verification per component is documented in PRD §14.
