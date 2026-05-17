# Claude Code Context for transcript-rag

## What this project is

A strictly local, open-source RAG (retrieval-augmented generation) system over ~5 TB of Whisper-generated transcripts. The user queries it through Open WebUI; chat models (Qwen 2.5 7B / DeepSeek-R1 7B) call custom function tools that perform hybrid retrieval (Qdrant dense + Tantivy BM25 + bge-reranker-v2-m3 reranking) and Postgres-backed analytics.

**Full spec is in `PRD.md`. Read it first, every time you start a new phase.**

## Hard rules

1. **Read PRD.md before each phase.** It is the single source of truth.
2. **Open-source only.** No OpenAI, Anthropic, Cohere, Pinecone, Voyage, Weaviate Cloud, etc. Every dependency must be on the green-lit list in PRD §14, or have its license verified before use.
3. **One phase at a time.** Implement only what the current phase asks for. No skipping ahead, no bundling phases.
4. **Acceptance criteria are mandatory.** After implementing a phase, run its acceptance steps and show me the output. Don't move on until they pass.
5. **Commit per phase.** Use the exact commit message in the PRD. One phase = one commit (or one squash-merged branch).
6. **Never silently swallow exceptions.** Every `except` must log with file/context/reason. Errors are detected and isolated, not hidden.
7. **Never promise zero errors at scale.** A 14-day ingestion over hundreds of millions of records will hit *something*. Promise: detected, logged, isolated, resumable.
8. **Default to the simplest thing that satisfies the criteria.** No premature abstraction, no extra config knobs, no "while I'm at it" refactors.
9. **Ask before deviating from PRD §3 locked decisions.** If you think Qdrant should be Weaviate, or bge-m3 should be E5, stop and ask first. Don't substitute silently.

## Where to start

Run the prompt in PRD §18. Begin with Phase 0.

## When stuck or uncertain

1. Re-read the current phase in PRD.md, including Implementation notes.
2. Check `docs/_context/` if I've dropped reference material there.
3. Ask me. Do not guess on architectural choices.

## Conventions

- **Python 3.11+** with type hints on all public functions.
- **Line length 100**, `ruff` for lint + format.
- **Pytest** for tests; fixtures in `tests/fixtures/`.
- **Logs to stdout AND `ingest.log`** (rotating, 100 MB × 10 backups).
- **Config via `.env`**; `.env` is gitignored. `.env.example` is committed.
- **Bash scripts**: `set -euo pipefail` at the top, executable bit set.
- **No comments explaining the obvious.** Comments explain *why*, not *what*.
- **Docstrings on every function tool exposed to Open WebUI** — the LLM reads them to decide when to call.

## Service ports (all localhost)

| Service | Port |
|---|---|
| Open WebUI | 8080 |
| Ollama | 11434 |
| Qdrant REST | 6333 |
| Qdrant gRPC | 6334 |
| Tantivy sidecar | 8765 |
| Infinity reranker | 7997 |
| Postgres | 5432 |
| Redis | 6379 |

## Container vs host names

- Inside docker-compose network: services reach each other by service name (`http://qdrant:6333`, `http://ollama:11434`).
- From host: use `localhost:<port>`.
- From the Open WebUI container to the host (where the Tantivy sidecar runs): `http://host.docker.internal:8765`.

## File of last resort

If something is ambiguous and the PRD is silent, ask me. Do not make assumptions on:
- Vector store choice
- Embedding model choice
- Quantization settings
- Chunking parameters
- Whether to add features beyond the PRD

You may make your own call on:
- Code structure within a single module
- Test scaffolding details
- Logging format details
- Helper function naming

## What "done" looks like for any phase

A phase is done when:
1. All files in its *Deliverables* list exist with the specified content/behavior.
2. All bullet points in its *Acceptance criteria* are satisfied — and I've seen the output proving it.
3. The commit is made with the specified message.
4. You can describe in 2 sentences what was built and how to verify it still works.

If any of these are weak, the phase is not done.
