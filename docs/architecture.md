# Architecture

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

The system is one host, one GPU, one docker-compose stack plus one
host-side sidecar. Everything is open-source.

---

## Query path (chat → answer)

1. User asks a question in **Open WebUI**.
2. The chat model (Qwen 2.5 7B or DeepSeek-R1 7B, served by Ollama)
   decides whether to invoke a tool. The two Phase 7 tools are
   `search_transcripts` / `find_quote` (retrieval) and `count_mentions`
   / `list_transcripts_mentioning` / `top_speakers_for_topic` (analytics).
3. **For retrieval tools:** the tool embeds the query with bge-m3 via
   Ollama → fetches top-40 dense neighbors from Qdrant **and** top-40
   BM25 matches from the Tantivy sidecar → fuses with weighted RRF
   (k=60, bm25_weight=0.65 by default) → reranks the top candidates with
   bge-reranker-v2-m3 hosted by Infinity → returns the top-8 formatted
   chunks back to the chat model.
4. **For analytics tools:** the tool runs a single parameterized SQL
   query against Postgres (`chunk_meta` table with GIN indexes on
   `speakers` and `to_tsvector('english', text)`) with a 10s
   `statement_timeout`.
5. The chat model composes the user-facing answer, citing the chunks
   surfaced by retrieval (`self.citation = True` on the tool surfaces
   citation badges in the UI).

Redis is the Open WebUI cache; nothing in the retrieval path depends on
it for correctness.

---

## Ingestion path (raw transcripts → searchable)

1. **Chunkers** (host process) split each transcript into ~450-token
   chunks. `chunker_json.py` handles whisperX/whisper JSON with
   speaker-aware flushing and timestamps; `chunker_text.py` handles
   plain `.txt` with sentence-overlap. Each chunk gets a header naming
   source, timestamps, and speakers.
2. **`bulk_ingest_hardened.py`** reads the chunk JSON, embeds each batch
   via Ollama, and writes three places transactionally per file:
   - **Qdrant** — 1024-d int8-quantized vectors keyed by UUIDv5
     (deterministic from `chunk_id`), payload includes `source_file`,
     `speakers`, `start_sec`, `end_sec`, `text`.
   - **Tantivy** — BM25 index on `(chunk_id, source_file, speakers, text)`
     for keyword retrieval. The same on-disk dir is read by the sidecar.
   - **Postgres** (`chunk_meta`, `file_meta`) — analytics-friendly view
     of the same chunks plus per-file aggregates.
3. **Progress** is tracked in `ingest_progress.sqlite`; files marked
   `ok` are skipped on resume, `failed` files go to
   `dead_letter/<reason>/<filename>` with the reason in the SQLite row.
4. **`verify_ingestion.py`** samples N random files post-ingest and
   confirms each chunk's UUID exists in both Qdrant and Tantivy. Phase
   10 preflight requires ≥99.9% present.

---

## Per-component role and failure modes

### Open WebUI
**Role.** Chat UI; orchestrates LLM ↔ function-tool calls; handles
authentication and per-user model attachment. Backed by Postgres for
user/model state, Redis for caching.

**Failure modes.** Container crash → no chat at all (other services keep
working). Function tool not visible → see
[troubleshooting.md](troubleshooting.md). The container talks to other
services by docker-compose DNS name (`http://qdrant:6333`), not
`localhost`.

### Ollama
**Role.** Hosts bge-m3 (embeddings, always loaded, ~1.2 GB VRAM) and
the chat models (Qwen 2.5 7B / DeepSeek-R1 7B, ~5 GB VRAM each,
auto-unloaded after 10 min idle). Single REST API on `:11434`.

**Failure modes.** CUDA OOM under load — usually because bge-m3 + a chat
model + parallel-request slots exceed the 12 GB budget. Mitigations in
[troubleshooting §Ollama OOM](troubleshooting.md#ollama-oom-during-embedding):
drop `--batch-size`, set `OLLAMA_NUM_PARALLEL=1`, don't chat during
heavy ingestion.

### Qdrant
**Role.** Vector store. Collection `transcripts`, 1024-d, **int8 scalar
quantization** with `always_ram=true`, HNSW (`m=32`, `ef_construct=256`),
`on_disk_payload=true`. Stores the embedding and the chunk payload
(source, speakers, times, text).

**Failure modes.** Returns 503 when low on disk (needs ~1.5× scratch for
segment merges) or when the optimizer is far behind. See
[troubleshooting §Qdrant 503](troubleshooting.md#qdrant-returns-503-service-unavailable).
The int8 quantization makes vectors ~4× smaller but cuts recall by ~1–2%
at the default candidate pool sizes — acceptable for this workload
because the cross-encoder reranker fixes most of the gap.

### Tantivy
**Role.** Pure-Rust full-text BM25 index. The host process
`services/tantivy_server/tantivy_server.py` exposes it over HTTP on
`:8765` (`/health`, `/search`, `/reload`) so the Open WebUI container
can query it via `host.docker.internal`. The bulk ingester writes to the
same on-disk dir; the sidecar opens it **read-only**.

**Failure modes.** Lock contention if two writers point at the same dir
or a previous ingester crashed without closing
([troubleshooting §Tantivy lock error](troubleshooting.md#tantivy-lock-error-lockbusy--cannot-acquire-writer-lock)).
Stale searcher after ingestion — call `POST /reload` (see
[runbook §3.2](runbook.md#32-reload-after-additional-ingestion)). Index
corruption is rare in our use; if it happens, rebuild from chunk JSON by
re-running the ingester with `--reset-tantivy` (not implemented as of
Phase 4 — file an issue if you hit this).

### Infinity reranker
**Role.** Cross-encoder reranker (`BAAI/bge-reranker-v2-m3`). Open WebUI
and the Phase 7 search tool both call its `/rerank` endpoint with
`(query, [candidates])` and get back relevance scores. Reranking 40
candidates → 8 returned to the LLM is the default.

**Failure modes.** OOM under combined GPU pressure (chat + embed + rerank
concurrent). The Phase 7 `_rerank` falls back to RRF order on
`requests.RequestException` so the UI keeps working with degraded
quality.

### Postgres
**Role.** Two unrelated workloads on the same instance:
- **Open WebUI app database** (`DATABASE_URL`).
- **Analytics** (`chunk_meta`, `file_meta` — created in Phase 6). GIN
  index on `speakers TEXT[]` for speaker filters and on
  `to_tsvector('english', text)` for full-text counts.

**Failure modes.** Analytics returns 0 for everything → schema not
applied or ingester ran before Phase 6 schema existed
([troubleshooting §analytics returns 0](troubleshooting.md#bonus-analytics-returns-0-for-everything)).
Slow analytics queries → drop the 10s `statement_timeout` further or add
specific indexes for new query patterns.

### Redis
**Role.** Open WebUI session and model cache. Nothing in the retrieval
or ingestion paths depends on it.

**Failure modes.** Container down → Open WebUI logs cache misses but
keeps working. Safe to restart at any time.

### Host-side Python workers
**Role.** Everything that *writes* (the chunkers, the bulk ingester,
`verify_ingestion`, `retry_dead_letter`) runs on the host, not in
containers. Reasons: GPU embedding throughput is best with direct
host-side Ollama calls (no double-network hop), and the Tantivy index
dir needs a single writer that we can manage with systemd / tmux. The
sidecar (`tantivy_server.py`) is also host-side because it shares that
index dir.

**Failure modes.** Any of: signal-induced shutdown (130), OOM kill from
the host, GPU starvation. The bulk ingester is fully resumable —
[runbook §2](runbook.md#2-resuming-a-halted-ingestion).

---

## VRAM budget (12 GB total)

| Component | VRAM | Loaded when |
|---|---|---|
| bge-m3 embedding | ~1.2 GB | Always |
| bge-reranker-v2-m3 (Infinity) | ~1.2 GB | Always (pinned) |
| Qwen 2.5 7B Q4_K_M | ~5.0 GB | When chatting (auto-unloaded after 10 min) |
| KV cache @ 16k context | ~2.5 GB | During chat |
| **Total during query** | **~10 GB** | Within budget |
| **Total during ingestion-only** | **~2.4 GB** | Chat model unloaded |

Ingestion + chat concurrently can spike past 12 GB and trigger CUDA OOM
— this is why the [troubleshooting](troubleshooting.md) section says
"don't chat during heavy ingestion."

---

## Network surface (all bound to localhost on the host)

| Service | Port | Notes |
|---|---|---|
| Open WebUI | 8080 | The only one a human points a browser at |
| Ollama | 11434 | REST: `/api/embed`, `/api/generate`, `/api/tags` |
| Qdrant | 6333 (REST), 6334 (gRPC) | API key in `.env` |
| Tantivy sidecar | 8765 | Host process, reached from compose via `host.docker.internal` |
| Infinity reranker | 7997 | `/rerank` is the only endpoint we call |
| Postgres | 5432 | `owui` user; password in `.env` |
| Redis | 6379 | No auth; loopback only |

Inside the compose network, services use service names (`http://qdrant:6333`,
`postgresql://owui:...@postgres:5432/openwebui`). From the host, use
`localhost:<port>`. From an Open WebUI tool → the Tantivy sidecar:
`http://host.docker.internal:8765`.
