# Installing the Open WebUI Function Tools

These instructions add the Phase 7 retrieval and analytics tools to Open
WebUI so chat models can call them. One-time setup, ~5 minutes per tool.

## Prerequisites

- Open WebUI is running at <http://localhost:8080> (see Phase 1).
- Phase 4 ingestion has populated Qdrant (`transcripts` collection) and the
  Tantivy index dir.
- Phase 5 Tantivy sidecar is running on the host at port 8765 (per the
  systemd unit in [runbook.md](runbook.md), or via
  `bash scripts/run_tantivy_server.sh`).
- Phase 6 Postgres schema is applied (`bash scripts/03_init_postgres.sh`).

## Step 1 — Install `search_transcripts`

1. Open <http://localhost:8080>, sign in as the admin user.
2. **Admin Panel → Functions → "+ New Function"**.
3. Paste the entire contents of
   [`open_webui_functions/search_transcripts.py`](../open_webui_functions/search_transcripts.py)
   into the editor.
4. Click **Save**, then toggle the function **on**.
5. Click the gear icon to open **Valves** and confirm:
   - `qdrant_url`: `http://qdrant:6333` (docker network name, NOT
     `localhost` — Open WebUI lives inside the same compose stack)
   - `qdrant_key`: paste the `QDRANT_API_KEY` value from your `.env`
   - `qdrant_collection`: `transcripts`
   - `ollama_url`: `http://ollama:11434`
   - `embed_model`: `bge-m3`
   - `reranker_url`: `http://reranker:7997/rerank`
   - `reranker_model`: `BAAI/bge-reranker-v2-m3`
   - `tantivy_proxy_url`: `http://host.docker.internal:8765`
   - `candidates_per_source`: `40`
   - `final_top_k`: `8`
   - `bm25_weight`: `0.65`

> **Why `host.docker.internal` for Tantivy and `qdrant`/`ollama` for the
> others?** Qdrant, Ollama, and the reranker run inside the same docker
> compose stack as Open WebUI, so they're reachable by service name. The
> Tantivy sidecar runs on the host because the same on-disk index dir is
> written by the bulk ingester — `host.docker.internal` is the docker-side
> alias for the host machine.

## Step 2 — Install `analytics`

1. Repeat the **+ New Function** flow.
2. Paste [`open_webui_functions/analytics.py`](../open_webui_functions/analytics.py).
3. Save + toggle on.
4. **Valves**:
   - `pg_dsn`: `postgresql://owui:<POSTGRES_PASSWORD>@postgres:5432/openwebui`
     where `<POSTGRES_PASSWORD>` is the value from `.env`. (Service name
     `postgres` resolves inside the docker network.)
   - `statement_timeout_ms`: `10000`

## Step 3 — Attach to chat models

For each chat model (Qwen 2.5 7B and DeepSeek-R1 7B):

1. **Workspace → Models → Edit**.
2. Under **Tools**, attach both `search_transcripts` and `analytics`.
3. Confirm **Function Calling = native** (not "default").
4. Save.

Phase 8 documents the full per-model settings (context length, system
prompt, etc.).

## Step 4 — Smoke test

In the chat UI with a tools-attached model:

- "find when someone mentioned the platform team" → should invoke
  `find_quote` and return chunks with citation badges.
- "how many times do we talk about latency?" → should invoke
  `count_mentions`.
- "search for content about distributed consistency" → should invoke
  `search_transcripts`.

## Troubleshooting

- **Tool not invoked / model answers from memory:** confirm the model has
  Function Calling = native and the tools are toggled on per-model. Some
  base models (older Qwen tags) don't support tool calling — switch to a
  newer instruct variant.
- **`ConnectionError` to `host.docker.internal`:** the Tantivy sidecar
  isn't running on the host, or Docker Desktop is on an older Linux setup
  that doesn't auto-resolve `host.docker.internal`. Add
  `extra_hosts: ["host.docker.internal:host-gateway"]` to the `open-webui`
  service in `docker-compose.override.yml` and restart.
- **`No results.`:** confirm ingestion has actually populated the
  `transcripts` collection. Run
  `curl -s -H "api-key: $QDRANT_API_KEY" http://localhost:6333/collections/transcripts | jq '.result.points_count'`
  — if it's 0, run the chunkers + bulk ingester first.
- **Analytics returns 0 for everything:** the `chunk_meta` table is empty.
  Confirm Phase 6 `scripts/03_init_postgres.sh` ran and the ingester logged
  `Postgres chunk_meta writes enabled` (NOT "disabled — table not yet
  created").
