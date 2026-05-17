# Troubleshooting

Common failure modes you'll hit running transcript-rag. Each entry has a
symptom, root cause, and fix. Pair with the [runbook](runbook.md) for
ops procedures.

---

## Ollama OOM during embedding

**Symptom.** `bulk_ingest_hardened.py` logs repeated
`HTTPError: 500 Internal Server Error` from `/api/embed`, or the Ollama
container dies. `docker logs ollama` shows `CUDA out of memory` /
`cudaMalloc failed`.

**Cause.** bge-m3 plus the chat model plus Ollama's parallel request
slots exceeded the 12 GB VRAM budget. Most common when a chat model is
still resident (10 min keep-alive) and a 32-doc embedding batch lands on
top of it.

**Fix.**

1. Reduce the embedding batch:
   ```bash
   python -m ingestion.bulk_ingest_hardened \
       --chunks-dir /data/processed --batch-size 8
   ```
2. Force single-request serialization in Ollama. Edit `docker-compose.yml`:
   ```yaml
   ollama:
     environment:
       OLLAMA_NUM_PARALLEL: "1"     # was "2"
       OLLAMA_MAX_LOADED_MODELS: "1"  # was "2"
   ```
   Then `docker compose up -d ollama`.
3. While ingesting at scale, **don't chat** through Open WebUI — the
   7B chat model evicts bge-m3 from VRAM and embedding stalls.

The retry decorator (Phase 4 `utils/retries.py`) will back off and retry
transient OOMs automatically; if they're persistent, the file ends up in
`dead_letter/embed_failed/`.

---

## Qdrant returns 503 Service Unavailable

**Symptom.** Ingestion logs `qdrant_client.http.exceptions.UnexpectedResponse: 503`,
or `/collections` returns 503 from a browser.

**Cause.** Two common ones:

1. **Disk pressure.** Qdrant needs scratch space for segment merges
   (typically 1.5× the on-disk index size). When the volume holding
   `data/qdrant/` drops below that headroom, Qdrant rejects writes.
2. **Segment merge backlog.** A long ingestion overran the merger;
   logs show `optimizer: pending segments: N` with N growing.

**Fix.**

1. Free disk:
   ```bash
   df -h data/qdrant
   # If < 20% free: prune old snapshots, move data/qdrant to a larger volume,
   # or shrink the working set (delete an old collection).
   ```
2. Throttle ingestion to let the optimizer catch up:
   ```bash
   # Stop the ingester
   pkill -TERM -f bulk_ingest_hardened
   # Watch the segment count drop
   curl -s -H "api-key: $QDRANT_API_KEY" \
       http://localhost:6333/collections/transcripts | jq '.result.segments_count'
   # Resume when stable
   ```
3. Lower the parallel optimizer count if you have CPU pressure:
   ```yaml
   qdrant:
     environment:
       QDRANT__STORAGE__OPTIMIZERS__MAX_OPTIMIZATION_THREADS: "2"
   ```

---

## Tantivy lock error (`LockBusy` / `Cannot acquire writer lock`)

**Symptom.** `bulk_ingest_hardened.py` exits at startup or mid-run with
`LockBusy` or `Cannot acquire lock`. Or the Tantivy sidecar refuses to
start with a similar message.

**Cause.** Tantivy allows exactly one writer per index directory. A
previous bulk-ingest process crashed without dropping the lock, or two
ingesters are pointing at the same `TANTIVY_DIR`, or the sidecar opened
the dir in write mode (it shouldn't — verify it uses `Index.open`, not
`Index.create_in_dir`).

**Fix.**

1. Identify the holder:
   ```bash
   pgrep -af bulk_ingest_hardened
   sudo lsof "${TANTIVY_DIR:-/data/tantivy}" | grep -E '\.lock|meta\.json'
   ```
2. If there's a live process, kill it cleanly:
   ```bash
   pkill -TERM -f bulk_ingest_hardened
   ```
3. If no holder remains, remove the stale lock:
   ```bash
   rm -f "${TANTIVY_DIR:-/data/tantivy}"/.tantivy-writer.lock
   ```
4. Restart the ingester. The sidecar (read-only) does not hold the
   writer lock, so it can run concurrently with ingestion — just call
   `POST /reload` after the ingester commits.

This was the regression caught between Phase 4 and Phase 5 — the original
`TantivyWriter.commit()` reopened a new writer immediately, racing the
not-yet-released old one for the lock. The current implementation closes
the writer on shutdown only.

---

## Reranker timeout / 502 from Infinity

**Symptom.** `search_transcripts` falls back to RRF order (the Phase 7
graceful degrade path), or you see `requests.exceptions.ReadTimeout` /
`502 Bad Gateway` from `http://reranker:7997/rerank`.

**Cause.** Infinity OOMed (bge-reranker-v2-m3 needs ~1.2 GB pinned), or
the GPU is starved by the chat model + embeddings concurrently.

**Fix.**

1. Check Infinity is alive:
   ```bash
   curl -s http://localhost:7997/health
   docker compose logs --tail=100 reranker | grep -iE 'error|oom|cuda'
   ```
2. Restart it:
   ```bash
   docker compose restart reranker
   ```
3. If it keeps OOMing, lower the batch:
   ```yaml
   reranker:
     command: >
       v2
       --model-id BAAI/bge-reranker-v2-m3
       --port 7997
       --engine torch
       --batch-size 16    # was 32
   ```
4. As a workaround until you have headroom, lower
   `candidates_per_source` on the `search_transcripts` Valves (e.g.
   20 → fewer pairs sent to the reranker per query).

`search_transcripts._rerank` is intentionally fault-tolerant — if the
reranker is unreachable it returns RRF order so the user still gets
results. Quality drops but the UI doesn't break.

---

## Open WebUI doesn't see the function tool

**Symptom.** Chat with a tool-attached model and ask a question that
should trigger the tool; the model answers from training data instead,
or replies "I don't have access to that information."

**Cause (in rough likelihood order).**

1. The tool is installed but not **toggled on** in Admin → Functions.
2. The tool is on, but not **attached to this specific model**
   (Workspace → Models → Edit → Tools).
3. The model has Function Calling set to `default` instead of `native`.
   Some older Qwen / DeepSeek tags don't support native tool calling at
   all — switch to the instruct variant pinned in
   [`docs/model_config.md`](model_config.md).
4. Valves point to the wrong host. Inside the Open WebUI container,
   docker network names work (`http://qdrant:6333`, `http://ollama:11434`);
   `localhost` resolves to the container itself, not the host.

**Fix.** Walk the four checks above. If everything looks right, watch the
Open WebUI logs while you send the query:

```bash
docker compose logs -f open-webui | grep -iE 'tool|function|search_transcripts'
```

If you see the function being called but no results, jump to:

- **`No results.`** → ingestion didn't populate the collection. Run the
  health check from the runbook:
  ```bash
  curl -s -H "api-key: $QDRANT_API_KEY" \
      http://localhost:6333/collections/transcripts | jq '.result.points_count'
  ```
- **`ConnectionError` on `host.docker.internal`** → the Tantivy sidecar
  isn't running. Start it (`bash scripts/run_tantivy_server.sh`) or fix
  the systemd unit. If your Docker version doesn't auto-resolve that
  host, add the override below to `docker-compose.override.yml`:
  ```yaml
  services:
    open-webui:
      extra_hosts:
        - "host.docker.internal:host-gateway"
  ```
  Then `docker compose up -d open-webui`.

---

## Bonus: analytics returns 0 for everything

**Symptom.** `count_mentions("anything")` consistently returns 0.

**Cause.** The `chunk_meta` table is empty. Either Phase 6 schema wasn't
applied, or the ingester logged
`Postgres chunk_meta writes disabled — table not yet created` because it
ran before `scripts/03_init_postgres.sh`.

**Fix.**

```bash
# Apply schema (idempotent)
bash scripts/03_init_postgres.sh

# Confirm the tables exist
docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" postgres \
    psql -U owui -d openwebui -c "\dt chunk_meta file_meta"

# Re-run ingestion (existing files are skipped via SQLite, but you'll need
# to clear progress to backfill chunk_meta for already-ingested files):
sqlite3 ingest_progress.sqlite \
    "UPDATE ingest_status SET status='pending';"
python -m ingestion.bulk_ingest_hardened --chunks-dir /data/processed
```

---

## Bonus: ingestion exits 130 unexpectedly

**Symptom.** `bulk_ingest_hardened.py` returns exit code 130 without
your sending SIGTERM/SIGINT.

**Cause.** Something else sent the signal. Common culprits: the OOM
killer (host RAM exhausted; check `dmesg | tail`), a parent shell exiting
(if you didn't run under `tmux` or `nohup`), or a service-manager
timeout.

**Fix.**

1. Always run long ingestions under `tmux` or `screen`:
   ```bash
   tmux new -s ingest
   python -m ingestion.bulk_ingest_hardened --chunks-dir /data/processed
   # Ctrl-B D to detach
   ```
2. If `dmesg` shows `Killed process X (python)` with a memory line,
   you're hitting host RAM limits. The Phase 4 memory monitor aborts at
   95% RAM use to avoid this — if you're still being killed externally,
   drop `--batch-size` and/or close other heavy processes.
3. The script is fully resumable — just rerun the same command. See
   [runbook §2](runbook.md#2-resuming-a-halted-ingestion).
