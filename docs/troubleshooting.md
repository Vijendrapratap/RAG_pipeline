# Troubleshooting

> ⚠️ **Phase F note.** Entries that mention Open WebUI, Redis, or the
> Tantivy `:8765` sidecar describe the retired stack. The dashboard now
> serves on `:8080` via the `rag-api` container; Tantivy runs in-process
> inside it. Symptoms for Ollama / Qdrant / Postgres / Infinity below are
> still accurate.

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

## Dashboard 502s + "relation does not exist" after a Windows / Docker restart

**Symptom.** After rebooting Windows, restarting Docker Desktop, or
returning from sleep, the dashboard breaks:

- `/api/query`, `/api/analytics/*` return `502 Bad Gateway`
- `/api/health` reports `bm25_enabled: false` and `tantivy_docs: 0`
- Postgres errors: `relation "file_meta" does not exist` or
  `relation "chunk_meta" does not exist`
- Sidebar shows "no chats yet" even though there were entries yesterday

Inside the containers, the bind-mounted dirs look fresh-initialised
(empty Qdrant collections, fresh `initdb`-style Postgres files dated
today, empty Ollama `models/manifests/`). **But on the WSL host,
`~/transcript-rag-data/{postgres,qdrant,ollama}` is still fully
populated** with yesterday's data and old timestamps.

**Cause.** Docker Desktop on Windows + WSL2 + bind mounts has a startup
race. The daemon runs in the `docker-desktop` WSL distro; bind-mount
sources live in `Ubuntu-24.04` (the user's home). On a cold start, the
containers can come up *before* the 9P bridge between those two distros
is fully re-established. Each container then sees an empty version of
its mount target and acts on it:

- **Postgres** finds no `PG_VERSION`, but by the time it would run
  `initdb` the bridge has partially recovered — it ends up in a
  confused half-loaded state where some tables are visible and others
  aren't. Analytics queries hit the missing ones and 502.
- **Qdrant** starts with an empty `storage/collections/` and writes a
  fresh `raft_state.json` on top, clobbering the host-side directory.
  Points show up as zero.
- **Ollama / Tantivy** bind mounts may not reconnect at all — the
  container operates on its own writable layer, blind to the host data.
  `tantivy_docs: 0`, `manifests/` is empty.

**Fix.** Recreate the containers — fresh containers re-establish fresh
bind-mount connections, and they pick up the populated host dirs.

```bash
# Run from inside WSL Ubuntu-24.04, NOT Git Bash
cd /mnt/d/Vishvas-rag-pipeline
docker compose down                    # NOT `down -v` — see warning below
docker compose up -d
sleep 60                                # let reranker reload its model
curl http://localhost:8080/api/health   # tantivy_docs should match Qdrant points_count
```

If `docker compose down` errors with *"service 'open-webui' has neither
an image nor a build context"*, your `docker-compose.override.yml` still
references services that no longer exist in the base compose. Edit the
override to drop the dead service sections (and remember `!override`,
not `!reset`, when replacing the volumes list).

**Verify the data really is intact before assuming the worst.** From WSL
Ubuntu, an Alpine helper container can read the bind-mount source
without needing `sudo`:

```bash
docker run --rm -v /home/$USER/transcript-rag-data:/data alpine sh -c \
  'du -sh /data/postgres /data/qdrant /data/ollama; \
   cat /data/postgres/PG_VERSION; \
   ls /data/qdrant/collections; \
   find /data/ollama/models/manifests -type f | head'
```

Expect: ~10 GB of Ollama, several hundred MB of Qdrant, the `transcripts`
collection, and a populated `PG_VERSION`. If any of those are missing
from the host as well, the data really is gone — recovery means
re-ingestion.

> ⚠️ **Never run `docker compose down -v`** in this stack. The `-v` flag
> removes named volumes. Bind mounts (everything migrated to
> `~/transcript-rag-data/`) are NOT affected by `-v`, but if any named
> volume ever creeps back into the compose file `-v` will wipe it
> silently. Plain `down` is enough — it removes containers and the
> network, never the data.

---

## Dashboard / frontend changes don't appear after edits

**Symptom.** You edited a file under `rag_api/` or `frontend/src/` and
ran `docker compose up -d`, but the running stack is still on the old
behaviour:

- New API endpoint returns `404 Not Found`
- New React component (e.g. a `Sidebar`) doesn't render
- `POST /api/history` succeeds in `app.py` source but the container says
  *"Not Found"*
- `/openapi.json` doesn't list the new route
- Browser's loaded `<script src="/assets/index-XXXX.js">` hash hasn't
  changed

**Cause.** `docker compose up -d` reuses the cached image. The `rag-api`
image is multi-stage and bakes BOTH the Python module AND the built
React `dist/` bundle at build time (`services/rag_api/Dockerfile`).
Without the `--build` flag, neither updates — the container starts
yesterday's snapshot of your code.

This is most painful when you've added something brand-new (a route, a
component, a SQL migration) and the on-disk file is there but the
container has never seen it.

**Fix.** Rebuild only `rag-api` — Ollama, Qdrant, Postgres, and the
reranker stay running and untouched:

```bash
cd /mnt/d/Vishvas-rag-pipeline
docker compose up -d --build rag-api
```

First build is slow (~5–10 min: Node `npm ci` + `vite build`, then
Python `pip install`). Subsequent rebuilds with only a Python change are
fast because Docker caches the Node stage.

**For frontend-only iteration**, skip the rebuild loop entirely — run
the Vite dev server on `:5173`, which hot-reloads on edit and proxies
`/api` to the container on `:8080`. See [frontend/README.md](../frontend/README.md):

```bash
cd frontend
npm install                          # one-time
npm run dev                          # http://localhost:5173
```

You only need `--build` to ship the change into the dashboard container.

**How to confirm the rebuild took.** Three quick checks:

```bash
# 1. Image build time should match "now"
docker image inspect vishvas-rag-pipeline-rag-api --format '{{.Created}}'

# 2. New file present in the container
docker exec rag-api ls /app/rag_api/    # should include your new module

# 3. New route in the OpenAPI spec
curl -s http://localhost:8080/openapi.json | grep -oE '"/api/[^"]*"' | sort -u
```

If a frontend route still serves stale HTML, hard-refresh the browser
(`Ctrl+Shift+R`) to bypass cached asset hashes — the new bundle has a
new hash but `index.html` may be locally cached.

**Note on migrations.** SQL migrations under
`infra/postgres/migrations/` are NOT applied by `docker compose up`.
Run them manually against the live DB:

```bash
docker exec -i postgres psql -U owui -d openwebui \
  < infra/postgres/migrations/002_conversations.sql
```

The same applies to one-shot Python init scripts under `scripts/`.

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
