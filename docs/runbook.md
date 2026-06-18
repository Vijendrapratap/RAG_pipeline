# Runbook — daily ops, backup, recovery

> ⚠️ **Phase F note.** Sections that mention Open WebUI, Redis, or the
> Tantivy `:8765` sidecar describe the retired stack. The systemd unit
> for the host-side sidecar is no longer needed: Tantivy now runs
> in-process inside the `rag-api` container, reading the same
> `data/tantivy/` directory ingestion writes to. Backup (`pg_dump`,
> Qdrant snapshots, Tantivy rsync) is unchanged. See
> [dashboard.md](dashboard.md) for the current service layout.

Operational reference for a running `transcript-rag` deployment.
Architecture: [architecture.md](architecture.md). Failure modes:
[troubleshooting.md](troubleshooting.md). One-time Open WebUI function
install: [install_functions.md](install_functions.md). Per-model chat
settings: [model_config.md](model_config.md).

All paths and ports below assume the layout in PRD §5 and the defaults in
[`.env.example`](../.env.example). Substitute your own paths where
appropriate.

---

## 0. Operating & accessing the app (desktop)

> This section is the **user-facing front end** — how a non-technical person
> runs and uses the system. Sections 1+ below are operator/backend tasks.

### 0.1 What the app is

**Vishvas Foundation — Discourse Archive** is a small Windows desktop app
(`desktop/`) that launches and supervises the whole stack, then shows the search
dashboard in a normal window. The user never opens Docker, a terminal, or WSL —
they double-click the icon and search. It does not bundle its own UI; the
`rag-api` container serves the dashboard and the app points a window at it
(`http://localhost:8081` here, canonically `8080`).

### 0.2 Launching and accessing

- **Open the app** from the Start menu / desktop shortcut. After a short branded
  splash (cold start) or instantly (warm), the archive window opens.
- **Access** is the app window itself — no URL to type. As a fallback, the same
  dashboard opens in any browser at **http://localhost:8081** (8080 is taken by
  MiniTool ShadowMaker's `MTAgentService` on this box, so the override publishes
  8081; the launcher probes both automatically).
- **Login:** if `DASHBOARD_PASSWORD` is set in `.env`, the dashboard shows a
  login screen. The value lives in `.env` (gitignored) — never printed by
  `/api/health` (which only exposes `auth_required`). Empty = no auth (dev only).
- **Closing the window hides it to the tray** (stack stays warm); double-click
  the tray icon or *Open Vishvas Archive* to reopen instantly. Single-instance:
  re-clicking the icon just focuses the existing window.

### 0.3 The system tray menu

| Item | What it does |
|---|---|
| **Open Vishvas Archive** | Show/focus the window (same as double-clicking the tray icon) |
| **● status line** | Read-only health, refreshed every 15s from `/api/health` — "All systems online" or "N services offline" |
| **Restart services** | `docker compose restart` (in WSL) — first thing to try when the archive misbehaves |
| **Quit (keep services warm)** | Closes the shell but **leaves containers running** — model stays in VRAM, next open is instant. The normal way to exit |
| **Quit & stop services** | `docker compose stop` then quits — the **only** path that tears the stack down; next launch is a cold start |

### 0.4 How startup works (and the bug it fixes)

The launcher is the **single, authoritative starter**. All five services are set
`restart: "no"` in `docker-compose.override.yml`, so nothing auto-starts at boot.
When the app launches and the stack isn't already warm, it runs
`docker compose up -d` **inside the Ubuntu WSL distro**
(`wsl -d Ubuntu-24.04 … docker compose …`).

This matters because the data bind-mounts use absolute WSL paths
(`/home/pc/transcript-rag-data/…`, native ext4) that resolve to the **real data
only when compose runs from inside Ubuntu**. The old `restart: unless-stopped`
let Docker Desktop auto-start the stack at boot from its *own* VM, where those
paths are **empty** — so everything came up blank (no models, vectors, or
transcripts). The fix is two-part: `restart: "no"` stops the bad auto-start, and
the launcher always starts from WSL. It even self-corrects leftovers: if a
service answers but Ollama has **0 models**, `boot()` force-recreates from WSL
("Repairing the archive…").

Splash statuses you may see, in order: *Starting Docker → Starting services* (or
*Repairing the archive*) *→ Waking the archive → Warming the answer model*.

### 0.5 `/api/health` fields

`GET http://localhost:8081/api/health` (unauthenticated by design; the tray polls
it every 15s):

| Field | Meaning |
|---|---|
| `ok` | Go/no-go: chat backend **and** Qdrant reachable (minimum to answer) |
| `services` | `{ollama, qdrant, reranker}` reachability — drives the tray status line |
| `bm25_enabled` / `tantivy_docs` | Tantivy BM25 loaded; doc count (`0` = lexical index didn't mount) |
| `retrieval_backend` | Default pipeline (`hybrid`) |
| `pageindex_trees` | PageIndex section trees on disk |
| `auth_required` | `true` when `DASHBOARD_PASSWORD` is set (never exposes it) |
| `chat_provider` / `chat_model` | `ollama` (local) or `openrouter`; active answer model (`qwen3.5:9b`) |
| `embed_model` | Dense embedding model (`bge-m3`) |

### 0.6 Desktop troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Archive opens but shows **no transcripts** / empty results | Empty-mount: stack came up on Docker Desktop's VM context where the data dirs are empty | Let the app self-heal (it force-recreates from WSL when Ollama has 0 models). If not, *Quit & stop services* → relaunch. Confirm `/api/health` shows `tantivy_docs > 0` |
| First launch after reboot sits on the splash a long time | Cold start: Docker daemon down and/or pulling/loading the model | Wait (Docker ≤120s, health ≤240s). Every later launch is warm. If it errors, click *Try again* |
| Splash error: "Could not start the archive…" | Docker Desktop missing/wrong path, WSL distro unavailable, or project files not where expected | Verify Docker Desktop installed (or set `VISHVAS_DOCKER_DESKTOP`), the `Ubuntu-24.04` distro exists, and `docker-compose.yml` + `.env` are present; click *Try again* |
| Tray reads "● N services offline" | A container crashed or is slow | *Restart services*; re-check after ~15s. Still down → *Quit & stop services* → relaunch |
| `http://localhost:8080` won't load on this box | Port 8080 held by `MTAgentService`; dashboard is on 8081 | Use **http://localhost:8081**, or just launch via the desktop app (probes both) |
| First query slow though the window opened fast | Model wasn't resident yet (prewarm is best-effort) | No action; first query loads it, the rest are fast (`OLLAMA_KEEP_ALIVE=-1`) |

---

## 1. Daily ops

### 1.1 Log review

The bulk ingester logs to **stdout** and to `ingest.log` (rotating,
100 MB × 10 backups via `RotatingFileHandler`). Anything serious is
WARNING or ERROR.

```bash
# Live tail of an ongoing ingestion
tail -F ingest.log

# Quick health scan — only errors and warnings from the last hour
grep -E '^\S+ \S+ (ERROR|WARNING)' ingest.log | tail -50

# Per-file failures recorded today
grep -E 'quarantined|failed after' ingest.log | tail -50
```

Open WebUI, Qdrant, Ollama, Postgres, Redis, and the reranker all log to
their containers:

```bash
docker compose logs --tail=200 -f open-webui
docker compose logs --tail=200 -f qdrant
docker compose logs --tail=200 -f ollama
docker compose logs --tail=200 -f reranker
docker compose logs --tail=200 -f postgres
docker compose logs --tail=200 -f redis
```

The Tantivy sidecar (host process) logs to journald when run under
systemd (see §6) or to stdout when run via `scripts/run_tantivy_server.sh`.

```bash
journalctl -u tantivy-sidecar -f
```

### 1.2 Dead-letter triage

Files that exceed the per-file timeout, fail validation, or repeatedly
fail to embed/upsert are moved to
[`dead_letter/<reason>/<filename>`](../dead_letter/). Inspect daily:

```bash
# Counts per failure reason
ls dead_letter/ | while read d; do
  printf '%-30s %d\n' "$d" "$(ls dead_letter/$d | wc -l)"
done

# Look at the most recent failures
ls -lt dead_letter/*/ | head -20

# What does ingest_progress.sqlite say about them?
sqlite3 ingest_progress.sqlite "SELECT file, status, reason, attempts
  FROM ingest_status WHERE status='failed' ORDER BY updated_at DESC LIMIT 20;"
```

To re-attempt after fixing the root cause (e.g. you increased the per-file
timeout, fixed a corrupted file, restored a service):

```bash
# Re-run only the dead-lettered files
python -m ingestion.retry_dead_letter \
    --chunks-dir /data/processed --batch-size 32

# OR run the whole ingester with --retry-failed (skips ok, retries failed)
python -m ingestion.bulk_ingest_hardened \
    --chunks-dir /data/processed --retry-failed
```

### 1.3 Disk monitoring

Five things grow over time. Watch them:

```bash
# Compose-managed volumes
du -sh data/qdrant data/postgres data/redis data/open-webui data/ollama \
       data/infinity-cache

# Tantivy index (host path from .env)
du -sh "${TANTIVY_DIR:-/data/tantivy}"

# Logs + progress DB
du -sh ingest.log* ingest_progress.sqlite dead_letter/

# Raw + processed corpora
du -sh "${RAW_TRANSCRIPTS_DIR:-/data/raw-transcripts}" \
       "${PROCESSED_CHUNKS_DIR:-/data/processed}"
```

Rule of thumb: keep at least **20% headroom** on the partition holding
`data/qdrant` and the Tantivy index. Qdrant returns 503 well before it
runs out of space (segment merges need scratch room) — see
[troubleshooting.md](troubleshooting.md).

---

## 2. Resuming a halted ingestion

`bulk_ingest_hardened.py` is **idempotent and resumable.** It records
each input file's status in `ingest_progress.sqlite`. On restart:

- files with `status='ok'` → skipped
- files with `status='failed'` → skipped unless `--retry-failed`
- files with `status='in_progress'` (the process died mid-file) →
  reprocessed; UUIDv5 chunk IDs are stable so any partial Qdrant upserts
  are overwritten cleanly

There is nothing to "resume from" by hand — just re-run the same command:

```bash
# Same args you used originally
python -m ingestion.bulk_ingest_hardened \
    --chunks-dir /data/processed --batch-size 32
```

If you suspect a specific file's state is wrong (e.g. the SIGTERM landed
exactly between Qdrant upsert and `mark_ok`), force it back to pending:

```bash
sqlite3 ingest_progress.sqlite \
    "UPDATE ingest_status SET status='pending' WHERE file='whatever.chunks.json';"
```

After a clean stop (SIGTERM / SIGINT) the script exits 130. After an
unclean stop (OOM kill, host reboot), no special action is needed —
the next run picks up where the SQLite says it left off.

---

## 3. Tantivy: hosting, reload, lock-recovery

### 3.1 systemd unit (production)

The Tantivy sidecar runs on the **host**, not inside docker-compose,
because the Phase 4 bulk ingester writes to the same on-disk index
directory. The Open WebUI container reaches it via
`http://host.docker.internal:8765`.

```ini
# /etc/systemd/system/tantivy-sidecar.service
[Unit]
Description=Tantivy BM25 search sidecar (transcript-rag)
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=transcript-rag
Group=transcript-rag
WorkingDirectory=/opt/transcript-rag
Environment=TANTIVY_DIR=/opt/transcript-rag/data/tantivy
ExecStart=/opt/transcript-rag/.venv/bin/uvicorn \
    services.tantivy_server.tantivy_server:app \
    --host 0.0.0.0 --port 8765 \
    --workers 1 --log-level info
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

A copy of this unit lives at
[`services/tantivy_server/tantivy-sidecar.service`](../services/tantivy_server/tantivy-sidecar.service).
Install it with:

```bash
sudo cp services/tantivy_server/tantivy-sidecar.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tantivy-sidecar
sudo systemctl status tantivy-sidecar
journalctl -u tantivy-sidecar -f
```

For development, run it in the foreground instead:

```bash
bash scripts/run_tantivy_server.sh
```

### 3.2 Reload after additional ingestion

The sidecar opens the index in **read-only** mode and caches the
searcher. After `bulk_ingest_hardened.py` adds new documents you must
force the sidecar to reopen the index:

```bash
curl -X POST http://localhost:8765/reload
```

Verify the doc count after reload:

```bash
curl -s -X POST http://localhost:8765/search \
    -H 'Content-Type: application/json' \
    -d '{"query":"*","limit":1}' | jq '.total_estimated // .num_hits'
```

### 3.3 Recovering from a stale Tantivy lock

Only one writer may hold the index dir. If the bulk ingester crashed
mid-commit, you'll see `LockBusy` on the next run. Check for stale
processes first:

```bash
# Anyone holding the index?
sudo lsof "${TANTIVY_DIR:-/data/tantivy}" | grep -E '\.lock|meta\.json'

# Any orphaned bulk-ingest process?
pgrep -af bulk_ingest_hardened
```

If there is no live writer but the lock file remains, remove it:

```bash
ls -la "${TANTIVY_DIR:-/data/tantivy}"/.tantivy-writer.lock 2>/dev/null
rm -f "${TANTIVY_DIR:-/data/tantivy}"/.tantivy-writer.lock
```

---

## 4. Backup

All state is on the host filesystem. Stop write-side services before
snapshotting to get a consistent point in time:

```bash
# Pause ingestion (the bulk script exits cleanly on SIGTERM)
pkill -TERM -f bulk_ingest_hardened || true
sudo systemctl stop tantivy-sidecar

# Now snapshot each store. Reads stay live for Qdrant/Postgres if needed.
```

### 4.1 Qdrant — snapshot API

```bash
# Trigger a snapshot of the transcripts collection
curl -X POST -H "api-key: $QDRANT_API_KEY" \
    http://localhost:6333/collections/transcripts/snapshots

# List existing snapshots
curl -s -H "api-key: $QDRANT_API_KEY" \
    http://localhost:6333/collections/transcripts/snapshots | jq

# Snapshots land in data/qdrant/snapshots/transcripts/<name>.snapshot
# rsync to your backup target:
rsync -av data/qdrant/snapshots/ /backup/qdrant-snapshots/

# Full whole-storage backup (alternative): bring qdrant down first,
# then tar the volume.
docker compose stop qdrant
tar czf /backup/qdrant-$(date +%F).tar.gz -C data qdrant
docker compose start qdrant
```

Restore by copying the snapshot file back into
`data/qdrant/snapshots/<collection>/` and POSTing:

```bash
curl -X PUT -H "api-key: $QDRANT_API_KEY" \
    "http://localhost:6333/collections/transcripts/snapshots/recover" \
    -H 'Content-Type: application/json' \
    -d '{"location": "file:///qdrant/storage/snapshots/transcripts/<name>.snapshot"}'
```

### 4.2 Tantivy — rsync the index directory

The Tantivy index is a flat directory; once the sidecar (writer) is
stopped, it's just files.

```bash
sudo systemctl stop tantivy-sidecar
rsync -av --delete "${TANTIVY_DIR:-/data/tantivy}/" /backup/tantivy/
sudo systemctl start tantivy-sidecar
```

For a hot backup (sidecar still running, no ingester writing): the
sidecar opens the index read-only, so `rsync` of the dir is safe as long
as nothing else writes. Consider `rsync -av --link-dest` for daily
hardlink snapshots.

### 4.3 Postgres — `pg_dump`

```bash
# Logical dump (recommended — restorable across Postgres minor versions)
docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" postgres \
    pg_dump -U owui -d openwebui -Fc \
    > /backup/postgres-$(date +%F).dump

# Restore on a fresh container
docker exec -i -e PGPASSWORD="$POSTGRES_PASSWORD" postgres \
    pg_restore -U owui -d openwebui --clean --if-exists \
    < /backup/postgres-2026-05-18.dump
```

The `chunk_meta` and `file_meta` tables (Phase 6) plus the Open WebUI
application tables all live in this database, so the dump is the single
backup unit.

### 4.4 SQLite progress DB

It's tiny — just copy it:

```bash
cp ingest_progress.sqlite /backup/ingest_progress-$(date +%F).sqlite
```

You only need this for in-flight ingestions; once a run completes you
can throw it away and start fresh next time.

### 4.5 What does NOT need backup

- `data/ollama/` — re-pullable from Ollama registry (see §6).
- `data/infinity-cache/` — re-pulled from HuggingFace on first request.
- `data/redis/` — cache only; safe to lose.
- `data/open-webui/` — contains uploaded files & SQLite chat history.
  Back it up if you care about chat history; Postgres holds the user/model
  metadata.

---

## 5. Log rotation

The bulk ingester rotates `ingest.log` itself (`RotatingFileHandler`,
100 MB × 10 backups). Nothing to set up.

Docker container logs rotate per your Docker daemon config — by default
they grow unbounded. Configure once in `/etc/docker/daemon.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "10"
  }
}
```

Then `sudo systemctl restart docker` (warning: this restarts every
container on the host). Existing container log files are not retroactively
trimmed — you can `truncate -s 0` them while containers are stopped.

The Tantivy sidecar under systemd logs to journald; `journalctl
--vacuum-size=2G` caps the journal.

---

## 6. Updating models

bge-m3 (embeddings) and the chat models live in the Ollama volume. To
update:

```bash
# Pull latest tag (or a specific version)
docker exec ollama ollama pull bge-m3
docker exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M
docker exec ollama ollama pull deepseek-r1:7b-qwen-distill-q4_K_M

# Confirm what's loaded
docker exec ollama ollama list

# Restart Ollama so it drops cached model weights (optional — keep-alive
# expires in ~10 min anyway)
docker compose restart ollama
```

**⚠️ If you change the embedding model**, the existing Qdrant collection
is invalidated — the vectors were produced by the old model. You must:

1. Re-create the Qdrant collection with the new vector dim
   (`bash scripts/02_init_qdrant.sh` after editing dim if needed).
2. Re-run ingestion (`bulk_ingest_hardened.py`) — it will re-embed every
   chunk.

The chat model can be swapped freely; nothing on-disk depends on it.

The reranker (`BAAI/bge-reranker-v2-m3`, hosted by Infinity) is updated
by changing the `--model-id` in `docker-compose.yml` and running
`docker compose up -d --force-recreate reranker`. Infinity re-downloads
on first request.

---

## 7. Adding a new Open WebUI function tool

The pattern set by [`open_webui_functions/search_transcripts.py`](../open_webui_functions/search_transcripts.py)
and [`open_webui_functions/analytics.py`](../open_webui_functions/analytics.py):

1. **Write the tool.** A Python file with `class Tools:` and an inner
   `class Valves(BaseModel)` for user-configurable settings. Each public
   method on `Tools` becomes a callable function. Docstrings are
   important — the LLM reads them to decide when to call.

   ```python
   class Tools:
       class Valves(BaseModel):
           my_setting: str = "default"
       def __init__(self) -> None:
           self.valves = self.Valves()
           self.citation = True   # show citation badges in the UI

       def my_new_tool(self, arg: str) -> str:
           """One-line description the LLM uses for tool selection."""
           ...
   ```

2. **Install via the UI** (mirrors [install_functions.md](install_functions.md)):
   - Admin → Functions → "+ New Function" → paste the file → Save → toggle on.
   - Click the gear, set the Valves (especially URLs — use docker network
     names like `http://qdrant:6333`, not `localhost`, since Open WebUI is
     a container).

3. **Attach to chat models.** Workspace → Models → Edit → Tools → check
   the new tool → confirm Function Calling = native → Save.

4. **Smoke test.** Ask a chat that should obviously trigger the tool. If
   the model answers from memory instead, the docstring isn't selling it
   — rewrite the first line to be a clearer trigger.

5. **Add to eval.** If the tool has measurable correctness (returns
   facts, not just opinion), add 3–5 queries to
   [`eval/golden_queries.yaml`](../eval/golden_queries.yaml) and extend
   `eval/run_eval.py::_score_analytics` to parse the new tool's output
   format.

For a host-side tool that talks to data outside docker (like Tantivy),
use `http://host.docker.internal:<port>` as the URL valve. Add
`extra_hosts: ["host.docker.internal:host-gateway"]` to the `open-webui`
service in `docker-compose.override.yml` if your Docker setup doesn't
auto-resolve that hostname.
