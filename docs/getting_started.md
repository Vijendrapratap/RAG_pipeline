# Getting started — transcript-rag

A two-script first run. Everything is local, free, and open-source.
For day-to-day operations after first run, see [runbook.md](runbook.md).
When something breaks, see [troubleshooting.md](troubleshooting.md).

---

## TL;DR

Inside your **WSL2 Ubuntu** shell at the repo root:

```bash
bash scripts/setup_env.sh         # creates/fixes .env, generates secrets
bash scripts/start_pipeline.sh    # brings everything up, ~10–30 min the first time
```

When `start_pipeline.sh` prints **`✅ READY`** the stack is up and
indexed; you're ready to chunk and ingest your transcripts.

---

## Before you start — five things to check

1. **You're inside WSL2 Ubuntu** (not PowerShell). The repo lives at
   `/mnt/d/Vishvas-rag-pipeline/` from WSL.
2. **GPU is visible:** `nvidia-smi` shows your card.
3. **Docker can use the GPU:** `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` returns the GPU table.
4. **Disk:** `df -h .` shows roughly **3 TB free** for indices.
5. **Transcript data is reachable from WSL.** Your data lives on the
   Windows drive at `D:\GuruAudio\Output`, which appears in WSL as
   `/mnt/d/GuruAudio/Output`. Open it once to confirm:

   ```bash
   ls /mnt/d/GuruAudio/Output | head
   ```

   If your transcripts are somewhere else, edit `RAW_TRANSCRIPTS_DIR`
   (and `RAW_TRANSCRIPTS_BASE_DIR`) in `.env` to point at the right
   path — or just re-run `bash scripts/setup_env.sh` after editing,
   which will sanity-check the new path.

---

## What `setup_env.sh` does

- Creates `.env` from `.env.example` if missing.
- Generates real secrets (`WEBUI_SECRET_KEY`, `POSTGRES_PASSWORD`,
  `QDRANT_API_KEY`) using `openssl rand -hex 32` whenever they're
  blank or still `replace_me`.
- Fills in any missing required key with a default (see
  [`.env.example`](../.env.example) for the full list).
- Warns if your data paths don't exist on this host yet.

Idempotent — re-run any time. It never overwrites a real value.

```bash
bash scripts/setup_env.sh           # fix anything missing
bash scripts/setup_env.sh --check   # validate only; exit 1 if broken
```

---

## What `start_pipeline.sh` does

Eight steps, each idempotent, with a health gate between them:

| # | Step | Time |
|---|---|---|
| 1 | Check docker, GPU, NVIDIA Container Toolkit | <1s |
| 2 | Validate `.env` (calls `setup_env.sh`) | <1s |
| 3 | `docker compose up -d` (Qdrant, Ollama, reranker, Postgres, Redis, Open WebUI) | 30–60s |
| 4 | Poll each service until it responds | up to 2 min |
| 5 | Pull Ollama models: `bge-m3`, `qwen2.5:7b`, `deepseek-r1:7b` | 10–30 min first time, instant after |
| 6 | Create Qdrant `transcripts` collection + Postgres analytics schema | 1–2s |
| 7 | Launch Tantivy BM25 sidecar on host (port 8765, logs to `logs/tantivy.log`) | 2s |
| 8 | Final health check — six ✅ | <1s |

If any step fails, the script prints the failing command and exits
non-zero. Re-run after fixing — it picks up where it left off.

---

## After `✅ READY` — the manual steps

These are intentionally **not** auto-run, because each takes hours-to-days:

### 1. Chunk your transcripts (minutes to hours)

```bash
# Plain text:
python -m ingestion.chunker_text "$RAW_TRANSCRIPTS_DIR" "$PROCESSED_CHUNKS_DIR"
# whisperX JSON:
python -m ingestion.chunker_json "$RAW_TRANSCRIPTS_DIR" "$PROCESSED_CHUNKS_DIR"
```

Output: one `<stem>.chunks.json` per input in `$PROCESSED_CHUNKS_DIR`.
Failures land in `$PROCESSED_CHUNKS_DIR/_failed/`.

### 2. Preflight on a 100 GB slice (8–24 hours) — **mandatory**

```bash
bash scripts/preflight.sh
```

Full automation of the 5-step rehearsal: random slice → chunk → ingest
→ verify → eval. Must pass these gates before you trust a full run:

- ≥ 95% files `status='ok'`
- ≥ 99.9% chunk presence in Qdrant **and** Tantivy
- Eval Hit@5 ≥ 80% on quote-finding queries

### 3. Full ingestion (8–14 days, resumable)

```bash
tmux new -s ingest
python -m ingestion.bulk_ingest_hardened --chunks-dir "$PROCESSED_CHUNKS_DIR"
# Ctrl-B D to detach. Re-attach: tmux a -t ingest
```

The ingester is signal-safe and resumable from `ingest_progress.sqlite`.
Logs to stdout **and** `ingest.log` (rotating, 100 MB × 10).

After it finishes, refresh the BM25 index:

```bash
curl -X POST http://localhost:8765/reload
```

Then verify:

```bash
python -m ingestion.verify_ingestion --chunks-dir "$PROCESSED_CHUNKS_DIR"
```

### 4. Wire up Open WebUI (15 minutes of clicking)

Open <http://localhost:8080> (first sign-in becomes admin). Then:

1. [install_functions.md](install_functions.md) — paste the two tool
   files into Admin → Functions, set Valves to docker-network URLs
   (`http://qdrant:6333`, not `localhost`), attach to your chat models.
2. [model_config.md](model_config.md) — set context length, system
   prompt, and **Function Calling = native** for both models.

### 5. First query

In a chat with the function tools attached:

> *"find when someone mentioned [topic in your data]"*

You should see a tool call (`find_quote` / `search_transcripts`) and
chunk citations in the response. If the model answers from training
data instead of calling a tool, you forgot Function Calling = native —
see [troubleshooting](troubleshooting.md#open-webui-doesnt-see-the-function-tool).

---

## When you change anything that touches retrieval

Re-run the eval:

```bash
python -m eval.run_eval --queries eval/golden_queries.yaml
# Exits non-zero if quote Hit@5 < 0.80.
```

---

## Where to go next

- **Daily ops + backup** — [runbook.md](runbook.md) sections 1, 2, 4.
- **When things break** — [troubleshooting.md](troubleshooting.md).
- **Mental model** — [architecture.md](architecture.md) (query path + ingestion path diagrams).
