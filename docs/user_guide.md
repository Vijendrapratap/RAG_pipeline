# User Guide — transcript-rag

> ⚠️ **Phase F note.** This guide predates the custom-dashboard cutover.
> Open WebUI and the Tantivy `:8765` sidecar are gone — the UI now lives
> at `http://localhost:8080/` (served by the `rag-api` container), and
> Tantivy runs in-process inside that container. Ingestion, indexing,
> chunking, and the eval harness still work as described. For the
> dashboard-side flow see [DASHBOARD.md](../DASHBOARD.md); for upgrading
> an existing deployment see [../UPGRADE.md](../UPGRADE.md).

End-to-end walkthrough from **"all code committed, nothing running"**
to **"asking questions in Open WebUI and getting cited answers."**

This guide covers what's left to do today. Setup mechanics
(`setup_env.sh`, `start_pipeline.sh`) are documented separately in
[getting_started.md](getting_started.md); daily ops are in
[runbook.md](runbook.md). When something breaks, see
[troubleshooting.md](troubleshooting.md).

---

## Where you are right now

| Layer | Status |
|---|---|
| Code (Phases 0–13) | ✅ Committed on `main` |
| `.env` | ✅ Points at `D:\GuruAudio\Output Transcribe` |
| Path parser | ✅ Handles your `_isolation` / `_model-...` / `turbo/` layout |
| Docker stack (Qdrant, Postgres, Redis, Ollama, reranker, Open WebUI) | ✅ All 6 containers running (per Docker Desktop) |
| Ollama models pulled (`bge-m3`, `qwen2.5:7b`, `deepseek-r1:7b`) | ❓ Verify — see Step 1 |
| Qdrant collection + Postgres schema | ❓ Verify — see Step 1 |
| Tantivy BM25 sidecar | ❓ Verify — runs on the **host**, not in Docker |
| Transcripts chunked (40 JSON files in `Output Transcribe/`) | ❌ 0 chunked |
| Bulk ingest run (chunks → embeddings → Qdrant + Tantivy + Postgres) | ❌ 0 ingested |
| Phase 13 content tags written | ❌ 0 tagged |
| Open WebUI function tools installed (`search_transcripts`, `analytics`) | ❌ Not installed |

> **About the ❓ rows:** if you ran `bash scripts/start_pipeline.sh` end-to-end,
> those are ✅. If you only ran `docker compose up -d` (or Docker Desktop
> brought the containers up for you), the containers exist but the
> models, schemas, and Tantivy sidecar are not yet set up — see Step 1.

All commands below run inside **WSL2 Ubuntu** at
`/mnt/d/Vishvas-rag-pipeline/`. Open it with:

```bash
wsl -d Ubuntu
cd /mnt/d/Vishvas-rag-pipeline
```

---

## Step 1 — Finish bringing the stack up

Your 6 Docker containers are running. Three things still need to be
verified (and probably run): Ollama models pulled, Qdrant + Postgres
schemas initialized, Tantivy sidecar started on the host.

**Easiest path — let `start_pipeline.sh` do everything it hasn't done.**
It's idempotent and will skip work that's already complete:

```bash
bash scripts/setup_env.sh        # only needed if .env has issues
bash scripts/start_pipeline.sh   # the big orchestrator
```

What it does (8 steps, each skipped if already complete): prerequisite
checks → docker compose up → service health waits → Ollama model pulls
→ Qdrant + Postgres init → Tantivy sidecar launch → final health check.
First-run cost is dominated by ~20 GB of Ollama model weights; re-runs
are seconds because the containers are already up.

When it prints **`✅ READY`**, you're done with Step 1.

**Manual verification** (or to check after the script runs):

```bash
# 1. Models pulled?  Expect bge-m3, qwen2.5:7b, deepseek-r1:7b in the list.
curl -s http://localhost:11434/api/tags | python3 -c "import json,sys; print('\n'.join(m['name'] for m in json.load(sys.stdin)['models']))"

# 2. Qdrant collection exists?  Expect "transcripts" in the result.
curl -s -H "api-key: $QDRANT_API_KEY" http://localhost:6333/collections

# 3. Postgres schema applied?  Expect tables: chunk_meta, file_meta, ingest_runs.
docker compose exec -T postgres psql -U owui -d openwebui -c "\dt"

# 4. Tantivy sidecar running on the host?  Expect HTTP 200.
curl -s http://localhost:8765/health

# Or just run the all-in-one check:
bash scripts/00_health_check.sh
```

Any failure → run the matching script directly:

| Failure | Fix |
|---|---|
| Models missing | `bash scripts/01_pull_models.sh` |
| Qdrant collection missing | `bash scripts/02_init_qdrant.sh` |
| Postgres schema missing | `bash scripts/03_init_postgres.sh` |
| Tantivy not responding | `nohup bash scripts/run_tantivy_server.sh > logs/tantivy.log 2>&1 &` |

---

## Step 2 — Chunk your transcripts (~1–2 min for 40 files)

Splits each whisper JSON into retrieval-sized passages and attaches the
Phase 12 path metadata (event, date, location, track type, season).

```bash
python -m ingestion.chunker_json \
  --recursive \
  --format whisper \
  --base-dir "$RAW_TRANSCRIPTS_BASE_DIR" \
  "$RAW_TRANSCRIPTS_DIR" \
  "$PROCESSED_CHUNKS_DIR"
```

What happens:
- Recursively walks `Output Transcribe/` and finds every `*.json`.
- Parses path metadata for each file — your `_isolation` / `_model-...`
  / `turbo/` layout is handled automatically.
- Writes one `<stem>.chunks.json` per input into
  `data/processed/` (set by `$PROCESSED_CHUNKS_DIR`).
- Failures land in `data/processed/_failed/<file>.error.txt` and the
  run continues — never aborts on one bad file.

Re-running is safe — files that already have a `.chunks.json` are
skipped. To force a re-chunk after tuning parameters, pass
`--no-skip-existing`.

**Sanity check before moving on:**

```bash
ls "$PROCESSED_CHUNKS_DIR" | grep -c '\.chunks\.json$'   # expect 40
ls "$PROCESSED_CHUNKS_DIR/_failed" 2>/dev/null           # expect: empty/missing
```

---

## Step 3 — Bulk ingest into Qdrant + Tantivy + Postgres (~10–30 min for 40 files)

Embeds each chunk with `bge-m3`, writes the vector to Qdrant, the text
to Tantivy (BM25), and the row to Postgres analytics. Resumable via
`ingest_progress.sqlite` — re-running skips chunks already written.

**Run inside `tmux` so the session survives any disconnect:**

```bash
tmux new -s ingest
python -m ingestion.bulk_ingest_hardened \
  --chunks-dir "$PROCESSED_CHUNKS_DIR" \
  --batch-size 32
# Detach: Ctrl-B then D.  Re-attach: tmux a -t ingest
```

For your 40-file corpus this takes 10–30 minutes depending on GPU.
Progress logs go to stdout **and** `ingest.log` (rotating, 100 MB ×
10 backups).

After it finishes, reload the BM25 index so the sidecar picks up the
newly-written files:

```bash
curl -X POST http://localhost:8765/reload
```

**Verify ingestion integrity** — exits non-zero if any chunk is missing
in any of the three stores:

```bash
python -m ingestion.verify_ingestion --chunks-dir "$PROCESSED_CHUNKS_DIR"
```

> **Why no preflight?** `scripts/preflight.sh` exists to rehearse a
> 100 GB random slice before committing to a multi-day ingest of 5 TB.
> Your current 40-file corpus is well under that threshold — skip
> preflight and just run the ingest directly. When you eventually point
> this at the full 5 TB, preflight becomes mandatory.

---

## Step 4 — Add Phase 13 content tags (optional, ~10–30 min for 40 files)

Per-file enrichment via Qwen 2.5 7B: event type, primary language,
topics, people/places/scriptures named, timing/location clues, and full
Hindi + English summaries. Lets you filter searches by content, not
just folder name.

**Always try a dry run first:**

```bash
bash scripts/06_enrich_tags.sh --limit 5 --dry-run
```

Then for real (resumable — skips already-tagged files):

```bash
bash scripts/06_enrich_tags.sh --limit 5     # 5 files first, sanity check
bash scripts/06_enrich_tags.sh               # tag everything untagged
```

What to watch on the first real run:
1. **First-file timing** — printed in the log. ~20 s per file on a
   4090, ~30 s on a 3090, much longer on CPU.
2. **`summary_hindi` is Devanagari** — open one in Postgres:
   ```bash
   docker compose exec postgres psql -U owui -d openwebui -c \
     "SELECT source_file, event_type, summary_hindi FROM file_meta WHERE tagged_at IS NOT NULL LIMIT 3;"
   ```
   Romanized Hindi means the prompt didn't take — re-run with
   `--retry-failed` after fixing.
3. **`dead_letter/tag_failures/` should be empty** — any file in here
   had bad JSON output from the model and needs retry.

You can run Step 4 in parallel with Step 5 if Step 3 is already done —
they touch different services.

---

## Step 5 — Install the Open WebUI function tools (one-time, ~10 min)

Open WebUI runs at <http://localhost:8080>. The first sign-in becomes
the admin user.

You need to paste **two Python files** into Admin → Functions and set
the Valves correctly. Full step-by-step (with the exact Valve values to
use) is in [install_functions.md](install_functions.md). Quick version:

1. **Functions** → **+ New Function** → paste
   `open_webui_functions/search_transcripts.py` → Save → toggle ON →
   gear icon → set Valves (`qdrant_url=http://qdrant:6333`,
   `tantivy_proxy_url=http://host.docker.internal:8765`, etc.).
2. Repeat for `open_webui_functions/analytics.py` (set `pg_dsn` to the
   compose-network DSN: `postgresql://owui:<password>@postgres:5432/openwebui`).
3. **Workspace → Models → Edit** for each chat model
   (`qwen2.5:7b`, `deepseek-r1:7b`):
   - Attach both functions
   - Set **Function Calling = native**
   - Bump context length per [model_config.md](model_config.md)

> **Critical gotcha:** if Function Calling is left at the default
> ("default" / "json"), the model will answer from training data
> instead of calling your tools — no citations, no real retrieval.

---

## Step 6 — Ask your first query

In a new chat with the `search_transcripts` tool attached, try a query
that mentions content from your transcripts:

> *"What did Swami ji say about meditation as a form of love?"*

You should see:
- A tool call to `search_transcripts(query="...")` in the chat UI
- 3–8 chunk citations with `[Source: 04 PRAVACHAN.json | 00:01:23 →
  00:03:45 | Speakers: Swami ji]` headers
- An answer that quotes from those chunks

After Step 4 (content tags) is done, you can also filter by content:

> *"Find a satsang where someone asked about karma yoga"*

The model will pass `event_type="satsang"` and `topics=["karma-yoga"]`
to the tool, and only those chunks come back.

**If the model answers without calling the tool:** Function Calling
isn't set to `native`. See [troubleshooting.md](troubleshooting.md#open-webui-doesnt-see-the-function-tool).

---

## What "done" looks like

After all six steps:
- `bash scripts/00_health_check.sh` → 6 green
- `ls data/processed | wc -l` → 40 (plus any `_failed/`)
- Qdrant collection size matches chunk count
- Open WebUI chat returns citations from your real transcripts
- Postgres `file_meta` rows have `tagged_at IS NOT NULL`

---

## Daily ops cheat sheet

```bash
# Add new transcripts after they've been processed by D:\GuruAudio:
python -m ingestion.chunker_json --recursive --format whisper \
  --base-dir "$RAW_TRANSCRIPTS_BASE_DIR" \
  "$RAW_TRANSCRIPTS_DIR" "$PROCESSED_CHUNKS_DIR"

python -m ingestion.bulk_ingest_hardened --chunks-dir "$PROCESSED_CHUNKS_DIR"
curl -X POST http://localhost:8765/reload
bash scripts/06_enrich_tags.sh

# Stop everything:
docker compose down
# Tantivy sidecar runs on the host, not in compose:
kill "$(cat logs/tantivy.pid)"

# Bring back up:
bash scripts/start_pipeline.sh
```

The chunker, ingester, and tagger are all **resumable** — they skip
work that's already done, so re-running after adding new files only
processes the new files.

---

## When something breaks

- **Health check fails** → [troubleshooting.md](troubleshooting.md)
- **Search returns nothing** → check `verify_ingestion` output,
  reload Tantivy, confirm Function Calling = native
- **Hindi summaries come out romanized** → re-tag with
  `bash scripts/06_enrich_tags.sh --retry-failed`
- **Hit@5 < 80% on an eval** → see [best_practices.md](best_practices.md)
  §11.5 for prompt-tuning checklist

---

## Where to dig deeper

- **Architecture** (query path, ingestion path) →
  [architecture.md](architecture.md)
- **Phase-by-phase implementation log** → [doc.md](doc.md) *(gitignored
  local file)*
- **Source of truth** → [PRD.md](../PRD.md)
