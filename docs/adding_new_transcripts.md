# Adding New Transcripts to the Search System

A copy-paste guide for loading **newly transcribed audio** into the search
system. It is **incremental** — it processes only the new files and skips
everything already loaded. Running it again does no harm.

> Do this whenever WhisperX has produced new tracks in
> `D:\Transcription whisperx\Output`.

---

## The simple way — one command

Open the **Ubuntu** app (WSL) from the Start menu and run:

```bash
cd /mnt/d/Vishvas-rag-pipeline
bash scripts/add_transcripts.sh
```

That's it. The script does everything: it finds new tracks anywhere under
`D:\Transcription whisperx\Output`, prepares them, loads them into the search
system, refreshes the app, and verifies. You'll see four labelled steps and end
with:

```
==> Done. New transcripts are live.
    Open the Vishvas app, or browse http://localhost:8081 and search.
```

**Only want one folder?** Pass it as an argument (still incremental):

```bash
bash scripts/add_transcripts.sh "/mnt/d/Transcription whisperx/Output/Live Masters 2010_isolation"
```

That's the whole job. The sections below are only for when you want to
understand it, run the steps by hand, or fix a problem.

---

## Before you start (one-time check)

The search system must be **running**. It no longer starts on its own with the
PC — it starts when you **open the Vishvas app** (which is the recommended way),
or you can start it from Ubuntu:

```bash
cd /mnt/d/Vishvas-rag-pipeline
docker compose ps        # should list 5 services as "Up"
docker compose up -d     # run this if they are not up, then wait ~1 min
```

The five services are `rag-api`, `ollama`, `qdrant`, `postgres`, `reranker`.

> Why it doesn't auto-start anymore: auto-start at boot used to bring the system
> up **empty** (a Windows/Docker quirk). Now the app starts everything the
> correct way. See [runbook.md](runbook.md) → *Operating & accessing the app*.

---

## What the one command does (the manual steps)

If you ever want to run it by hand, this is exactly what the script runs.

### Step 1 — Load settings (passwords/keys)

```bash
cd /mnt/d/Vishvas-rag-pipeline
set -a; source .env; set +a
```

> **Required.** If you skip it, Step 3 fails with an "Unauthorized" / `401` error.

### Step 2 — Prepare the new transcripts

```bash
.venv/bin/python -m ingestion.chunker_cleaned \
  "/mnt/d/Transcription whisperx/Output" \
  data/processed \
  --base-dir "/mnt/d/Transcription whisperx/Output" -r
```

You'll see a line like:

```
incremental scan: 156 tracks found — 150 already processed (skip), 6 new to process
...
Summary: 156 tracks found | 6 processed (new) -> 19 chunks | 150 skipped | 0 failed
```

If everything is already loaded it says `0 new` — that's fine, nothing to do.

### Step 3 — Load them into the search system

```bash
.venv/bin/python -m ingestion.bulk_ingest_hardened --chunks-dir data/processed
```

Final line:

```
ingest summary: processed=6 skipped=150 failed=0 total_chunks=19 ...
```

`processed` = newly searchable recordings. `failed=0` means success.

### Step 4 — Refresh the app

```bash
docker compose restart rag-api
```

Wait ~10 seconds. The new content is now live in the dashboard.

---

## Check it worked (optional)

```bash
.venv/bin/python -m ingestion.verify_ingestion --chunks-dir data/processed
```

A healthy result looks like:

```
Sampled 290 chunks. Qdrant: 100.0% present. Tantivy: 100.0% present.
```

Both at **100%** = everything loaded correctly. Then open the Vishvas app (or
**http://localhost:8081**) and try a search.

---

## Important: why you don't hand-edit the paths

The script pins `--base-dir` to `/mnt/d/Transcription whisperx/Output` on every
run. **That must stay constant** — it's how the system recognises which tracks
are already loaded. If you change it, every recording looks "new" and you get
**duplicates**. The one-command script removes this risk; if you run the steps
by hand, keep `--base-dir` exactly as shown above.

The default **input** folder is the same root, scanned recursively (`-r`), so
new tracks in *any* event folder are picked up automatically — you don't name a
specific event unless you want to limit the run to one.

---

## If something goes wrong

| What you see | What it means | Fix |
|---|---|---|
| `401` / `Unauthorized` | Settings not loaded | Re-run Step 1 (`set -a; source .env; set +a`), then retry — or just use the one-command script, which does this for you |
| `input folder not found` | The source path is wrong/missing | Check `D:\Transcription whisperx\Output` exists; pass the correct folder as the argument |
| `health check failed` / `exiting` | A service is down | Open the Vishvas app (or `docker compose up -d`), wait a minute, retry |
| `0 new to process` | Nothing new to load | Normal — you're already up to date |
| A few `failed=N` in Step 3 | Those specific files had a problem | They're logged in `dead_letter/`; safe to re-run — only the failed ones retry (or `bulk_ingest_hardened --retry-failed`) |
| Dashboard shows old results | App hasn't refreshed | `docker compose restart rag-api` |

Nothing here deletes anything. Re-running is always safe — already-loaded
recordings are skipped, not duplicated.

---

## Advanced: enriched search features (occasional, not per-batch)

Basic search (the steps above) makes new transcripts findable immediately. A few
**optional** indexes power extra features and are rebuilt only occasionally, not
every batch:

- **File-summary / two-stage search** — `python -m ingestion.build_summary_index`
  (needs file summaries populated first). See [upgrade.md](upgrade.md) → Step 2.
- **PageIndex tree search** — `python -m ingestion.build_pageindex`.
- **Catalog (sheet) enrichment** — see [catalog_enrichment.md](catalog_enrichment.md).

Skip these for routine "add new transcripts" work.
