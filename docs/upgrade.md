# Upgrade Guide — applying the dashboard-track changes to a running system

**Who this is for:** the operator of an *existing* `transcript-rag` deployment
— a machine that has already run the original pipeline and has a populated
Postgres database. This guide brings that machine up to date with the
dashboard-track changes (Phases A–D). It is a handoff document: follow it
top to bottom.

For *what* changed and *why*, see [changes.md](changes.md) and
[dashboard.md](dashboard.md). This file is only the operational steps.

---

## TL;DR

| Step | Required? | Effect if skipped |
|---|---|---|
| 0. Update the code | yes | You don't have the new features at all |
| 1. Postgres FTS migration | **required if you use analytics** | New analytics queries do slow sequential scans |
| 2. Build the summary index | optional | `summaries` / `two_stage` retrieval unavailable |
| 3. Run the new dashboard API | optional (preview) | You keep using Open WebUI |

**Nothing here removes or breaks the existing Open WebUI setup.** All changes
are additive. The Open WebUI → dashboard cutover is a *later* phase (F) and is
not part of this upgrade.

---

## Step 0 — Update the repository

Pull the latest code (or copy the updated tree) onto the machine. New and
changed files are listed in [changes.md](changes.md). No service needs to be
stopped for this step.

Confirm these new paths exist afterwards:

```bash
ls rag_api/                                  # query_parse.py, analytics.py, ...
ls infra/postgres/migrations/                # 001_hindi_fts.sql
```

---

## Step 1 — Postgres full-text-search migration  **(the only breaking change)**

### Why

The corpus is Hindi. The analytics full-text index was built with the
`'english'` text-search config, which applies English stemming and stopwords
to Devanagari — wrong results. Phase D fixes the analytics queries to use the
correct `'simple'` config. **Postgres only uses a full-text index when the
query config matches the index config**, so the index must be rebuilt to
`'simple'` as well, or every new analytics query falls back to a full
sequential scan of `chunk_meta`.

> Re-running `analytics_schema.sql` does **not** fix this — `CREATE INDEX IF
> NOT EXISTS` sees the old index name already exists and skips it. The
> migration below is the only thing that swaps the index.

### 1a. Check what you have

```bash
psql "$POSTGRES_DSN" -c \
  "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_chunk_text_fts';"
```

| Result | What it means | Do |
|---|---|---|
| contains `to_tsvector('english'` | Old index | Run the migration (1b) |
| contains `to_tsvector('simple'` | Already migrated | Skip Step 1 |
| no rows returned | Index not created yet | Skip the migration; just apply `infra/postgres/analytics_schema.sql` once — it now creates the correct index |

If `$POSTGRES_DSN` is not set in your shell, build it from `.env`:

```bash
export $(grep -v '^#' .env | xargs)
export POSTGRES_DSN="postgresql://owui:${POSTGRES_PASSWORD}@localhost:5432/openwebui"
```

### 1b. Estimate the cost first

```bash
psql "$POSTGRES_DSN" -c "SELECT count(*) FROM chunk_meta;"
```

- **Thousands–millions of rows:** the rebuild takes seconds to a few minutes.
- **Hundreds of millions of rows:** it can take hours and is I/O-heavy. The
  migration uses `CREATE INDEX CONCURRENTLY`, so it does **not block
  ingestion or reads** — but run it inside `tmux` / `screen` / `nohup` so a
  dropped SSH session doesn't kill it.

During the rebuild window, analytics full-text queries are slow (sequential
scan) but still correct. Normal retrieval (Qdrant + Tantivy) is unaffected.

### 1c. Run the migration

**Option A — `psql` available on the host:**

```bash
psql "$POSTGRES_DSN" -f infra/postgres/migrations/001_hindi_fts.sql
```

**Option B — Postgres runs in a docker-compose container:**

```bash
docker compose exec -T postgres \
  psql -U owui -d openwebui < infra/postgres/migrations/001_hindi_fts.sql
```

Do **not** wrap this in a transaction and do **not** pass `--single-transaction`
/ `-1` — `CREATE INDEX CONCURRENTLY` cannot run inside a transaction block.

### 1d. Verify

The index definition should now say `simple`:

```bash
psql "$POSTGRES_DSN" -c \
  "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_chunk_text_fts';"
```

Confirm a real query *uses* the index (look for `Bitmap Index Scan on
idx_chunk_text_fts`, not `Seq Scan`):

```bash
psql "$POSTGRES_DSN" -c \
  "EXPLAIN SELECT count(*) FROM chunk_meta
   WHERE to_tsvector('simple', text) @@ plainto_tsquery('simple', 'धर्म');"
```

### 1e. If the migration was interrupted

A failed `CREATE INDEX CONCURRENTLY` can leave an *invalid* index behind.
Check and clean up, then re-run Step 1c:

```bash
psql "$POSTGRES_DSN" -c \
  "SELECT c.relname, i.indisvalid
   FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid
   WHERE c.relname = 'idx_chunk_text_fts';"
# if indisvalid = f:
psql "$POSTGRES_DSN" -c "DROP INDEX CONCURRENTLY IF EXISTS idx_chunk_text_fts;"
```

---

## Step 2 — Build the file-summary index  *(optional)*

This enables `scope=summaries` and `scope=two_stage` retrieval (file-level /
topic search). It only works if the Phase 13 enrichment pass has populated
`summary_english` / `summary_hindi` in `file_meta`.

Check first:

```bash
psql "$POSTGRES_DSN" -c \
  "SELECT count(*) FROM file_meta
   WHERE summary_english IS NOT NULL OR summary_hindi IS NOT NULL;"
```

If that count is `0`, skip this step — there is nothing to index yet. Otherwise:

```bash
python -m ingestion.build_summary_index           # full build
# python -m ingestion.build_summary_index --dry-run   # preview first
```

It is resumable and idempotent — safe to re-run after more files are enriched.

---

## Step 3 — Run the new dashboard API  *(optional preview)*

The new FastAPI backend (`rag_api`) can run **alongside** Open WebUI — it does
not replace anything yet. To try it:

```bash
pip install -e .                                    # adds fastapi, uvicorn, ...
# set DASHBOARD_PASSWORD in .env  (empty = no auth, local only)
uvicorn rag_api.app:app --host 0.0.0.0 --port 8080  # or: python -m rag_api.app

curl localhost:8080/api/health
```

See [dashboard.md](dashboard.md) for the full endpoint reference. The React UI
and the removal of Open WebUI are still upcoming (Phases E and F).

---

## What NOT to do

- **Do not remove Open WebUI or the Tantivy `:8765` sidecar.** They are
  retired in a later phase (F), not here.
- **Do not delete the old `open_webui_functions/` tools.** Still in use.
- **Do not run the migration twice in parallel.** One run only.

---

## Rollback

The only change to existing data is the FTS index. To revert it to the
previous (English) behaviour:

```bash
psql "$POSTGRES_DSN" -c "DROP INDEX CONCURRENTLY IF EXISTS idx_chunk_text_fts;"
psql "$POSTGRES_DSN" -c \
  "CREATE INDEX CONCURRENTLY idx_chunk_text_fts
   ON chunk_meta USING GIN (to_tsvector('english', text));"
```

Everything else in this upgrade is additive code / new collections — reverting
just means not running the new service.
