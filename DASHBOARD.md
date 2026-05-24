# Custom Dashboard Track

Progress + reference for replacing **Open WebUI** with a lightweight custom
dashboard. Companion to [PRD.md](PRD.md) (the original 13-phase build, still
the source of truth for ingestion/retrieval internals).

---

## Why

Open WebUI is heavy (full Node+Python app + its own Postgres + Redis). The
goal: a light **FastAPI backend + React dashboard** that reuses the existing
hybrid-retrieval engine, adds **bilingual (Hindi/English) answer synthesis**,
and makes the **file summaries searchable**.

Retrieval *quality* is unchanged at the core (same Qdrant + Tantivy + RRF +
reranker). Gains come from: always-retrieve flow (no flaky 7B tool-calling),
language-aware answers, summary-level search, query-time filter extraction,
optional HyDE expansion, and Hindi-correct full-text analytics.

---

## Architecture change

```
BEFORE:  Browser → Open WebUI → function tool → Qdrant/Tantivy/Infinity/Postgres → Ollama
AFTER:   Browser → React dashboard → rag-api (FastAPI) → Qdrant/Tantivy/Infinity/Postgres → Ollama
```

- **Removed (Phase F):** `open-webui`, `redis` containers.
- **Kept:** `ollama`, `qdrant`, `reranker` (Infinity), `postgres`.
- **Added:** `rag-api` FastAPI service (Phase F also serves the built
  React `dist/` from `/` inside this container). Tantivy BM25 runs
  **in-process** inside `rag-api` — the `:8765` sidecar is gone.

---

## Phase status

| Phase | Scope | Status |
|---|---|---|
| A | Backend scaffold + retrieval extraction | ✅ done |
| B | Bilingual answer synthesis | ✅ done |
| C | Summary-level retrieval | ✅ done |
| D | Retrieval quality upgrades (filter extraction, HyDE, Hindi FTS, analytics) | ✅ done |
| E | React/Vite dashboard UI | ✅ done |
| F | docker-compose cutover + docs | ✅ done |

---

## What exists now (`rag_api/` package)

| File | Role |
|---|---|
| `rag_api/config.py` | Settings from env / `.env` (tiny built-in loader). |
| `rag_api/retrieval.py` | Hybrid retrieval lifted out of the Open WebUI tool: dense (Qdrant) + BM25 (in-process Tantivy) + weighted RRF + Infinity rerank. Plus summary + two-stage search, and HyDE vector blending (`mean_pool`). |
| `rag_api/synthesis.py` | Citation-grounded answer generation via Ollama `/api/chat` (streaming + non-streaming). |
| `rag_api/lang.py` | Hindi/English detection (Devanagari ratio). |
| `rag_api/query_parse.py` | Deterministic filter extraction — pulls years / dates / season / place / topic signals out of the query text. Pure, vocab-grounded. |
| `rag_api/expand.py` | Optional HyDE query expansion — chat model writes a hypothetical passage to steer the dense vector. Opt-in. |
| `rag_api/analytics.py` | Postgres corpus analytics (mention counts, speaker / transcript ranking). Hindi-correct `'simple'` full-text search. |
| `rag_api/db.py` | Postgres reads — distinct values for filter dropdowns, plus `VocabCache` (TTL cache feeding query-time filter extraction). |
| `rag_api/app.py` | FastAPI app, endpoints, shared-password auth. |
| `ingestion/build_summary_index.py` | Backfill: embeds file summaries → `transcript_summaries` Qdrant collection. Resumable, idempotent. |

`open_webui_functions/` is left untouched — Phase F retires it.

### Frontend (`frontend/` — Phase E)

| File | Role |
|---|---|
| `frontend/index.html`, `vite.config.ts`, `tsconfig.json`, `package.json` | Vite + TypeScript scaffold. Dev port 5173 (matches backend `RAG_CORS_ORIGINS` default). |
| `src/types.ts` | TypeScript mirror of the rag-api JSON contract. |
| `src/api.ts` | Typed fetch client — auth header, SSE stream parser for `/api/query`. |
| `src/filters.ts` | Facet metadata + detection-to-filter helpers shared by panel and chips. |
| `src/App.tsx` | Health probe, auth gate, tab switch (Search / Analytics). |
| `src/components/Login.tsx` | Shared-password gate. |
| `src/components/Header.tsx` | Title, tabs, upstream-service dots, model name, sign-out. |
| `src/components/SearchView.tsx` | Orchestrates query/filter/response state for the Search tab. |
| `src/components/QueryBar.tsx` | Query input + mode / scope / top-K / HyDE / quote / auto-filter / language controls. |
| `src/components/FilterPanel.tsx` | One dropdown per `file_meta` facet, plus date range + speaker. |
| `src/components/DetectedFilters.tsx` | Strong = auto-applied chip; soft = one-click "+ apply" chip; HyDE badge. |
| `src/components/AnswerPane.tsx` | Streaming answer with clickable `[N]` citations that scroll-to + flash the citation card. |
| `src/components/ResultList.tsx`, `ResultCard.tsx` | `/api/search` result set and the shared chunk/summary card. |
| `src/components/AnalyticsView.tsx` | Mention count / top speakers / transcripts ranking. |
| `src/styles.css` | One stylesheet — plain CSS, light theme, Devanagari-aware font stack. |

See `frontend/README.md` for run + build instructions.

---

## API endpoints

Base port `8080` (configurable). Auth: header `X-Dashboard-Password` must
match `DASHBOARD_PASSWORD` (empty password ⇒ auth disabled, dev only).

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/health` | no | Service reachability + index status. |
| POST | `/api/search` | yes | Retrieval only — structured results, no LLM. |
| POST | `/api/query` | yes | Full RAG: retrieve → synthesise a bilingual answer. |
| POST | `/api/tantivy/reload` | yes | Re-open the BM25 index after new ingestion. |
| GET | `/api/filters` | yes | Distinct metadata values for UI dropdowns. |
| GET | `/api/analytics/mentions` | yes | Full-text mention count for a term (`?term=&speaker=`). |
| GET | `/api/analytics/speakers` | yes | Speakers ranked by mentions of a term (`?term=&limit=`). |
| GET | `/api/analytics/transcripts` | yes | Transcripts ranked by mentions of a term (`?term=&limit=`). |

### `POST /api/search` / `POST /api/query` body

```jsonc
{
  "query": "karma yoga in 2015",
  "find_quote": false,                // true => BM25-heavy exact-phrase hunt
  "scope": "chunks",                  // chunks | summaries | two_stage
  "top_k": 8,
  "filters": { "season": "monsoon", "track_type": ["discourse"], ... },
  "auto_filters": true,               // extract filters from the query text
  "expand_query": false,              // true => HyDE (one extra LLM call)
  "answer_language": "auto",          // query only: auto | hindi | english
  "stream": false                     // query only: true => SSE
}
```

- **`scope=chunks`** — hybrid chunk search (default, unchanged behaviour).
- **`scope=summaries`** — one result per file, ranked by summary match. Best
  for cross-corpus / topic questions.
- **`scope=two_stage`** — summary search picks the top files, then hybrid
  chunk search runs only within them.

Both responses also carry `detected_filters` (every signal found in the
query, each with `confidence` strong/soft), `applied_filters` (what actually
filtered retrieval), and `expanded` (whether HyDE ran).

`/api/query` returns `{answer, answer_language, citations[], count, …}` — or,
with `stream:true`, Server-Sent Events: `meta` (language + citations +
filters) → `token` deltas → `done`/`error`. Answers cite passages as `[N]`
mapped to `citations`.

---

## New Qdrant collection: `transcript_summaries`

One vector per transcript file (1024-d bge-m3, int8 quantized). Vector =
embedding of `summary_english + summary_hindi + topics/scriptures/people`.
Payload-indexed on `source_file`, `event_type`, `primary_language`, `season`,
`location`, `event_id`, `track_type`, `topics`, `scriptures_referenced`,
`people_named`, `session_date`. Built by `ingestion.build_summary_index`.

---

## Phase D — retrieval quality

Three levers, all reusing the existing Qdrant/Tantivy/RRF/reranker core:

**1. Query-time filter extraction** (`rag_api/query_parse.py`). A query like
"karma yoga discourses from 2015 at noida" carries filterable signals. They
are pulled out with regex + vocabulary matching (no LLM) and classified:

- *strong* — bare years, ISO dates. Unambiguous → auto-applied (when
  `auto_filters` is true, the default).
- *soft* — a season / place / topic-tag word found in the query. Reported as
  a suggestion, never auto-applied — "winter" is as often the *subject* of a
  question as a filter on it. The UI turns soft detections into one-click
  chips.

Every detection is returned in `detected_filters`, so a guess is visible and
removable, not a silent regression. The vocabulary comes from Postgres
`file_meta` via a 10-minute TTL cache (`VocabCache`); a DB outage degrades to
date-only extraction, never an error.

**2. HyDE expansion** (`rag_api/expand.py`, opt-in via `expand_query`). The
chat model writes a short hypothetical discourse passage answering the query;
its embedding is mean-pooled with the query embedding before dense search.
The hypothetical *looks like the corpus*, so the blended vector lands closer
to the right chunks. Cost: one extra chat call (~1–3 s) — hence opt-in. BM25
and the reranker still use the raw query. Generation failure degrades to
query-only retrieval.

**3. Hindi-correct full-text search** (`rag_api/analytics.py`). The corpus is
Hindi; the analytics queries (ported from `open_webui_functions/analytics.py`)
used `to_tsvector('english', …)`, which applies English stemming + stopwords
to Devanagari — wrong lexemes, missed matches. Now `'simple'` (lowercase +
tokenise, no stemming/stopwords). The GIN index must match the query config,
so `analytics_schema.sql` is updated **and** existing databases must run
`infra/postgres/migrations/001_hindi_fts.sql` to rebuild the index.

### How retrieval compares to the Open WebUI setup

| | Open WebUI | Dashboard API (A–D) |
|---|---|---|
| Tool invocation | 7B model decides to call the search tool — flaky | Always retrieves; no tool-calling gamble |
| Filters | Model fills tool args from the query — unreliable | Deterministic extraction + explicit UI filters |
| Vague queries | Query embedding only | Optional HyDE blend |
| Topic / corpus questions | Chunk search only | `summaries` + `two_stage` scopes |
| Analytics on Hindi text | `'english'` FTS — broken | `'simple'` FTS — correct |
| Answer language | Whatever the model picks | Detected, or forced, per request |

The retrieval *core* (bge-m3 dense + Tantivy BM25 + weighted RRF + bge-
reranker-v2-m3) is unchanged — the gains are in feeding it better queries and
filters, and in not depending on a 7B model to decide when to retrieve.

---

## New env vars (see `.env.example`)

```bash
CHAT_MODEL=qwen2.5:7b-instruct-q4_K_M   # answer model; swap to 26B fine-tune here
DASHBOARD_PASSWORD=replace_me           # shared password; empty = auth off
RAG_API_HOST=0.0.0.0
RAG_API_PORT=8080
RAG_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
QDRANT_SUMMARY_COLLECTION=transcript_summaries
RAG_CANDIDATES=40
RAG_TOP_K=8
RAG_BM25_WEIGHT=0.65
RAG_QUOTE_BM25_WEIGHT=0.85
RAG_SUMMARY_TOP_FILES=6                 # files kept by stage-1 summary search
CHAT_NUM_CTX=8192
CHAT_TEMPERATURE=0.2
CHAT_TIMEOUT_S=300
```

`pyproject.toml` package discovery now includes `rag_api*`.

---

## Run + test (in WSL2, Docker stack up)

```bash
pip install -e .                                    # installs fastapi, tantivy, etc.
pytest tests/unit -q                                # rag_api/track tests + existing suite
uvicorn rag_api.app:app --host 0.0.0.0 --port 8080  # or: python -m rag_api.app

curl localhost:8080/api/health
curl -X POST localhost:8080/api/query \
  -H 'X-Dashboard-Password: <yours>' -H 'Content-Type: application/json' \
  -d '{"query":"कर्म योग क्या है","scope":"two_stage","expand_query":true}'

curl 'localhost:8080/api/analytics/mentions?term=dharma' \
  -H 'X-Dashboard-Password: <yours>'

# Build the summary index (after Phase 13 enrichment has populated file_meta):
python -m ingestion.build_summary_index

# One-time, on a pre-existing DB — rebuild the FTS index for Hindi:
psql "$POSTGRES_DSN" -f infra/postgres/migrations/001_hindi_fts.sql
```

**Tests:** 67 unit tests across `test_rag_api_retrieval.py`,
`test_rag_api_lang.py`, `test_rag_api_synthesis.py`,
`test_rag_api_query_parse.py`, `test_rag_api_analytics.py`,
`test_rag_api_expand.py`, `test_build_summary_index.py` — all passing. Live
API acceptance needs the Docker stack + pulled models.

---

## Phase F — cutover (what changed)

Five-service stack now: **`rag-api`** (FastAPI + the built React dashboard,
Tantivy in-process) + `ollama` + `qdrant` + `reranker` + `postgres`. No
more `open-webui`, no more `redis`, no more `:8765` Tantivy sidecar.

- `services/rag_api/Dockerfile` — multi-stage: Node builds `frontend/dist/`,
  then a Python 3.11 slim runtime installs the `rag_api` package and copies
  the built UI to `/app/frontend/dist`. FastAPI's `StaticFiles` (`html=True`)
  serves it at `/`; `/api/*` is the same FastAPI app.
- `docker-compose.yml` — rewritten. `rag-api` reads `data/tantivy/`
  read-only (ingestion writes), and gets every retrieval / chat / auth
  knob from the `.env` file.
- Frontend: same-origin by default (`API_BASE=""`). Dev mode keeps Vite on
  `:5173` and proxies `/api` → `:8080`, so no env var is required for
  either dev or prod.
- `scripts/00_health_check.sh` now checks the rag-api `/api/health`
  endpoint and drops the Open WebUI / Redis / Tantivy-sidecar checks.
- `scripts/start_pipeline.sh` is 7 steps instead of 8 — the sidecar launch
  step is gone.
- `services/tantivy_server/`, `scripts/run_tantivy_server.sh` — **deleted**.
- `open_webui_functions/` is kept on disk for the legacy `eval/run_eval.py`
  harness, with a `DEPRECATED` notice in its `__init__.py`. Nothing in the
  deployed stack imports it.
- Operator docs (`docs/install_functions.md`, `user_guide.md`,
  `architecture.md`, `troubleshooting.md`, `runbook.md`, `model_config.md`)
  carry a Phase F banner pointing at DASHBOARD.md. `README.md` got a full
  quickstart rewrite.

### Run

```bash
docker compose up -d --build       # builds the rag-api image first time
bash scripts/00_health_check.sh    # all five services should be ✅
# dashboard: http://localhost:8080
```

For dashboard-only iteration (hot reload, no rebuild loop), see
[frontend/README.md](frontend/README.md).

## Notes / caveats

- **Postgres FTS migration:** Phase D changes the analytics full-text config
  from `'english'` to `'simple'`. A fresh DB from `analytics_schema.sql` is
  already correct; an **existing** DB must run
  `infra/postgres/migrations/001_hindi_fts.sql` once — until then analytics
  full-text queries do a sequential scan (slow, but correct). PRD.md §Phase 6
  still shows the old `'english'` index — flag for the PRD owner.
- **Git:** the repo root is the whole `D:\` drive — per-phase commits are
  messy until that's fixed (`git init` inside the project folder). No commits
  made on this track yet.
- **26B model:** at q4 (~15–16 GB) it will not fit the 12 GB GPU alongside
  embed+reranker. `CHAT_MODEL` is config-only so the swap is one line — but
  plan the hardware. Qwen 2.5 7B until then. HyDE adds a second chat call per
  query when enabled — weigh that against the VRAM/latency budget.
- SSE auth uses a header, so the React app must consume `/api/query` streams
  via `fetch` (ReadableStream), not `EventSource`.
