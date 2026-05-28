# Change Log — Custom Dashboard Track

A running record of every change made while replacing **Open WebUI** with a
custom **FastAPI backend + React dashboard**. For the *why* and the
architecture, see [dashboard.md](dashboard.md); for retrieval/ingestion
internals see [PRD.md](../PRD.md). To apply these changes to an
already-running deployment, follow [upgrade.md](upgrade.md).

Phases A → F are complete — the Open WebUI cutover is done. **No git commits
made on this track yet** — the git repo root is the whole `D:\` drive.

Legend: 🆕 new file · ✏️ modified · 🗑️ removed

---

## Phase A — Backend scaffold + retrieval extraction ✅

Stood up the FastAPI service and lifted the hybrid-retrieval engine out of the
Open WebUI function tool into a plain, testable package.

| | File | Change |
|---|---|---|
| 🆕 | `rag_api/__init__.py` | Package marker. |
| 🆕 | `rag_api/config.py` | Settings from env / `.env` (built-in 12-line loader, no `python-dotenv`). |
| 🆕 | `rag_api/retrieval.py` | Hybrid retrieval — dense (Qdrant) + BM25 + weighted RRF + Infinity rerank — extracted from `open_webui_functions/search_transcripts.py`. Tantivy now runs **in-process** (the `:8765` sidecar is no longer needed). |
| 🆕 | `rag_api/app.py` | FastAPI app: `GET /api/health`, `POST /api/search`, `POST /api/tantivy/reload`, shared-password auth. |
| 🆕 | `tests/unit/test_rag_api_retrieval.py` | Unit tests for the pure retrieval helpers. |
| ✏️ | `pyproject.toml` | Added `rag_api*` to package discovery. |
| ✏️ | `.env.example` | Added the dashboard-API env section. |

---

## Phase B — Bilingual answer synthesis ✅

Added the answer-generation layer Open WebUI used to own — now bilingual
(Hindi/English) and citation-grounded.

| | File | Change |
|---|---|---|
| 🆕 | `rag_api/lang.py` | Hindi/English detection by Devanagari-vs-Latin ratio. |
| 🆕 | `rag_api/synthesis.py` | Citation-grounded answer generation via Ollama `/api/chat` — streaming + non-streaming, language-aware prompts, canned "no context" message. |
| 🆕 | `tests/unit/test_rag_api_lang.py` | Language-detection tests. |
| 🆕 | `tests/unit/test_rag_api_synthesis.py` | Prompt-builder tests. |
| ✏️ | `rag_api/app.py` | Added `POST /api/query` — full RAG turn, with Server-Sent-Events streaming. |

---

## Phase C — Summary-level retrieval ✅

Made the per-file Hindi/English summaries (written into Postgres `file_meta` by
Phase 13 enrichment) **searchable** — they were never embedded before.

| | File | Change |
|---|---|---|
| 🆕 | `rag_api/db.py` | Postgres reads — distinct `file_meta` values for the dashboard filter dropdowns. |
| 🆕 | `ingestion/build_summary_index.py` | Backfill script: embeds each file's summary + tags → new `transcript_summaries` Qdrant collection (one vector per file). Resumable, idempotent. |
| 🆕 | `tests/unit/test_build_summary_index.py` | Tests for the summary-text / payload builders. |
| ✏️ | `rag_api/config.py` | Added summary-collection + Postgres settings. |
| ✏️ | `rag_api/retrieval.py` | Added `search_summaries` (one result per file) and `search_two_stage` (summary search picks files → chunk search within them). |
| ✏️ | `rag_api/app.py` | Added the `scope` selector (`chunks` / `summaries` / `two_stage`) and `GET /api/filters`. |
| ✏️ | `.env.example` | Added `QDRANT_SUMMARY_COLLECTION` and retrieval-tuning vars. |

---

## Phase D — Retrieval quality upgrades ✅

Four levers that feed the retrieval core better queries and filters. The core
itself (bge-m3 dense + Tantivy BM25 + weighted RRF + bge-reranker-v2-m3) is
unchanged.

### New files

| | File | Change |
|---|---|---|
| 🆕 | `rag_api/query_parse.py` | **Deterministic filter extraction.** Regex + vocabulary matching (no LLM) pulls signals out of the query text. *Strong* signals (years, ISO dates) are auto-applied; *soft* signals (a season/place/topic word) are reported as suggestions only. |
| 🆕 | `rag_api/expand.py` | **HyDE query expansion** (opt-in). The chat model writes a hypothetical discourse passage; its embedding is blended into the dense vector. One extra LLM call — off by default. |
| 🆕 | `rag_api/analytics.py` | **Corpus analytics**, ported from `open_webui_functions/analytics.py`: mention counts, speaker ranking, transcript ranking. Full-text search switched to the Hindi-correct `'simple'` config. |
| 🆕 | `infra/postgres/migrations/001_hindi_fts.sql` | One-time migration to rebuild the `idx_chunk_text_fts` GIN index with the `'simple'` config on an existing database. |
| 🆕 | `tests/unit/test_rag_api_query_parse.py` | Filter-extraction tests. |
| 🆕 | `tests/unit/test_rag_api_analytics.py` | Analytics shaping + FTS-config tests. |
| 🆕 | `tests/unit/test_rag_api_expand.py` | HyDE prompt-builder tests. |

### Modified files

| | File | Change |
|---|---|---|
| ✏️ | `rag_api/retrieval.py` | Added `mean_pool`, batched embedding (`_embed_batch`), and `_query_vector` (blends query + HyDE embeddings). `search` / `search_summaries` / `search_two_stage` take an optional `dense_text`. |
| ✏️ | `rag_api/db.py` | Added `VocabCache` — a 10-minute TTL cache of the filter vocabulary, feeding query-time filter extraction; resilient to DB outages. |
| ✏️ | `rag_api/app.py` | Added `auto_filters` + `expand_query` request fields; `_prepare` helper (detection + expansion); `detected_filters` / `applied_filters` / `expanded` in responses; three `GET /api/analytics/*` endpoints. |
| ✏️ | `infra/postgres/analytics_schema.sql` | `idx_chunk_text_fts` rebuilt as `to_tsvector('simple', text)` — correct for the Hindi corpus (was `'english'`). |
| ✏️ | `docs/architecture.md` | Updated two stale references to the old `'english'` FTS config. |
| ✏️ | `tests/unit/test_rag_api_retrieval.py` | Added `mean_pool` tests. |

### Open items from Phase D

- **Postgres FTS migration must be run once** on any pre-existing database:
  `psql "$POSTGRES_DSN" -f infra/postgres/migrations/001_hindi_fts.sql`.
  A fresh DB built from `analytics_schema.sql` is already correct.
- `PRD.md` §Phase 6 still shows the old `to_tsvector('english', …)` index —
  left untouched (PRD is the locked spec); the PRD owner should reconcile it.

---

## Phase E — React/Vite dashboard UI ✅

Replaced the Open WebUI chat surface with a lightweight bilingual dashboard.
TypeScript, plain CSS, no component library. One `fetch`-based client speaks
the rag-api JSON contract; `/api/query` SSE is parsed in-browser (auth needs
a header, so `EventSource` cannot be used). `npm run build` is clean —
`tsc` passes, the gzipped bundle is ~52 KB.

### Scaffold

| | File | Change |
|---|---|---|
| 🆕 | `frontend/package.json` | React 18 + Vite 5 + TypeScript 5. Three dev scripts (`dev`, `build`, `preview`). |
| 🆕 | `frontend/index.html` | Single `<div id="root">` mount. |
| 🆕 | `frontend/vite.config.ts` | Dev server pinned to port 5173 (matches backend `RAG_CORS_ORIGINS` default). |
| 🆕 | `frontend/tsconfig.json`, `tsconfig.node.json` | Strict TS, bundler resolution, `react-jsx`. |
| 🆕 | `frontend/.env.example`, `.gitignore` | `VITE_API_BASE` for the rag-api base URL. |
| 🆕 | `frontend/README.md` | Run / build / layout reference. |
| 🆕 | `frontend/src/vite-env.d.ts` | `import.meta.env` typing for `VITE_API_BASE`. |
| 🆕 | `frontend/src/main.tsx`, `styles.css` | React entry + a single light-theme stylesheet with a Devanagari font fallback. |

### App shell + API layer

| | File | Change |
|---|---|---|
| 🆕 | `frontend/src/types.ts` | TypeScript mirror of the rag-api contract: Health, FilterModel, FilterOptions, Detection, RetrievalResult, SearchResponse, QueryMeta, analytics shapes. |
| 🆕 | `frontend/src/api.ts` | Typed client: `getHealth`, `getFilters`, `search`, `streamQuery` (fetch + ReadableStream + SSE-block parser), `analyticsMentions / Speakers / Transcripts`. `ApiError` carries the HTTP status so callers can react to 401s. Password lives in sessionStorage. |
| 🆕 | `frontend/src/filters.ts` | `FACETS` config (option-key → field → label → list?), `cleanFilters`, `applyDetection`, `detectionActive`. |
| 🆕 | `frontend/src/App.tsx` | Health probe → auth gate → tab switch (Search / Analytics). Stored password is silently validated against an authed endpoint; a 401 from anywhere drops back to Login. |
| 🆕 | `frontend/src/components/Login.tsx` | Shared-password gate; validates by calling `/api/filters`. |
| 🆕 | `frontend/src/components/Header.tsx` | Title, tabs, three upstream-service status dots (ollama / qdrant / reranker), chat-model name, sign-out. |

### Search tab

| | File | Change |
|---|---|---|
| 🆕 | `frontend/src/components/SearchView.tsx` | Two-column layout (filter sidebar + main pane). Holds all query/filter/response state. Mode switches between streaming `/api/query` (answer + citations) and `/api/search` (results only). AbortController-backed Stop button. |
| 🆕 | `frontend/src/components/QueryBar.tsx` | Textarea + `mode` / `scope` / `top_k` / `find_quote` / `auto_filters` / `expand_query` / `answer_language` controls. Ctrl/Cmd+Enter submits. `find_quote` disables the semantic switches the backend would ignore. |
| 🆕 | `frontend/src/components/FilterPanel.tsx` | One dropdown per `file_meta` facet (rendered only when options arrive), plus a date-range picker and a free-text speaker field. Warns inline if Postgres is unreachable. |
| 🆕 | `frontend/src/components/DetectedFilters.tsx` | Phase-D transparency surface: strong signals as non-interactive "auto" chips, soft signals as one-click "+ apply" chips, applied-filter chips, HyDE-expansion badge. |
| 🆕 | `frontend/src/components/AnswerPane.tsx` | Streaming answer with clickable `[N]` citations — clicks scroll to + briefly flash the citation card. Out-of-range markers render disabled. Tokens accumulate into a state string for incremental render. |
| 🆕 | `frontend/src/components/ResultList.tsx`, `ResultCard.tsx` | `/api/search` result set and the shared chunk-/summary-card (source, timestamp range, score, speakers, metadata pills). |

### Analytics tab

| | File | Change |
|---|---|---|
| 🆕 | `frontend/src/components/AnalyticsView.tsx` | One form, three buttons → `/api/analytics/mentions` / `/speakers` / `/transcripts`. Results render as a card (count) or a ranked table. |

### Docs

| | File | Change |
|---|---|---|
| ✏️ | `DASHBOARD.md` | Phase E marked ✅, frontend file table added, Pending reduced to Phase F. |

---

## Phase F — docker-compose cutover ✅

Five-service stack now: `rag-api` + `ollama` + `qdrant` + `reranker` +
`postgres`. Open WebUI, Redis, and the Tantivy `:8765` sidecar are gone.
`docker compose config` parses clean.

### Container + service

| | File | Change |
|---|---|---|
| 🆕 | `services/rag_api/Dockerfile` | Multi-stage: Node builds `frontend/dist/`; Python 3.11 slim installs the `rag_api` package and copies the built UI. FastAPI `StaticFiles(html=True)` serves it at `/`. |
| 🆕 | `services/rag_api/.dockerignore` | Skinny build context — drops `data/`, `tests/`, `ingestion/`, `open_webui_functions/`, etc. |
| ✏️ | `docker-compose.yml` | 🗑️ `open-webui`, `redis`. 🆕 `rag-api` service (port 8080, mounts `data/tantivy` read-only, takes every retrieval/chat/auth knob via env). Obsolete `version:` removed. |
| ✏️ | `rag_api/app.py` | Mounts `frontend/dist/` at `/` when present (after the API routes, so `/api/*` is never shadowed). Logs whether the dashboard build was found. |

### Frontend wiring

| | File | Change |
|---|---|---|
| ✏️ | `frontend/vite.config.ts` | Dev server proxies `/api` → backend (target via `VITE_API_TARGET`, default `:8080`) so the app works with same-origin relative URLs in both dev and prod. |
| ✏️ | `frontend/src/api.ts` | Default `API_BASE` is `""` (same-origin). `VITE_API_BASE` only needed for hosting the UI on a different origin. |
| ✏️ | `frontend/.env.example`, `src/vite-env.d.ts` | Document `VITE_API_TARGET` (dev proxy) vs `VITE_API_BASE` (cross-origin override). |

### Retirements

| | File | Change |
|---|---|---|
| 🗑️ | `services/tantivy_server/` | Deleted — Tantivy runs in-process inside `rag-api`. |
| 🗑️ | `scripts/run_tantivy_server.sh` | Deleted — no sidecar to launch. |
| ✏️ | `open_webui_functions/__init__.py` | Deprecation notice. Directory kept on disk for the legacy `eval/run_eval.py` harness only; nothing in the deployed stack imports it. |

### Scripts + config

| | File | Change |
|---|---|---|
| ✏️ | `scripts/00_health_check.sh` | Now checks `rag-api` `/api/health`. Dropped Open WebUI, Redis, and Tantivy sidecar checks. |
| ✏️ | `scripts/start_pipeline.sh` | 7 steps instead of 8 — sidecar launch step gone. Final message points at the dashboard. |
| ✏️ | `.env.example` | 🗑️ `WEBUI_SECRET_KEY`, `TANTIVY_URL`. |
| ✏️ | `.gitignore` | Adds `frontend/dist/` alongside `node_modules/`. |
| ✏️ | `pyproject.toml` | `services*` removed from `packages.find` include. |

### Docs

| | File | Change |
|---|---|---|
| ✏️ | `README.md` | Full rewrite: 5-service architecture paragraph, dashboard quickstart, frontend section, links to DASHBOARD.md / UPGRADE.md / PRD.md as the layered documentation. |
| ✏️ | `docs/install_functions.md`, `model_config.md` | "Retired" banner — fully obsolete now. |
| ✏️ | `docs/user_guide.md`, `architecture.md`, `troubleshooting.md`, `runbook.md` | "Phase F note" banner — flags the Open WebUI / sidecar parts that no longer apply; ingestion and retrieval-internals sections still accurate. |
| ✏️ | `DASHBOARD.md` | Phase F marked ✅, new "Phase F — cutover (what changed)" section. |

---

## Test status

**67 unit tests passing** across the dashboard-track suites
(`test_rag_api_retrieval.py`, `test_rag_api_lang.py`,
`test_rag_api_synthesis.py`, `test_rag_api_query_parse.py`,
`test_rag_api_analytics.py`, `test_rag_api_expand.py`,
`test_build_summary_index.py`). Live API acceptance still needs the Docker
stack with models pulled.

---

## Still to come

The dashboard-track scope is complete. Loose ends:

- **PRD.md reconciliation** — PRD §Phase 6 still shows the old
  `to_tsvector('english', text)` index expression (line ~680). Left
  untouched (CLAUDE.md rule 9 — PRD is the locked spec); the PRD owner
  should reconcile it.
- **Eval harness port** — `eval/run_eval.py` still imports the deprecated
  `open_webui_functions/` package and calls `_embed` / `_dense` / `_bm25`
  / `_rrf` / `_rerank` on the `SearchTools` class. A follow-up should
  rewrite it against `rag_api.retrieval.Retriever` (or against the
  `/api/search` HTTP endpoint), after which `open_webui_functions/` can
  be deleted outright.
- **26B answer model** — `CHAT_MODEL` swap is a one-line `.env` change,
  but the model itself does not fit the 12 GB GPU at q4 alongside the
  embedder + reranker. Plan VRAM / hardware before flipping.
- **Live acceptance** — A–F unit-side is verified (67 backend tests + a
  clean `tsc && vite build`; `docker compose config` parses). A live
  end-to-end pass against the Docker stack (models pulled, ingestion
  done, dashboard queried) still remains for the operator to run.
