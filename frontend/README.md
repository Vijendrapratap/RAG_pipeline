# Transcript RAG — Dashboard (Phase E)

Lightweight **React + Vite + TypeScript** UI for the `rag-api` backend. Plain
CSS, no component library. Speaks Hindi and English. Replaces the Open WebUI
chat surface with: a typed query box, scope / language / HyDE controls, a
filter sidebar, soft-filter suggestion chips from `detected_filters`, a
streaming answer pane with clickable `[N]` citations, and a Postgres-backed
analytics tab.

## Prerequisites

- Node 18+ (developed against Node 24).
- The `rag-api` backend running and reachable. See the root
  [docs/dashboard.md](../docs/dashboard.md). For local dev:
  ```bash
  uvicorn rag_api.app:app --host 0.0.0.0 --port 8080
  ```
  The default `RAG_CORS_ORIGINS` already whitelists
  `http://localhost:5173`, which is the Vite dev port.

## Configure

Copy `.env.example` to `.env` and set the API base URL if it is not the
default `http://localhost:8080`:

```bash
cp .env.example .env
# edit VITE_API_BASE if needed
```

Auth: the password is the backend's `DASHBOARD_PASSWORD`. Empty on the
backend ⇒ no login screen.

## Develop

```bash
npm install
npm run dev          # http://localhost:5173
```

## Build

```bash
npm run build        # type-check, then bundle to dist/
npm run preview      # serve the bundle locally
```

Phase F will mount `dist/` from the `rag-api` container; for now the dev
server is the canonical way to run the UI.

## Layout

```
src/
  main.tsx           # React entry
  App.tsx            # health probe + auth gate + tab switch
  api.ts             # typed client (fetch + SSE stream parser)
  types.ts           # TypeScript mirror of the backend JSON contract
  filters.ts         # facet metadata + detection-to-filter helpers
  styles.css         # one stylesheet, light theme, Devanagari font fallback
  components/
    Header.tsx           # title, tabs, service dots, logout
    Login.tsx            # shared-password gate
    SearchView.tsx       # orchestrates query/filter/response state
    QueryBar.tsx         # input + scope/HyDE/quote/language controls
    FilterPanel.tsx      # one dropdown per file_meta facet + date range
    DetectedFilters.tsx  # auto / soft chips + applied-filter chips
    AnswerPane.tsx       # streamed answer + [N] citation refs
    ResultList.tsx       # /api/search result set
    ResultCard.tsx       # one chunk- or summary-level hit
    AnalyticsView.tsx    # mentions / speakers / transcripts
```

## Notes

- SSE streams require the auth header, so `/api/query` is consumed via
  `fetch` + `ReadableStream` (an `EventSource` cannot set headers).
- Strong detections (years, ISO dates) are reported as `auto`-applied — you
  see what the backend used and can override via the filter panel. Soft
  detections (season / topic words) are one-click chips that add to the
  active filters.
- `Find quote` disables scope / auto-filters / HyDE — the backend ignores
  them for exact-phrase BM25 hunts, and the UI matches.
