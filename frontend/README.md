# Frontend — Vishvas Foundation Discourse Archive

Lightweight **React + Vite + TypeScript** dashboard for the `rag-api`
backend. Plain CSS, no UI library. Speaks Hindi and English. Ships
inside the `rag-api` container at `/` in production; here you'll mostly
run it as a hot-reloading dev server while iterating on the UI.

## Prerequisites

- Node 18+ (developed against Node 24).
- The `rag-api` backend running and reachable. See the operator-facing
  [docs/dashboard.md](../docs/dashboard.md) for the API contract.
  For local dev:
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

`dist/` is what the production rag-api image picks up via multi-stage
Docker build — see [services/rag_api/Dockerfile](../services/rag_api/Dockerfile).

## Layout

```
src/
  main.tsx           # React entry
  App.tsx            # health probe + auth gate + sidebar/tab orchestrator
  api.ts             # typed client (fetch + SSE stream parser)
  types.ts           # TypeScript mirror of the backend JSON contract
  filters.ts         # facet metadata + detection-to-filter helpers
  styles.css         # one stylesheet — design tokens, components, responsive
  components/
    Sidebar.tsx          # brand, new-chat, tab nav, conversation history, health
    Login.tsx            # shared-password gate
    SearchView.tsx       # orchestrates query/filter/response state, welcome state
    QueryBar.tsx         # composer: textarea + advanced disclosure
    FilterPanel.tsx      # slide-out drawer with one control per file_meta facet
    DetectedFilters.tsx  # auto / soft chips + applied-filter chips
    AnswerPane.tsx       # streamed answer + [N] citation refs + sources panel
    ResultList.tsx       # /api/search result set
    ResultCard.tsx       # one chunk- or summary-level hit
    AnalyticsView.tsx    # mentions / speakers / transcripts
```

## Design notes

- The visual system is warm cream + deep saffron, with a serif voice on
  headings and answer text. All tokens live in `:root` at the top of
  `styles.css` — change a colour or radius there to retheme the whole
  app.
- The brand mark is the Devanagari syllable **वि** (the first syllable of
  *Vishvas*), in a saffron→rust gradient. Reused as the sidebar logo,
  login crest, welcome mark, and assistant avatar.
- SSE streams require the auth header, so `/api/query` is consumed via
  `fetch` + `ReadableStream` (an `EventSource` cannot set headers).
- Strong detections (years, ISO dates) are reported as `auto`-applied —
  you see what the backend used and can override via the filter drawer.
  Soft detections (season / topic words) are one-click chips that add to
  the active filters.
- `Find exact quote` disables scope / auto-filters / HyDE — the backend
  ignores them for exact-phrase BM25 hunts, and the UI matches.
