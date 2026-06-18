"""FastAPI backend for the transcript-RAG dashboard.

Endpoints:
  GET  /api/health              — service reachability + index status (no auth)
  POST /api/search              — hybrid retrieval, structured JSON results
  POST /api/query               — full RAG turn, bilingual answer (+ streaming)
  POST /api/tantivy/reload      — re-open the BM25 index after fresh ingestion
  GET  /api/filters             — distinct metadata values for UI dropdowns
  GET  /api/analytics/mentions  — full-text mention count for a term
  GET  /api/analytics/speakers  — speakers ranked by mentions of a term
  GET  /api/analytics/transcripts — transcripts ranked by mentions of a term

Search / query requests carry two Phase-D quality switches: `auto_filters`
(extract metadata filters from the query text) and `expand_query` (HyDE).

Run locally:
    uvicorn rag_api.app:app --host 0.0.0.0 --port 8080
or:
    python -m rag_api.app
"""
from __future__ import annotations

import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_api.analytics import Analytics
from rag_api.config import Settings, get_settings
from rag_api.db import VocabCache
from rag_api.history import History
from rag_api.lang import resolve_language
from rag_api.pageindex import PageIndexRetriever
from rag_api.query_parse import (
    detect_quote, detect_signals, merge_filters, signals_to_filters,
)
from rag_api.retrieval import Retriever
from rag_api.synthesis import Synthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("rag_api.app")


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------


class FilterModel(BaseModel):
    """Optional metadata filters. All default to None (no constraint)."""

    speaker: str | None = None
    source_file: str | None = None
    season: str | None = None
    track_type: list[str] | str | None = None
    location: str | None = None
    event_id: str | None = None
    date_range: tuple[str, str] | None = None
    event_type: str | None = None
    primary_language: str | None = None
    topics: list[str] | None = None
    people_named: list[str] | None = None
    scriptures_referenced: list[str] | None = None
    # Phase 14: curated-catalog performer credits (singers/chorus).
    performers: list[str] | str | None = None
    # Phase 14: camp year (sheet column CampYear). Translated to a session_date
    # range so it filters both transcript and catalog by year.
    year: str | int | None = None


# Retrieval scope:
#   chunks    — hybrid chunk search (default; transcripts + blended catalog)
#   summaries — file-level summary search (cross-corpus / topic questions)
#   two_stage — summary search picks files, then chunk search within them
#   catalog   — curated-catalog only (performer/title lookups, accuracy A/B)
RetrievalScope = Literal["chunks", "summaries", "two_stage", "catalog"]

# Retrieval backend:
#   hybrid    — locked Qdrant + Tantivy + Infinity path (default, unchanged)
#   pageindex — local LLM tree-reasoning over pre-built section trees (opt-in
#               A/B comparison; see rag_api.pageindex). None => env default
#               (RETRIEVAL_BACKEND), resolved in `_resolve_backend`.
RetrievalBackend = Literal["hybrid", "pageindex"]


class SearchRequest(BaseModel):
    # min_length 0: an empty query + filters is a valid "browse by facet"
    # request (Phase 14) — return the catalog rows matching the filters.
    query: str = Field("", min_length=0)
    find_quote: bool = False
    scope: RetrievalScope = "chunks"
    backend: RetrievalBackend | None = None
    top_k: int = Field(8, ge=1, le=40)
    filters: FilterModel = Field(default_factory=FilterModel)
    # Extract metadata filters from the query text. Strong signals (years,
    # ISO dates) are applied; soft ones (season/topic words) are only
    # reported as suggestions. See rag_api.query_parse.
    auto_filters: bool = True
    # Expand the query with a HyDE hypothetical passage before dense search.
    # One extra chat-model call — opt-in.
    expand_query: bool = False
    # Blend curated-catalog results into the chunk search (Phase 14). Active
    # by default; a missing catalog collection degrades silently. Ignored for
    # non-chunk scopes (use scope="catalog" for catalog-only).
    include_catalog: bool = True


class QueryRequest(BaseModel):
    """A full RAG turn: retrieve, then synthesise a bilingual answer."""

    query: str = Field(..., min_length=1)
    find_quote: bool = False
    scope: RetrievalScope = "chunks"
    backend: RetrievalBackend | None = None
    top_k: int = Field(8, ge=1, le=40)
    filters: FilterModel = Field(default_factory=FilterModel)
    auto_filters: bool = True
    expand_query: bool = False
    # Blend curated-catalog results into the chunk search (Phase 14).
    include_catalog: bool = True
    # 'auto' detects Hindi/English from the query; the others force it.
    answer_language: Literal["auto", "hindi", "english"] = "auto"
    # When true, the answer is streamed back as Server-Sent Events.
    stream: bool = False
    # Optional per-request model override. Both must be set together and
    # must match an entry in /api/models — see `Settings.is_model_allowed`.
    # When omitted, the env-configured default is used.
    provider: Literal["ollama", "openrouter"] | None = None
    model: str | None = None


# --------------------------------------------------------------------------
# App + lifespan
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.retriever = Retriever(settings)
    # Additive, opt-in reasoning backend. Reuses the hybrid retriever for its
    # stage-1 doc selection + reranking. Never engaged unless a request asks
    # for backend=pageindex (or RETRIEVAL_BACKEND flips the default).
    app.state.pageindex = PageIndexRetriever(settings, app.state.retriever)
    app.state.synthesizer = Synthesizer(settings)
    app.state.analytics = Analytics(settings.pg_dsn)
    app.state.history = History(settings.pg_dsn)
    app.state.vocab_cache = VocabCache()
    log.info(
        "rag_api up — collection=%s bm25=%s chat=%s/%s backend=%s "
        "pageindex_trees=%d auth=%s",
        settings.qdrant_collection,
        app.state.retriever.bm25_enabled,
        app.state.synthesizer.provider,
        app.state.synthesizer.active_model,
        settings.retrieval_backend,
        len(app.state.pageindex.available_docs()),
        "on" if settings.dashboard_password else "OFF (dev)",
    )
    yield


app = FastAPI(title="transcript-rag dashboard API", version="0.1.0",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Auth — shared password via X-Dashboard-Password header
# --------------------------------------------------------------------------


def require_auth(
    x_dashboard_password: str | None = Header(default=None),
) -> None:
    """Reject requests whose password header doesn't match the configured
    shared password. If no password is configured, auth is disabled (dev)."""
    expected = get_settings().dashboard_password
    if not expected:
        return
    if not x_dashboard_password or not secrets.compare_digest(
        x_dashboard_password, expected
    ):
        raise HTTPException(status_code=401,
                            detail="invalid or missing dashboard password")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


def _probe(url: str, headers: dict[str, str] | None = None) -> bool:
    try:
        r = requests.get(url, headers=headers or {}, timeout=3)
        return r.status_code < 500
    except requests.RequestException:
        return False


def _check_services(s: Settings) -> dict[str, bool]:
    return {
        "ollama": _probe(f"{s.ollama_url}/api/tags"),
        "qdrant": _probe(
            f"{s.qdrant_url}/collections",
            headers={"api-key": s.qdrant_key} if s.qdrant_key else None,
        ),
        "reranker": _probe(f"{s.reranker_url.rstrip('/')}/health"),
    }


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    """Selectable models for the dashboard's Model dropdown.

    Unauthenticated by design — same posture as /api/health — so the login
    screen / public probe can read it. Returns only model IDs and labels;
    no API keys, no URLs, no secrets.
    """
    s: Settings = app.state.settings
    return {"models": s.available_models()}


def _resolve_model_choice(
    req: QueryRequest | SearchRequest,
) -> tuple[str | None, str | None]:
    """Validate the per-request provider+model override.

    Returns ``(provider, model)`` to pass through to the Synthesizer, or
    ``(None, None)`` when no override was supplied (use env defaults).
    Raises 400 on a malformed override (one field without the other, or a
    model not on the env allowlist).
    """
    p, m = getattr(req, "provider", None), getattr(req, "model", None)
    if p is None and m is None:
        return None, None
    if (p is None) != (m is None):
        raise HTTPException(
            status_code=400,
            detail="provider and model must be set together",
        )
    s: Settings = app.state.settings
    if not s.is_model_allowed(p, m):
        raise HTTPException(
            status_code=400,
            detail=f"model {p}/{m} is not in the configured allowlist",
        )
    return p, m


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Service reachability + index status. Unauthenticated by design so a
    monitor / the login screen can probe it."""
    s: Settings = app.state.settings
    retriever: Retriever = app.state.retriever
    pageindex: PageIndexRetriever = app.state.pageindex
    synth: Synthesizer = app.state.synthesizer
    services = _check_services(s)
    # When CHAT_PROVIDER=openrouter, Ollama may still be running for
    # embeddings — keep the health check honest about what's actually
    # required for /api/query to work.
    chat_reachable = (
        services["ollama"] if synth.provider == "ollama"
        else bool(s.openrouter_api_key)
    )
    return {
        "ok": chat_reachable and services["qdrant"],
        "services": services,
        "bm25_enabled": retriever.bm25_enabled,
        "tantivy_docs": retriever.tantivy_doc_count(),
        "retrieval_backend": s.retrieval_backend,
        "pageindex_trees": len(pageindex.available_docs()),
        "auth_required": bool(s.dashboard_password),
        "chat_provider": synth.provider,
        "chat_model": synth.active_model,
        "embed_model": s.embed_model,
    }


def _resolve_backend(req: SearchRequest | QueryRequest) -> str:
    """The effective retrieval backend: the per-request override if given, else
    the env default (RETRIEVAL_BACKEND, normally 'hybrid')."""
    return req.backend or app.state.settings.retrieval_backend


def _retrieve(
    retriever: Retriever, pageindex: PageIndexRetriever, query: str,
    find_quote: bool, scope: str, backend: str,
    filters: dict[str, Any], top_k: int, dense_text: str,
    include_catalog: bool = False,
) -> list[dict[str, Any]]:
    """Route a retrieval request to the right pipeline. `find_quote` is always
    a chunk-level lexical quote hunt (hybrid only — PageIndex has no lexical
    arm), so backend/scope are ignored when it is set. `backend=pageindex`
    routes to LLM tree-reasoning; scope/HyDE do not apply there. `scope=catalog`
    searches only the curated catalog; otherwise `include_catalog` blends it
    into the chunk search (Phase 14)."""
    # Filter-only browse: no query text, just facets -> return the matching
    # catalog rows directly (the sheet's data; the audio transcript once mapped).
    if not (query or "").strip():
        if any(v for v in filters.values()):
            return retriever.browse_catalog(filters, top_k)
        return []
    if find_quote:
        return retriever.find_quote(query, top_k)
    if backend == "pageindex":
        return pageindex.search(query, filters, top_k)
    dt = dense_text or None
    if scope == "summaries":
        return retriever.search_summaries(query, filters, top_k, dense_text=dt)
    if scope == "two_stage":
        return retriever.search_two_stage(query, filters, top_k, dense_text=dt)
    if scope == "catalog":
        return retriever.search_catalog(query, filters, top_k, dense_text=dt)
    return retriever.search(query, filters, top_k, dense_text=dt,
                            include_catalog=include_catalog)


def _prepare(req: SearchRequest | QueryRequest) -> tuple[
    dict[str, Any], list[dict[str, Any]], str, bool, str
]:
    """Phase-D pre-retrieval planning, shared by /api/search and /api/query.

    Returns ``(effective_filters, detections, dense_text, find_quote, query)``:
      * effective_filters — explicit request filters merged with auto-extracted
        strong signals (explicit always wins);
      * detections — every signal found in the query text, for the UI to show
        as removable / applyable chips;
      * dense_text — a HyDE hypothetical passage when `expand_query` is set,
        else "";
      * find_quote — effective quote-mode flag: the request's, OR auto-detected
        when the query is a pasted Devanagari verbatim passage;
      * query — the text to retrieve with: the original, or (when a quote is
        auto-detected) the quote with its trailing romanized question stripped.

    A `find_quote` request — explicit or auto-detected — skips filters and HyDE:
    it is a lexical exact-phrase hunt, so metadata filters and semantic
    expansion do not apply. (Synthesis/language detection still use the caller's
    original query, so the model answers the user's actual question.)
    """
    find_quote, query = req.find_quote, req.query
    if not find_quote:
        qd = detect_quote(req.query)
        if qd.is_quote:
            find_quote, query = True, qd.query

    if find_quote:
        return {}, [], "", find_quote, query

    s: Settings = app.state.settings
    explicit = {k: v for k, v in req.filters.model_dump().items()
                if v is not None}
    vocab = app.state.vocab_cache.get(s.pg_dsn)
    detections = detect_signals(req.query, vocab)
    auto = signals_to_filters(detections) if req.auto_filters else {}
    effective = merge_filters(explicit, auto)

    # Translate an explicit `year` facet into a session_date range (reuses all
    # the existing date plumbing, and works on both transcript and catalog).
    if effective.get("year") and not effective.get("date_range"):
        y = str(effective["year"]).strip()
        if y.isdigit():
            effective["date_range"] = (f"{y}-01-01", f"{y}-12-31")
    effective.pop("year", None)

    dense_text = ""
    if req.expand_query:
        dense_text = app.state.retriever.make_expansion(req.query)

    return effective, detections, dense_text, find_quote, query


@app.post("/api/search", dependencies=[Depends(require_auth)])
def search(req: SearchRequest) -> dict[str, Any]:
    """Retrieval only — structured results, no LLM synthesis (that is
    POST /api/query). `scope` selects chunk / summary / two-stage retrieval."""
    retriever: Retriever = app.state.retriever
    pageindex: PageIndexRetriever = app.state.pageindex
    backend = _resolve_backend(req)
    effective, detections, dense_text, find_quote, rquery = _prepare(req)
    t0 = time.monotonic()
    try:
        results = _retrieve(retriever, pageindex, rquery, find_quote,
                            req.scope, backend, effective, req.top_k, dense_text,
                            include_catalog=req.include_catalog)
    except requests.RequestException as e:
        raise HTTPException(status_code=502,
                            detail=f"upstream service error: {e}")
    except Exception as e:  # noqa: BLE001 - surface, don't swallow
        log.error("search failed for %r: %s", req.query, e)
        raise HTTPException(status_code=500, detail=f"retrieval failed: {e}")
    retrieval_ms = round((time.monotonic() - t0) * 1000.0, 1)
    return {
        "query": req.query,
        "find_quote": find_quote,
        "auto_quote": find_quote and not req.find_quote,
        "scope": req.scope,
        "backend": backend,
        "retrieval_ms": retrieval_ms,
        "count": len(results),
        "results": results,
        "detected_filters": detections,
        "applied_filters": effective,
        "expanded": bool(dense_text),
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event. ensure_ascii=False keeps Devanagari
    intact on the wire (the response is UTF-8)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/query", dependencies=[Depends(require_auth)])
def query(req: QueryRequest):
    """Full RAG turn: retrieve transcript passages, then synthesise a
    bilingual, citation-grounded answer.

    Non-streaming (`stream=false`): returns one JSON object with `answer`
    and `citations`. Streaming (`stream=true`): returns Server-Sent Events —
    a `meta` event (language + citations), then `token` events with answer
    deltas, then `done` (or `error`).
    """
    retriever: Retriever = app.state.retriever
    pageindex: PageIndexRetriever = app.state.pageindex
    synthesizer: Synthesizer = app.state.synthesizer

    # Per-request model override (400 if malformed). None,None = env default.
    override_provider, override_model = _resolve_model_choice(req)

    backend = _resolve_backend(req)
    effective, detections, dense_text, find_quote, rquery = _prepare(req)
    t0 = time.monotonic()
    try:
        results = _retrieve(retriever, pageindex, rquery, find_quote,
                            req.scope, backend, effective, req.top_k, dense_text,
                            include_catalog=req.include_catalog)
    except requests.RequestException as e:
        raise HTTPException(status_code=502,
                            detail=f"upstream service error: {e}")
    except Exception as e:  # noqa: BLE001 - surface, don't swallow
        log.error("query retrieval failed for %r: %s", req.query, e)
        raise HTTPException(status_code=500, detail=f"retrieval failed: {e}")
    retrieval_ms = round((time.monotonic() - t0) * 1000.0, 1)

    answer_language = resolve_language(req.answer_language, req.query)

    if not req.stream:
        try:
            answer = synthesizer.generate(
                req.query, results, answer_language,
                provider=override_provider, model=override_model,
            )
        except requests.RequestException as e:
            raise HTTPException(status_code=502,
                                detail=f"answer model error: {e}")
        return {
            "query": req.query,
            "find_quote": find_quote,
            "auto_quote": find_quote and not req.find_quote,
            "scope": req.scope,
            "backend": backend,
            "retrieval_ms": retrieval_ms,
            "answer_language": answer_language,
            "answer": answer,
            "count": len(results),
            "citations": results,
            "detected_filters": detections,
            "applied_filters": effective,
            "expanded": bool(dense_text),
        }

    def event_stream():
        yield _sse("meta", {
            "query": req.query,
            "find_quote": find_quote,
            "auto_quote": find_quote and not req.find_quote,
            "scope": req.scope,
            "backend": backend,
            "retrieval_ms": retrieval_ms,
            "answer_language": answer_language,
            "count": len(results),
            "citations": results,
            "detected_filters": detections,
            "applied_filters": effective,
            "expanded": bool(dense_text),
        })
        try:
            for delta in synthesizer.stream(
                req.query, results, answer_language,
                provider=override_provider, model=override_model,
            ):
                yield _sse("token", {"text": delta})
        except requests.RequestException as e:
            log.error("answer streaming failed for %r: %s", req.query, e)
            yield _sse("error", {"detail": f"answer model error: {e}"})
        else:
            yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/tantivy/reload", dependencies=[Depends(require_auth)])
def tantivy_reload() -> dict[str, Any]:
    """Re-open the BM25 index — call after ingesting new transcripts."""
    retriever: Retriever = app.state.retriever
    docs = retriever.reload_tantivy()
    return {"ok": True, "bm25_enabled": retriever.bm25_enabled, "docs": docs}


@app.get("/api/filters", dependencies=[Depends(require_auth)])
def filter_options() -> dict[str, Any]:
    """Distinct metadata values for the dashboard's filter dropdowns,
    sourced from Postgres file_meta via the shared TTL cache (also used by
    query-time filter extraction). db_ok=false means the cache has never
    loaded — the dashboard still renders, with empty dropdowns."""
    s: Settings = app.state.settings
    cache: VocabCache = app.state.vocab_cache
    options = cache.get(s.pg_dsn)
    return {"db_ok": cache.ok, "options": options}


# --------------------------------------------------------------------------
# Analytics — Postgres-backed corpus statistics (no LLM)
# --------------------------------------------------------------------------


def _run_analytics(label: str, fn, *args: Any) -> dict[str, Any]:
    """Call an analytics method, mapping failures to a 502 instead of a 500 —
    analytics is a Postgres dependency, like the retrieval upstreams."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - surface, don't swallow
        log.error("analytics %s failed for %r: %s", label, args, e)
        raise HTTPException(status_code=502, detail=f"analytics query failed: {e}")


@app.get("/api/analytics/mentions", dependencies=[Depends(require_auth)])
def analytics_mentions(
    term: str = Query(..., min_length=1),
    speaker: str | None = None,
) -> dict[str, Any]:
    """How many chunks mention `term` (full-text), optionally for one speaker."""
    a: Analytics = app.state.analytics
    return _run_analytics("mentions", a.count_mentions, term, speaker)


@app.get("/api/analytics/speakers", dependencies=[Depends(require_auth)])
def analytics_speakers(
    term: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    """Speakers ranked by how many `term`-matching chunks they appear in."""
    a: Analytics = app.state.analytics
    return _run_analytics("speakers", a.top_speakers, term, limit)


@app.get("/api/analytics/transcripts", dependencies=[Depends(require_auth)])
def analytics_transcripts(
    term: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=200),
) -> dict[str, Any]:
    """Transcript files ranked by how many `term`-matching chunks they hold."""
    a: Analytics = app.state.analytics
    return _run_analytics("transcripts", a.list_transcripts, term, limit)


# --------------------------------------------------------------------------
# Conversation history — Postgres-backed Q&A log for the sidebar
# --------------------------------------------------------------------------


class HistorySaveRequest(BaseModel):
    """One completed Q&A turn the dashboard wants to persist.

    Mirrors the data the UI already has after a /api/query response — we just
    forward it to Postgres. The full citation list and applied/detected
    filters ride along as JSONB so the read-only viewer can re-render the
    exact past answer without re-running retrieval.
    """

    question: str = Field(..., min_length=1)
    answer: str = ""
    mode: Literal["answer", "search"] = "answer"
    scope: RetrievalScope = "chunks"
    top_k: int = Field(8, ge=1, le=40)
    find_quote: bool = False
    expanded: bool = False
    answer_language: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    detected_filters: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)


def _run_history(label: str, fn, *args: Any) -> Any:
    """Call a history method, mapping infra failures to 502 like analytics."""
    try:
        return fn(*args)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 - surface, don't swallow
        log.error("history %s failed for %r: %s", label, args, e)
        raise HTTPException(status_code=502, detail=f"history query failed: {e}")


@app.get("/api/history", dependencies=[Depends(require_auth)])
def history_list(
    limit: int = Query(200, ge=1, le=500),
) -> dict[str, Any]:
    """Sidebar listing — id, title, created_at, mode, scope (no bodies)."""
    h: History = app.state.history
    items = _run_history("list", h.list_summaries, limit)
    return {"items": items, "count": len(items)}


@app.get("/api/history/{conversation_id}", dependencies=[Depends(require_auth)])
def history_get(conversation_id: str) -> dict[str, Any]:
    """Full conversation record for the read-only viewer."""
    h: History = app.state.history
    record = _run_history("get", h.get, conversation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return record


@app.post("/api/history", dependencies=[Depends(require_auth)])
def history_save(req: HistorySaveRequest) -> dict[str, Any]:
    """Persist one completed Q&A turn. Returns the saved record."""
    h: History = app.state.history
    return _run_history("save", h.save, req.model_dump())


@app.delete("/api/history/{conversation_id}",
            dependencies=[Depends(require_auth)])
def history_delete(conversation_id: str) -> dict[str, Any]:
    """Remove one conversation by id."""
    h: History = app.state.history
    deleted = _run_history("delete", h.delete, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"ok": True, "id": conversation_id}


# --------------------------------------------------------------------------
# Static dashboard — Phase F. The Vite build output is served at `/` so the
# same container ships both the API and the UI. Mounted *after* the API
# routes so /api/* is never shadowed; absent in dev (where Vite serves the
# UI on :5173 and proxies /api → :8080).
# --------------------------------------------------------------------------

class _DashboardStatic(StaticFiles):
    """StaticFiles that tells the browser never to cache the HTML shell.

    Vite fingerprints JS/CSS (new filename per build) so those are safe to cache
    forever — but `index.html` references them, and a browser that caches the
    old index.html keeps loading the old bundle, so a redeploy appears to do
    nothing (stale logo / UI). Marking HTML `no-cache` forces a revalidate, so
    each deploy is picked up on a normal reload."""

    async def get_response(self, path: str, scope: Any) -> Any:
        resp = await super().get_response(path, scope)
        if resp.headers.get("content-type", "").startswith("text/html"):
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


_STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _STATIC_DIR.is_dir():
    # `html=True` makes StaticFiles return `index.html` for `/` and any
    # directory request — enough for this single-page UI (no client router).
    app.mount("/", _DashboardStatic(directory=str(_STATIC_DIR), html=True),
              name="dashboard")
    log.info("dashboard static files mounted from %s", _STATIC_DIR)
else:
    log.info("no dashboard build at %s — running API only (dev mode)",
             _STATIC_DIR)


def main() -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run("rag_api.app:app", host=s.api_host, port=s.api_port,
                log_level="info")


if __name__ == "__main__":
    main()
