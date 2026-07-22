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
from urllib.parse import urlencode

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_api import tracks
from rag_api.analytics import Analytics
from rag_api.best import BestSittings
from rag_api.config import Settings, get_settings
from rag_api.corpus import CorpusReader
from rag_api.db import VocabCache, record_play
from rag_api.history import History
from rag_api.lang import resolve_language
from rag_api.pageindex import PageIndexRetriever
from rag_api.query_parse import (
    detect_quote, detect_signals, merge_filters, signals_to_filters,
)
from rag_api.followup import FollowupRewriter
from rag_api.ground import GroundChecker
from rag_api.planner import INTENT_BEST, INTENT_LIST, INTENT_VERBATIM, Planner
from rag_api.retrieval import Retriever
from rag_api.route import route
from rag_api.synthesis import (
    Synthesizer, build_best_question, build_list_question,
    build_verbatim_question, chitchat_reply, filter_min_score,
)

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
    top_k: int = Field(4, ge=1, le=40)
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
    top_k: int = Field(4, ge=1, le=40)
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
    # Prior conversation turns, oldest first. Each: {question, answer} (the
    # dashboard's shape) or {role, content}. When present and the router tags
    # the query `followup`, it is rewritten to a standalone query before
    # retrieval (Stage 4.3, gated by RAG_FOLLOWUP_REWRITE). Empty = no rewrite.
    history: list[dict[str, Any]] = Field(default_factory=list)


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
    app.state.followup = FollowupRewriter(settings)
    app.state.synthesizer = Synthesizer(settings)
    app.state.ground = GroundChecker(settings)
    app.state.planner = Planner(settings)
    app.state.best = BestSittings(settings)
    app.state.analytics = Analytics(settings.pg_dsn)
    app.state.history = History(settings.pg_dsn)
    app.state.corpus = CorpusReader(settings.pg_dsn)
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
        # File-level summary index (scope=summaries / two_stage). 0 until
        # ingestion.build_summary_index has run.
        "summary_docs": retriever.summary_doc_count(),
        "retrieval_backend": s.retrieval_backend,
        "pageindex_trees": len(pageindex.available_docs()),
        "auth_required": bool(s.dashboard_password),
        "chat_provider": synth.provider,
        "chat_model": synth.active_model,
        "embed_model": s.embed_model,
        # Whether the two `:ro` media mounts actually landed. Docker silently
        # creates an empty directory for a missing bind source, so the track
        # panel degrades to "audio unavailable" instead of 404-ing every click.
        "transcripts_mounted": tracks.tree_is_mounted(Path(s.transcripts_dir)),
        "isolated_mounted": tracks.tree_is_mounted(Path(s.isolated_dir)),
    }


def _resolve_backend(req: SearchRequest | QueryRequest) -> str:
    """The effective retrieval backend: the per-request override if given, else
    the env default (RETRIEVAL_BACKEND, normally 'hybrid')."""
    return req.backend or app.state.settings.retrieval_backend


def _retrieve(
    retriever: Retriever, pageindex: PageIndexRetriever, query: str,
    find_quote: bool, scope: str, backend: str,
    filters: dict[str, Any], top_k: int, dense_text: str,
    include_catalog: bool = False, bm25_weight: float | None = None,
    plan: dict[str, Any] | None = None,
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
    # Agentic plan (chunk scope only — `_prepare` guarantees that): list
    # intent retrieves at the sitting level (one summary result per file);
    # best intent re-ranks sittings by relevance + mentions + length + plays;
    # verbatim intent is a chunk search with a boosted budget (the answer
    # quotes exact lines, so coverage matters); a multi-query plan runs every
    # variant and merges.
    if plan:
        s: Settings = app.state.settings
        if plan["intent"] == INTENT_BEST:
            return app.state.best.rank(
                retriever, plan["queries"], filters,
                top_k=max(top_k, plan["n"] or top_k), pool=s.list_top_k)
        if plan["intent"] == INTENT_LIST:
            return retriever.search_summaries_multi(
                plan["queries"], filters, top_k=max(top_k, plan["n"] or top_k))
        if plan["intent"] == INTENT_VERBATIM:
            # Always search the plan's queries: the planner stripped the
            # summary meta-words from them ("summary of X" reranks ~0 against
            # transcript prose), so `query` here is NOT the right search text.
            return retriever.search_multi(
                plan["queries"], filters, max(top_k, s.verbatim_top_k),
                bm25_weight=bm25_weight, dense_text=dt,
                include_catalog=include_catalog)
        if len(plan["queries"]) > 1:
            return retriever.search_multi(
                plan["queries"], filters, top_k, bm25_weight=bm25_weight,
                dense_text=dt, include_catalog=include_catalog)
    return retriever.search(query, filters, top_k, bm25_weight=bm25_weight,
                            dense_text=dt, include_catalog=include_catalog)


def _route_info(
    query_class: str | None, bm25_weight: float | None,
    include_catalog: bool, rewritten: str | None, retrieve: bool = True,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The routing summary `_prepare` returns to the handlers (echoed in the
    response and used by `_retrieve`). ``retrieve=False`` (chitchat only) tells
    the handler to answer with a fixed reply — no retrieval, no synthesis.
    ``plan`` is the agentic planner's verdict (``{intent, queries, n}``, n
    already clamped) or None when the planner is off / had nothing to add."""
    return {
        "query_class": query_class, "bm25_weight": bm25_weight,
        "include_catalog": include_catalog, "rewritten_query": rewritten,
        "retrieve": retrieve, "plan": plan,
    }


def _prepare(req: SearchRequest | QueryRequest) -> tuple[
    dict[str, Any], list[dict[str, Any]], str, bool, str, dict[str, Any]
]:
    """Phase-D pre-retrieval planning, shared by /api/search and /api/query.

    Returns ``(effective_filters, detections, dense_text, find_quote, query,
    route_info)``:
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
      * route_info — ``{query_class, bm25_weight, include_catalog}``. When the
        Stage 4.1 router is off (default), it just echoes the request's
        `include_catalog` with no class/weight override, so the pipeline is
        byte-for-byte unchanged. When on, per-class settings from
        `rag_api.route` take effect (catalog off, per-class BM25 weight).

    A `find_quote` request — explicit or auto-detected — skips filters and HyDE:
    it is a lexical exact-phrase hunt, so metadata filters and semantic
    expansion do not apply. (Synthesis/language detection still use the caller's
    original query, so the model answers the user's actual question.)
    """
    s: Settings = app.state.settings
    find_quote, query = req.find_quote, req.query
    query_class: str | None = None
    bm25_weight: float | None = None
    include_catalog = req.include_catalog
    route_expand = False
    rewritten: str | None = None
    history = getattr(req, "history", None) or []

    if s.router_enabled and not find_quote:
        decision = route(req.query, s, history_present=bool(history))
        query_class = decision.query_class
        if query_class == "followup" and s.followup_rewrite_enabled:
            # Rewrite the context-dependent follow-up to a standalone query,
            # then re-route it so it inherits its underlying class (4.3).
            rewritten = app.state.followup.rewrite(req.query, history)
            decision = route(rewritten, s, history_present=False)
            query_class = decision.query_class
        if not decision.retrieve:
            # Chitchat: never reaches the retriever — the reranker scores a
            # bare "hi" at ~0.55, far above any sane citation floor, so a
            # greeting must be gated here, not filtered downstream.
            return {}, [], "", False, req.query, _route_info(
                query_class, None, False, rewritten, retrieve=False)
        if decision.find_quote:
            find_quote, query = True, decision.query
        else:
            query = rewritten if rewritten is not None else query
            bm25_weight = decision.bm25_weight
            include_catalog = decision.include_catalog
            route_expand = decision.expand_query
    elif not find_quote:
        qd = detect_quote(req.query)
        if qd.is_quote:
            find_quote, query = True, qd.query

    if find_quote:
        return {}, [], "", find_quote, query, _route_info(
            query_class or "quote", None, False, rewritten)

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

    # Agentic planner (RAG_PLANNER): detects list intent ("10 sittings for
    # rain") and adds bilingual/synonym retrieval variants ("barish" → बारिश /
    # rain). Only for the router's retrieval classes — chitchat/quote already
    # returned above — and only on the default chunk scope (an explicit scope
    # is the caller's choice). Fail-open: no plan = today's pipeline.
    plan: dict[str, Any] | None = None
    if (s.planner_enabled and query_class in ("thematic", "name", "analytic")
            and req.scope == "chunks" and (query or "").strip()):
        p = app.state.planner.plan(query)
        if p is not None:
            n = (min(p.n or 10, s.list_top_k)
                 if p.intent in (INTENT_LIST, INTENT_BEST) else None)
            plan = {"intent": p.intent, "queries": p.queries, "n": n}

    # HyDE fires when the caller asks (expand_query) or when the router flags a
    # thematic query and RAG_HYDE_THEMATIC is on (4.2). One ~1 s chat round-trip.
    # A multi-query plan supersedes it: the variants already bridge vocabulary,
    # and list/best-intent retrieval is summary-level — either way a second LLM
    # call would buy nothing.
    dense_text = ""
    multi = bool(plan and len(plan["queries"]) > 1)
    summary_intent = bool(plan and plan["intent"] in (INTENT_LIST, INTENT_BEST))
    if not (multi or summary_intent) and (
            req.expand_query or (route_expand and s.hyde_thematic_enabled)):
        dense_text = app.state.retriever.make_expansion(req.query)

    return effective, detections, dense_text, find_quote, query, _route_info(
        query_class, bm25_weight, include_catalog, rewritten, plan=plan)


@app.post("/api/search", dependencies=[Depends(require_auth)])
def search(req: SearchRequest) -> dict[str, Any]:
    """Retrieval only — structured results, no LLM synthesis (that is
    POST /api/query). `scope` selects chunk / summary / two-stage retrieval."""
    retriever: Retriever = app.state.retriever
    pageindex: PageIndexRetriever = app.state.pageindex
    backend = _resolve_backend(req)
    effective, detections, dense_text, find_quote, rquery, route_info = _prepare(req)
    if not route_info["retrieve"]:
        return {
            "query": req.query, "plan": None, "find_quote": False,
            "auto_quote": False,
            "query_class": route_info["query_class"], "scope": req.scope,
            "backend": backend, "retrieval_ms": 0.0, "count": 0, "results": [],
            "detected_filters": [], "applied_filters": {}, "expanded": False,
        }
    t0 = time.monotonic()
    try:
        results = _retrieve(retriever, pageindex, rquery, find_quote,
                            req.scope, backend, effective, req.top_k, dense_text,
                            include_catalog=route_info["include_catalog"],
                            bm25_weight=route_info["bm25_weight"],
                            plan=route_info["plan"])
    except requests.RequestException as e:
        raise HTTPException(status_code=502,
                            detail=f"upstream service error: {e}")
    except Exception as e:  # noqa: BLE001 - surface, don't swallow
        log.error("search failed for %r: %s", req.query, e)
        raise HTTPException(status_code=500, detail=f"retrieval failed: {e}")
    retrieval_ms = round((time.monotonic() - t0) * 1000.0, 1)
    return {
        "query": req.query,
        "plan": route_info["plan"],
        "find_quote": find_quote,
        "auto_quote": find_quote and not req.find_quote,
        "query_class": route_info["query_class"],
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


def _chitchat_response(req: QueryRequest, backend: str,
                       route_info: dict[str, Any],
                       provider: str | None = None, model: str | None = None):
    """Reply for the router's chitchat class — no retrieval, no citations.
    With RAG_CHITCHAT_LLM (default on) the chat model writes a natural reply
    (capability system prompt + recent turns, think=false); any model failure
    falls back to the fixed bilingual string, so this path never 5xxes.
    Mirrors the normal response/SSE shape (0 citations) so the frontend needs
    no special case."""
    s: Settings = app.state.settings
    synthesizer: Synthesizer = app.state.synthesizer
    answer_language = resolve_language(req.answer_language, req.query)
    history = getattr(req, "history", None) or []
    base = {
        "query": req.query, "plan": None, "find_quote": False,
        "auto_quote": False,
        "query_class": route_info["query_class"],
        "rewritten_query": route_info["rewritten_query"],
        "scope": req.scope, "backend": backend, "retrieval_ms": 0.0,
        "answer_language": answer_language, "count": 0, "citations": [],
        "detected_filters": [], "applied_filters": {}, "expanded": False,
    }
    if not req.stream:
        answer = ""
        if s.chitchat_llm_enabled:
            try:
                answer = synthesizer.chitchat_generate(
                    req.query, answer_language, history,
                    provider=provider, model=model)
            except requests.RequestException as e:
                log.warning("chitchat LLM failed for %r: %s — fixed reply",
                            req.query, e)
        return {**base, "answer": answer or chitchat_reply(answer_language)}

    def event_stream():
        yield _sse("meta", base)
        emitted = False
        if s.chitchat_llm_enabled:
            try:
                for delta in synthesizer.chitchat_stream(
                        req.query, answer_language, history,
                        provider=provider, model=model):
                    emitted = True
                    yield _sse("token", {"text": delta})
            except requests.RequestException as e:
                # Fall back only when nothing streamed yet — appending the
                # canned reply after partial model output would read as noise.
                log.warning("chitchat LLM stream failed for %r: %s — %s",
                            req.query, e,
                            "fixed reply" if not emitted else "partial kept")
        if not emitted:
            yield _sse("token", {"text": chitchat_reply(answer_language)})
        yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
    effective, detections, dense_text, find_quote, rquery, route_info = _prepare(req)
    if not route_info["retrieve"]:
        return _chitchat_response(req, backend, route_info,
                                  override_provider, override_model)
    t0 = time.monotonic()
    try:
        results = _retrieve(retriever, pageindex, rquery, find_quote,
                            req.scope, backend, effective, req.top_k, dense_text,
                            include_catalog=route_info["include_catalog"],
                            bm25_weight=route_info["bm25_weight"],
                            plan=route_info["plan"])
    except requests.RequestException as e:
        raise HTTPException(status_code=502,
                            detail=f"upstream service error: {e}")
    except Exception as e:  # noqa: BLE001 - surface, don't swallow
        log.error("query retrieval failed for %r: %s", req.query, e)
        raise HTTPException(status_code=500, detail=f"retrieval failed: {e}")
    retrieval_ms = round((time.monotonic() - t0) * 1000.0, 1)

    # Citation floor (chat only): keep only passages the reranker scored above
    # the configured threshold — those are what gets cited, shown in the UI, and
    # fed to the answer model. Applied here (not in /api/search) so the model's
    # numbered context and the displayed citation cards stay in lockstep.
    # Best-sitting results are exempt: their `score` is the composite rank, not
    # a relevance — the semantic floor was already applied inside best.rank().
    s: Settings = app.state.settings
    plan = route_info["plan"]
    if not (plan and plan["intent"] == INTENT_BEST):
        n_before = len(results)
        results = filter_min_score(results, s.cite_min_score, s.allow_abstain)
        if len(results) != n_before:
            log.info(
                "cite floor %.2f (abstain=%s) dropped %d/%d passages for %r",
                s.cite_min_score, s.allow_abstain, n_before - len(results),
                n_before, req.query,
            )

    answer_language = resolve_language(req.answer_language, req.query)

    # Plan-intent synthesis wrappers (grounding / citation rules unchanged):
    # list — enumerate the retrieved sittings; best — ranked best-first list
    # with the Mentions/Length/Plays rationale; verbatim — bullet quotes copied
    # word-for-word. All other intents synthesise the raw query.
    synth_query = req.query
    if plan and results:
        if plan["intent"] == INTENT_LIST:
            synth_query = build_list_question(
                req.query, plan["n"] or len(results), answer_language)
        elif plan["intent"] == INTENT_BEST:
            synth_query = build_best_question(
                req.query, plan["n"] or len(results), answer_language)
        elif plan["intent"] == INTENT_VERBATIM:
            synth_query = build_verbatim_question(req.query, answer_language)

    if not req.stream:
        try:
            answer = synthesizer.generate(
                synth_query, results, answer_language,
                provider=override_provider, model=override_model,
            )
        except requests.RequestException as e:
            raise HTTPException(status_code=502,
                                detail=f"answer model error: {e}")
        if s.groundedness_enabled and results and answer:
            # Opt-in post-check (RAG_GROUNDEDNESS): drop answer sentences the
            # judge finds unsupported by the passages. Fail-open inside check().
            answer, dropped = app.state.ground.check(answer, results)
            if dropped:
                log.info("groundedness dropped %d sentence(s) for %r",
                         dropped, req.query)
        return {
            "query": req.query,
            "plan": plan,
            "find_quote": find_quote,
            "auto_quote": find_quote and not req.find_quote,
            "query_class": route_info["query_class"],
            "rewritten_query": route_info["rewritten_query"],
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
            "plan": plan,
            "find_quote": find_quote,
            "auto_quote": find_quote and not req.find_quote,
            "query_class": route_info["query_class"],
            "rewritten_query": route_info["rewritten_query"],
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
                synth_query, results, answer_language,
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
# Corpus map — the archive as a tree. Read-only; Postgres is the only source.
#
# Declared here, well above the `app.mount("/", StaticFiles(...))` at the foot
# of this module: the mount matches every path after it, so a route registered
# below it is unreachable.
# --------------------------------------------------------------------------


@app.get("/api/corpus/summary", dependencies=[Depends(require_auth)])
def corpus_summary() -> dict[str, Any]:
    """Whole-archive totals plus a `version` the map polls for live refresh.

    `version` moves when (n_files, n_chunks, max_ingested_at) moves — so an
    ingest that only *replaces* a file's chunks still registers.
    """
    c: CorpusReader = app.state.corpus
    return _run_analytics("corpus_summary", c.summary)


@app.get("/api/corpus/state", dependencies=[Depends(require_auth)])
def corpus_state(
    prefix: str = Query("", max_length=512),
    depth: int = Query(1, ge=1, le=4),
) -> dict[str, Any]:
    """The nodes at most `depth` levels below `prefix` ("" is the archive root).

    depth=3 from the root draws the container skeleton (collections, events,
    sessions) in one request; depth=1 from a session reveals its tracks.
    """
    c: CorpusReader = app.state.corpus
    return _run_analytics("corpus_state", c.state, prefix, depth)


# --------------------------------------------------------------------------
# One track — its transcript, and its isolated vocals.
#
# Read-only, like the corpus map. Two `:ro` bind mounts are the entire new
# surface. `file_meta` is the allowlist: a key that is not a row is a 404
# before any path is built, so there is nothing to traverse.
# --------------------------------------------------------------------------


@app.get("/api/track/transcript", dependencies=[Depends(require_auth)])
def track_transcript(source_file: str = Query(..., max_length=512)) -> dict[str, Any]:
    """A track's segments, its cleaned text, and a ticketed URL for its audio."""
    s: Settings = app.state.settings
    c: CorpusReader = app.state.corpus

    if not _run_analytics("track_lookup", c.has_track, source_file):
        log.info("track not indexed: %r", source_file)
        raise HTTPException(status_code=404, detail="no such track in the archive")

    try:
        raw_path = tracks.transcript_path(source_file, Path(s.transcripts_dir))
    except ValueError as e:
        # An indexed key that cannot be turned into a path means the ingest
        # convention drifted. Loud: the parity test should have caught it first.
        log.error("indexed key is not a usable path %r: %s", source_file, e)
        raise HTTPException(status_code=404, detail="track has no readable transcript")

    raw = tracks.read_json(raw_path)
    if raw is None:
        log.warning("transcript missing or unreadable on disk: %s", raw_path)
        raise HTTPException(status_code=404, detail="transcript not on disk")

    out = tracks.slim(raw)
    out["source_file"] = source_file

    cleaned = tracks.read_json(tracks.cleaned_path(source_file, Path(s.transcripts_dir)))
    out["cleaned_text"] = (cleaned or {}).get("cleaned_text")
    # Set when the Qwen polish was discarded (summarized, or a repetition loop).
    out["clean_note"] = (cleaned or {}).get("clean_note")

    wav = tracks.audio_path(raw.get("audio_file"), source_file, Path(s.isolated_dir))
    if wav is None:
        log.info("no isolated audio for %r", source_file)
        out["audio_url"] = None
    else:
        q = urlencode({"source_file": source_file, "t": tracks.mint_ticket(source_file)})
        out["audio_url"] = f"/api/track/audio?{q}"
    return out


@app.get("/api/track/audio")
def track_audio(
    source_file: str = Query(..., max_length=512),
    # Defaulted, not required: an omitted ticket is an authorization failure, and
    # should read as one. `Query(...)` would answer 422 with a validation schema.
    t: str = Query("", max_length=128),
) -> FileResponse:
    """Stream the isolated vocals. Ticket-gated, because `<audio src>` cannot
    send the `X-Dashboard-Password` header and the password must never ride in a
    query string. The ticket was minted by `/api/track/transcript`, which is
    header-authed and checks `file_meta` — so reaching here proves both, and no
    second database round-trip is needed on each of the Range requests Chrome
    fires while the user drags the scrubber.

    `FileResponse` answers Range with a 206 on its own; there is nothing to add.
    """
    if not tracks.check_ticket(source_file, t):
        log.info("rejected audio ticket for %r", source_file)
        raise HTTPException(status_code=403, detail="expired or invalid audio ticket")

    s: Settings = app.state.settings
    raw = tracks.read_json(tracks.transcript_path(source_file, Path(s.transcripts_dir)))
    wav = tracks.audio_path((raw or {}).get("audio_file"), source_file,
                            Path(s.isolated_dir))
    if wav is None:
        log.warning("ticket valid but no isolated audio on disk for %r", source_file)
        raise HTTPException(status_code=404, detail="isolated audio not on disk")

    return FileResponse(wav, media_type="audio/wav", filename=wav.name)


class PlayedRequest(BaseModel):
    """One listen of a recording, reported by the dashboard's track player."""

    source_file: str = Field(..., max_length=512)


@app.post("/api/track/played", dependencies=[Depends(require_auth)])
def track_played(req: PlayedRequest) -> dict[str, Any]:
    """Record one play of a recording (migration 004's `play_events`).

    Feeds the best-sitting ranking's popularity signal. The frontend fires
    this once per listen and ignores failures — playback never depends on it.
    """
    sf = (req.source_file or "").strip()
    if not sf:
        raise HTTPException(status_code=400, detail="source_file is required")
    try:
        record_play(app.state.settings.pg_dsn, sf)
    except Exception as e:  # noqa: BLE001 - surface, don't swallow
        log.warning("play event failed for %r: %s", sf, e)
        raise HTTPException(status_code=502, detail=f"play event failed: {e}")
    return {"ok": True, "source_file": sf}


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
    top_k: int = Field(4, ge=1, le=40)
    find_quote: bool = False
    expanded: bool = False
    answer_language: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    detected_filters: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    # Append this turn to an existing thread. None (or absent) starts a new
    # conversation; the save response echoes the id to reuse for later turns.
    conversation_id: str | None = None


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
