"""Configuration for the RAG dashboard API.

Values come from environment variables, with a minimal `.env` loader so the
service can run as a host process without the bash export dance. Container
deployments get their env from docker-compose and the `.env` file is simply
absent / ignored.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_flag(name: str, default: str = "off") -> bool:
    """Truthy env var — accepts 1/true/yes/on (case-insensitive)."""
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a `.env` file without overriding existing vars.

    A 12-line loader avoids a python-dotenv dependency for one tiny job. Only
    plain `KEY=VALUE` lines are honoured; `${VAR}` interpolation is not — keep
    real interpolation out of `.env` values this service reads.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(_REPO_ROOT / ".env")


def _resolve_pg_dsn() -> str:
    """Postgres DSN for analytics + filter-option queries.

    Prefers an explicit PG_DSN / POSTGRES_DSN, but only if it has no
    unexpanded `${VAR}` placeholder (the committed `.env.example` carries
    one). Otherwise builds the DSN from POSTGRES_PASSWORD — same shape
    `open_webui_functions/analytics.py` uses.
    """
    explicit = os.environ.get("PG_DSN") or os.environ.get("POSTGRES_DSN")
    if explicit and "${" not in explicit:
        return explicit
    return "postgresql://owui:{}@localhost:5432/openwebui".format(
        os.environ.get("POSTGRES_PASSWORD", "")
    )


def _dedup_with_default(default_model: str, raw_csv: str) -> list[str]:
    """Split a comma-separated env var into a clean, dedup'd model list.

    The default model is always position-zero so the dropdown opens on
    whichever model the operator picked as the deployment default.
    """
    extras = [m.strip() for m in (raw_csv or "").split(",") if m.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for m in [default_model] + extras:
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


class Settings(BaseModel):
    """Resolved runtime configuration. Construct via `get_settings()`."""

    # --- Qdrant ---
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_key: str = os.environ.get("QDRANT_API_KEY", "")
    qdrant_collection: str = os.environ.get("QDRANT_COLLECTION", "transcripts")
    # One vector per file: bge-m3 embedding of the file's summary + tags.
    qdrant_summary_collection: str = os.environ.get(
        "QDRANT_SUMMARY_COLLECTION", "transcript_summaries"
    )
    # Phase 14: standalone curated-catalog collection (hand transcriptions +
    # canonical titles). Searched alongside transcripts as a separate, labelled
    # source. Missing collection (404) degrades to transcript-only silently.
    qdrant_catalog_collection: str = os.environ.get(
        "QDRANT_CATALOG_COLLECTION", "catalog"
    )

    # --- Chat provider ---
    # "ollama" (default — strictly local, per PRD) or "openrouter" (opt-in
    # cloud fallback for A/B testing larger models that don't fit the GPU).
    # OpenRouter is the only exception to the "strictly local" rule, gated
    # behind this env var. Default deployment stays local.
    chat_provider: str = os.environ.get("CHAT_PROVIDER", "ollama").lower()

    # --- Ollama (local default) ---
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    embed_model: str = os.environ.get("EMBED_MODEL", "bge-m3")
    # Answer-generation model. Kept configurable so swapping in the fine-tuned
    # 26B model later is a one-line .env change — no code edit.
    chat_model: str = os.environ.get("CHAT_MODEL", "qwen3.5:9b")
    chat_num_ctx: int = int(os.environ.get("CHAT_NUM_CTX", "8192"))
    # Prompt-processing batch size (ollama `num_batch`). RAG prompts are large
    # (system prompt + retrieved passages near num_ctx), so prefill dominates
    # time-to-first-token. A wider batch lets the GPU chew through the prompt
    # faster — a pure throughput knob with no effect on output quality. 512 is
    # ollama's default; 1024 noticeably cuts TTFT on the RTX 5090.
    chat_num_batch: int = int(os.environ.get("CHAT_NUM_BATCH", "1024"))
    chat_temperature: float = float(os.environ.get("CHAT_TEMPERATURE", "0.2"))
    # Anti-repetition penalty (ollama `repeat_penalty`). 1.0 = off. A mild 1.1
    # stops the rare degenerate loop where a small model repeats a sentence for
    # hundreds of tokens (observed on qwen3.5:9b) — bounds worst-case latency
    # without truncating good answers or harming normal phrasing. Quality-safe.
    chat_repeat_penalty: float = float(
        os.environ.get("CHAT_REPEAT_PENALTY", "1.1")
    )
    # Before synthesis, drop passages scoring below this fraction of the top hit.
    # On a precise hit (e.g. a located quote: top ~0.8, rest ~0.04) the tail is
    # noise, and a smaller chat model otherwise refuses ("not found"). 0.2 keeps a
    # located quote's single match while still keeping the cluster of relevant
    # passages on a semantic query (top ~0.93 keeps everything >= ~0.19). 0 = off.
    synthesis_min_score_ratio: float = float(
        os.environ.get("SYNTH_MIN_SCORE_RATIO", "0.2")
    )
    # Citation floor for the chat answer flow: only passages scoring strictly
    # above this (reranker relevance, ~[0,1]) are cited and shown in the UI, and
    # only they are fed to the answer model. Calibrated from the §0.2 histogram
    # (eval/results/0_2_score_histogram.md): known-good top hits span 0.04–0.997
    # (Devanagari median 0.93, cross-script Latin as low as 0.04), so the old 0.75
    # dropped the single best chunk for ~30% of legitimate queries. 0.03 keeps
    # 100% of known-good top hits while still trimming the very weakest tail.
    # 0 disables it. The Stage 3.3 negative-set sweep gives the floor menu:
    # 0.03–0.10 catches ~40% of invented-entity queries (0% false abstention),
    # 0.20 → ~60% (~6% false), 0.50 → ~80% (~6% false; the false abstentions
    # are low-scoring cross-script hits). Deployment ships 0.2 via .env; the
    # code default stays the permissive 0.03.
    cite_min_score: float = float(os.environ.get("RAG_CITE_MIN_SCORE", "0.03"))
    # When true, drop the "keep the single best passage" safety net in
    # filter_min_score so a query with nothing above cite_min_score truly
    # abstains (→ NO_CONTEXT_MESSAGE, 0 citations). Code default false (the
    # always-answer net); deployment turns it ON via .env now that the floor
    # is certified against the §3.3 negative set — without it the model is
    # force-fed one irrelevant passage and answers anyway.
    allow_abstain: bool = _env_flag("RAG_ALLOW_ABSTAIN", "false")
    # Answer generation can take a while — separate, longer timeout.
    chat_timeout_s: float = float(os.environ.get("CHAT_TIMEOUT_S", "300"))
    # Reasoning ("thinking") control for the local ollama provider. Thinking
    # models (Qwen3.x, DeepSeek-R1) emit a hidden <think> chain before the
    # answer — more deliberate, but it adds seconds of latency before the
    # first visible token. "off" disables it for a large speed win; "on"
    # forces it; "auto" (default) leaves the model's own behaviour untouched.
    # NOTE: "off"/"on" require a thinking-capable model — Ollama rejects the
    # flag on a plain model (e.g. qwen2.5). Keep "auto" if you run those.
    chat_think: str = os.environ.get("CHAT_THINK", "auto").lower()

    # --- OpenRouter (optional cloud alternate; only read when chat_provider
    #     == "openrouter" OR when an OpenRouter model is picked from the
    #     frontend dropdown). NEVER commit a real API key — it lives in
    #     `.env` which is gitignored. ---
    openrouter_url: str = os.environ.get(
        "OPENROUTER_URL", "https://openrouter.ai/api/v1"
    )
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.environ.get(
        "OPENROUTER_MODEL", "qwen/qwen3.6-27b"
    )
    # Cap on output tokens (incl. any <think> block). 4096 covers a thinking
    # model's reasoning + a long bilingual answer; raise if answers truncate.
    openrouter_max_tokens: int = int(
        os.environ.get("OPENROUTER_MAX_TOKENS", "4096")
    )
    # Optional attribution headers OpenRouter uses for its analytics page.
    # Empty = not sent. Safe to leave as-is.
    openrouter_referer: str = os.environ.get("OPENROUTER_REFERER", "")
    openrouter_title: str = os.environ.get(
        "OPENROUTER_TITLE", "transcript-rag"
    )

    # --- Frontend model dropdown ---
    # Comma-separated extras the dashboard offers in its "Model" dropdown.
    # The defaults (chat_model / openrouter_model) are merged in automatically
    # so the dropdown always exposes whichever provider the operator picked.
    # Empty lists collapse the dropdown to a single hardcoded model.
    available_ollama_models_raw: str = os.environ.get(
        "AVAILABLE_OLLAMA_MODELS", ""
    )
    available_openrouter_models_raw: str = os.environ.get(
        "AVAILABLE_OPENROUTER_MODELS", ""
    )

    # --- Reranker (Infinity) — base URL; `/rerank` is appended in code ---
    reranker_url: str = os.environ.get("RERANKER_URL", "http://localhost:7997")
    reranker_model: str = os.environ.get(
        "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
    )

    # --- Tantivy — opened in-process (read-only), not the HTTP sidecar ---
    tantivy_dir: str = os.environ.get("TANTIVY_DIR", "./data/tantivy")

    # --- Media roots for the archive map's track panel (both mounted :ro) ---
    # Container mount points; the host paths behind them are TRANSCRIPTS_DIR /
    # ISOLATED_DIR in `.env`. Absent bind source => Docker makes an empty dir,
    # so /api/health probes for content rather than existence.
    #
    # ISOLATED_DIR must point at `GuruAudio/Output` — the folder holding the
    # per-model `.ckpt` trees, NOT one of them. Three tracks were transcribed
    # from the second model's tree, and rooting at a single model makes their
    # audio unreachable while the files sit right there.
    transcripts_dir: str = os.environ.get(
        "RAG_TRANSCRIPTS_MOUNT", "/app/data/transcripts"
    )
    isolated_dir: str = os.environ.get("RAG_ISOLATED_MOUNT", "/app/data/isolated")

    # --- PageIndex (alternate, opt-in retrieval backend for A/B comparison) ---
    # Reasoning-based "vectorless" retrieval: per-document LLM-built section
    # trees navigated by LLM reasoning instead of dense+BM25 fusion. This is an
    # ADDITIVE benchmark backend — the default stays "hybrid", so the locked
    # Qdrant+Tantivy+Infinity path is untouched unless a request opts in via
    # `backend=pageindex`. Trees are JSON files under `pageindex_dir`, built
    # offline by `ingestion.build_pageindex` over a bounded sample of files
    # (you cannot tree-build the whole 5 TB corpus). The reasoning LLM is the
    # local Ollama chat model — stays strictly local, no data leaves the box.
    retrieval_backend: str = os.environ.get("RETRIEVAL_BACKEND", "hybrid").lower()
    pageindex_dir: str = os.environ.get("PAGEINDEX_DIR", "./data/pageindex")
    # Ollama model for tree-building (titles/summaries) and reasoning retrieval.
    # Defaults to the chat model so a single local model serves both.
    pageindex_model: str = os.environ.get(
        "PAGEINDEX_MODEL",
        os.environ.get("CHAT_MODEL", "qwen3.5:9b"),
    )
    # Stage-1 fan-out: how many candidate documents (trees) the reasoning step
    # considers per query. Summary search picks them; falls back to on-disk
    # trees when the summary index is empty.
    pageindex_candidate_docs: int = int(
        os.environ.get("PAGEINDEX_CANDIDATE_DOCS", "5")
    )
    # Section granularity: consecutive chunks are grouped into one tree node
    # until this token budget is hit. One LLM title+summary call per node, so
    # smaller nodes = finer retrieval but a slower, costlier build.
    pageindex_max_tokens_per_node: int = int(
        os.environ.get("PAGEINDEX_MAX_TOKENS_PER_NODE", "1200")
    )

    # --- Retrieval tuning ---
    candidates_per_source: int = int(os.environ.get("RAG_CANDIDATES", "40"))
    final_top_k: int = int(os.environ.get("RAG_TOP_K", "4"))
    bm25_weight: float = float(os.environ.get("RAG_BM25_WEIGHT", "0.65"))
    quote_bm25_weight: float = float(
        os.environ.get("RAG_QUOTE_BM25_WEIGHT", "0.85")
    )
    # Per-class BM25 weight for the `name` (person-lookup) class — lexical, so
    # a touch heavier than the thematic default. Stage 4.1 starting value.
    name_bm25_weight: float = float(
        os.environ.get("RAG_NAME_BM25_WEIGHT", "0.75")
    )
    # Stage 4.1 query router. Default off: the deterministic classifier +
    # per-class retrieval settings only engage when this is on, so the default
    # pipeline is unchanged until an A/B proves the win. See rag_api.route.
    router_enabled: bool = _env_flag("RAG_ROUTER")
    # Stage 4.2: let the router turn HyDE expansion on for the `thematic` class
    # only (helps vague conceptual queries, hurts precise quote/name lookups).
    # Requires router_enabled. Default off: HyDE adds a ~1 s chat round-trip per
    # thematic query, and the accuracy win is unproven without a discriminating
    # thematic golden set — so it stays off until an A/B certifies it.
    hyde_thematic_enabled: bool = _env_flag("RAG_HYDE_THEMATIC")
    # Stage 4.3: rewrite a context-dependent follow-up into a standalone query
    # (using request `history`) before retrieval. Requires router_enabled. One
    # think=false chat round-trip, and only when history is present AND the
    # router tags the query `followup`. Default off until an A/B certifies it.
    followup_rewrite_enabled: bool = _env_flag("RAG_FOLLOWUP_REWRITE")
    # Post-synthesis groundedness check: one extra think=false chat call that
    # judges each answer sentence against the retrieved passages and drops
    # unsupported ones (rag_api.ground). Non-streaming /api/query only — a
    # token stream cannot be post-edited. Default off: it adds ~1-2 s per
    # answer, and the faithfulness eval decides whether the fidelity win is
    # worth it. Fail-open: judge failures never break answering.
    groundedness_enabled: bool = _env_flag("RAG_GROUNDEDNESS")
    # Agentic query planner (rag_api.planner): one think=false chat call per
    # thematic/name/analytic query that (a) detects list intent ("10 sittings
    # for rain" → sitting-level retrieval + numbered-list answer) and (b) adds
    # bilingual/synonym retrieval variants ("barish" → बारिश + rain) searched
    # in parallel and merged. Requires router_enabled. Fail-open: an LLM
    # failure degrades to the unchanged single-query pipeline. Default off;
    # deployment flips it via .env.
    planner_enabled: bool = _env_flag("RAG_PLANNER")
    # Cap on retrieval query variants per plan (original + N-1 variants).
    planner_max_queries: int = int(os.environ.get("RAG_PLANNER_MAX_QUERIES", "3"))
    # Cap on list-intent result count ("100 sittings" still returns <= this).
    # Also the candidate-pool size for best-sitting ranking.
    list_top_k: int = int(os.environ.get("RAG_LIST_TOP_K", "20"))
    # Chitchat via the chat model (default ON): the router's chitchat class is
    # answered by a short no-retrieval think=false LLM call instead of a fixed
    # string, so "hi hello what can you do" gets a natural reply. Fail-open:
    # any model error falls back to the fixed bilingual reply.
    chitchat_llm_enabled: bool = _env_flag("RAG_CHITCHAT_LLM", "on")
    # Chunk budget for verbatim-summary turns ("summary of the sitting on X").
    # The answer quotes exact lines, so more passages = better coverage of the
    # sitting; the effective top_k is max(request top_k, this).
    verbatim_top_k: int = int(os.environ.get("RAG_VERBATIM_TOP_K", "10"))
    http_timeout_s: float = float(os.environ.get("HTTP_TIMEOUT_S", "30"))
    # Two-stage retrieval: how many files stage 1 (summary search) keeps
    # before drilling into their chunks in stage 2.
    summary_top_files: int = int(os.environ.get("RAG_SUMMARY_TOP_FILES", "6"))

    # --- Postgres (analytics + dashboard filter options) ---
    pg_dsn: str = _resolve_pg_dsn()

    # --- API ---
    # Shared password gate. Empty string => auth disabled (local dev only).
    dashboard_password: str = os.environ.get("DASHBOARD_PASSWORD", "")
    api_host: str = os.environ.get("RAG_API_HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("RAG_API_PORT", "8080"))
    # CORS origins for the React dev server. Comma-separated.
    cors_origins: str = os.environ.get(
        "RAG_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )

    @property
    def reranker_endpoint(self) -> str:
        return self.reranker_url.rstrip("/") + "/rerank"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def available_ollama_models(self) -> list[str]:
        """Parsed Ollama model list, with the default first and dedup'd."""
        return _dedup_with_default(
            self.chat_model, self.available_ollama_models_raw
        )

    @property
    def available_openrouter_models(self) -> list[str]:
        """Parsed OpenRouter model list, with the default first and dedup'd.

        If no API key is configured, OpenRouter options are not advertised —
        offering them would just produce 401 errors.
        """
        if not self.openrouter_api_key:
            return []
        return _dedup_with_default(
            self.openrouter_model, self.available_openrouter_models_raw
        )

    def available_models(self) -> list[dict[str, Any]]:
        """All offer-able chat models, in dropdown order.

        Each entry: ``{"provider", "model", "label", "is_default"}``.
        ``is_default`` matches the currently active CHAT_PROVIDER + its
        default model.
        """
        out: list[dict[str, Any]] = []
        provider = (self.chat_provider or "ollama").lower()
        for m in self.available_ollama_models:
            out.append({
                "provider": "ollama",
                "model": m,
                "label": f"{m} (local)",
                "is_default": (provider == "ollama" and m == self.chat_model),
            })
        for m in self.available_openrouter_models:
            out.append({
                "provider": "openrouter",
                "model": m,
                "label": f"{m} (cloud)",
                "is_default": (
                    provider == "openrouter" and m == self.openrouter_model
                ),
            })
        # If no entry was flagged default (e.g. provider=openrouter but no
        # OpenRouter key, so all OpenRouter options were filtered out), fall
        # back to the first option.
        if out and not any(e["is_default"] for e in out):
            out[0]["is_default"] = True
        return out

    def is_model_allowed(self, provider: str, model: str) -> bool:
        """Per-request override gate: only models in the dropdown are accepted.

        Stops a curious caller from proxying any arbitrary OpenRouter model
        through the API and burning the operator's credits / key on a model
        the operator never approved.
        """
        prov = (provider or "").lower()
        if prov == "ollama":
            return model in self.available_ollama_models
        if prov == "openrouter":
            return model in self.available_openrouter_models
        return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Cached so env is read exactly once."""
    return Settings()
