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
    chat_model: str = os.environ.get("CHAT_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    chat_num_ctx: int = int(os.environ.get("CHAT_NUM_CTX", "8192"))
    chat_temperature: float = float(os.environ.get("CHAT_TEMPERATURE", "0.2"))
    # Answer generation can take a while — separate, longer timeout.
    chat_timeout_s: float = float(os.environ.get("CHAT_TIMEOUT_S", "300"))

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

    # --- Retrieval tuning ---
    candidates_per_source: int = int(os.environ.get("RAG_CANDIDATES", "40"))
    final_top_k: int = int(os.environ.get("RAG_TOP_K", "8"))
    bm25_weight: float = float(os.environ.get("RAG_BM25_WEIGHT", "0.65"))
    quote_bm25_weight: float = float(
        os.environ.get("RAG_QUOTE_BM25_WEIGHT", "0.85")
    )
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
