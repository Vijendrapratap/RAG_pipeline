"""Pre-ingestion health checks. Per PRD §6 Phase 4 deliverable 4."""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")
TANTIVY_URL = os.environ.get("TANTIVY_URL", "http://localhost:8765")

TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT_S", "5"))


def _try(url: str, headers: dict[str, str] | None = None) -> bool:
    try:
        r = requests.get(url, headers=headers or {}, timeout=TIMEOUT)
        return r.status_code < 500
    except requests.RequestException as e:
        log.debug("health check failed for %s: %s", url, e)
        return False


def check_all(include_tantivy: bool = True) -> dict[str, bool]:
    """Return {service_name: reachable}. Ollama and Qdrant are mandatory;
    Tantivy is optional (the sidecar may not be up yet during Phase 4
    integration tests — set include_tantivy=False to skip)."""
    out = {
        "ollama": _try(f"{OLLAMA_URL}/api/tags"),
        "qdrant": _try(
            f"{QDRANT_URL}/collections",
            headers={"api-key": QDRANT_KEY} if QDRANT_KEY else None,
        ),
    }
    if include_tantivy:
        out["tantivy"] = _try(f"{TANTIVY_URL}/health")
    return out
