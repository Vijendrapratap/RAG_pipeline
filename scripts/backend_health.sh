#!/usr/bin/env bash
# Exit 0 iff Stage 3 can ACTUALLY ingest from here -- the WSL shell where
# ingestion runs. Two checks, both from this shell:
#   1. Qdrant reachable (utils.health.check_all).
#   2. The configured embed model (EMBED_MODEL, e.g. bge-m3) actually EMBEDS --
#      not just that Ollama answers /api/tags. A reachable Ollama with the model
#      un-pulled returns 200 on /api/tags but 404 on /api/embed, so a reachability
#      probe passes while every embed fails. The staleness sweep deletes vectors
#      BEFORE re-embedding, so a false-pass here means "deleted, can't rebuild".
#      This probe closes that hole by embedding a token end-to-end.
#
# Called by the orchestrator (run_pipeline.ps1) and by add_transcripts.sh's
# [1/5] gate. Prints a one-line status; nonzero exit means "do not ingest".
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && source .env; set +a
exec .venv/bin/python - <<'PY'
import os, sys, json, urllib.request
sys.path.insert(0, ".")
from ingestion.utils.health import check_all

h = check_all(include_tantivy=False)
qdrant_ok = bool(h.get("qdrant"))

ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
model = os.environ.get("EMBED_MODEL", "bge-m3")
embed_ok = False
detail = ""
try:
    body = json.dumps({"model": model, "input": "ping"}).encode()
    req = urllib.request.Request(ollama_url + "/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        doc = json.loads(r.read())
    vecs = doc.get("embeddings") or ([doc["embedding"]] if doc.get("embedding") else [])
    embed_ok = bool(vecs) and bool(vecs[0])
    detail = f"dim={len(vecs[0])}" if embed_ok else "no vector in response"
except Exception as e:
    detail = f"{type(e).__name__}: {e}"

print(f"qdrant={qdrant_ok} embed[{model}]={embed_ok} ({detail})")
sys.exit(0 if (qdrant_ok and embed_ok) else 1)
PY
