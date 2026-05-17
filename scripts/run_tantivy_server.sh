#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec uvicorn services.tantivy_server.tantivy_server:app \
    --host 0.0.0.0 --port 8765 \
    --workers 1 --log-level info
