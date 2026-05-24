# Multi-stage build for the rag-api service.
#
# Stage 1 — build the React dashboard with Node. Its output (`dist/`) is
# copied into the runtime image, so the same container ships both the API
# and the UI (served at `/` by FastAPI StaticFiles).
#
# Stage 2 — Python 3.11 runtime. Installs only the rag-api package + its
# transitive deps; the heavy ingestion modules are deliberately not copied,
# keeping the image small. Tantivy runs in-process inside this container —
# there is no separate `:8765` sidecar in Phase F.

# ---- Stage 1: frontend build --------------------------------------------
FROM node:20-alpine AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python runtime --------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# README is referenced by pyproject.toml (PEP 621 `readme`); pip install .
# fails without it. The other top-level docs are not needed at runtime.
COPY pyproject.toml README.md ./
COPY rag_api/ ./rag_api/

# Install the project — pyproject's `packages.find` only matches rag_api*
# inside this image because the other source trees are not copied.
RUN pip install --no-cache-dir .

# Drop in the built dashboard so FastAPI's StaticFiles mount picks it up.
COPY --from=frontend-build /build/dist /app/frontend/dist

# `transcripts` Qdrant collection, Tantivy index dir, etc. are passed via
# environment variables — see docker-compose.yml.
EXPOSE 8080
CMD ["uvicorn", "rag_api.app:app", "--host", "0.0.0.0", "--port", "8080"]
