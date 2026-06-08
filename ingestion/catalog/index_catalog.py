"""Index the curated catalog into its own Qdrant 'catalog' collection.

A SEPARATE data source — not a merge. Two document kinds are produced:

  * ``sitting_detail`` — the per-sitting hand transcription (`DetailContents`),
    sentence-chunked like a normal transcript, one or more points per sitting.
  * ``track_title``   — one point per track carrying the canonical (Devanagari)
    song/bhajan title plus its metadata.

Every point gets a header line (location / date / performers) so the embedding
model sees the metadata verbatim — the same precision trick the Phase-12
chunker uses. Payloads carry the catalog facets (performers, location,
session_date, season, track_type) and the join_key, so a catalog hit can be
deduped against or cross-referenced with a 'transcripts' hit later.

Embedding + upsert require Ollama + Qdrant; both are gated behind ``--commit``.
Dry-run (the default) builds every document offline and reports counts, so the
chunking/payload logic is fully testable without services.

Run (dry-run):
    python -m ingestion.catalog.index_catalog --xlsx ".../Camp Record Export.xlsx"
Commit:
    python -m ingestion.catalog.index_catalog --xlsx ... \
        --qdrant http://localhost:6333 --ollama http://localhost:11434 --commit
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ingestion.catalog.normalize import CatalogTrack, join_key_from_fields, rows_to_tracks
from ingestion.chunker_text import estimate_tokens, split_sentences

log = logging.getLogger("ingestion.catalog.index_catalog")

# Match the transcript chunker so catalog chunks are comparable in size.
TARGET_TOKENS = 450
MAX_TOKENS = 700
OVERLAP_SENTENCES = 2

# Fixed namespace so point IDs are deterministic across re-runs (idempotent
# upserts). Hard-coded constant — never regenerate.
_NAMESPACE = uuid.UUID("ca7a1091-0e1f-4a2b-8c3d-9f0a1b2c3d4e")


def chunk_detail(text: str) -> list[str]:
    """Greedy sentence grouping to ~TARGET tokens with a 2-sentence overlap.
    Mirrors chunker_text.chunk_sentences semantics but returns plain strings."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks: list[str] = []
    i, n = 0, len(sentences)
    while i < n:
        buf: list[str] = []
        buf_tokens = 0
        while i < n:
            s = sentences[i]
            s_tok = estimate_tokens(s)
            if buf and buf_tokens + s_tok > MAX_TOKENS:
                break
            buf.append(s)
            buf_tokens += s_tok
            i += 1
            if buf_tokens >= TARGET_TOKENS:
                break
        chunks.append(" ".join(buf).strip())
        if i < n and len(buf) > OVERLAP_SENTENCES:
            i -= OVERLAP_SENTENCES  # overlap, unless it would stall progress
    return chunks


@dataclass
class CatalogDoc:
    point_id: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)


def _header(location: str | None, date_iso: str | None, performers: list[str]) -> str:
    bits: list[str] = ["Catalog"]
    if location:
        bits.append(location)
    if date_iso:
        bits.append(date_iso)
    if performers:
        bits.append("Performers: " + ", ".join(performers))
    return "[" + " | ".join(bits) + "]"


def _point_id(*parts: Any) -> str:
    return str(uuid.uuid5(_NAMESPACE, "|".join("" if p is None else str(p) for p in parts)))


def _sitting_key(t: CatalogTrack) -> str | None:
    return join_key_from_fields(t.session_date, t.session_seq, None)


def build_documents(tracks: list[CatalogTrack]) -> list[CatalogDoc]:
    """Pure: turn normalised catalog tracks into embeddable documents.

    Produces one or more `sitting_detail` docs per sitting (chunked hand
    transcription) and one `track_title` doc per track that has a title.
    """
    docs: list[CatalogDoc] = []

    # --- per-sitting hand transcription -----------------------------------
    seen_sittings: dict[str, CatalogTrack] = {}
    sitting_performers: dict[str, list[str]] = {}
    for t in tracks:
        sk = _sitting_key(t)
        if sk is None:
            continue
        if sk not in seen_sittings and t.detail_contents:
            seen_sittings[sk] = t
        bucket = sitting_performers.setdefault(sk, [])
        for p in t.performers:
            if p not in bucket:
                bucket.append(p)

    for sk, t in seen_sittings.items():
        date_iso = t.session_date.isoformat() if t.session_date else None
        performers = sitting_performers.get(sk, [])
        header = _header(t.location, date_iso, performers)
        for idx, body in enumerate(chunk_detail(t.detail_contents or "")):
            payload = {
                "source_type": "catalog",
                "doc_type": "sitting_detail",
                "sitting_key": sk,
                "location": t.location,
                "camp_place": t.camp_place,
                "session_date": date_iso,
                "season": t.season,
                "performers": performers,
                "release_ref": t.release_ref,
                "source_file": f"catalog:{sk}",
                "text": f"{header}\n{body}",
            }
            docs.append(CatalogDoc(_point_id("detail", sk, idx), payload["text"], payload))

    # --- per-track canonical title ----------------------------------------
    for t in tracks:
        if not t.track_title or not t.join_key:
            continue
        date_iso = t.session_date.isoformat() if t.session_date else None
        header = _header(t.location, date_iso, t.performers)
        text = f"{header}\nTitle: {t.track_title}"
        payload = {
            "source_type": "catalog",
            "doc_type": "track_title",
            "join_key": t.join_key,
            "sitting_key": _sitting_key(t),
            "location": t.location,
            "camp_place": t.camp_place,
            "session_date": date_iso,
            "season": t.season,
            "performers": t.performers,
            "track_no": t.track_no,
            "track_title": t.track_title,
            "track_type": t.track_type,
            "duration_sec": t.duration_sec,
            "release_ref": t.release_ref,
            "source_file": f"catalog:{t.join_key}",
            "text": text,
        }
        docs.append(CatalogDoc(_point_id("title", t.join_key), text, payload))

    return docs


# --- live embed + upsert (commit only) -------------------------------------

def _embed_batch(texts: list[str], ollama_url: str, model: str, timeout: float) -> list[list[float]]:
    import requests  # noqa: PLC0415

    r = requests.post(f"{ollama_url}/api/embed",
                      json={"model": model, "input": texts}, timeout=timeout)
    r.raise_for_status()
    embs = r.json().get("embeddings") or []
    if len(embs) != len(texts):
        raise RuntimeError(f"embed count mismatch: {len(embs)} != {len(texts)}")
    return embs


def _upsert(docs: list[CatalogDoc], qdrant_url: str, collection: str, api_key: str | None,
            ollama_url: str, model: str, batch: int, timeout: float) -> int:
    import requests  # noqa: PLC0415

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    url = f"{qdrant_url}/collections/{collection}/points"
    pushed = 0
    for start in range(0, len(docs), batch):
        window = docs[start:start + batch]
        vectors = _embed_batch([d.text for d in window], ollama_url, model, timeout)
        points = [{"id": d.point_id, "vector": v, "payload": d.payload}
                  for d, v in zip(window, vectors)]
        r = requests.put(url, json={"points": points}, headers=headers,
                         params={"wait": "true"}, timeout=timeout)
        r.raise_for_status()
        pushed += len(points)
        log.info("upserted %d / %d", pushed, len(docs))
    return pushed


def _read_rows(xlsx_path: Path) -> list[dict[str, Any]]:
    import pandas as pd  # noqa: PLC0415

    df = pd.read_excel(xlsx_path)
    return df.where(df.notna(), None).to_dict("records")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Index the catalog into its own Qdrant collection.")
    ap.add_argument("--xlsx", required=True, type=Path)
    ap.add_argument("--qdrant", default=None)
    ap.add_argument("--collection", default="catalog")
    ap.add_argument("--qdrant-key", default=None)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--postgres", default=None,
                    help="Postgres DSN. When given, also upsert catalog_sitting/"
                    "catalog_track so the dashboard's performers facet populates.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--http-timeout", type=float, default=60.0)
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args(argv)

    if not args.xlsx.exists():
        log.error("xlsx not found: %s", args.xlsx)
        return 2

    tracks = rows_to_tracks(_read_rows(args.xlsx))
    docs = build_documents(tracks)
    n_detail = sum(1 for d in docs if d.payload["doc_type"] == "sitting_detail")
    n_title = sum(1 for d in docs if d.payload["doc_type"] == "track_title")
    log.info("built %d catalog documents (%d sitting_detail, %d track_title)",
             len(docs), n_detail, n_title)

    if not args.commit:
        log.info("DRY-RUN (no embed, no upsert). Pass --commit to apply.")
        log.info("  qdrant:   %s", f"would upsert {len(docs)} points into {args.collection!r}"
                 if args.qdrant else "skipped (no --qdrant)")
        log.info("  postgres: %s", "would upsert catalog_sitting/catalog_track"
                 if args.postgres else "skipped (no --postgres)")
        if docs:
            sample = docs[0]
            log.info("  sample id=%s doc_type=%s", sample.point_id, sample.payload["doc_type"])
            log.info("  sample text[:200]=%r", sample.text[:200])
        return 0

    if args.postgres:
        from ingestion.catalog.backfill import _upsert_postgres  # noqa: PLC0415

        n_s, n_t = _upsert_postgres(tracks, args.postgres)
        log.info("postgres: upserted %d sittings, %d tracks (performers facet ready)",
                 n_s, n_t)
    if args.qdrant:
        pushed = _upsert(docs, args.qdrant, args.collection, args.qdrant_key,
                         args.ollama, args.embed_model, args.batch, args.http_timeout)
        log.info("done: upserted %d points into %r", pushed, args.collection)
    elif not args.postgres:
        log.error("--commit requires --qdrant and/or --postgres")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
