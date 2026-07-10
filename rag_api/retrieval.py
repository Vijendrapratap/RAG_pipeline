"""Hybrid retrieval core for the dashboard API.

This is the retrieval logic from `open_webui_functions/search_transcripts.py`
lifted out of the Open WebUI `Tools`/`Valves` wrapper into a plain module:

    dense (Qdrant / bge-m3)  +  BM25 (Tantivy)  ->  weighted RRF  ->  rerank

Two changes from the original:
  1. Tantivy is opened **in-process** (read-only) instead of via the HTTP
     sidecar on :8765 — one fewer service to run. If the index directory is
     missing, BM25 is disabled and search degrades to dense-only.
  2. `search()` / `find_quote()` return structured dicts (not preformatted
     text), so the API can hand JSON to the frontend and the synthesis layer.

The fusion / filter / result helpers are module-level pure functions so they
can be unit-tested without a live stack.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from rag_api.config import Settings
from rag_api.expand import QueryExpander

try:  # tantivy is a hard dep, but degrade gracefully if the wheel is absent
    import tantivy  # type: ignore[import-not-found]
    _TANTIVY_AVAILABLE = True
except ImportError:  # pragma: no cover
    tantivy = None
    _TANTIVY_AVAILABLE = False

log = logging.getLogger("rag_api.retrieval")

# Payload keys surfaced to the frontend under result["metadata"]. Phase 12
# (path) + Phase 13 (content tags). Only non-null keys are included per result.
_META_FIELDS: tuple[str, ...] = (
    "collection", "year", "event_id", "location",
    "session_date", "session_time", "track_no", "track_title", "track_type",
    "season", "event_type", "primary_language",
    "topics", "people_named", "places_named", "scriptures_referenced",
    # Phase 14 catalog facets — present on catalog-source results.
    "performers", "release_ref", "doc_type",
)


# --------------------------------------------------------------------------
# Pure helpers (no network, no I/O — unit-testable)
# --------------------------------------------------------------------------


def build_qdrant_filter(filters: dict[str, Any]) -> dict[str, Any] | None:
    """Translate a flat filter dict into a Qdrant REST `filter` clause.

    Array filters (track_type / topics / people_named / scriptures_referenced)
    use `MatchAny` — logical OR within a filter, AND across filters. Returns
    None when no filter is set.
    """
    must: list[dict[str, Any]] = []
    f = filters

    if f.get("speaker"):
        must.append({"key": "speakers", "match": {"value": f["speaker"]}})
    if f.get("source_file"):
        sf = f["source_file"]
        if isinstance(sf, str):
            must.append({"key": "source_file", "match": {"value": sf}})
        else:
            # A list of files (two-stage retrieval) -> OR across them.
            must.append({"key": "source_file", "match": {"any": list(sf)}})
    if f.get("location"):
        must.append({"key": "location",
                     "match": {"value": str(f["location"]).upper()}})
    if f.get("event_id"):
        must.append({"key": "event_id", "match": {"value": f["event_id"]}})
    if f.get("season"):
        must.append({"key": "season", "match": {"value": f["season"]}})
    if f.get("track_type"):
        tt = f["track_type"]
        tt = [tt] if isinstance(tt, str) else list(tt)
        must.append({"key": "track_type", "match": {"any": tt}})
    if f.get("date_range"):
        gte, lte = f["date_range"]
        must.append({"key": "session_date", "range": {"gte": gte, "lte": lte}})
    if f.get("event_type"):
        must.append({"key": "event_type", "match": {"value": f["event_type"]}})
    if f.get("primary_language"):
        must.append({"key": "primary_language",
                     "match": {"value": f["primary_language"]}})
    if f.get("topics"):
        must.append({"key": "topics", "match": {"any": list(f["topics"])}})
    if f.get("people_named"):
        must.append({"key": "people_named",
                     "match": {"any": list(f["people_named"])}})
    if f.get("scriptures_referenced"):
        must.append({"key": "scriptures_referenced",
                     "match": {"any": list(f["scriptures_referenced"])}})
    if f.get("performers"):
        # Phase 14: performer credits backfilled from the curated catalog.
        perf = f["performers"]
        perf = [perf] if isinstance(perf, str) else list(perf)
        must.append({"key": "performers", "match": {"any": perf}})

    return {"must": must} if must else None


def _rrf_accumulate(
    scored: dict[str, list[Any]],
    hits: list[dict[str, Any]],
    weight: float,
    id_key: str,
    payload_fn: Any,
    k: int,
) -> None:
    """Add one ranked list's RRF contribution (weight/(k+rank+1)) into `scored`."""
    for rank, h in enumerate(hits):
        cid = str(h[id_key])
        entry = scored.setdefault(cid, [0.0, None])
        entry[0] += weight / (k + rank + 1)
        if entry[1] is None:
            entry[1] = payload_fn(h)


def rrf_fuse(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    bm25_weight: float,
    k: int = 60,
    catalog_hits: list[dict[str, Any]] | None = None,
    catalog_weight: float | None = None,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Weighted reciprocal rank fusion over N ranked lists.

    Returns [(chunk_id, fused_score, payload), ...] sorted by score desc.
    Dense hits keep their Qdrant payload; BM25-only hits get a payload-shaped
    dict synthesised from the Tantivy doc so downstream code is uniform.

    The optional catalog arm is fused **by rank**, not by raw cosine: a catalog
    row scores ``catalog_weight/(k+rank+1)`` on the same scale as every other
    arm. Concatenating its raw cosine (~0.5–0.95) into an RRF list (max ~1/61 ≈
    0.016) instead let a single catalog row outweigh — or, after positional
    truncation, be dropped ahead of — the entire transcript fusion. Its default
    weight matches the dense arm (it is a semantic/dense source).
    """
    scored: dict[str, list[Any]] = {}
    _rrf_accumulate(scored, dense_hits, 1 - bm25_weight, "id",
                    lambda h: h.get("payload") or {}, k)
    _rrf_accumulate(scored, bm25_hits, bm25_weight, "chunk_id",
                    lambda h: {
                        "text": h.get("text", ""),
                        "source_file": h.get("source_file", ""),
                        "speakers": [],
                        "start_sec": None,
                        "end_sec": None,
                    }, k)
    if catalog_hits:
        cw = (1 - bm25_weight) if catalog_weight is None else catalog_weight
        _rrf_accumulate(scored, catalog_hits, cw, "id",
                        lambda h: h.get("payload") or {}, k)

    merged = [(cid, float(s), pl or {}) for cid, (s, pl) in scored.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


def apply_post_filters(
    fused: list[tuple[str, float, dict[str, Any]]],
    filters: dict[str, Any],
) -> list[tuple[str, float, dict[str, Any]]]:
    """Enforce metadata filters against fused results.

    Qdrant already filtered the dense side server-side; BM25-only hits never
    saw a filter and carry no metadata. When any filter is active they are
    dropped here — the user explicitly asked for a filtered subset, so a hit
    we can't prove belongs in it doesn't belong in it.
    """
    f = filters
    tt_set: set[str] | None = None
    if f.get("track_type"):
        tt = f["track_type"]
        tt_set = {tt} if isinstance(tt, str) else set(tt)
    sf_set: set[str] | None = None
    if f.get("source_file") and not isinstance(f["source_file"], str):
        sf_set = set(f["source_file"])
    topics_set = set(f["topics"]) if f.get("topics") else None
    people_set = set(f["people_named"]) if f.get("people_named") else None
    scriptures_set = (set(f["scriptures_referenced"])
                      if f.get("scriptures_referenced") else None)
    performers_set: set[str] | None = None
    if f.get("performers"):
        perf = f["performers"]
        performers_set = {perf} if isinstance(perf, str) else set(perf)

    out: list[tuple[str, float, dict[str, Any]]] = []
    for cid, score, pl in fused:
        if f.get("speaker") and f["speaker"] not in (pl.get("speakers") or []):
            continue
        if f.get("source_file"):
            sf = f["source_file"]
            if isinstance(sf, str):
                if pl.get("source_file") != sf:
                    continue
            elif pl.get("source_file") not in sf_set:  # type: ignore[operator]
                continue
        if f.get("season") and pl.get("season") != f["season"]:
            continue
        if tt_set is not None and pl.get("track_type") not in tt_set:
            continue
        if f.get("location") and (
            (pl.get("location") or "").upper() != str(f["location"]).upper()
        ):
            continue
        if f.get("event_id") and pl.get("event_id") != f["event_id"]:
            continue
        if f.get("date_range") and pl.get("session_date"):
            gte, lte = f["date_range"]
            if not (gte <= pl["session_date"] <= lte):
                continue
        if f.get("event_type") and pl.get("event_type") != f["event_type"]:
            continue
        if f.get("primary_language") and (
            pl.get("primary_language") != f["primary_language"]
        ):
            continue
        if topics_set is not None and not (
            topics_set & set(pl.get("topics") or [])
        ):
            continue
        if people_set is not None and not (
            people_set & set(pl.get("people_named") or [])
        ):
            continue
        if scriptures_set is not None and not (
            scriptures_set & set(pl.get("scriptures_referenced") or [])
        ):
            continue
        if performers_set is not None and not (
            performers_set & set(pl.get("performers") or [])
        ):
            continue
        out.append((cid, score, pl))
    return out


def to_result(cid: str, score: float, payload: dict[str, Any]) -> dict[str, Any]:
    """Shape one fused/reranked chunk hit into the API result schema.

    `source` distinguishes the curated catalog from Whisper transcripts so the
    UI can label them and the user can compare (Phase 14 separate source).
    """
    is_catalog = payload.get("source_type") == "catalog"
    return {
        "result_type": "catalog" if is_catalog else "chunk",
        "source": "catalog" if is_catalog else "transcript",
        "chunk_id": cid,
        "score": round(float(score), 4),
        "text": payload.get("text", ""),
        "source_file": payload.get("source_file"),
        "start_sec": payload.get("start_sec"),
        "end_sec": payload.get("end_sec"),
        "speakers": payload.get("speakers") or [],
        "metadata": {
            k: payload[k] for k in _META_FIELDS if payload.get(k) is not None
        },
    }


def summary_text_of(payload: dict[str, Any]) -> str:
    """The text used to represent a file-level summary result. English is
    preferred (the answer model reads it fine and writes in any language);
    Hindi is the fallback."""
    return (payload.get("summary_english")
            or payload.get("summary_hindi") or "").strip()


def to_summary_result(
    cid: str, score: float, payload: dict[str, Any]
) -> dict[str, Any]:
    """Shape one file-level summary hit. Same schema as `to_result` so the
    frontend and synthesis layer treat chunk and summary results uniformly —
    `text` carries the summary, timestamps/speakers are absent."""
    return {
        "result_type": "summary",
        "chunk_id": cid,
        "score": round(float(score), 4),
        "text": summary_text_of(payload),
        "source_file": payload.get("source_file"),
        "start_sec": None,
        "end_sec": None,
        "speakers": [],
        "summary_english": payload.get("summary_english"),
        "summary_hindi": payload.get("summary_hindi"),
        "metadata": {
            k: payload[k] for k in _META_FIELDS if payload.get(k) is not None
        },
    }


def mean_pool(vectors: list[list[float]]) -> list[float]:
    """Element-wise mean of equal-length vectors.

    Used to blend the query embedding with a HyDE hypothetical-document
    embedding into one dense search vector. Pure — unit-tested.
    """
    if not vectors:
        return []
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


def collect_sitting_bodies(
    candidates: list[tuple[str, float, dict[str, Any]]],
) -> dict[str, str]:
    """Map ``sitting_key -> joined sitting_detail prose`` harvested from the
    candidate pool.

    A catalog ``track_title`` row stores only a metadata header + title, but the
    actual discourse text lives in the ``sitting_detail`` rows of the same
    sitting — which the catalog dense search usually surfaces alongside it. This
    lets `rerank_text` score a title row against its real transcription with no
    extra network round-trip.
    """
    bodies: dict[str, list[str]] = {}
    for _cid, _score, pl in candidates:
        if (pl.get("source_type") == "catalog"
                and pl.get("doc_type") == "sitting_detail"):
            sk = pl.get("sitting_key")
            text = pl.get("text", "")
            if sk and text:
                bodies.setdefault(sk, []).append(text)
    return {sk: "\n".join(parts) for sk, parts in bodies.items()}


def rerank_text(
    payload: dict[str, Any],
    sitting_bodies: dict[str, str] | None = None,
) -> str:
    """Text handed to the cross-encoder for one candidate.

    Transcript chunks and catalog ``sitting_detail`` rows already carry prose, so
    their stored ``text`` is used verbatim. A catalog ``track_title`` row stores
    only ``[Catalog | LOCATION | DATE | Performers: …]\\nTitle: …`` — bracketed
    metadata the reranker cannot judge against a natural-language query (it rates
    a genuinely relevant title near zero, which surfaces as a misleadingly tiny
    score). For those rows we compose a natural-language sentence from the catalog
    facets and, when the sitting's transcription is available in the candidate
    pool, append it so the reranker scores against real content.

    Display text is untouched — this only affects what the reranker *reads*.
    """
    text = payload.get("text", "")
    if not (payload.get("source_type") == "catalog"
            and payload.get("doc_type") == "track_title"):
        return text

    title = payload.get("track_title")
    if not title:
        return text  # no facets worth composing from

    lead = f"{payload.get('track_type') or 'track'}: {title}"
    ctx: list[str] = []
    if payload.get("location"):
        ctx.append(f"at {payload['location']}")
    if payload.get("session_date"):
        ctx.append(f"on {payload['session_date']}")
    if payload.get("season"):
        ctx.append(f"during {payload['season']} season")
    performers = payload.get("performers") or []
    if performers:
        ctx.append("performed by " + ", ".join(performers[:8]))
    sentence = lead + ((", " + " ".join(ctx)) if ctx else "")

    if sitting_bodies:
        body = sitting_bodies.get(payload.get("sitting_key") or "")
        if body:
            sentence = f"{sentence}. {body}"
    return sentence


# --------------------------------------------------------------------------
# Retriever — holds clients and orchestrates the pipeline
# --------------------------------------------------------------------------


class Retriever:
    """Stateful retrieval engine. One instance per process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = requests.Session()
        self._tantivy_index = self._open_tantivy()
        self._expander = QueryExpander(settings)

    # ---- query expansion (HyDE, optional) -------------------------------

    def make_expansion(self, query: str) -> str:
        """Generate a HyDE hypothetical passage for `query`, or '' on failure.

        Call this once per request and pass the result as `dense_text` to the
        search methods — that way two-stage retrieval reuses one generation
        instead of paying for the chat-model call twice.
        """
        return self._expander.hypothetical(query)

    # ---- Tantivy (in-process, read-only) --------------------------------

    def _open_tantivy(self) -> Any:
        """Open the Tantivy index read-only. Returns None (BM25 disabled) if
        the wheel is missing or the index directory does not exist yet."""
        if not _TANTIVY_AVAILABLE:
            log.warning("tantivy not importable — BM25 disabled, dense-only")
            return None
        from pathlib import Path

        idx_dir = Path(self.settings.tantivy_dir)
        if not idx_dir.exists():
            log.warning(
                "Tantivy index dir %s not found — BM25 disabled, dense-only. "
                "Run ingestion, then POST /api/tantivy/reload.", idx_dir,
            )
            return None
        try:
            index = tantivy.Index.open(str(idx_dir))
            index.reload()
            log.info("Tantivy index opened: %s (%d docs)",
                     idx_dir, index.searcher().num_docs)
            return index
        except Exception as e:
            log.error("failed to open Tantivy index %s: %s — BM25 disabled",
                      idx_dir, e)
            return None

    @property
    def bm25_enabled(self) -> bool:
        return self._tantivy_index is not None

    def tantivy_doc_count(self) -> int:
        if self._tantivy_index is None:
            return 0
        try:
            return self._tantivy_index.searcher().num_docs
        except Exception as e:  # pragma: no cover - defensive
            log.error("tantivy doc count failed: %s", e)
            return 0

    def summary_doc_count(self) -> int:
        """Points in the file-level summary collection (0 if it does not exist
        yet). Mirrors tantivy_doc_count for /api/health — lets an operator see
        that scope=summaries / scope=two_stage are actually backed by an index
        instead of silently degrading to a chunk search."""
        headers = {"Content-Type": "application/json"}
        if self.settings.qdrant_key:
            headers["api-key"] = self.settings.qdrant_key
        try:
            r = self._session.post(
                f"{self.settings.qdrant_url}/collections/"
                f"{self.settings.qdrant_summary_collection}/points/count",
                json={"exact": True}, headers=headers,
                timeout=self.settings.http_timeout_s,
            )
            if r.status_code == 404:
                return 0
            r.raise_for_status()
            return int((r.json().get("result") or {}).get("count", 0))
        except Exception as e:  # pragma: no cover - defensive
            log.error("summary doc count failed: %s", e)
            return 0

    def reload_tantivy(self) -> int:
        """Re-open + reload the index (call after fresh ingestion). Returns
        the new doc count; attempts a cold open if it was disabled before."""
        if self._tantivy_index is None:
            self._tantivy_index = self._open_tantivy()
        elif self._tantivy_index is not None:
            self._tantivy_index.reload()
        return self.tantivy_doc_count()

    # ---- stage helpers --------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batched embeddings via Ollama bge-m3."""
        r = self._session.post(
            f"{self.settings.ollama_url}/api/embed",
            json={"model": self.settings.embed_model, "input": texts},
            timeout=self.settings.http_timeout_s,
        )
        r.raise_for_status()
        embs = r.json().get("embeddings") or []
        if len(embs) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(embs)} embeddings for {len(texts)} inputs"
            )
        return embs

    def _embed(self, text: str) -> list[float]:
        """Single-query embedding via Ollama bge-m3."""
        return self._embed_batch([text])[0]

    def _query_vector(
        self, query: str, dense_text: str | None = None,
    ) -> list[float]:
        """Build the dense search vector.

        With `dense_text` (a HyDE hypothetical passage) the query and
        hypothetical embeddings are mean-pooled — that is the HyDE blend.
        Falls back to a query-only vector if the paired embed call fails, so
        a flaky expansion never breaks retrieval.
        """
        if dense_text:
            try:
                return mean_pool(self._embed_batch([query, dense_text]))
            except (requests.RequestException, RuntimeError) as e:
                log.warning("HyDE embed failed (%s) — query-only vector", e)
        return self._embed(query)

    def _dense(
        self, vec: list[float], filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Qdrant ANN search. Returns raw [{id, score, payload}, ...]."""
        body: dict[str, Any] = {
            "vector": vec,
            "limit": self.settings.candidates_per_source,
            "with_payload": True,
            "with_vector": False,
        }
        qfilter = build_qdrant_filter(filters)
        if qfilter:
            body["filter"] = qfilter
        headers = {"Content-Type": "application/json"}
        if self.settings.qdrant_key:
            headers["api-key"] = self.settings.qdrant_key
        r = self._session.post(
            f"{self.settings.qdrant_url}/collections/"
            f"{self.settings.qdrant_collection}/points/search",
            json=body, headers=headers, timeout=self.settings.http_timeout_s,
        )
        r.raise_for_status()
        return r.json().get("result", [])

    def _bm25(self, query: str) -> list[dict[str, Any]]:
        """In-process Tantivy BM25 search. Returns [] if disabled or if the
        query parser chokes — BM25 is one of two arms, so a failure here
        degrades to dense-only rather than failing the whole request."""
        if self._tantivy_index is None:
            return []
        try:
            idx = self._tantivy_index
            parser = idx.parse_query(query, ["text"])
            searcher = idx.searcher()
            hits = searcher.search(
                parser, limit=self.settings.candidates_per_source
            ).hits
            out: list[dict[str, Any]] = []
            for score, addr in hits:
                doc = searcher.doc(addr)
                out.append({
                    "chunk_id": doc["chunk_id"][0],
                    "text": doc["text"][0],
                    "source_file": doc["source_file"][0],
                    "score": float(score),
                })
            return out
        except Exception as e:
            log.warning("BM25 search failed for %r: %s — dense-only for this "
                        "query", query, e)
            return []

    def _rerank(
        self, query: str, candidates: list[tuple[str, float, dict[str, Any]]],
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Cross-encoder rerank via Infinity. Falls back to RRF order if the
        reranker is unreachable — degraded results still beat no results."""
        if not candidates:
            return []
        # Catalog title rows carry only metadata text; give the cross-encoder a
        # composed sentence plus the sitting's actual transcription so it scores
        # them on real content instead of rating relevant rows near zero. Bodies
        # already in the pool are free; the rest are fetched in one batched call.
        sitting_bodies = collect_sitting_bodies(candidates)
        missing = sorted({
            pl.get("sitting_key") for _c, _s, pl in candidates
            if pl.get("source_type") == "catalog"
            and pl.get("doc_type") == "track_title"
            and pl.get("sitting_key")
            and pl.get("sitting_key") not in sitting_bodies
        })
        if missing:
            sitting_bodies = {**self._fetch_sitting_bodies(missing), **sitting_bodies}
        texts = [rerank_text(pl, sitting_bodies) for _, _, pl in candidates]
        try:
            r = self._session.post(
                self.settings.reranker_endpoint,
                json={
                    "model": self.settings.reranker_model,
                    "query": query,
                    "documents": texts,
                    "return_documents": False,
                },
                timeout=self.settings.http_timeout_s,
            )
            r.raise_for_status()
            results = r.json().get("results") or []
        except requests.RequestException as e:
            log.warning("reranker unreachable (%s) — returning RRF order", e)
            return candidates[: self.settings.final_top_k]
        rescored: list[tuple[str, float, dict[str, Any]]] = []
        for item in results:
            idx = int(item["index"])
            score = float(item["relevance_score"])
            cid, _, pl = candidates[idx]
            rescored.append((cid, score, pl))
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored

    def _fetch_sitting_bodies(self, sitting_keys: list[str]) -> dict[str, str]:
        """Batch-fetch ``sitting_detail`` prose for the given sitting_keys from
        the catalog collection, so metadata-only ``track_title`` candidates can be
        reranked against their sitting's actual transcription.

        One scroll for the whole set. Returns ``{}`` on any error / missing
        collection — affected title rows then fall back to their composed text.
        """
        if not sitting_keys:
            return {}
        body = {
            "limit": len(sitting_keys) * 8,
            "with_payload": ["sitting_key", "text"],
            "with_vector": False,
            "filter": {"must": [
                {"key": "doc_type", "match": {"value": "sitting_detail"}},
                {"key": "sitting_key", "match": {"any": sitting_keys}},
            ]},
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.qdrant_key:
            headers["api-key"] = self.settings.qdrant_key
        try:
            r = self._session.post(
                f"{self.settings.qdrant_url}/collections/"
                f"{self.settings.qdrant_catalog_collection}/points/scroll",
                json=body, headers=headers, timeout=self.settings.http_timeout_s,
            )
            r.raise_for_status()
            points = r.json().get("result", {}).get("points", [])
        except requests.RequestException as e:
            log.warning("sitting-body fetch failed (%s) — title rows use "
                        "composed text", e)
            return {}
        bodies: dict[str, list[str]] = {}
        for p in points:
            pl = p.get("payload") or {}
            sk, text = pl.get("sitting_key"), pl.get("text", "")
            if sk and text:
                bodies.setdefault(sk, []).append(text)
        return {sk: "\n".join(parts) for sk, parts in bodies.items()}

    # ---- public API -----------------------------------------------------

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 8,
        bm25_weight: float | None = None,
        dense_text: str | None = None,
        include_catalog: bool = False,
    ) -> list[dict[str, Any]]:
        """Hybrid semantic + BM25 search with optional metadata filters.

        `dense_text` is an optional HyDE hypothetical passage; when given it is
        blended into the dense vector (see `_query_vector`). BM25 always uses
        the raw query — lexical match should track the user's actual words.

        When `include_catalog` is set, curated-catalog points are folded into
        the candidate pool and reranked *together* with the transcript hits, so
        the two sources interleave by relevance and each result is labelled by
        `source`. A missing catalog collection degrades silently to
        transcript-only.

        Returns up to `top_k` result dicts (see `to_result`), best first.
        """
        filters = filters or {}
        weight = self.settings.bm25_weight if bm25_weight is None else bm25_weight

        vec = self._query_vector(query, dense_text)
        dense = self._dense(vec, filters)
        bm25 = self._bm25(query)
        cat = self._catalog_dense(vec, filters) if include_catalog else None
        # Fuse the catalog arm by rank (see rrf_fuse) so it competes on the RRF
        # scale and survives the positional truncation below, then re-sort is
        # done inside rrf_fuse.
        fused = rrf_fuse(dense, bm25, weight, catalog_hits=cat)

        if any(v for v in filters.values()):
            fused = apply_post_filters(fused, filters)

        candidates = fused[: self.settings.candidates_per_source * (2 if include_catalog else 1)]
        reranked = self._rerank(query, candidates)
        cap = min(top_k, self.settings.candidates_per_source)
        return [to_result(c, s, pl) for c, s, pl in reranked[:cap]]

    # ---- catalog-source retrieval (Phase 14, separate source) -----------

    def _catalog_dense(
        self, vec: list[float], filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """ANN search over the standalone catalog collection. Returns [] (not
        an error) if the collection does not exist yet — run
        `ingestion.catalog.index_catalog --commit` to populate it."""
        body: dict[str, Any] = {
            "vector": vec,
            "limit": self.settings.candidates_per_source,
            "with_payload": True,
            "with_vector": False,
        }
        qfilter = build_qdrant_filter(filters)
        if qfilter:
            body["filter"] = qfilter
        headers = {"Content-Type": "application/json"}
        if self.settings.qdrant_key:
            headers["api-key"] = self.settings.qdrant_key
        r = self._session.post(
            f"{self.settings.qdrant_url}/collections/"
            f"{self.settings.qdrant_catalog_collection}/points/search",
            json=body, headers=headers, timeout=self.settings.http_timeout_s,
        )
        if r.status_code == 404:
            log.warning(
                "catalog collection %r not found — run "
                "ingestion.catalog.index_catalog to enable catalog search",
                self.settings.qdrant_catalog_collection,
            )
            return []
        r.raise_for_status()
        return r.json().get("result", [])

    def search_catalog(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 8,
        dense_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Catalog-only search (the `catalog` scope): dense over the curated
        collection + rerank. Use for performer/title/sitting-summary lookups
        and Whisper-vs-catalog comparison."""
        filters = filters or {}
        vec = self._query_vector(query, dense_text)
        hits = self._catalog_dense(vec, filters)
        candidates = [
            (str(h["id"]), float(h.get("score", 0.0)), dict(h.get("payload") or {}))
            for h in hits
        ]
        if any(v for v in filters.values()):
            candidates = apply_post_filters(candidates, filters)
        reranked = self._rerank(query, candidates[: self.settings.candidates_per_source])
        cap = min(top_k, self.settings.candidates_per_source)
        return [to_result(c, s, pl) for c, s, pl in reranked[:cap]]

    def browse_catalog(
        self, filters: dict[str, Any] | None = None, top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """Filter-only browse over the catalog — no query vector, no reranker.
        Returns the catalog rows that match the metadata facets (place, year,
        performer, track-type, season, date), newest first. This is what powers
        "use the sheet's columns to get data directly": pick facets, get the
        matching sheet entries (and, once mapped, the audio transcript).

        Uses Qdrant scroll with the filter. A missing catalog collection (404)
        degrades to []. Track-title rows are preferred over the long
        hand-transcription chunks so the browse list stays scannable."""
        filters = filters or {}
        qfilter = build_qdrant_filter(filters)
        body: dict[str, Any] = {
            "limit": max(top_k * 4, self.settings.candidates_per_source),
            "with_payload": True, "with_vector": False,
        }
        if qfilter:
            body["filter"] = qfilter
        headers = {"Content-Type": "application/json"}
        if self.settings.qdrant_key:
            headers["api-key"] = self.settings.qdrant_key
        r = self._session.post(
            f"{self.settings.qdrant_url}/collections/"
            f"{self.settings.qdrant_catalog_collection}/points/scroll",
            json=body, headers=headers, timeout=self.settings.http_timeout_s,
        )
        if r.status_code == 404:
            log.warning("catalog collection not found — browse returns nothing")
            return []
        r.raise_for_status()
        points = r.json().get("result", {}).get("points", [])
        # Prefer canonical-title rows (one per track) for a scannable list;
        # fall back to detail rows only if no titles match.
        titles = [p for p in points if (p.get("payload") or {}).get("doc_type") == "track_title"]
        chosen = titles or points
        chosen.sort(
            key=lambda p: (p.get("payload") or {}).get("session_date") or "",
            reverse=True,
        )
        return [to_result(str(p["id"]), 0.0, dict(p.get("payload") or {}))
                for p in chosen[:top_k]]

    def find_quote(self, partial_quote: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Locate an exact / near-exact phrase. BM25 is weighted heavily
        because lexical match matters more than semantics for quote recovery.
        """
        return self.search(
            partial_quote,
            filters=None,
            top_k=top_k,
            bm25_weight=self.settings.quote_bm25_weight,
        )

    # ---- summary-level retrieval (Phase C) ------------------------------

    def _summary_dense(
        self, vec: list[float], filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """ANN search over the file-level summary collection. Returns [] (not
        an error) if the collection does not exist yet — run
        `ingestion.build_summary_index` to populate it."""
        body: dict[str, Any] = {
            "vector": vec,
            "limit": self.settings.candidates_per_source,
            "with_payload": True,
            "with_vector": False,
        }
        qfilter = build_qdrant_filter(filters)
        if qfilter:
            body["filter"] = qfilter
        headers = {"Content-Type": "application/json"}
        if self.settings.qdrant_key:
            headers["api-key"] = self.settings.qdrant_key
        r = self._session.post(
            f"{self.settings.qdrant_url}/collections/"
            f"{self.settings.qdrant_summary_collection}/points/search",
            json=body, headers=headers, timeout=self.settings.http_timeout_s,
        )
        if r.status_code == 404:
            log.warning(
                "summary collection %r not found — run "
                "ingestion.build_summary_index to enable summary search",
                self.settings.qdrant_summary_collection,
            )
            return []
        r.raise_for_status()
        return r.json().get("result", [])

    def search_summaries(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 8,
        dense_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """File-level search: one result per transcript, ranked by how well
        its summary matches the query. Best for cross-corpus / topic
        questions ("which discourses cover karma yoga"). `dense_text` is an
        optional HyDE hypothetical passage blended into the dense vector."""
        filters = filters or {}
        vec = self._query_vector(query, dense_text)
        hits = self._summary_dense(vec, filters)

        candidates: list[tuple[str, float, dict[str, Any]]] = []
        for h in hits:
            pl = dict(h.get("payload") or {})
            # The reranker scores (query, text); give it the summary as text.
            pl["text"] = summary_text_of(pl)
            candidates.append((str(h["id"]), float(h.get("score", 0.0)), pl))

        reranked = self._rerank(
            query, candidates[: self.settings.candidates_per_source]
        )
        cap = min(top_k, self.settings.candidates_per_source)
        return [to_summary_result(c, s, pl) for c, s, pl in reranked[:cap]]

    def search_two_stage(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 8,
        dense_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Two-stage retrieval: stage 1 finds the most relevant *files* via
        summary search, stage 2 runs the normal hybrid chunk search confined
        to those files. Sharpens topic questions by anchoring chunk retrieval
        to files the corpus-level summary already says are on-topic.

        `dense_text` (a HyDE hypothetical passage) is reused across both
        stages — the caller generates it once. Degrades to a plain chunk
        search when the summary index is empty, or when the caller already
        pinned a single `source_file`.
        """
        filters = filters or {}
        if filters.get("source_file"):
            # Caller already pinned a file — stage 1 would be redundant.
            return self.search(query, filters, top_k, dense_text=dense_text)

        summaries = self.search_summaries(
            query, filters, top_k=self.settings.summary_top_files,
            dense_text=dense_text,
        )
        files = [s["source_file"] for s in summaries if s.get("source_file")]
        if not files:
            log.info("two-stage: no summary hits — falling back to chunk search")
            return self.search(query, filters, top_k, dense_text=dense_text)

        chunk_filters = dict(filters)
        chunk_filters["source_file"] = files  # list -> OR filter in stage 2
        return self.search(query, chunk_filters, top_k, dense_text=dense_text)
