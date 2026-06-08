"""title: Catalog Search (curated discourse archive)
author: transcript-rag
version: 1.0.0
license: MIT
description: Semantic search over the curated 'Camp Record Export' catalog —
    a 34-year human index of the discourse corpus (per-sitting hand
    transcriptions + canonical song/bhajan titles, with performer credits).
    This is a SEPARATE source from the Whisper transcript search: use it to
    find performers, canonical titles, release references, and human-verified
    sitting summaries — and to cross-check Whisper accuracy.

This file is an Open WebUI **Function Tool**. Install via Admin Panel →
Functions → "+ New Function" → paste contents → toggle on → attach to chat
models. It does not touch the 'transcripts' collection or its tool.

Per docs/catalog_enrichment.md (Phase 14, separate-source design).
"""
from __future__ import annotations

import os
from typing import Any

import requests
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        qdrant_url: str = Field(
            default=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
            description="Qdrant base URL.",
        )
        qdrant_key: str = Field(
            default=os.environ.get("QDRANT_API_KEY", ""),
            description="Qdrant API key.",
        )
        qdrant_collection: str = Field(
            default=os.environ.get("QDRANT_CATALOG_COLLECTION", "catalog"),
            description="Catalog collection name.",
        )
        ollama_url: str = Field(
            default=os.environ.get("OLLAMA_URL", "http://ollama:11434"),
            description="Ollama base URL for the embedding model.",
        )
        embed_model: str = Field(
            default=os.environ.get("EMBED_MODEL", "bge-m3"),
            description="Embedding model tag (must match the indexer).",
        )
        reranker_url: str = Field(
            default=os.environ.get("RERANKER_URL", "http://reranker:7997/rerank"),
            description="Infinity reranker endpoint.",
        )
        reranker_model: str = Field(
            default=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            description="Reranker model identifier.",
        )
        candidates: int = Field(
            default=40, description="Candidates pulled from Qdrant before reranking."
        )
        final_top_k: int = Field(
            default=8, description="Results returned after reranking."
        )
        http_timeout_s: float = Field(default=30.0, description="Per-HTTP-call timeout.")

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.citation = True

    # ---- internal helpers --------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        r = requests.post(
            f"{self.valves.ollama_url}/api/embed",
            json={"model": self.valves.embed_model, "input": [text]},
            timeout=self.valves.http_timeout_s,
        )
        r.raise_for_status()
        embs = r.json().get("embeddings") or []
        if not embs:
            raise RuntimeError("Ollama returned no embeddings for query")
        return embs[0]

    @staticmethod
    def _build_filter(
        performers: list[str] | str | None,
        location: str | None,
        date_range: tuple[str, str] | None,
        season: str | None,
        track_type: list[str] | str | None,
        doc_type: str | None,
    ) -> dict[str, Any] | None:
        """Translate catalog filters into a Qdrant `filter` clause. Pure, so it
        is unit-testable without any services."""
        must: list[dict[str, Any]] = []
        if performers:
            pf = [performers] if isinstance(performers, str) else list(performers)
            must.append({"key": "performers", "match": {"any": pf}})
        if location:
            must.append({"key": "location", "match": {"value": location.upper()}})
        if season:
            must.append({"key": "season", "match": {"value": season}})
        if track_type:
            tt = [track_type] if isinstance(track_type, str) else list(track_type)
            must.append({"key": "track_type", "match": {"any": tt}})
        if doc_type:
            must.append({"key": "doc_type", "match": {"value": doc_type}})
        if date_range:
            gte, lte = date_range
            must.append({"key": "session_date", "range": {"gte": gte, "lte": lte}})
        return {"must": must} if must else None

    def _dense(self, vec: list[float], qfilter: dict[str, Any] | None) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "vector": vec, "limit": self.valves.candidates,
            "with_payload": True, "with_vector": False,
        }
        if qfilter:
            body["filter"] = qfilter
        headers = {"Content-Type": "application/json"}
        if self.valves.qdrant_key:
            headers["api-key"] = self.valves.qdrant_key
        r = requests.post(
            f"{self.valves.qdrant_url}/collections/"
            f"{self.valves.qdrant_collection}/points/search",
            json=body, headers=headers, timeout=self.valves.http_timeout_s,
        )
        r.raise_for_status()
        return r.json().get("result", [])

    def _rerank(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not hits:
            return []
        texts = [(h.get("payload") or {}).get("text", "") for h in hits]
        try:
            r = requests.post(
                self.valves.reranker_url,
                json={"model": self.valves.reranker_model, "query": query,
                      "documents": texts, "return_documents": False},
                timeout=self.valves.http_timeout_s,
            )
            r.raise_for_status()
            results = r.json().get("results") or []
        except requests.RequestException:
            return hits[: self.valves.final_top_k]
        ordered: list[dict[str, Any]] = []
        for item in results:
            h = dict(hits[int(item["index"])])
            h["_score"] = float(item["relevance_score"])
            ordered.append(h)
        ordered.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        return ordered[: self.valves.final_top_k]

    @staticmethod
    def _format(hits: list[dict[str, Any]]) -> str:
        if not hits:
            return "No catalog results."
        out: list[str] = []
        for i, h in enumerate(hits, start=1):
            pl = h.get("payload") or {}
            score = h.get("_score", h.get("score", 0.0))
            out.append(f"--- Catalog result {i} (score: {score:.3f}) ---")
            bits: list[str] = ["Source: curated catalog"]
            if pl.get("camp_place") or pl.get("location"):
                bits.append(f"Place: {pl.get('camp_place') or pl.get('location')}")
            if pl.get("session_date"):
                bits.append(f"Date: {pl['session_date']}")
            if pl.get("track_title"):
                bits.append(f"Track: {pl['track_title']}")
            if pl.get("performers"):
                bits.append(f"Performers: {', '.join(pl['performers'])}")
            if pl.get("release_ref"):
                bits.append(f"Ref: {pl['release_ref']}")
            out.append(" | ".join(bits))
            out.append(pl.get("text", ""))
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    # ---- public method (LLM-callable) --------------------------------------

    def search_catalog(
        self,
        query: str,
        top_k: int = 8,
        *,
        performers: list[str] | str | None = None,
        location: str | None = None,
        date_range: tuple[str, str] | None = None,
        season: str | None = None,
        track_type: list[str] | str | None = None,
        doc_type: str | None = None,
    ) -> str:
        """Search the curated discourse catalog (a separate, human-verified
        source from the Whisper transcripts). Use this for performer credits,
        canonical song/bhajan titles, release references, human sitting
        summaries, or to sanity-check what the Whisper transcript says.

        :param query: Natural-language question or title. Hindi (Devanagari or
            romanized) and English both work.
        :param top_k: Results after reranking (default 8).
        :param performers: Singer/chorus name(s) credited in the catalog, e.g.
            ["Abhipsa"]. Logical OR. Use for "bhajans sung by Abhipsa".
        :param location: City label, e.g. "NOIDA", "DELHI" (folded to upper).
        :param date_range: (start_iso, end_iso), e.g. ("2010-01-01","2010-01-31").
        :param season: "winter" | "summer" | "monsoon" | "post-monsoon".
        :param track_type: "discourse" | "address" | "meditation" | "invocation"
            | "music" | "bhajan". String or list.
        :param doc_type: Restrict to "track_title" (canonical titles only) or
            "sitting_detail" (hand-transcribed sitting content only). Omit for both.
        :return: Formatted catalog results, each labelled "Source: curated
            catalog" with place/date/title/performers/reference and the text.

        Examples:
        - "Who sang the bhajan about the guru's grace in 2005?"
            → (no filter; semantic) or performers/date filters
        - "Bhajans sung by Abhipsa at the Noida camp"
            → performers=["Abhipsa"], location="NOIDA", doc_type="track_title"
        - "Human transcription of the monsoon meditation sittings"
            → season="monsoon", doc_type="sitting_detail"
        """
        effective_top_k = min(top_k, self.valves.final_top_k)
        vec = self._embed(query)
        qfilter = self._build_filter(
            performers, location, date_range, season, track_type, doc_type
        )
        hits = self._dense(vec, qfilter)
        reranked = self._rerank(query, hits)
        return self._format(reranked[:effective_top_k])
