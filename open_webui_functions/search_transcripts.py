"""title: Transcript Search (hybrid dense + BM25 + reranker)
author: transcript-rag
version: 1.0.0
license: MIT
description: Hybrid retrieval over Whisper transcripts. Dense (Qdrant/bge-m3)
    + BM25 (Tantivy sidecar) + RRF fusion + bge-reranker-v2-m3 reranking.
    Exposes two callable methods: search_transcripts (for concept/topic
    queries) and find_quote (for exact-phrase hunts; BM25-weighted higher).

This file is an Open WebUI **Function Tool**. Install via Admin Panel →
Functions → "+ New Function" → paste contents → toggle on → attach to chat
models. See docs/install_functions.md.

Per PRD §6 Phase 7.
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
            description="Qdrant base URL (inside docker network).",
        )
        qdrant_key: str = Field(
            default=os.environ.get("QDRANT_API_KEY", ""),
            description="Qdrant API key (mirrors QDRANT_API_KEY in .env).",
        )
        qdrant_collection: str = Field(
            default=os.environ.get("QDRANT_COLLECTION", "transcripts"),
            description="Qdrant collection name to search.",
        )
        ollama_url: str = Field(
            default=os.environ.get("OLLAMA_URL", "http://ollama:11434"),
            description="Ollama base URL for the embedding model.",
        )
        embed_model: str = Field(
            default=os.environ.get("EMBED_MODEL", "bge-m3"),
            description="Embedding model tag in Ollama.",
        )
        reranker_url: str = Field(
            default=os.environ.get("RERANKER_URL", "http://reranker:7997/rerank"),
            description="Infinity reranker endpoint.",
        )
        reranker_model: str = Field(
            default=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            description="Reranker model identifier.",
        )
        tantivy_proxy_url: str = Field(
            default=os.environ.get(
                "TANTIVY_URL", "http://host.docker.internal:8765"
            ),
            description="Tantivy BM25 sidecar (runs on host, not in docker).",
        )
        candidates_per_source: int = Field(
            default=40,
            description="How many candidates to pull from dense AND BM25 "
            "before fusion.",
        )
        final_top_k: int = Field(
            default=8,
            description="How many final results to return after reranking.",
        )
        bm25_weight: float = Field(
            default=0.65,
            description="Weight on BM25 in reciprocal rank fusion. "
            "0.5 = balanced; >0.5 favors lexical/exact-string; "
            "<0.5 favors semantic. find_quote overrides this to 0.85.",
        )
        http_timeout_s: float = Field(
            default=30.0, description="Per-HTTP-call timeout."
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        # Show citations in the Open WebUI chat surface.
        self.citation = True

    # ---- internal helpers --------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Single-query embedding via Ollama bge-m3 /api/embed."""
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

    def _dense(
        self,
        vec: list[float],
        speaker: str | None,
        source_file: str | None,
    ) -> list[dict[str, Any]]:
        """Qdrant ANN search. Returns list of {id, score, payload}."""
        must: list[dict[str, Any]] = []
        if speaker:
            must.append({"key": "speakers", "match": {"value": speaker}})
        if source_file:
            must.append({"key": "source_file", "match": {"value": source_file}})
        body: dict[str, Any] = {
            "vector": vec,
            "limit": self.valves.candidates_per_source,
            "with_payload": True,
            "with_vector": False,
        }
        if must:
            body["filter"] = {"must": must}
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

    def _bm25(self, query: str) -> list[dict[str, Any]]:
        """Tantivy sidecar /search. Returns list of {chunk_id, text,
        source_file, score}."""
        r = requests.get(
            f"{self.valves.tantivy_proxy_url}/search",
            params={"q": query, "k": self.valves.candidates_per_source},
            timeout=self.valves.http_timeout_s,
        )
        r.raise_for_status()
        return r.json().get("hits", [])

    def _rrf(
        self,
        dense_hits: list[dict[str, Any]],
        bm25_hits: list[dict[str, Any]],
        bm25_weight: float,
        k: int = 60,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Weighted reciprocal rank fusion.
        Returns [(chunk_id, fused_score, payload-or-bm25-hit), ...] sorted desc.
        For dense hits, the payload is preserved; for BM25-only hits, the
        BM25 doc dict is preserved so the reranker / formatter has the text."""
        scored: dict[str, list[float | dict[str, Any] | None]] = {}
        for rank, h in enumerate(dense_hits):
            cid = str(h["id"])
            entry = scored.setdefault(cid, [0.0, None])
            entry[0] = float(entry[0]) + (1 - bm25_weight) / (k + rank + 1)
            if entry[1] is None:
                entry[1] = h.get("payload") or {}
        for rank, h in enumerate(bm25_hits):
            cid = str(h["chunk_id"])
            entry = scored.setdefault(cid, [0.0, None])
            entry[0] = float(entry[0]) + bm25_weight / (k + rank + 1)
            if entry[1] is None:
                # BM25-only hit: synthesize a payload-shaped dict from the
                # Tantivy doc so downstream formatting still works.
                entry[1] = {
                    "text": h.get("text", ""),
                    "source_file": h.get("source_file", ""),
                    "speakers": [],
                    "start_sec": None,
                    "end_sec": None,
                }
        merged = [(cid, float(s), pl or {}) for cid, (s, pl) in scored.items()]
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    def _rerank(
        self, query: str, candidates: list[tuple[str, float, dict[str, Any]]],
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Cross-encoder rerank via Infinity. Returns the top final_top_k
        candidates sorted by reranker score desc. Falls back to RRF order
        if the reranker is unreachable (logged-via-comment scenario)."""
        if not candidates:
            return []
        texts = [pl.get("text", "") for _, _, pl in candidates]
        try:
            r = requests.post(
                self.valves.reranker_url,
                json={
                    "model": self.valves.reranker_model,
                    "query": query,
                    "documents": texts,
                    "return_documents": False,
                },
                timeout=self.valves.http_timeout_s,
            )
            r.raise_for_status()
            results = r.json().get("results") or []
        except requests.RequestException:
            # If reranker is down, return RRF order truncated. Don't kill the
            # whole search — degraded results are still useful.
            return candidates[: self.valves.final_top_k]
        # Pair reranker scores back to candidates and sort desc.
        rescored: list[tuple[str, float, dict[str, Any]]] = []
        for item in results:
            idx = int(item["index"])
            score = float(item["relevance_score"])
            cid, _, pl = candidates[idx]
            rescored.append((cid, score, pl))
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[: self.valves.final_top_k]

    @staticmethod
    def _format(results: list[tuple[str, float, dict[str, Any]]]) -> str:
        """Format per PRD §6 Phase 7 output spec.

        --- Result N (score: X.XXX) ---
        Source: <file> | <start>s -> <end>s | Speakers: A, B
        <chunk text body>

        (blank line between results)
        """
        if not results:
            return "No results."
        out: list[str] = []
        for i, (_cid, score, pl) in enumerate(results, start=1):
            source = pl.get("source_file") or "?"
            start = pl.get("start_sec")
            end = pl.get("end_sec")
            if start is not None and end is not None:
                ts = f"{start}s -> {end}s"
            else:
                ts = "timestamps unavailable"
            speakers = pl.get("speakers") or []
            spk = ", ".join(speakers) if speakers else "unknown"
            out.append(f"--- Result {i} (score: {score:.3f}) ---")
            out.append(f"Source: {source} | {ts} | Speakers: {spk}")
            out.append(pl.get("text", ""))
            out.append("")  # blank line separator
        return "\n".join(out).rstrip() + "\n"

    # ---- public methods (LLM-callable) -------------------------------------

    def search_transcripts(
        self,
        query: str,
        speaker: str | None = None,
        source_file: str | None = None,
        top_k: int = 8,
    ) -> str:
        """Hybrid semantic + BM25 search over transcripts.

        Use this when the user asks about a topic, theme, or concept ("what
        did we discuss about latency?", "find content about the new dashboard").
        For exact-quote hunts, prefer `find_quote` instead.

        :param query: Natural-language question or topic. Example:
            "discussion of organizational changes in Q3 planning".
        :param speaker: Optional. If provided, restrict to chunks containing
            this speaker label (e.g. "SPEAKER_00"). Useful for "what did X
            say about Y" questions.
        :param source_file: Optional. Restrict to a specific transcript
            filename (e.g. "2024_03_15_quarterly_review.json").
        :param top_k: Number of final results after reranking (default 8;
            falls back to valves.final_top_k if higher).
        :return: Formatted text with one block per result, including source
            file, timestamps, speakers, BM25/dense fused + reranked score,
            and the chunk text. Citations are surfaced in the chat UI.

        Example user query: "what did we say about the platform team launch?"
        """
        # Allow caller to dial top_k down per-call; cap at valves to keep
        # rerank batch reasonable.
        effective_top_k = min(top_k, self.valves.final_top_k)
        vec = self._embed(query)
        dense = self._dense(vec, speaker, source_file)
        bm25 = self._bm25(query)
        fused = self._rrf(dense, bm25, self.valves.bm25_weight)
        # Optional post-fusion filter when only BM25 hits would otherwise
        # leak through ignoring the filter args.
        if speaker or source_file:
            fused = [(c, s, pl) for c, s, pl in fused
                     if (not speaker or speaker in (pl.get("speakers") or []))
                     and (not source_file or pl.get("source_file") == source_file)]
        reranked = self._rerank(query, fused[: self.valves.candidates_per_source])
        return self._format(reranked[:effective_top_k])

    def find_quote(self, partial_quote: str, top_k: int = 5) -> str:
        """Locate a specific phrase or near-quote in the transcripts.

        Use this when the user is hunting for an exact line, a remembered
        snippet, or "who said X". BM25 is weighted heavily (0.85) because
        lexical match matters more than semantic similarity for quote
        recovery.

        :param partial_quote: The phrase the user remembers (can be partial
            or approximate). Example: "we have to focus on retention".
        :param top_k: Number of final results (default 5).
        :return: Formatted text per `search_transcripts` output spec.

        Example user query: "find when someone said 'mobile rewrite is on
        track'".
        """
        # Override bm25 weight per PRD spec for quote-finding mode.
        prior = self.valves.bm25_weight
        try:
            self.valves.bm25_weight = 0.85
            vec = self._embed(partial_quote)
            dense = self._dense(vec, speaker=None, source_file=None)
            bm25 = self._bm25(partial_quote)
            fused = self._rrf(dense, bm25, self.valves.bm25_weight)
            reranked = self._rerank(
                partial_quote, fused[: self.valves.candidates_per_source]
            )
            return self._format(reranked[: min(top_k, self.valves.final_top_k)])
        finally:
            self.valves.bm25_weight = prior
