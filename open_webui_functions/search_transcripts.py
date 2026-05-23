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
        *,
        date_range: tuple[str, str] | None = None,
        season: str | None = None,
        track_type: list[str] | str | None = None,
        location: str | None = None,
        event_id: str | None = None,
        event_type: str | None = None,
        primary_language: str | None = None,
        topics: list[str] | None = None,
        people_named: list[str] | None = None,
        scriptures_referenced: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Qdrant ANN search. Returns list of {id, score, payload}.

        Phase 12 adds path-metadata filters (date_range, season, track_type,
        location, event_id). Phase 13 adds content-tag filters (event_type,
        primary_language, topics, people_named, scriptures_referenced).
        Array filters (topics/people_named/scriptures_referenced) use
        MatchAny — logical OR within the filter, AND across filters.
        """
        must: list[dict[str, Any]] = []
        if speaker:
            must.append({"key": "speakers", "match": {"value": speaker}})
        if source_file:
            must.append({"key": "source_file", "match": {"value": source_file}})
        if location:
            must.append({"key": "location", "match": {"value": location.upper()}})
        if event_id:
            must.append({"key": "event_id", "match": {"value": event_id}})
        if season:
            must.append({"key": "season", "match": {"value": season}})
        if track_type:
            tt = [track_type] if isinstance(track_type, str) else list(track_type)
            must.append({"key": "track_type", "match": {"any": tt}})
        if date_range:
            gte, lte = date_range
            must.append({"key": "session_date", "range": {"gte": gte, "lte": lte}})
        # --- Phase 13 content-tag filters ---
        if event_type:
            must.append({"key": "event_type", "match": {"value": event_type}})
        if primary_language:
            must.append({"key": "primary_language", "match": {"value": primary_language}})
        if topics:
            must.append({"key": "topics", "match": {"any": list(topics)}})
        if people_named:
            must.append({"key": "people_named", "match": {"any": list(people_named)}})
        if scriptures_referenced:
            must.append({"key": "scriptures_referenced",
                         "match": {"any": list(scriptures_referenced)}})

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
            # Phase 12 metadata line — only printed when at least one field
            # is present. Lets the LLM cite event/date/track naturally.
            meta_bits: list[str] = []
            if pl.get("event_id"):
                meta_bits.append(f"Event: {pl['event_id']}")
            elif pl.get("location"):
                meta_bits.append(f"Location: {pl['location']}")
            if pl.get("session_date"):
                meta_bits.append(f"Date: {pl['session_date']}")
            if pl.get("track_title"):
                meta_bits.append(f"Track: {pl['track_title']}")
            if pl.get("track_type"):
                meta_bits.append(f"Type: {pl['track_type']}")
            if pl.get("season"):
                meta_bits.append(f"Season: {pl['season']}")
            # Phase 13 content tags — only printed when present.
            if pl.get("event_type"):
                meta_bits.append(f"EventType: {pl['event_type']}")
            if pl.get("primary_language"):
                meta_bits.append(f"Lang: {pl['primary_language']}")
            if pl.get("topics"):
                meta_bits.append(f"Topics: {', '.join(pl['topics'])}")
            if pl.get("scriptures_referenced"):
                meta_bits.append(
                    f"Scriptures: {', '.join(pl['scriptures_referenced'])}"
                )
            if meta_bits:
                out.append(" | ".join(meta_bits))
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
        *,
        date_range: tuple[str, str] | None = None,
        season: str | None = None,
        track_type: list[str] | str | None = None,
        location: str | None = None,
        event_id: str | None = None,
        event_type: str | None = None,
        primary_language: str | None = None,
        topics: list[str] | None = None,
        people_named: list[str] | None = None,
        scriptures_referenced: list[str] | None = None,
    ) -> str:
        """Hybrid semantic + BM25 search over transcripts, with optional
        metadata filters. Two filter families:
        - Path-based (Phase 12): when/where it was recorded. date_range,
          season, track_type, location, event_id.
        - Content-based (Phase 13): what the audio is about. event_type,
          primary_language, topics, people_named, scriptures_referenced.

        Use this when the user asks about a topic, theme, or concept ("what
        did we discuss about latency?", "find content about the new dashboard").
        For exact-quote hunts, prefer `find_quote` instead.

        :param query: Natural-language question or topic. Example:
            "discussion of organizational changes in Q3 planning".
        :param speaker: Optional. If provided, restrict to chunks containing
            this speaker label (e.g. "Swami ji", "SPEAKER_00").
        :param source_file: Optional. Restrict to a specific transcript
            filename (e.g. "04 PRAVACHAN.json").
        :param top_k: Number of final results after reranking (default 8;
            capped at valves.final_top_k).
        :param date_range: Optional (start_iso, end_iso) date pair, e.g.
            ("2010-01-01", "2010-12-31"). Restricts to chunks whose
            session_date falls in the range, inclusive.
        :param season: Optional. One of "winter" (Jan-Feb), "summer"
            (Mar-May), "monsoon" (Jun-Sep), "post-monsoon" (Oct-Dec).
            Use for queries like "what did Swami ji say on a monsoon day"
            — pass season="monsoon".
        :param track_type: Optional. One of "discourse" (PRAVACHAN),
            "address" (SAMBODHAN), "meditation", "invocation", "music",
            "bhajan". Pass a single string or a list. Recommended default
            for teaching queries: ["discourse", "address"] to exclude
            bhajans, music, and meditation tracks.
        :param location: Optional location/city label, e.g. "NOIDA",
            "RISHIKESH". Case-insensitive on the caller side; folded to
            uppercase before the filter.
        :param event_id: Optional event folder name, e.g.
            "01 NOIDA 7 - 10 JAN 2010". Pin to a specific camp/event.
        :param event_type: Optional content tag. One of "satsang", "bhajan",
            "meditation", "qa", "discourse", "mixed", "unknown". Differs
            from track_type (path-derived) by reflecting what the model
            actually heard, not what folder named the file.
        :param primary_language: Optional. One of "hindi", "sanskrit",
            "mixed". Useful when user wants only Hindi summaries or only
            Sanskrit chant content.
        :param topics: Optional list of topic tags. Logical OR within the
            list. Tags are lowercase hyphenated (e.g. "karma-yoga",
            "self-inquiry", "guru-bhakti").
        :param people_named: Optional list of names. Returns files where
            Guruji *mentions* any of these people. Match is exact on the
            tag string — ASR transliterations may need spelling variants.
        :param scriptures_referenced: Optional list. e.g.
            ["Bhagavad Gita", "Upanishads", "Yoga Sutras"]. Logical OR.
        :return: Formatted text with one block per result, including source
            file, timestamps, speakers, BM25/dense fused + reranked score,
            and the chunk text. Citations are surfaced in the chat UI.

        Example queries this enables:
        - "What did Swami ji say on a monsoon day about barsat?"
            → season="monsoon", track_type=["discourse", "address"]
        - "Find discourses from the Noida camp in January 2010"
            → location="NOIDA", date_range=("2010-01-01", "2010-01-31"),
              track_type="discourse"
        - "When did Guruji talk about the Bhagavad Gita?"
            → scriptures_referenced=["Bhagavad Gita"]
        - "Find satsangs where he mentions Anush"
            → event_type="satsang", people_named=["Anush"]
        - "Discourses about karma yoga"
            → topics=["karma-yoga"], track_type="discourse"
        """
        # Allow caller to dial top_k down per-call; cap at valves to keep
        # rerank batch reasonable.
        effective_top_k = min(top_k, self.valves.final_top_k)
        vec = self._embed(query)
        dense = self._dense(
            vec, speaker, source_file,
            date_range=date_range, season=season, track_type=track_type,
            location=location, event_id=event_id,
            event_type=event_type, primary_language=primary_language,
            topics=topics, people_named=people_named,
            scriptures_referenced=scriptures_referenced,
        )
        bm25 = self._bm25(query)
        fused = self._rrf(dense, bm25, self.valves.bm25_weight)
        # Post-fusion filter: BM25-only hits don't carry payload metadata,
        # so any filter that's set is enforced here against the dense-side
        # payloads we do have. BM25-only hits without payload are dropped
        # when a metadata filter is active (consistent with user intent —
        # they asked for a filtered subset).
        any_filter = (
            speaker or source_file or season or track_type
            or location or event_id or date_range
            or event_type or primary_language or topics
            or people_named or scriptures_referenced
        )
        if any_filter:
            tt_set: set[str] | None = None
            if track_type:
                tt_set = {track_type} if isinstance(track_type, str) else set(track_type)
            topics_set = set(topics) if topics else None
            people_set = set(people_named) if people_named else None
            scriptures_set = (set(scriptures_referenced)
                              if scriptures_referenced else None)
            filtered: list[tuple[str, float, dict[str, Any]]] = []
            for c, s, pl in fused:
                if speaker and speaker not in (pl.get("speakers") or []):
                    continue
                if source_file and pl.get("source_file") != source_file:
                    continue
                if season and pl.get("season") != season:
                    continue
                if tt_set is not None and pl.get("track_type") not in tt_set:
                    continue
                if location and (pl.get("location") or "").upper() != location.upper():
                    continue
                if event_id and pl.get("event_id") != event_id:
                    continue
                if date_range and pl.get("session_date"):
                    gte, lte = date_range
                    sd = pl.get("session_date")
                    if not (gte <= sd <= lte):
                        continue
                if event_type and pl.get("event_type") != event_type:
                    continue
                if primary_language and pl.get("primary_language") != primary_language:
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
                filtered.append((c, s, pl))
            fused = filtered
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
