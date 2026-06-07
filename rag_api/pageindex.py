"""PageIndex-style reasoning retrieval — an additive, opt-in backend.

This is a local, from-scratch implementation of the *method* popularised by
VectifyAI's PageIndex (https://github.com/VectifyAI/PageIndex, MIT): instead of
dense + BM25 fusion, a transcript is turned into a small **tree of sections**
(each with an LLM-written title + summary), and retrieval is the LLM *reasoning*
over that tree to pick the relevant sections — "similarity != relevance".

Why a reimplementation and not the upstream package:
  * the upstream OSS repo ships tree *building* only (PDF/heading oriented);
    the reasoning/tree-search step is its cloud service.
  * our documents are flat WhisperX segment streams — no headings/pages — so we
    build sections by grouping consecutive transcript chunks to a token budget.
  * we must stay strictly local: the reasoning LLM is the local Ollama model.

Trees are JSON files under `settings.pageindex_dir`, written offline by
`ingestion.build_pageindex`. The default deployment never touches this — only a
request with `backend=pageindex` routes here, so the locked Qdrant+Tantivy+
Infinity path is unchanged.

Node schema (2-level: a root description over a flat list of section nodes)::

    {
      "source_file": str,
      "model": str,
      "chunk_count": int,
      "doc_description": str,
      "nodes": [
        {
          "node_id": "0001",
          "title": str,
          "summary": str,
          "source_file": str,
          "start_sec": float | None,
          "end_sec": float | None,
          "speakers": [str, ...],
          "chunk_ids": [str, ...],
          "text": str            # the section's transcript text
        },
        ...
      ]
    }

The fusion/grouping/prompt/parse helpers are module-level pure functions so they
can be unit-tested without Ollama or a filesystem.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests

from rag_api.config import Settings

log = logging.getLogger("rag_api.pageindex")

# 1 token ~= 4 chars (rough, fine for the English+Hindi mix), same heuristic
# `ingestion.enrich_content_tags` uses for its budgeting.
CHARS_PER_TOKEN = 4


# --------------------------------------------------------------------------
# Pure helpers (no network, no I/O — unit-testable)
# --------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def safe_tree_name(source_file: str) -> str:
    """Filesystem-safe stem for a tree JSON file. Deterministic so the builder
    and the retriever agree on the path for a given `source_file`. Mirrors the
    sanitisation in `ingestion.enrich_content_tags.dead_letter`."""
    return "".join(
        c if c.isalnum() or c in "-_." else "_" for c in source_file
    )[:120]


def tree_path(pageindex_dir: str | Path, source_file: str) -> Path:
    return Path(pageindex_dir) / f"{safe_tree_name(source_file)}.json"


def group_chunks_into_sections(
    chunk_rows: list[dict[str, Any]], max_tokens: int,
) -> list[list[dict[str, Any]]]:
    """Group consecutive transcript chunks into sections by a token budget.

    Each `chunk_row` is `{chunk_id, text, start_sec, end_sec, speakers}` in
    transcript order. A new section starts once adding the next chunk would push
    the running section past `max_tokens` — except a section is never empty, so
    a single oversized chunk still forms its own section. Reusing the original
    chunks means each section already carries time/speaker provenance.
    """
    sections: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    running = 0
    for row in chunk_rows:
        toks = estimate_tokens(row.get("text") or "")
        if current and running + toks > max_tokens:
            sections.append(current)
            current = []
            running = 0
        current.append(row)
        running += toks
    if current:
        sections.append(current)
    return sections


def section_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse a section's chunk rows into node-level provenance: the section
    text, its time span, the union of speakers, and the source chunk ids."""
    starts = [r["start_sec"] for r in rows if r.get("start_sec") is not None]
    ends = [r["end_sec"] for r in rows if r.get("end_sec") is not None]
    speakers: list[str] = []
    for r in rows:
        for sp in (r.get("speakers") or []):
            if sp not in speakers:
                speakers.append(sp)
    return {
        "text": "\n".join((r.get("text") or "") for r in rows).strip(),
        "start_sec": min(starts) if starts else None,
        "end_sec": max(ends) if ends else None,
        "speakers": speakers,
        "chunk_ids": [str(r["chunk_id"]) for r in rows],
        "source_file": rows[0].get("source_file") if rows else None,
    }


def build_tree_reasoning_prompt(
    query: str, doc_description: str, nodes: list[dict[str, Any]],
) -> str:
    """Prompt the LLM to pick relevant section node_ids by reasoning over the
    tree's titles + summaries (not embeddings). The model must answer with a
    strict JSON object so the response is machine-parseable."""
    lines = [
        "You are navigating one transcript to answer a user's question.",
        "Below is the transcript's table of contents: numbered sections, each",
        "with a short title and summary. Decide which sections most likely",
        "contain the answer — reason about relevance, not just keyword overlap.",
        "",
        f"Transcript: {doc_description or '(no description)'}",
        "",
        "Sections:",
    ]
    for n in nodes:
        lines.append(
            f"[{n['node_id']}] {n.get('title', '').strip()} — "
            f"{n.get('summary', '').strip()}"
        )
    lines += [
        "",
        f"Question: {query}",
        "",
        'Reply ONLY with JSON: {"node_ids": ["0001", ...]} listing the ids of',
        "the relevant sections, most relevant first. Pick none rather than",
        'irrelevant ones — return {"node_ids": []} if nothing fits.',
    ]
    return "\n".join(lines)


def parse_node_selection(raw: str, valid_ids: set[str]) -> list[str]:
    """Extract the chosen node_ids from the reasoning model's JSON reply.

    Tolerant of extra prose around the JSON object. Keeps only ids that exist
    in the tree, preserving the model's order and dropping duplicates. Returns
    [] on any parse failure — the caller logs and degrades, never crashes.
    """
    obj: Any = None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return []
    if not isinstance(obj, dict):
        return []
    ids = obj.get("node_ids")
    if not isinstance(ids, list):
        return []
    out: list[str] = []
    for i in ids:
        s = str(i).strip()
        if s in valid_ids and s not in out:
            out.append(s)
    return out


def node_to_result(node: dict[str, Any], score: float) -> dict[str, Any]:
    """Shape one selected section node into the API result schema — the same
    shape as `rag_api.retrieval.to_result`, so the frontend and synthesis layer
    treat PageIndex sections exactly like hybrid chunk hits. The section title /
    summary ride along under `metadata` for display."""
    return {
        "result_type": "chunk",
        "chunk_id": f"{node.get('source_file')}#{node['node_id']}",
        "score": round(float(score), 4),
        "text": node.get("text", ""),
        "source_file": node.get("source_file"),
        "start_sec": node.get("start_sec"),
        "end_sec": node.get("end_sec"),
        "speakers": node.get("speakers") or [],
        "metadata": {
            "backend": "pageindex",
            "section_title": node.get("title"),
            "section_summary": node.get("summary"),
            "node_id": node.get("node_id"),
        },
    }


# --------------------------------------------------------------------------
# Ollama JSON call (shared by builder + retriever)
# --------------------------------------------------------------------------


def ollama_generate_json(
    session: requests.Session, ollama_url: str, model: str, prompt: str,
    num_ctx: int, timeout: float,
) -> str:
    """POST /api/generate with format=json. Returns the raw response string.
    Mirrors the call shape in `ingestion.enrich_content_tags`."""
    r = session.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"num_ctx": num_ctx, "temperature": 0.0},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "")


# --------------------------------------------------------------------------
# Tree builder — offline indexing (one LLM call per section + one per doc)
# --------------------------------------------------------------------------


class PageIndexBuilder:
    """Builds a section tree for one transcript via the local Ollama model.

    Stateless across documents; construct once and call `build` per file. All
    LLM failures raise (the CLI catches + dead-letters them) — never silently
    produce a half-built tree.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = requests.Session()

    def _ollama_json(self, prompt: str) -> str:
        return ollama_generate_json(
            self._session, self.settings.ollama_url, self.settings.pageindex_model,
            prompt, self.settings.chat_num_ctx, self.settings.chat_timeout_s,
        )

    def _summarize_section(self, text: str) -> tuple[str, str]:
        """Return (title, summary) for one section. Falls back to a truncated
        title and an empty summary if the model returns unparseable JSON — a
        weak node still indexes; it just reasons slightly worse."""
        prompt = (
            "Summarise this transcript excerpt for a table of contents. "
            'Reply ONLY with JSON {"title": "...", "summary": "..."} where '
            "title is a short phrase (<=8 words) and summary is 1-2 sentences "
            "naming the key topics/people. Use the excerpt's own language.\n\n"
            f"Excerpt:\n{text}"
        )
        raw = self._ollama_json(prompt)
        try:
            obj = json.loads(raw)
            title = str(obj.get("title") or "").strip()
            summary = str(obj.get("summary") or "").strip()
            if title or summary:
                return title or text[:60].strip(), summary
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            log.warning("section summary parse failed (%s) — using fallback", e)
        return text[:60].strip(), ""

    def _describe_doc(self, nodes: list[dict[str, Any]]) -> str:
        """One-line description of the whole transcript from its section
        summaries. Empty string on failure — the tree is still usable."""
        toc = "\n".join(
            f"- {n.get('title', '')}: {n.get('summary', '')}" for n in nodes
        )
        prompt = (
            "Here is a transcript's section outline. Reply ONLY with JSON "
            '{"description": "..."} giving a 1-sentence description of the '
            "whole transcript.\n\n" + toc
        )
        try:
            raw = self._ollama_json(prompt)
            return str(json.loads(raw).get("description") or "").strip()
        except (requests.RequestException, json.JSONDecodeError, TypeError) as e:
            log.warning("doc description failed (%s) — leaving empty", e)
            return ""

    def build(
        self, source_file: str, chunk_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build and return the tree dict for `source_file`. `chunk_rows` are
        ordered `{chunk_id, text, start_sec, end_sec, speakers}` records (from
        chunk_meta). Raises on a hard LLM/transport failure."""
        sections = group_chunks_into_sections(
            chunk_rows, self.settings.pageindex_max_tokens_per_node
        )
        nodes: list[dict[str, Any]] = []
        for i, rows in enumerate(sections, start=1):
            prov = section_provenance(rows)
            for r in rows:
                r.setdefault("source_file", source_file)
            prov.setdefault("source_file", source_file)
            title, summary = self._summarize_section(prov["text"])
            nodes.append({
                "node_id": f"{i:04d}",
                "title": title,
                "summary": summary,
                **prov,
                "source_file": source_file,
            })
        return {
            "source_file": source_file,
            "model": self.settings.pageindex_model,
            "chunk_count": len(chunk_rows),
            "doc_description": self._describe_doc(nodes),
            "nodes": nodes,
        }


# --------------------------------------------------------------------------
# Retriever — reasoning over stored trees
# --------------------------------------------------------------------------


class PageIndexRetriever:
    """Reasoning retrieval over the pre-built section trees on disk.

    Reuses the hybrid `Retriever` for two things it already does well: stage-1
    document selection (summary search) and cross-encoder reranking of the
    selected sections. Reasoning over each candidate's tree is a local Ollama
    call. Degrades gracefully at every step — a missing summary index, an
    unreachable reranker, or a flaky reasoning call each narrow results rather
    than failing the request.
    """

    def __init__(self, settings: Settings, hybrid: Any) -> None:
        self.settings = settings
        self._hybrid = hybrid
        self._session = requests.Session()
        # source_file -> (mtime, tree). Avoids re-reading a tree every query.
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    # ---- tree loading ---------------------------------------------------

    def available_docs(self) -> list[str]:
        """source_files that have a tree on disk (for /api/health visibility)."""
        d = Path(self.settings.pageindex_dir)
        if not d.is_dir():
            return []
        out: list[str] = []
        for p in sorted(d.glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8"))["source_file"])
            except (json.JSONDecodeError, KeyError, OSError) as e:
                log.warning("skipping unreadable tree %s: %s", p, e)
        return out

    def _load_tree(self, source_file: str) -> dict[str, Any] | None:
        path = tree_path(self.settings.pageindex_dir, source_file)
        if not path.is_file():
            return None
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        cached = self._cache.get(source_file)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            tree = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.error("failed to load tree %s: %s", path, e)
            return None
        self._cache[source_file] = (mtime, tree)
        return tree

    # ---- stage 1: candidate documents -----------------------------------

    def _candidate_docs(
        self, query: str, filters: dict[str, Any],
    ) -> list[str]:
        """Pick which transcripts' trees to reason over.

        Priority: an explicitly pinned `source_file` wins; else summary search
        ranks documents by topical match; else (no summary index) fall back to
        every tree on disk (the indexed sample is small by construction).
        """
        sf = filters.get("source_file")
        if isinstance(sf, str) and sf:
            return [sf]
        if isinstance(sf, list) and sf:
            return list(sf)

        try:
            summaries = self._hybrid.search_summaries(
                query, filters, top_k=self.settings.pageindex_candidate_docs,
            )
            docs = [s["source_file"] for s in summaries if s.get("source_file")]
        except (requests.RequestException, KeyError) as e:
            log.warning("pageindex stage-1 summary search failed (%s) — "
                        "falling back to all on-disk trees", e)
            docs = []

        # Keep only docs we actually have a tree for; backfill from disk.
        docs = [d for d in docs if tree_path(self.settings.pageindex_dir, d).is_file()]
        if not docs:
            docs = self.available_docs()
        return docs[: self.settings.pageindex_candidate_docs]

    # ---- stage 2: reasoning over a tree ---------------------------------

    def _select_nodes(self, query: str, tree: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the section nodes the model judges relevant, in its order.
        Empty list (not an error) when the model picks nothing or misbehaves."""
        nodes = tree.get("nodes") or []
        if not nodes:
            return []
        prompt = build_tree_reasoning_prompt(
            query, tree.get("doc_description", ""), nodes
        )
        try:
            raw = ollama_generate_json(
                self._session, self.settings.ollama_url,
                self.settings.pageindex_model, prompt,
                self.settings.chat_num_ctx, self.settings.chat_timeout_s,
            )
        except requests.RequestException as e:
            log.warning("pageindex reasoning call failed for %r on %s: %s",
                        query, tree.get("source_file"), e)
            return []
        by_id = {n["node_id"]: n for n in nodes}
        chosen = parse_node_selection(raw, set(by_id))
        return [by_id[i] for i in chosen]

    # ---- public API -----------------------------------------------------

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """Reasoning retrieval: pick candidate documents, reason over each
        tree to select sections, then rerank the union and return up to `top_k`
        results in the standard result shape. Returns [] when no trees are
        indexed yet — the caller surfaces that as an empty result set."""
        filters = filters or {}
        docs = self._candidate_docs(query, filters)
        if not docs:
            log.info("pageindex: no indexed trees for query %r — empty result", query)
            return []

        selected: list[dict[str, Any]] = []
        for source_file in docs:
            tree = self._load_tree(source_file)
            if tree is None:
                continue
            selected.extend(self._select_nodes(query, tree))

        if not selected:
            return []

        # Rerank the selected sections with the shared cross-encoder so the
        # score is comparable to the hybrid backend. Payload carries the node
        # under "text" (what the reranker scores) plus the node itself.
        candidates: list[tuple[str, float, dict[str, Any]]] = []
        for n in selected:
            cid = f"{n.get('source_file')}#{n['node_id']}"
            candidates.append((cid, 0.0, {"text": n.get("text", ""), "_node": n}))
        reranked = self._hybrid._rerank(query, candidates)

        cap = min(top_k, len(reranked))
        return [node_to_result(pl["_node"], s) for _, s, pl in reranked[:cap]]
