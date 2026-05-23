"""title: Transcript Analytics
author: transcript-rag
version: 1.0.0
license: MIT
description: Postgres-backed analytics over chunk_meta / file_meta. Counts
    mentions, ranks speakers by topic, lists transcripts that touch a topic.
    Uses tsvector full-text search (idx_chunk_text_fts) and the
    speakers TEXT[] GIN index (idx_chunk_speakers) populated by Phase 4/6
    ingestion.

This file is an Open WebUI **Function Tool**. Install via Admin Panel →
Functions → "+ New Function" → paste contents → toggle on → attach to chat
models. See docs/install_functions.md.

Per PRD §6 Phase 7.
"""
from __future__ import annotations

import os
from typing import Any

import psycopg2
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        pg_dsn: str = Field(
            default=os.environ.get(
                "PG_DSN",
                "postgresql://owui:{}@postgres:5432/openwebui".format(
                    os.environ.get("POSTGRES_PASSWORD", "")
                ),
            ),
            description="Postgres DSN. Default uses the docker network name "
            "'postgres'; from the host use 'localhost'.",
        )
        statement_timeout_ms: int = Field(
            default=10_000,
            description="Per-query Postgres statement_timeout (milliseconds).",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.citation = True

    # ---- internal --------------------------------------------------------

    def _query(self, sql: str, args: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        conn = psycopg2.connect(self.valves.pg_dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SET statement_timeout = {self.valves.statement_timeout_ms}"
                )
                cur.execute(sql, args)
                return cur.fetchall()
        finally:
            conn.close()

    # ---- public methods (LLM-callable) -----------------------------------

    def count_mentions(self, term: str, speaker: str | None = None) -> str:
        """Count how many chunks mention `term` across all transcripts.

        Use this for "how often" / "how many times" questions. Backed by the
        English tsvector GIN index on chunk_meta.text, so it handles
        morphological variants (e.g. "discuss" matches "discussed",
        "discussing").

        :param term: Word or short phrase. Quoted phrases are supported via
            `phraseto_tsquery`-like behavior: pass them as-is.
        :param speaker: Optional. Restrict to chunks where the given speaker
            label appears.
        :return: A one-line summary like
            'Term "platform" mentioned in 47 chunks (SPEAKER_00: filter on).'

        Example user query: "how many times did we talk about the dashboard?"
        """
        if speaker:
            rows = self._query(
                """SELECT COUNT(*) FROM chunk_meta
                   WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)
                     AND %s = ANY(speakers)""",
                (term, speaker),
            )
            label = f"(filtered to speaker {speaker!r})"
        else:
            rows = self._query(
                """SELECT COUNT(*) FROM chunk_meta
                   WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)""",
                (term,),
            )
            label = "(across all speakers)"
        n = int(rows[0][0]) if rows else 0
        return f'Term "{term}" mentioned in {n} chunks {label}.'

    def top_speakers_for_topic(self, term: str, limit: int = 10) -> str:
        """Rank speakers by how often they appear in chunks mentioning `term`.

        Use this for "who talks most about X" / "which speaker discusses X"
        questions. Returns up to `limit` speaker labels with their chunk counts.

        :param term: Word or short phrase to search for.
        :param limit: Max number of speakers to return (default 10).
        :return: Multi-line formatted ranking, e.g.
            'Top speakers for "latency":\\n  SPEAKER_02: 14 chunks\\n  SPEAKER_00: 9 chunks'

        Example user query: "who talks most about distributed systems?"
        """
        rows = self._query(
            """SELECT speaker, COUNT(*) AS n
               FROM (
                   SELECT UNNEST(speakers) AS speaker
                   FROM chunk_meta
                   WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)
               ) s
               GROUP BY speaker
               ORDER BY n DESC
               LIMIT %s""",
            (term, int(limit)),
        )
        if not rows:
            return f'No speakers found for term "{term}".'
        lines = [f'Top speakers for "{term}":']
        for speaker, n in rows:
            lines.append(f"  {speaker}: {n} chunks")
        return "\n".join(lines)

    def list_transcripts_mentioning(self, term: str, limit: int = 20) -> str:
        """List distinct source files that contain `term`.

        Use this for "which transcripts mention X" / "which files cover X"
        questions. Pairs nicely with `search_transcripts(query=..., source_file=...)`
        for drilling into a specific file.

        :param term: Word or short phrase to search for.
        :param limit: Max number of source files to return (default 20).
        :return: Multi-line list of filenames with their chunk counts, e.g.
            'Transcripts mentioning "platform" (3):\\n  meeting_2024_03.json (12)\\n  ...'

        Example user query: "which transcripts mention the platform team?"
        """
        rows = self._query(
            """SELECT source_file, COUNT(*) AS n
               FROM chunk_meta
               WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)
               GROUP BY source_file
               ORDER BY n DESC
               LIMIT %s""",
            (term, int(limit)),
        )
        if not rows:
            return f'No transcripts found for term "{term}".'
        lines = [f'Transcripts mentioning "{term}" ({len(rows)}):']
        for source_file, n in rows:
            lines.append(f"  {source_file} ({n})")
        return "\n".join(lines)
