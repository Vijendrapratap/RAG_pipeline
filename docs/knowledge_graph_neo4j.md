# Knowledge Graph on Neo4j — implementation plan (proposed Phase 15, engine-specific)

> **Status: PROPOSAL, not yet built.** Companion to
> [`docs/knowledge_graph.md`](knowledge_graph.md) — read that first for *why* a
> graph, the latent node/edge model, and the entity-resolution plan. This doc
> covers **only one thing**: if we pick **Neo4j** as the graph engine, exactly
> how it wires into the current stack.
>
> It is **additive**. It sits *beside* Qdrant/Tantivy/reranker, never replaces
> them. Postgres stays the source of truth; Neo4j is a **derived, rebuildable
> projection**. Nothing here runs until we explicitly open Phase 15, and the
> whole thing is gated behind a `GRAPH_ENABLED` flag that defaults **off** — so
> the running retrieval path is byte-for-byte unchanged until someone flips it.

---

## 0. When to pick Neo4j (vs Apache AGE / plain SQL)

The main doc §4 recommends **Apache AGE** (graph inside your existing Postgres)
as the lowest-friction on-ramp. Pick **Neo4j Community** instead only when:

- Traversals get deep/complex enough that Cypher's expressiveness and Neo4j's
  query planner clearly beat recursive CTEs / AGE, **or**
- You want the mature **Neo4j Browser** visual tooling for exploring the graph
  during development, **or**
- You want a clean separation: the graph is its own service with its own
  operational surface, independent of the analytics Postgres.

The cost you accept for that: **a second stateful service** (JVM, ~1–2 GB RAM
floor), two more ports, and a one-way Postgres→Neo4j sync job to keep current.
On the RTX 5090 box that's fine — Neo4j is CPU/RAM, competes with nothing on the
GPU (Ollama + reranker own that).

---

## 1. License & components — the open-source-only check (CLAUDE.md rule 2)

| Component | License | Verdict |
|---|---|---|
| **Neo4j Community Edition** (the DB) | **GPLv3** | ✅ OSI-approved open source. Runs in-container, strictly local — no data leaves the box. |
| **`neo4j` Python driver** (Bolt client) | **Apache-2.0** | ✅ Clean. This is the only new Python dependency. |
| **APOC Core** (utility procedures) | **Apache-2.0** | ✅ Optional; bundle only `apoc` (core), not `apoc-extended`. |
| **Neo4j GDS** (Graph Data Science — Louvain, PageRank) | ⚠️ **source-available, verify before use** | 🟡 Wanted only for community detection (§9 global summaries). **Do not adopt without an explicit license sign-off.** Fallback below uses OSI-clean `python-louvain`/`networkx` offline instead. |

**Bottom line:** Neo4j Community (GPLv3) + the Apache-2.0 driver clear rule 2.
Keep **GDS out of the default path** — treat it exactly like Memgraph-BSL was
treated in the main doc: gated on a license decision, with a clean fallback.

> GPLv3 note: we *use* Neo4j as an unmodified service over the network (Bolt).
> We are not linking Neo4j's server code into our Python, and we're not
> distributing Neo4j. This is ordinary use of a GPLv3 server — the same posture
> as running Postgres or any GPL DB. If the project's distribution model ever
> changes, re-check.

---

## 2. Where Neo4j sits in the architecture

```
                    ┌──────────────── unchanged retrieval path ────────────────┐
   query ─▶ query_parse ─▶ Qdrant (dense) + Tantivy (BM25) ─▶ bge-reranker ─▶ synthesis ─▶ answer
                    └──────────────────────────────┬───────────────────────────┘
                                                    │  (GraphRAG augmentation, §7)
                                                    ▼
   Postgres (source of truth)                  rag_api/graph.py ──Bolt──▶  ┌──────────┐
   catalog_sitting / catalog_track ──build──▶  named traversals            │  Neo4j   │
   chunk_meta / file_meta            (§5)                  ◀────────────────│ (graph)  │
                                                                           └──────────┘
        ingestion/graph/build_graph_neo4j.py  ── one-way sync (MERGE) ──────────▲
```

- **Read path is untouched.** Neo4j is consulted *only* by the new graph query
  layer, and only when `GRAPH_ENABLED=true`.
- **Source of truth = Postgres.** Neo4j holds *no* data that can't be rebuilt
  from the base tables. Drop the volume, re-run the builder, you're back.

### Ports (add to CLAUDE.md's port table)

| Service | Port | Purpose |
|---|---|---|
| Neo4j Bolt | **7687** | driver protocol (rag-api ↔ Neo4j) |
| Neo4j HTTP (Browser) | **7474** | visual browser, dev only |

Container DNS: rag-api reaches Neo4j at `bolt://neo4j:7687`. From the host:
`bolt://localhost:7687`, Browser at `http://localhost:7474`.

---

## 3. Deployment — docker-compose service

Add to [`docker-compose.yml`](../docker-compose.yml), matching the existing
service style. **No GPU reservation** (unlike `ollama`/`reranker`): Neo4j is
JVM/CPU.

```yaml
  neo4j:
    image: neo4j:5-community
    container_name: neo4j
    restart: unless-stopped
    ports: ["7474:7474", "7687:7687"]
    volumes:
      - ./data/neo4j/data:/data
      - ./data/neo4j/logs:/logs
    environment:
      # Auth: user is always "neo4j"; password from .env (gitignored).
      NEO4J_AUTH: "neo4j/${NEO4J_PASSWORD}"
      # Apache-2.0 core procedures only. Omit to run with zero plugins.
      NEO4J_PLUGINS: '["apoc"]'
      # Memory sizing — Neo4j does NOT autodetect well in containers. Set both.
      # Heap for query execution; pagecache for graph+index working set. The
      # projected graph here is small (entities + edges, NOT the 5 TB corpus),
      # so this is modest. Raise pagecache if traversals spill to disk.
      NEO4J_server_memory_heap_initial__size: "1G"
      NEO4J_server_memory_heap_max__size: "2G"
      NEO4J_server_memory_pagecache_size: "1G"
      # Bolt listens on all interfaces inside the container network.
      NEO4J_server_bolt_listen__address: "0.0.0.0:7687"
    healthcheck:
      # cypher-shell ships in the image; a trivial query proves Bolt is up.
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p $${NEO4J_PASSWORD} 'RETURN 1' || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 10
      start_period: 40s
```

And make `rag-api` wait for it (only matters when the flag is on):

```yaml
  rag-api:
    depends_on: [ollama, qdrant, postgres, reranker, neo4j]
    environment:
      # ... existing vars ...
      GRAPH_ENABLED: "${GRAPH_ENABLED:-false}"
      NEO4J_URI: "bolt://neo4j:7687"
      NEO4J_USER: "neo4j"
      NEO4J_PASSWORD: "${NEO4J_PASSWORD}"
      NEO4J_DATABASE: "neo4j"
      GRAPH_QUERY_TIMEOUT_S: "${GRAPH_QUERY_TIMEOUT_S:-10}"
```

> The graph projection is **entities and their relationships**, not transcript
> text. It's small (thousands of sittings/tracks, tens of thousands of
> person/place/topic nodes) — this fits comfortably in the memory above. The
> 5 TB of chunk *text* stays in Postgres/Qdrant; Neo4j stores only `chunk_id`
> references, never the passage bodies.

### `.env.example` additions (committed; real values live in gitignored `.env`)

```dotenv
# --- Knowledge graph (Neo4j) — additive, off by default (Phase 15) ---
GRAPH_ENABLED=false
NEO4J_PASSWORD=${NEO4J_PASSWORD}
GRAPH_QUERY_TIMEOUT_S=10
```

---

## 4. Config wiring — `rag_api/config.py`

Add to the `Settings` class, in the same env-var style as the rest:

```python
    # --- Neo4j (knowledge graph — additive, Phase 15; off unless enabled) ---
    # GRAPH_ENABLED gates the whole feature. When false the graph query layer is
    # never constructed and the retrieval path is unchanged — safe default.
    graph_enabled: bool = os.environ.get("GRAPH_ENABLED", "false").lower() == "true"
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "")
    neo4j_database: str = os.environ.get("NEO4J_DATABASE", "neo4j")
    # Per-query wall-clock ceiling, mirroring analytics.py's statement_timeout.
    graph_query_timeout_s: float = float(os.environ.get("GRAPH_QUERY_TIMEOUT_S", "10"))
```

### `pyproject.toml` — one new dependency

```toml
    "neo4j>=5.20",   # Apache-2.0 Bolt driver; graph query layer only
```

---

## 5. The graph model in Neo4j

Same node/edge model as the main doc §2 — just expressed as Neo4j **labels** and
**relationship types**. Property keys carry the facets you already filter on.

### Constraints & indexes (run once, at build time)

```cypher
// Uniqueness = the canonical-identity guarantee. MERGE relies on these.
CREATE CONSTRAINT person_canonical  IF NOT EXISTS FOR (n:Person)    REQUIRE n.canonical  IS UNIQUE;
CREATE CONSTRAINT place_canonical   IF NOT EXISTS FOR (n:Place)     REQUIRE n.canonical  IS UNIQUE;
CREATE CONSTRAINT work_canonical    IF NOT EXISTS FOR (n:Work)      REQUIRE n.canonical  IS UNIQUE;
CREATE CONSTRAINT scripture_canon   IF NOT EXISTS FOR (n:Scripture) REQUIRE n.canonical  IS UNIQUE;
CREATE CONSTRAINT topic_canonical   IF NOT EXISTS FOR (n:Topic)     REQUIRE n.canonical  IS UNIQUE;
CREATE CONSTRAINT sitting_key       IF NOT EXISTS FOR (n:Sitting)   REQUIRE n.sitting_key IS UNIQUE;
CREATE CONSTRAINT track_joinkey     IF NOT EXISTS FOR (n:Track)     REQUIRE n.join_key   IS UNIQUE;
CREATE CONSTRAINT file_source       IF NOT EXISTS FOR (n:File)      REQUIRE n.source_file IS UNIQUE;
CREATE CONSTRAINT chunk_id          IF NOT EXISTS FOR (n:Chunk)     REQUIRE n.chunk_id   IS UNIQUE;

// Facet lookups used by the traversals in §6.
CREATE INDEX sitting_season IF NOT EXISTS FOR (n:Sitting) ON (n.season);
CREATE INDEX sitting_date   IF NOT EXISTS FOR (n:Sitting) ON (n.session_date);
CREATE INDEX file_source_ix IF NOT EXISTS FOR (n:File)    ON (n.source_file);
```

### Labels → source columns

| Neo4j label | Key property | Source |
|---|---|---|
| `:Person` | `canonical` | resolved from `catalog_*.performers[]` (seed) + `file_meta.people_named[]` (§8) |
| `:Place` | `canonical` | `location` / `camp_place` / `venue` / `places_named[]` |
| `:Work` | `canonical` | `catalog_track.track_title` |
| `:Scripture` | `canonical` | `file_meta.scriptures_referenced[]` |
| `:Topic` | `canonical` | `file_meta.topics[]` |
| `:Sitting` | `sitting_key` | `catalog_sitting.sitting_key` (+ props: `location, season, camp_year, session_date`) |
| `:Track` | `join_key` | `catalog_track.join_key` (+ props: `track_no, track_type, track_title`) |
| `:File` | `source_file` | `file_meta.source_file` (+ props: `session_date, primary_language, event_type`) |
| `:Chunk` | `chunk_id` | `chunk_meta.chunk_id` (reference only — **no text stored**) |

### Relationship types

```
(:Sitting)-[:HAS_TRACK]->(:Track)          catalog_track.sitting_key
(:Track)-[:PERFORMED_BY]->(:Person)        catalog_track.performers[]
(:Track)-[:ALIGNED_TO]->(:File)            catalog_track.matched_source_file
(:File)-[:HAS_CHUNK]->(:Chunk)             chunk_meta.source_file
(:Chunk)-[:SPOKEN_BY]->(:Person)           chunk_meta.speakers[]
(:File)-[:MENTIONS]->(:Person|:Place)      file_meta.people_named[] / places_named[]
(:File)-[:REFERENCES]->(:Scripture)        file_meta.scriptures_referenced[]
(:File)-[:ABOUT]->(:Topic)                 file_meta.topics[]
(:Sitting)-[:AT]->(:Place)                 catalog_sitting.location / venue
(:Person)-[:ALIAS_OF]->(:Person)           entity resolution (surface form → canonical), §8
```

---

## 6. Build / sync — `ingestion/graph/build_graph_neo4j.py`

One-way, idempotent projection from Postgres into Neo4j. Reads with `psycopg2`
(same as `analytics.py`), writes with the `neo4j` driver using **`MERGE` +
batched `UNWIND`** so a re-run changes nothing (upsert semantics).

```python
"""Project the latent Postgres graph into Neo4j. Idempotent and rebuildable.

Postgres is the source of truth; this writes a derived graph. Safe to re-run:
every node/edge is MERGE'd on its canonical key, so a second pass is a no-op.
Per CLAUDE.md rule 6: rows that fail to project are logged with context and
skipped, never silently dropped.
"""
from __future__ import annotations

import logging
import psycopg2
from neo4j import GraphDatabase

log = logging.getLogger("graph.build")

# Nodes+edges for one relationship, MERGE'd in a batch. Parameterized — never
# string-formatted — so Devanagari titles and quotes can't break the query.
_MERGE_TRACK_PERFORMER = """
UNWIND $rows AS row
MERGE (t:Track {join_key: row.join_key})
  ON CREATE SET t.track_no = row.track_no, t.track_type = row.track_type,
                t.track_title = row.track_title
MERGE (p:Person {canonical: row.performer})
MERGE (t)-[:PERFORMED_BY]->(p)
"""

def sync(pg_dsn: str, uri: str, user: str, password: str,
         database: str = "neo4j", batch: int = 1000) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        _apply_schema(driver, database)            # constraints + indexes (§5)
        with psycopg2.connect(pg_dsn) as pg:
            _sync_sittings(pg, driver, database, batch)
            _sync_tracks(pg, driver, database, batch)
            _sync_track_performers(pg, driver, database, batch)  # uses MERGE above
            _sync_file_alignments(pg, driver, database, batch)
            _sync_file_mentions(pg, driver, database, batch)     # people/places/scripture/topic
        log.info("graph sync complete")
    finally:
        driver.close()
```

Run it as an offline job (a Make target / one-shot container), the same way the
Tantivy index and PageIndex trees are built offline today. It is **not** part of
the request path.

Reconciliation (an acceptance check): after a sync, node/edge counts must match
source rows — e.g. `MATCH (:Track)-[:ALIGNED_TO]->(:File) RETURN count(*)` should
equal the number of `catalog_track` rows with a non-null `matched_source_file`.

---

## 7. Query layer — `rag_api/graph.py`

Mirrors the shape of [`rag_api/analytics.py`](../rag_api/analytics.py): a small
class, **guarded import**, own connection, a per-query timeout, and a fixed set
of **named, parameterized traversals** — *not* an open Cypher endpoint (never
pass user text into Cypher unparameterized). Returns plain dicts, same as
`Analytics`.

```python
"""Named graph traversals over Neo4j. Additive; constructed only when
Settings.graph_enabled is true. Mirrors analytics.Analytics: one connection,
bounded time, structured dicts, and NO open query surface.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    from neo4j import GraphDatabase          # Apache-2.0 driver
except ImportError:                          # optional dep; feature stays off
    GraphDatabase = None                     # type: ignore

log = logging.getLogger("rag.graph")


class Graph:
    def __init__(self, uri: str, user: str, password: str,
                 database: str = "neo4j", timeout_s: float = 10.0) -> None:
        if GraphDatabase is None:
            raise RuntimeError("neo4j driver not installed but GRAPH_ENABLED=true")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database
        self._timeout = timeout_s

    def close(self) -> None:
        self._driver.close()

    def _read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        # execute_query with routing_="r" is the modern (5.x) read path; the
        # transaction timeout bounds a runaway traversal, like statement_timeout.
        records, _, _ = self._driver.execute_query(
            cypher, params, database_=self._db, routing_="r",
        )
        return [r.data() for r in records]

    # §3.1 — performer → repertoire → scriptures in the same sittings
    def performer_repertoire_scriptures(self, performer: str,
                                        place: str | None = None) -> list[dict[str, Any]]:
        cypher = """
        MATCH (p:Person {canonical: $performer})<-[:PERFORMED_BY]-(t:Track)
              <-[:HAS_TRACK]-(s:Sitting)
        WHERE $place IS NULL OR (s)-[:AT]->(:Place {canonical: $place})
        OPTIONAL MATCH (s)-[:HAS_TRACK]->(:Track)-[:ALIGNED_TO]->(f:File)
                       -[:REFERENCES]->(sc:Scripture)
        RETURN t.track_title AS bhajan, s.session_date AS date,
               collect(DISTINCT sc.canonical) AS scriptures
        ORDER BY date
        """
        return self._read(cypher, performer=performer, place=place)

    # §3.2 — performer collaboration network, by season
    def collaborators(self, performer: str) -> list[dict[str, Any]]:
        cypher = """
        MATCH (p:Person {canonical: $performer})<-[:PERFORMED_BY]-(t:Track)
              -[:PERFORMED_BY]->(other:Person)
        WHERE other <> p
        MATCH (t)<-[:HAS_TRACK]-(s:Sitting)
        RETURN other.canonical AS collaborator, s.season AS season,
               count(*) AS shared_tracks
        ORDER BY shared_tracks DESC
        """
        return self._read(cypher, performer=performer)
```

Wire it up exactly like the main doc §6.3:

- Construct once at startup **only if** `settings.graph_enabled`, in
  [`rag_api/app.py`](../rag_api/app.py) (same place `Analytics` is built).
- Expose a few dashboard routes (`/graph/collaborators`, …).
- Expose an **Open WebUI function tool** with a docstring so the chat model knows
  when to call it (CLAUDE.md convention), e.g.
  `related_sittings(topic: str, performer: str)`.

---

## 8. Entity resolution (Neo4j landing)

The plan is identical to the main doc §5 — this only says *where the result
lands in Neo4j*:

1. Seed **canonical `:Person`/`:Work`** nodes from clean `catalog_*.performers[]`
   / `track_title`.
2. Deterministic normalization (strip `ji`/`जी`, honorifics) + `pg_trgm` fuzzy
   match happen **in Postgres** (that's where the text and `pg_trgm` live),
   producing an `entity_alias(surface_form, canonical, method, confidence)`
   table.
3. The builder (§6) reads that table and writes surface-form nodes linked by
   `(:Person {surface})-[:ALIAS_OF]->(:Person {canonical})`, **or** simply maps
   every mention straight to the canonical node and keeps the alias list as a
   property. Prefer the latter for query simplicity; keep `:ALIAS_OF` only if you
   want alias provenance walkable.
4. LLM-assisted merges for the tail (local Qwen 2.5 7B) go to a **review queue**,
   never auto-applied (CLAUDE.md rule 6). Unresolved mentions are logged and
   parked, not dropped.

Do resolution **in Postgres, before the Neo4j sync.** Neo4j receives already-
canonical identities — it is a projection target, not where you clean data.

---

## 9. GraphRAG augmentation + global-thematic questions

**Local augmentation** (the retrieval payoff), unchanged from main doc §6.4:
after Qdrant+Tantivy+reranker pick the top chunks, map chunk→file→entities,
traverse 1–2 hops in Neo4j for *related* sittings/scriptures, and feed a compact
"related context" block into [`rag_api/synthesis.py`](../rag_api/synthesis.py)
+ the source cards. Retrieval ranking is **not** altered — this is enrichment.

**Global-thematic** ("major themes across the 34-year corpus"): needs community
detection.

- **Preferred (license-clean):** compute communities **offline in Python** with
  OSI-licensed `networkx` + `python-louvain` over an exported edge list, write a
  `community_id` property back onto nodes, precompute per-community summaries
  with the local model. No GDS, no license question.
- **Only if GDS is license-approved:** `gds.louvain` / `gds.pageRank` in-DB.
  Gated on the §1 sign-off — do not reach for it by default.

---

## 10. Ops notes (strictly-local box)

- **Backup:** none required for correctness — Neo4j is a rebuildable projection.
  For convenience, `neo4j-admin database dump` to `./data/neo4j/backups`.
- **Startup order:** the `depends_on` + healthcheck make rag-api wait for Bolt.
  With `GRAPH_ENABLED=false`, rag-api never dials Neo4j even if it's slow/absent.
- **Failure isolation:** if Neo4j is down while the flag is on, the graph query
  layer must **degrade, not crash** — a traversal timeout/connection error is
  logged (rule 6) and the answer falls back to plain retrieval. Never let a
  graph outage take down the main RAG path.
- **Resource contention:** Neo4j is CPU/RAM only; it does not touch the GPU that
  Ollama + the reranker share. Memory floor ~1–2 GB heap + 1 GB pagecache (§3).

---

## 11. Acceptance criteria (when Phase 15 opens with Neo4j)

Per CLAUDE.md, done only when all are demonstrably true:

- [ ] `neo4j` service comes up healthy; Browser reachable at `:7474`, Bolt at
      `:7687`; auth from `.env`.
- [ ] `build_graph_neo4j.py` runs idempotently — a second sync reports zero
      created nodes/edges.
- [ ] Counts reconcile: `:Track`, `:Sitting`, and `:ALIGNED_TO` edge counts match
      the corresponding `catalog_*` / alignment rows in Postgres.
- [ ] `entity_alias` resolves a documented sample of `people_named` variants to
      the right canonical `:Person`; unresolved mentions are logged, not lost.
- [ ] At least the six §3 (main doc) queries return correct results through
      `rag_api/graph.py`.
- [ ] With `GRAPH_ENABLED=false`, the retrieval path and its outputs are
      **identical** to pre-Phase-15 (prove the flag is truly inert).
- [ ] With `GRAPH_ENABLED=true`, killing the Neo4j container degrades gracefully
      — the RAG answer still returns, the graph block is just absent + logged.
- [ ] Licenses verified against PRD §14: Neo4j Community GPLv3 ✅, `neo4j` driver
      Apache-2.0 ✅, GDS **not used** (or separately signed off).

---

## 12. Trade-offs vs Apache AGE (honest summary)

| | Apache AGE (main doc §4) | **Neo4j Community (this doc)** |
|---|---|---|
| New service / ports | None — inside Postgres | +1 JVM service, +2 ports |
| Query language | openCypher | Cypher (richer, mature planner) |
| Tooling / visual browser | Minimal | Excellent (Neo4j Browser, Bloom) |
| Sync job | None — same DB | One-way Postgres→Neo4j builder |
| Memory floor | ~0 (shares Postgres) | ~2–3 GB |
| License | Apache-2.0 | GPLv3 (still ✅ OSI) |
| Best when | Lowest friction, shallow-to-mid traversals | Deep traversals, want first-class graph tooling |

**Recommendation stands:** start with AGE / plain-SQL (main doc) unless the
tooling or deep-traversal argument above is compelling. This doc exists so that
if the team *does* choose Neo4j, the integration is fully specified and stays
additive, license-clean, and flag-gated.

---

### See also

- [`docs/knowledge_graph.md`](knowledge_graph.md) — why a graph, latent model, entity resolution, engine comparison (**read first**)
- [`infra/postgres/analytics_schema.sql`](../infra/postgres/analytics_schema.sql) — source of the node/edge model
- [`rag_api/analytics.py`](../rag_api/analytics.py) — the query-layer pattern `rag_api/graph.py` mirrors
- [`docker-compose.yml`](../docker-compose.yml) — where the `neo4j` service slots in
- [`docs/catalog_enrichment.md`](catalog_enrichment.md) — the join-key spine the graph reuses
