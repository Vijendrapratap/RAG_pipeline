# Knowledge Graph — design & implementation plan (proposed Phase 15)

> **Status: PROPOSAL, not yet built.** This is a design captured from a planning
> discussion. It is **additive** — it sits *beside* Qdrant, never replaces it.
> Nothing here is implemented until we explicitly open a Phase 15. The vector
> store is a PRD §3 locked decision; this document does not change it.

> **Phase 16 (the archive map) does not open this phase.** Every edge that map
> draws is *containment* — track inside sitting inside camp — derived from
> `file_meta.source_file` and nothing else. It needs no entity resolution, no new
> store, and no graph traversal. Should the map ever grow genuine cross-links
> (tracks sharing a performer, songs recurring across camps), that is this
> document's work and requires opening Phase 15 first. Measure before designing:
> `catalog_track.matched_source_file` is populated for **0 of 22,501 rows** as of
> 2026-07-08 — the catalog has never been aligned to a transcript, so such an
> edge would today connect nothing to nothing.

---

## TL;DR

- **You already have a knowledge graph.** Its nodes and edges live in Postgres
  today — as columns, arrays, and join keys across `catalog_sitting`,
  `catalog_track`, `chunk_meta`, and `file_meta`. A KG project does not *invent*
  a graph; it makes the relationships you already store **traversable**.
- **It does not replace Qdrant.** Semantic recall ("passages about surrender")
  is an embedding job. A graph answers a *different* class of question:
  multi-hop relationships, entity resolution, and global/thematic sense-making.
- **Cheapest honest on-ramp:** build the graph *inside your existing Postgres*
  with **Apache AGE** (Apache-2.0) — no new service, stays strictly local. Plain
  recursive SQL is an even-lower-dependency fallback for shallow traversals.
- **The hard part is entity resolution**, not storage: unifying messy,
  multilingual name mentions ("Guruji" / "गुरुजी" / "Guru ji") into one
  canonical node. That is a discrete, testable sub-project and where most of the
  value — and effort — lives.

---

## 1. Why a graph, and why *not* the alternatives

Recap of the options weighed:

| Option | Verdict |
|---|---|
| Keep Qdrant + Tantivy + reranker | ✅ Right tool for semantic recall at ~5 TB scale |
| **Obsidian** instead of Qdrant | ❌ Wrong category — a human note app: no semantic search, won't scale to millions of chunks, no retrieval API |
| **Knowledge graph** instead of Qdrant | ❌ Graphs match on *structure*, not *meaning* — they cannot do semantic retrieval |
| **Knowledge graph *beside* Qdrant** | 🟡 Worthwhile **if** there is real demand for multi-hop / entity / global-thematic questions |

The rule of thumb: a graph pays off for **traversal, entity resolution, and
global sense-making** — *not* for the single-field filters (season, location,
`event_id`) that [`rag_api/query_parse.py`](../rag_api/query_parse.py) + Postgres
already handle well. **Do not build a graph to do what the filter path already
does.**

---

## 2. The graph is already latent in your data

Everything below is derived directly from
[`infra/postgres/analytics_schema.sql`](../infra/postgres/analytics_schema.sql).

### Nodes you already have

| Node type | Source column(s) | Notes |
|---|---|---|
| **Person** | `catalog_sitting.performers[]`, `catalog_track.performers[]` (clean), `file_meta.people_named[]` (LLM-extracted, messy), `chunk_meta.speakers[]` (diarization) | Three provenance tiers — needs resolution (see §5) |
| **Place** | `location`, `camp_place`, `venue`, `file_meta.places_named[]` | `location` is a normalized city facet; `camp_place` is raw-for-display |
| **Work / Song** | `catalog_track.track_title` (canonical Devanagari song/bhajan title) | The catalog is authoritative here |
| **Scripture** | `file_meta.scriptures_referenced[]` | LLM-extracted |
| **Topic** | `file_meta.topics[]` | LLM-extracted |
| **Sitting / Event** | `catalog_sitting.sitting_key`, `chunk_meta.event_id`, `session_date`, `season`, `camp_year` | A sitting is one gathering |
| **Track** | `catalog_track.join_key`, `track_no`, `track_type` | One performance within a sitting |
| **File (transcript)** | `file_meta.source_file` | One recording |
| **Chunk** | `chunk_meta.chunk_id` | One retrievable passage |

### Edges you already have (as FKs / arrays / join keys)

```
Sitting ──has_track──▶ Track          catalog_track.sitting_key → catalog_sitting.sitting_key
Track   ──performed_by▶ Person        catalog_track.performers[]
Track   ──aligned_to──▶ File          catalog_track.matched_source_file → file_meta.source_file (set by backfill)
File    ──has_chunk───▶ Chunk         chunk_meta.source_file → file_meta.source_file
Chunk   ──spoken_by───▶ Person        chunk_meta.speakers[]
File    ──mentions────▶ Person        file_meta.people_named[]
File    ──mentions────▶ Place         file_meta.places_named[]
File    ──references──▶ Scripture     file_meta.scriptures_referenced[]
File    ──about───────▶ Topic         file_meta.topics[]
Sitting ──at──────────▶ Place         catalog_sitting.location / venue / camp_place
Sitting ──in_season──▶ Season         catalog_sitting.season
Sitting ──in_year────▶ Year           catalog_sitting.camp_year
```

The graph is *implicit*. A KG just makes it walkable. The **join key**
(`'YYYY-MM-DD|SEQ|TRACKNO'`, and `'YYYY-MM-DD|SEQ|'` for a sitting) is the spine
that already ties the catalog to transcripts — see
[`docs/catalog_enrichment.md`](catalog_enrichment.md).

### The picture

```
        (Season) (Year)                         (Topic)  (Scripture)
            ▲       ▲                                ▲        ▲
            │in     │in                        about │        │references
            └───(Sitting)──has_track──▶(Track)──aligned_to──▶(File)──has_chunk──▶(Chunk)
                   │ at        │performed_by      ▲              │mentions          │spoken_by
                   ▼           ▼                  │              ▼                  ▼
                (Place)     (Person)◀─────────────┴──────────(Person)          (Person)
                                        performed / mentioned / spoke
```

---

## 3. What a graph unlocks (that vectors and flat SQL don't)

Concrete questions, using your real entities. Vector search cannot express any
of these (they are about *relationships*, not passage meaning); flat SQL can do
1–2 hops but gets ugly at 3+ with array `UNNEST`.

1. **Cross-camp performer → repertoire → scripture**
   *"Which bhajans did performer X sing across all Noida camps, and which
   scriptures were referenced in those same sittings?"*
   `Person ─performed▶ Track ─in▶ Sitting ─references▶ Scripture`

2. **Collaboration network**
   *"Which performers appear together most often, and in which seasons?"*
   `Person ◀performed─ Track ─performed▶ Person`, grouped by `season`

3. **Thematic recurrence over time**
   *"Where does topic A recur with different performers over the 34-year span?"*
   `Topic ◀about─ File ─aligned◀ Track ─performed▶ Person`, ordered by `session_date`

4. **Scripture citation map**
   *"What discourses connect scripture B and topic A?"*
   `Scripture ◀references─ File ─about▶ Topic`

5. **Provenance expansion for a single answer** (ties to the source-card work in
   [`doc.md`](../doc.md) → 2026-07-02): given a retrieved chunk, surface *"this
   passage is from a sitting where X sang; 4 related sittings reference the same
   scripture"* — connective grounding under the answer.

6. **Global / thematic summaries** (GraphRAG communities): *"What are the major
   themes across the whole corpus and how do they relate?"* — a question vector
   RAG structurally cannot answer, because it only retrieves *local* passages.

---

## 4. What to use — technology choice

Hard constraint (PRD §14, CLAUDE.md rule 2): **open-source only, license
verified.**

| Engine | License | Fit | Verdict |
|---|---|---|---|
| **Apache AGE** (Postgres extension, openCypher) | Apache-2.0 ✅ | Graph lives *inside* your existing Postgres container — no new service, no new port, stays strictly local | **Recommended** |
| Plain Postgres (recursive CTEs + a materialized `graph_edge` table) | — (already in stack) | Zero new dependency; fine for shallow (≤2–3 hop) traversals; verbose for deep ones | **Fallback / v0** |
| Neo4j **Community** | GPLv3 ✅ (copyleft) | Mature, great tooling & visual browser, but a separate JVM service + port | Viable if traversal outgrows AGE — full integration spec in [`knowledge_graph_neo4j.md`](knowledge_graph_neo4j.md) |
| Memgraph Community | BSL 1.1 ⚠️ *(source-available, not OSI open-source)* | — | **Do not adopt without an explicit license sign-off** — likely disqualified by rule 2 |

**Recommendation: Apache AGE.** It keeps the graph in Postgres (one fewer moving
part in a strictly-local stack), speaks openCypher, and is cleanly licensed.

> **Caveat to verify first:** AGE tracks specific Postgres major versions. Check
> that a released AGE build matches the Postgres version in
> [`docker-compose.yml`](../docker-compose.yml) before committing — adopting AGE
> means switching the Postgres image to an AGE-enabled one (e.g. the `apache/age`
> image) or building the extension into a custom image. If that friction isn't
> worth it yet, ship the **plain-Postgres `graph_edge` table** first (§6.1) and
> upgrade to AGE later — the node/edge model is identical.

---

## 5. The hard part — entity resolution

Storage is easy; **canonicalizing entities is the real work**, and it's what
makes the graph trustworthy. `people_named[]` / `places_named[]` are free-text
LLM extractions across Hindi and English with spelling and honorific variants.
Plan, cheapest-first:

1. **Seed canonical nodes from clean data.** `catalog_*.performers[]` and
   `track_title` are human-curated — treat them as the authoritative canonical
   set. Build canonical Person / Work nodes from these first.
2. **Deterministic normalization.** Lowercase; strip honorifics (`ji` / `जी`,
   `guru` prefixes); fold Devanagari↔romanized where a transliteration is
   confident. This collapses the easy variants.
3. **Fuzzy match to the seed set.** Use Postgres `pg_trgm` (trigram similarity)
   / Levenshtein to link a messy `people_named` surface form to an existing
   canonical node above a similarity threshold. Keep an **alias table**
   (`surface_form → canonical_id`) so every mention stays auditable.
4. **LLM-assisted clustering for the tail** (optional, local). Reuse the Phase-13
   approach — Qwen 2.5 7B *locally* — to *propose* merges for ambiguous
   leftovers, written to a review queue, **never auto-applied**. Human confirms.
5. **Never silently drop.** Per CLAUDE.md rule 6, unresolved mentions are logged
   and parked as `unresolved`, not discarded — they surface for later linking.

Entity resolution is independently valuable and can ship before any fancy query
layer: even just canonical Person nodes + an alias table improves analytics and
the source-card provenance immediately.

---

## 6. How to implement — hand-in-hand with the current stack

Sequenced so each step is shippable and testable on its own. Every step is
**additive**: it reads existing tables and writes new ones; the running
retrieval path (`chunk_meta`, `file_meta`, Qdrant, Tantivy) is untouched, exactly
like the catalog tables are "standalone" today.

### 6.1 — Materialize the graph (builder script)

New module `ingestion/graph/build_graph.py`:

- Reads `catalog_sitting`, `catalog_track`, `chunk_meta`, `file_meta`.
- Emits two tables (plain Postgres to start — no AGE dependency yet):

  ```sql
  CREATE TABLE IF NOT EXISTS graph_node (
      node_id     BIGSERIAL PRIMARY KEY,
      node_type   TEXT NOT NULL,        -- person|place|work|scripture|topic|sitting|track|file|chunk
      canonical   TEXT NOT NULL,        -- canonical label
      attrs       JSONB DEFAULT '{}',   -- session_date, season, etc.
      UNIQUE (node_type, canonical)
  );
  CREATE TABLE IF NOT EXISTS graph_edge (
      src_id      BIGINT REFERENCES graph_node(node_id),
      dst_id      BIGINT REFERENCES graph_node(node_id),
      edge_type   TEXT NOT NULL,        -- has_track|performed_by|references|about|mentions|...
      attrs       JSONB DEFAULT '{}',
      PRIMARY KEY (src_id, dst_id, edge_type)
  );
  CREATE INDEX IF NOT EXISTS idx_gedge_src ON graph_edge(src_id, edge_type);
  CREATE INDEX IF NOT EXISTS idx_gedge_dst ON graph_edge(dst_id, edge_type);
  ```

- **Idempotent and resumable** (upserts on the unique keys) — same discipline as
  the ingestion pipeline. Rebuildable from scratch at any time; source of truth
  stays the Postgres base tables.

### 6.2 — Entity resolution (§5)

Populate an `entity_alias(surface_form, node_id, method, confidence)` table and
point messy mentions at canonical `graph_node` rows. Ship deterministic +
trigram first; LLM-assisted review queue later.

### 6.3 — Graph query layer

New module `rag_api/graph.py`, mirroring the shape of
[`rag_api/analytics.py`](../rag_api/analytics.py) (own connection per call,
`statement_timeout`, structured-dict returns). A handful of **parameterized,
named traversals** — *not* an open Cypher endpoint — one per question type in
§3. If AGE is adopted, these become openCypher; on plain Postgres they're
recursive CTEs. Same function signatures either way, so the engine swap is
invisible to callers.

Expose them:
- as dashboard API routes in [`rag_api/app.py`](../rag_api/app.py), and
- as an **Open WebUI function tool** (docstring'd so the chat model knows when to
  call it — CLAUDE.md convention), e.g. `related_sittings(topic, performer)`.

### 6.4 — GraphRAG augmentation (the payoff for retrieval)

Wire the graph into the existing hybrid flow *as an enrichment*, not a
replacement:

1. Qdrant + Tantivy + reranker retrieve top chunks (unchanged).
2. Map each chunk → file → its resolved entities (person / scripture / topic /
   sitting).
3. Traverse the graph one or two hops to find *related* sittings/entities.
4. Feed a compact **"related context"** block into
   [`rag_api/synthesis.py`](../rag_api/synthesis.py), and surface those links in
   the source-card provenance (builds on the 2026-07-02 source-card work).

For **global-thematic** questions, precompute community summaries offline and
answer from those.

### 6.5 — Visualization / navigation (optional, last)

Only after the data is trustworthy: a read-only graph view in the dashboard
(performer-collaboration or scripture-citation network). This — *not* the
corpus store — is the place an Obsidian-style graph *view* legitimately fits:
visualizing the KG, never storing the 5 TB.

### Where it plugs into the architecture

```
ingestion/graph/build_graph.py ──writes──▶ graph_node / graph_edge / entity_alias  (new Postgres tables)
                                                     │
rag_api/graph.py ──reads──▶ named traversals ◀───────┘
        │
        ├─▶ rag_api/app.py            (dashboard API routes)
        ├─▶ Open WebUI function tool  (chat-callable)
        └─▶ rag_api/synthesis.py      (GraphRAG "related context" block)
```

Qdrant, Tantivy, the reranker, and the base `chunk_meta`/`file_meta` tables are
**not modified** by any of this.

---

## 7. Cost, effort, and risk

| Item | Assessment |
|---|---|
| **Build (§6.1)** | Low — deterministic SQL over tables you already have |
| **Entity resolution (§5, §6.2)** | **Medium–High** — the real cost; multilingual, needs review tooling |
| **Query layer (§6.3)** | Low–Medium — a fixed set of named traversals |
| **GraphRAG augmentation (§6.4)** | Medium — touches synthesis + source cards |
| **New infra risk** | Low if AGE-in-Postgres or plain SQL; Medium if a separate Neo4j service |
| **Biggest risk** | A graph that *looks* authoritative but is built on unresolved/duplicated entities. Mitigate: ship resolution first, keep aliases auditable, never auto-merge. |

**When it's over-engineering:** if users mostly ask "find me passages about X"
and "what did they say on date Y," you already have the right stack. Build the
graph only when multi-hop / connective / global-thematic questions are a real,
recurring need.

---

## 8. Acceptance criteria (when Phase 15 is opened)

Per CLAUDE.md, a phase is done only when these are demonstrably true:

- [ ] `graph_node` / `graph_edge` build idempotently from the base tables; a
      re-run changes nothing.
- [ ] Node/edge counts reconcile against source rows (e.g. every
      `catalog_track` with a `matched_source_file` yields one `aligned_to` edge).
- [ ] `entity_alias` resolves a documented sample of `people_named` variants to
      the correct canonical performer; unresolved mentions are logged, not lost.
- [ ] At least the six §3 queries return correct results via `rag_api/graph.py`.
- [ ] The GraphRAG "related context" block appears in synthesis for a sample
      query **without** altering Qdrant data or the base retrieval ranking.
- [ ] Whichever engine is chosen has its license verified against PRD §14.

---

## 9. Decision required before any code

1. **Is there real demand** for multi-hop / entity / global-thematic questions?
   (If not, defer — the current stack is correct.)
2. **Engine:** Apache AGE (recommended) vs plain-Postgres-first vs Neo4j
   Community — pin the Postgres-version compatibility for AGE.
3. **Scope of v1:** resolution + query layer only, or include GraphRAG
   augmentation?

This is net-new scope beyond the current phases and the vector store is PRD
§3-locked — so nothing here proceeds without an explicit "yes, open Phase 15."

---

### See also

- [`docs/architecture.md`](architecture.md) — how the current retrieval stack fits together
- [`docs/catalog_enrichment.md`](catalog_enrichment.md) — the join-key spine this graph reuses
- [`infra/postgres/analytics_schema.sql`](../infra/postgres/analytics_schema.sql) — the source of the node/edge model
- [`doc.md`](../doc.md) → 2026-07-02 — the source-card provenance work this augments
