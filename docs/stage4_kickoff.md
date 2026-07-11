# Stage 4 Kickoff — Advanced Query-Side Technique (detailed, code-grounded)

> Companion to the canonical spec **`docs/plan_stage4_advanced_retrieval.md`**. That file is the
> *what/why*; this file is the *where/how* — verified against the code as of commit `9a71183`
> (Foundation Stage 0→3 complete + Stage 2 dedupe backfill done). Start the new section here.

---

## 0. Prerequisite status — MET

Stage 4's opening line: *"the entire foundation (Stage 0→3) must be done and its baseline
recorded."* As of this session:

- Stage 2 fully closed (enrichment 2.0/2.1/2.3, summary index 2.4, **dedupe backfill** `9a71183`).
- Stage 3 fully closed: eval harness (`eval/run_eval.py`), corpus golden set
  (`eval/golden_queries_corpus.yaml`), baseline + grid-search (`eval/results/baseline_stage3.md`).

**Stage 4 is unblocked.** Nothing here ships without beating that recorded baseline.

---

## 1. The bar to beat (recorded baseline — `eval/results/baseline_stage3.md`)

| metric | value | note |
|---|---|---|
| quote Hit@5 (doc2query verbatim) | **100%** | CI gate; do not regress |
| entity topic Hit@10 (match=any) | **62.5%** | the cross-script gap — 3/8 entities never surface |
| entity topic MRR | **0.487** | best config = per-type bm25, catalog **off** |
| abstention @ cite_min_score=0.03 | prec **100%** / recall **40%** / false-abstain **0%** | Stream C deferred |
| retrieval latency (hybrid) | **~319 ms** median | `bench_backends.py`, retrieval only |
| full `/api/query` turn | **~3.7–4.2 s** p50 observed | retrieval + **synthesis** (synthesis dominates) |

Two facts that shape every Stage 4 decision:

1. **`include_catalog=on` regresses** topic MRR (0.487 → 0.328). The live request default is
   `include_catalog=True` ([app.py:121,136](rag_api/app.py#L121)) — the router should turn it **off**
   except where a class wants it. This is free accuracy the baseline already proved.
2. The **2 s DoD budget is the whole turn**, not retrieval. Synthesis already eats most of it, so any
   *query-time* LLM call added here (router fallback, HyDE, follow-up rewrite) is spending against an
   already-thin margin. Measure p50 per class, per item — never assume.

---

## 2. Governing rules (do not skip)

- **Flag-gated, default off.** Each of 4.1–4.4 ships behind its own flag; the default pipeline is
  unchanged until an A/B proves a win.
- **A/B against the Stage 3 baseline** on `eval/golden_queries_corpus.yaml`. Ships only if it improves
  Hit@5 or abstention P/R **without** breaching the 2 s p50. Anything that ties or regresses is
  **reverted, not merged "to be safe."**
- **Deterministic first, LLM only when ambiguous.** Every LLM round-trip is latency you must justify.
- **`"think": false` on every structured LLM call.** `chat_model=qwen3.5:9b` ([config.py:106](rag_api/config.py#L106))
  is a thinking model; `format=json`/structured output + thinking ⇒ empty/garbage `response`. Same
  gotcha that dead-lettered enrichment. **See §7 — the existing HyDE call is already missing this.**
- **PRD §10 pre-authorises** routing, HyDE, step-back, follow-up rewrite, and eval-driven tuning.
  No PRD amendment needed for 4.1–4.4. (The Deferred list in §8 *does* need sign-off.)
- **One phase at a time, one commit per shipped item** (project rule). Do not bundle.

---

## 3. Code map — the exact seams Stage 4 plugs into

| seam | location | role in Stage 4 |
|---|---|---|
| `_prepare(req)` | [app.py:353](rag_api/app.py#L353) | **Router installs here.** Already does quote auto-detect + filter extraction + HyDE. Returns `(effective_filters, detections, dense_text, find_quote, query)`. 4.1 adds class inference + per-class settings to this return. |
| `_retrieve(...)` | [app.py:320](rag_api/app.py#L320) | Dispatches to pipeline by `find_quote`/`backend`/`scope`. The router feeds it the per-class `bm25_weight`, `include_catalog`, `dense_text`. |
| `Retriever.search(query, filters, top_k, bm25_weight, dense_text, include_catalog)` | [retrieval.py:662](rag_api/retrieval.py#L662) | **Per-class knobs already plumbed.** `bm25_weight` overrides the blend; `dense_text` is the HyDE hook (dense arm only — BM25 always uses raw query, [retrieval.py:675](rag_api/retrieval.py#L675)). |
| `detect_quote(query)` | [query_parse.py:191](rag_api/query_parse.py#L191) | Deterministic quote classifier — **reuse as-is** for the `quote` class. Already wired in `_prepare`. |
| `detect_signals(query, vocab)` | [query_parse.py:78](rag_api/query_parse.py#L78) | Filter/vocab extractor. **Caution:** its `performers` vocab matches catalog *singer* names (Latin), **not discourse subjects** — do **not** reuse it for `name`-class entity detection. Build a separate rule. |
| `QueryExpander.hypothetical(query)` / `retriever.make_expansion(query)` | [expand.py:63](rag_api/expand.py#L63) | HyDE, built. Opt-in via `expand_query`. 4.2 gates it to `thematic`. |
| per-class weights | [config.py:249-250](rag_api/config.py#L249) | `bm25_weight=0.65`, `quote_bm25_weight` today. Live API only distinguishes quote vs default; `BM25_WEIGHT_BY_TYPE` (`eval/run_eval.py`) is eval-only. 4.1 brings the full mapping into the API. |
| candidate budget / floors | [config.py:247,138,145](rag_api/config.py#L247) | `candidates_per_source=40`, `cite_min_score=0.03`, `allow_abstain` (default off). |
| conversation log | `infra/postgres/migrations/002_conversations.sql`, `rag_api/history.py` | Stores `question`+`citations`; **nothing reads it back yet** — 4.3 is its first consumer. |

---

## 4. Item 4.1 — Query router (build first; largest lever)

**Goal.** Classify each query into `name` / `quote` / `thematic` / `analytic` / `followup`, then route
to per-class retrieval settings instead of one-size-fits-all.

### 4.1.a — prerequisite: a labeled router eval set
The golden set is typed `quote/topic/negative/analytics` — that is **not** the router's label space.
Before you can measure router accuracy you need `eval/router_queries.yaml`: `{query, expected_class}`
across all five classes (seed from `golden_queries_corpus.yaml` + `conversation_history`). Without it,
4.1's "class-accuracy" acceptance is unmeasurable. **This is the first commit-sized task.**

### 4.1.b — new `rag_api/route.py`
Pure, deterministic core + optional LLM fallback:

```
classify(query, history_present) -> QueryClass         # deterministic; no I/O
  quote     : detect_quote(query).is_quote  (reuse query_parse:191)
  analytic  : regex — ^(how many|count|list all|kitne|kitni|which .* mention)
  name      : a person-name shape (see caution below) — NOT detect_signals performers
  followup  : history_present AND anaphora/short-context shape ("what about his ...", "uske baad")
  thematic  : default fallthrough
```

- **LLM fallback only when deterministic is ambiguous** — one `think:false` classify call, measured
  against latency; **skip entirely when a deterministic rule fires** (that is the common path, ~0 ms).
- `name` detection caution: discourse subjects are Devanagari entities (पुरन सिंह, विवेकानंद). A cheap
  starting rule: a short query dominated by a Devanagari proper-noun span with no thematic verb. Do
  **not** lean on `detect_signals` here (performers ≠ subjects).

### 4.1.c — per-class settings table (STARTING values — tune by A/B, do not treat as final)

| class | bm25_weight | include_catalog | HyDE | candidates | route to |
|---|---|---|---|---|---|
| `quote` | `quote_bm25_weight` (0.85) | off | no | default | `find_quote` path |
| `name` | ~0.75 (lexical) | off | no | ↑ (surface sparse entities) | `search` |
| `thematic` | ~0.5–0.65 | off | **yes (4.2)** | default | `search` |
| `analytic` | n/a | off | no | n/a | analytics tools (count/list/speakers) |
| `followup` | inherit underlying class after rewrite (4.3) | — | — | — | rewrite → reclassify |

### 4.1.d — wiring
`_prepare` ([app.py:353](rag_api/app.py#L353)) calls `route.classify(...)`, sets `bm25_weight` /
`include_catalog` / `expand_query` from the class table, and passes them through `_retrieve` →
`Retriever.search`. Gate the whole thing behind `RAG_ROUTER=off` default.

**Ship criterion.** Router class-accuracy reported on `router_queries.yaml`; per-class Hit@5 ≥ baseline
on ≥2 classes with no regression on the others; deterministic path adds ~0 ms, LLM fallback < ~150 ms p50.

**Commits (two):**
```
feat(rag): deterministic query router with per-class retrieval settings
feat(rag): LLM router fallback for ambiguous queries (think=false, latency-gated)
```
Ship the fallback only if it beats the deterministic-only router.

---

## 5. Item 4.2 — HyDE gated to `thematic`

**Goal.** Turn on `expand_query` **only** for the `thematic` class (per the router). HyDE writes a
hypothetical corpus-like passage and embeds *that* for the dense arm — helps vague conceptual queries,
**hurts** precise quote/name lookups (noisy vector + a full LLM round-trip). Keep the dense-only
asymmetry ([retrieval.py:675](rag_api/retrieval.py#L675)): BM25 stays on the raw query.

- Built already in [expand.py](rag_api/expand.py). **Fix `think:false` first (§7).**
- Ship criterion: thematic Hit@5 > baseline; thematic **p50 < 2 s** (this is the item most likely to
  breach latency — HyDE adds a generation round-trip on top of synthesis).

```
feat(rag): HyDE expansion gated to thematic queries
```

---

## 6. Items 4.3 / 4.4 — follow-up rewrite, step-back

### 4.3 Follow-up rewriting — `QueryRequest` at [app.py:124](rag_api/app.py#L124)
Add optional `history: list[dict]` to the request. When present, one `think:false` call rewrites a
context-dependent follow-up ("what about his early life?" / "uske baad kya hua") into a standalone query
**before** retrieval. First consumer of `conversation_history`. Round-trip only when `history` is
non-empty.
Ship criterion: standalone-rewrite Hit@5 beats raw-follow-up Hit@5 on a follow-up subset of the golden set.
```
feat(rag): conversational follow-up rewriting from history
```

### 4.4 Step-back prompting — last, maybe
For an over-narrow thematic query, generate a more general "step-back" question, retrieve for both, merge.
Highest latency cost. Gate to thematic queries the router flags as narrow, and **only if 4.2 didn't
already satisfy them**. If it only ties 4.2, do not ship the extra round-trip.
```
feat(rag): step-back prompting for narrow thematic queries
```

---

## 7. Known gotcha to fix before 4.2 — HyDE lacks `think:false`

[expand.py:74-92](rag_api/expand.py#L74) posts to `/api/chat` with `options`
(`temperature`/`num_ctx`/`num_predict`) but **no `"think": false`**. With `chat_model=qwen3.5:9b`
(a thinking model) the hypothetical passage may come back polluted with reasoning tokens, degrading the
embedding it is supposed to sharpen. `rag_api/pageindex.py` and the enrichment path already carry this
fix. **Add `"think": false` to the HyDE options as part of 4.2** (or a small precursor commit), and add
a unit test asserting the request body carries it.

---

## 8. Measurement harness (extend, don't rebuild)

- `eval/run_eval.py` retrieves via `Retriever.search` directly — it does **not** exercise the router or
  the `/api/query` LLM calls, and it does **not** measure turn latency. For Stage 4 you need either
  (a) import `route.py` into the eval to simulate class routing, or (b) A/B through the live API and
  read p50 from there. Record the full grid: **class × technique × Hit@5 × abstention P/R × p50** in
  `eval/results/` (`.md` committed, timestamped JSON gitignored — same convention as Stage 3).
- Reuse `eval/bench_backends.py` for latency; it already reports median/mean/recall@k.

---

## 9. Deferred — explicit sign-off, NOT Stage 4

- **Latin→Devanagari transliteration before BM25** — flagged as the single highest-impact *recall*
  lever (bigger than all of Stage 4), but needs a conjunct-safe tokenizer + full **Tantivy reindex**
  and header exclusion. Separate sign-off.
- **bge-m3 sparse + ColBERT heads** — needs serving bge-m3 off Ollama (PRD §3 locked) + reindex.
- **EN→HI query translation** — docs contradict; let Stage 0.2 histogram + 3.3 data decide, not a guess.
- **Phase 15 knowledge graph** — hard-blocked until enrichment populated `people_named` (now done),
  but still a separate plan + sign-off (`docs/knowledge_graph_neo4j.md`).
- **Reranker LoRA/fine-tuning** — PRD-banned (`PRD.md:55`); needs a PRD amendment.

---

## 10. First concrete step to open the section

1. Read `docs/plan_stage4_advanced_retrieval.md` (the canonical spec) + this file.
2. Build **4.1.a** — `eval/router_queries.yaml` (labeled `{query, expected_class}`, all five classes).
   This is the measurement ground truth everything else A/Bs against; commit it first.
3. Build **4.1.b** — `rag_api/route.py` deterministic core, behind `RAG_ROUTER=off`; wire into
   `_prepare`; A/B vs baseline. Commit `feat(rag): deterministic query router …`.
4. Only then decide on the LLM fallback, then 4.2 (fix `think:false` first), then 4.3, then maybe 4.4.

> Note: this doc + `plan_stage4_advanced_retrieval.md` are currently **untracked** (part of the
> unclassified doc swath). Commit them when you open the section.
