# Implementation Plan — Stage 4: Advanced Query-Side Technique

> **Prerequisite: the entire foundation (`docs/plan_foundation.md`, Stage 0→3) must be done
> and its baseline recorded.** Stage 4 is worthless without Stage 3 — every change here ships
> only if it beats the recorded baseline on Hit@5 or abstention P/R **without** breaching the
> 2 s p50 latency budget. Anything that doesn't is reverted, not merged "to be safe."

Everything in this document is **PRD-sanctioned** (PRD §10 pre-authorises HyDE, step-back
rewriting, and eval-driven tuning). Nothing here needs a PRD amendment. Items that *would*
(transliteration reindex, bge-m3 sparse/ColBERT heads, reranker LoRA, Phase 15 KG) are in the
**Deferred** section and require explicit sign-off.

---

## Why Stage 4 exists

The foundation makes the system **truthful and measurable** but not **smarter**. Today all
query classes — a located quote, a person lookup, a thematic question, an analytic count, a
follow-up — flow through **one** pipeline with one `bm25_weight` and one candidate budget. That
is the single largest accuracy lever left in the locked stack: different query shapes want
different retrieval behavior, and right now they don't get it.

Guiding principle: **deterministic first, LLM only when ambiguous.** Every added LLM round-trip
is latency against a 2 s budget and must be measured, not assumed.

---

## Constraints specific to Stage 4

| | |
|---|---|
| Latency | DoD p50 < 2 s; observed 3.7–4.2 s. Any query-time LLM call (routing fallback, HyDE, follow-up rewrite) is measured against this and gated per query class. |
| Thinking models | `qwen3.5:9b` routes `format=json` output to `thinking` → every structured router/HyDE call needs `"think": false` (same gotcha as enrichment). |
| No new heavy model | ~31 GB system RAM, one 32 GB GPU already holding bge-m3 + reranker + chat. Routing/HyDE reuse the resident chat model; no second large model. |
| Measurement | Each item ships behind a flag, default off, and is A/B'd against the Stage 3 baseline on the corpus golden set. |
| Existing asymmetry | HyDE moves the **dense** vector only; BM25 always uses the raw query (`retrieval.py:647`). Preserve this unless eval says otherwise. |

---

## 4.1 Query routing — new `rag_api/route.py`  *(build first; largest lever)*

Classify each query into `name` / `quote` / `thematic` / `analytic` / `followup`, then route to
per-class settings (bm25_weight, candidate budget, whether to run HyDE, whether to hit analytics).

- **Deterministic first.** Reuse `query_parse.detect_quote` (`rag_api/query_parse.py:191`) for
  quotes; add cheap regex/keyword rules for analytic ("how many", "count", "list all") and
  name-lookup shapes. Note `query_parse.detect_signals` matches catalog **performer** names
  (Latin singers), **not** discourse subjects — do not reuse it for entity detection as-is.
- **LLM fallback only when ambiguous** — one `think:false` classification call, measured against
  the latency budget; skip it entirely when the deterministic rules fire.
- Route to a per-class settings table. Today `BM25_WEIGHT_BY_TYPE` exists but the class is chosen
  by the caller, not inferred — this centralizes it.

**Ship criterion:** improves Hit@5 on ≥2 query classes without regressing others; router adds
< ~150 ms p50 (deterministic path adds ~0).

---

## 4.2 HyDE where it helps — `rag_api/expand.py` (built, `expand_query=false`)

Enable HyDE for **`thematic` only** (per the router). It generates a hypothetical answer and
embeds *that* for the dense arm — helps vague conceptual queries, hurts precise quote/name
lookups (adds a noisy vector + a full LLM round-trip). Keep the dense-only asymmetry
(`retrieval.py:647`): BM25 stays on the raw query.

**Ship criterion:** improves thematic Hit@5; measured p50 for thematic queries stays < 2 s
(HyDE adds a generation round-trip — this is the item most likely to breach latency).

---

## 4.3 Follow-up rewriting — `app.py` `QueryRequest` (`:121`)

Add optional `history: list[dict]` to the request. When present, one cheap `think:false` LLM call
rewrites a context-dependent follow-up ("what about his early life?") into a standalone query
before retrieval. `conversation_history` already stores the turns and **nothing reads them back** —
this is the first consumer.

**Ship criterion:** on a follow-up subset of the golden set, standalone-rewrite Hit@5 beats
raw-follow-up Hit@5; adds one round-trip only when `history` is non-empty.

---

## 4.4 Step-back prompting — for vague thematic queries

PRD §10 names it explicitly. For an over-narrow thematic query, generate a more general
"step-back" question, retrieve for both, and merge. Highest latency cost of the four — gate it to
thematic queries the router flags as narrow, and only if 4.2 didn't already satisfy them. Likely
the **last** item to ship, if at all.

**Ship criterion:** improves thematic Hit@5 beyond 4.2 alone, within latency budget. If it only
ties 4.2, don't ship the extra round-trip.

---

## Build order & measurement loop

```
4.1 router (deterministic core) ─► measure ─► 4.2 HyDE(thematic) ─► measure
        │                                              │
        └─► 4.3 follow-up (independent) ─► measure     └─► 4.4 step-back (only if 4.2 leaves a gap)
```

1. Build 4.1's deterministic core; A/B vs baseline. This alone should be the biggest win.
2. Add the LLM router fallback; keep only if it beats the deterministic-only router.
3. Layer 4.2, then 4.3, then (maybe) 4.4 — each behind its own flag, each A/B'd independently,
   each reverted if it doesn't beat baseline within latency.

Record the full grid (class × technique × Hit@5 × abstention P/R × p50) in `eval/results/`.

---

## Verification

| Item | Acceptance |
|---|---|
| 4.1 | Router class-accuracy on a labeled query set; per-class Hit@5 ≥ baseline on ≥2 classes; p50 delta measured. |
| 4.2 | Thematic Hit@5 > baseline; thematic p50 < 2 s. |
| 4.3 | Follow-up Hit@5 (rewritten) > raw; round-trip only when `history` present. |
| 4.4 | Thematic Hit@5 > (baseline + 4.2); p50 < 2 s — else reverted. |

Commit messages (one per shipped item):
```
feat(rag): deterministic query router with per-class retrieval settings
feat(rag): LLM router fallback for ambiguous queries (think=false, latency-gated)
feat(rag): HyDE expansion gated to thematic queries
feat(rag): conversational follow-up rewriting from history
feat(rag): step-back prompting for narrow thematic queries
```

---

## Deferred — explicit sign-off required (NOT Stage 4)

- **Transliteration (Latin→Devanagari) before BM25.** `docs/reranker_low_scores.md:164` calls it
  "Lever 1, the single highest-impact change." `indic-transliteration` is MIT. Buys **recall**, not
  just abstention — a bigger prize than anything in Stage 4. But it requires a conjunct-safe
  tokenizer + full **Tantivy reindex** and excluding the `[Source: … Speakers: …]` header from the
  indexed field. Larger than all of Stage 4 combined.
- **bge-m3 sparse + ColBERT heads.** Only the dense head is reachable via Ollama `/api/embed`. Needs
  serving bge-m3 off Ollama (a PRD §3 locked decision) + `sparse_vectors_config` + full reindex.
  ColBERT multivectors are infeasible at the 250–400 M-chunk target.
- **EN→HI query translation.** Our own docs contradict (`prob_faced.md §3.3` "obvious optional lever"
  vs `best_practices.md:328` "bge-m3 handles it; don't translate"). Stage 0.2's histogram + 3.3
  resolve it with data — decide there, not here.
- **Phase 15 knowledge graph** (`docs/knowledge_graph_neo4j.md`) — hard-blocked until enrichment
  (Foundation 2.3) populates `people_named`. Separate plan, separate sign-off.
- **Fine-tuning / LoRA on the reranker.** Strongest lever that exists; **banned** (`PRD.md:55`).
  Needs a PRD amendment.
