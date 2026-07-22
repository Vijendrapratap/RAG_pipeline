# Plan: query intelligence — planner + bilingual expansion + HyDE

Date: 2026-07-12. Status: implementing (user go).

## The problem, in the user's own examples

1. **"10 sitting for rain"** — the user wants a *list of sittings* (sessions)
   about a topic, with a count. Today the pipeline has no list intent: it does
   one chunk search with `top_k=4` and writes a single essay-style answer.
2. **"barish" vs "rain" vs "बारिश" vs "वर्षा"** — the same meaning in different
   words/scripts. Dense bge-m3 bridges this partially; BM25 not at all; short
   queries give the embedder little to work with.
3. **Complex / multi-part questions** — one blended retrieval can't cover two
   sub-topics at once.

## What already exists (do not rebuild)

* `Retriever.search_summaries` — one result per sitting (file-level summary
  collection `transcript_summaries`), reranked. This IS the "list sittings"
  retrieval primitive; it is just never routed to intelligently.
* HyDE (`rag_api/expand.py`) — built, gated by `RAG_HYDE_THEMATIC` (off).
* Deterministic router (`rag_api/route.py`) — chitchat/name/quote/thematic/
  analytic/followup. Untouched by this work (regression safety).

## Design

### A. Flip HyDE on for thematic queries

`.env`: `RAG_HYDE_THEMATIC=1`. Cost ~1 s per thematic query (think=false
already enforced in `ollama_chat.chat_text`). Helps "words differ, meaning
same" on the dense arm. When the planner produces multi-query variants for a
request, HyDE is skipped for that request (the variants already bridge
vocabulary; avoids paying two LLM calls).

### B. Agentic planner — ONE extra LLM call per eligible query

New module `rag_api/planner.py`, gated by `RAG_PLANNER` (default off in code,
on in deployment `.env` after tests).

* `Plan` = `{intent: "answer"|"list_sittings", queries: [1..3 strings], n}`.
* **Deterministic list-intent detector** (pure regex, bilingual): a count +
  sitting-word ("10 sittings/sessions/pravachan/satsang/प्रवचन/सत्संग/बैठक")
  or a list-verb + sitting-word ("suggest/give/list/batao/चाहिए …
  sittings"). Works even if the LLM is down; supplies `n` (default 10).
* **LLM planning call** (think=false, temp 0, JSON only): given the query,
  return intent, `n`, and 2–3 *retrieval query variants* — a Devanagari Hindi
  version, an English version, and a synonym-widened version. Names stay
  verbatim; variants must be search queries, not answers.
* **Fail-open**: any LLM failure/garbage → deterministic result only, or no
  plan at all → the pipeline behaves exactly as today.
* Eligible classes: `thematic`, `name`, `analytic` (router classes stay the
  outer gate; planner never sees chitchat/quote/followup-pre-rewrite).

### C. Multi-query retrieval merge

`Retriever.search_multi(queries, …)` — run the existing `search()` per
variant (HyDE dense_text only on the first), merge by `chunk_id` keeping max
reranker score, sort, cap. `search_summaries_multi` — same over
`search_summaries`, keyed by `source_file`. Merge logic is a pure function
(`merge_result_lists`) so it is unit-testable. The locked single-query
`search()` path is not modified.

Caveat (accepted): scores come from rerank-vs-variant, not rerank-vs-original;
max-merge over a shared [0,1] relevance scale is close enough and avoids a
second reranker round-trip.

### D. List-intent execution + synthesis

When `plan.intent == "list_sittings"`:
* retrieval = `search_summaries_multi(plan.queries, top_k=min(max(req.top_k, n), 20))`
  → one card per sitting (result_type "summary", date/event/track metadata).
* synthesis question is wrapped by `build_list_question(query, n, lang)` —
  instructs a numbered list: each entry = date/event/track from METADATA +
  one line on why it matches, citing [N]. Streaming unchanged (it's still
  one generate call). Citation floor + abstention apply as usual, so "10
  sittings about pizza" abstains instead of inventing.

### E. API surface

* `_prepare` gains a planner step; `route_info["plan"]` carries the plan.
* `_retrieve` gains a `plan` parameter; plan applies only on the default
  chunk scope + hybrid backend (explicit `scope=`/`backend=`/`find_quote`
  keep precedence).
* Response JSON gains a `"plan"` object (intent, queries, n) for the UI /
  debugging; frontend ignores unknown fields — no frontend change required.

## Config

| Flag | Default (code) | Deployment | Meaning |
|---|---|---|---|
| `RAG_HYDE_THEMATIC` | off | **1** | HyDE for thematic class |
| `RAG_PLANNER` | off | **1** | planner call + multi-query + list intent |
| `RAG_PLANNER_MAX_QUERIES` | 3 | 3 | cap on retrieval variants |
| `RAG_LIST_TOP_K` | 20 | 20 | cap on list-intent result count |

## Latency budget (RTX 5090, qwen3.5:9b, think=false)

* planner call ≈ 0.8–1.5 s; extra retrieval arms ≈ +0.4–0.8 s.
* Only thematic/name/analytic queries pay it; chitchat/quote unchanged.
* HyDE skipped when planner variants exist → no double LLM.

## Non-regression guarantees

* Both features flag-gated; flags off ⇒ byte-identical behaviour.
* Planner fail-open ⇒ LLM down = today's pipeline.
* Router, `search()`, floor/abstention, groundedness untouched in behaviour.
* Full unit suite must stay green (except the 2 known pre-existing catalog
  failures); new modules get their own tests.

## Acceptance probes (live, port 8081)

1. `"10 sitting for rain"` → `plan.intent=list_sittings`, ~10 summary
   results, numbered-list answer with dates/events.
2. `"barish ke baare mein swami ji ne kya kaha"` → planner variants include
   बारिश/rain; grounded chunk answer.
3. Non-regression: "hi" → chitchat 0 citations; gibberish → abstain;
   "puran singh ke baare mein" → name; real ध्यान query → grounded Hindi
   answer with citations.

## Explicitly out of scope (unchanged decisions)

* Transliteration + full Tantivy reindex (deferred item 4) — own phase.
* Iterative re-search loops (retrieve → reflect → re-retrieve) — next step
  after this lands, if eval shows recall still short.
* Model upgrade / OpenRouter A/B — orthogonal, gated separately.
