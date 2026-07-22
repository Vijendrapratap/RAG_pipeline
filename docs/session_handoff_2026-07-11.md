# Session Handoff — 2026-07-11

Hand this file to a fresh session. It is a snapshot of where the **Foundation plan
(`docs/plan_foundation.md`, Stages 0→3)** stands, what is committed, what is loose in the
working tree, and exactly what remains. Read `PRD.md` and `docs/plan_foundation.md` first —
this file points into them, it does not replace them.

---

## 0. Governing rules (do not violate)

- **Phase-gated.** One plan step = one commit with the plan's exact message. Do not advance a
  step without an explicit "go" from the user. Do not bundle steps.
- **Open-source only.** No paid APIs. `CHAT_PROVIDER=openrouter` is the sole gated exception.
- **One writer, ever.** Only ONE process may write Qdrant / Postgres / the Tantivy IndexWriter
  at a time. `set_payload`-by-`source_file` (enrichment) races point delete+upsert (re-chunk).
- **Never swallow an exception.** Every `except` logs file/context/reason and dead-letters.
- **Append to `doc.md`** (gitignored) after every meaningful action. It is the build log; it is
  currently ~2,550 lines and holds the full forensic trail for Stage 2.3.
- **Stack must come up from inside WSL2/Ubuntu** or bind mounts resolve to empty dirs.
- Python 3.11+, type hints on public functions, line length 100, `ruff`, `pytest`.
- Repo uses **psycopg2**, not psycopg3. Ollama tagging uses **`"think": false`** (qwen3.5:9b is a
  thinking model; without it, `format` output routes to `thinking` and `response` is empty).

### Environment / ops facts
- Ports: rag-api **8081** (local override; canonical 8080), Ollama 11434, Qdrant 6333/6334,
  Infinity 7997, Postgres 5432. PG user `owui`, db `openwebui`.
- Secrets come from `.env` (gitignored): `POSTGRES_PASSWORD`, `QDRANT_API_KEY`,
  `DASHBOARD_PASSWORD`. Never print or commit them. Source them inside a **script file**, not an
  inline `wsl.exe bash -lc "…"` — the outer shell eats `$VAR` and `$2`, and the Qdrant `api-key`
  header silently arrives empty (observed this session). Write probes to scratchpad and run
  `bash '/mnt/c/…/script.sh'`.
- Detached writers: use `setsid bash -c 'exec … >> log 2>&1' </dev/null & disown`. A plain
  `nohup … &` launched through `wsl.exe` **gets reaped when the call returns** (observed — the
  writer died with an empty stdout).
- **WSL2 clock drifts backwards under load.** `asctime` in logs is non-monotonic; one file logged
  `total -38s`. `time.monotonic()` elapsed is fine. **Postgres is ground truth for progress**, not
  `grep -c` on the log.

### Live state (measured 2026-07-11)
- `file_meta`: **9,324 / 9,335 tagged**. The 11 untagged are all **zero-chunk** (empty
  transcripts — the only genuine upstream/audio candidates, not a tagging problem).
- Qdrant collections: **`catalog`, `transcripts`** only. `transcript_summaries` **does not exist**
  (that is Stage 2.4). `transcripts` reconciles exactly: 24,414 chunk_meta rows of tagged files =
  24,414 points with `event_type`.
- `people_named`: 7,348 distinct / 31,071 mentions. Exports at repo root:
  `people_named_all.csv`, `places_named_all.csv` (2,243), `scriptures_referenced_all.csv` (753),
  `topics_all.csv` (3,293).
- `conversation_history`: **64 rows** (this gates how much of Stage 3.2 is free).
- Unit suite: **478 passed, 2 failed**. The 2 failures are pre-existing and platform-dependent
  (`tests/unit/test_catalog_normalize.py` hard-codes Windows backslash paths that `parse_path`
  can't split on Linux; failing since `31eb9fc`). Not a regression — do not chase them under a
  Stage-2/3 commit.

---

## 1. Committed state (git log)

Stages 0 and 1 (except optional 1.7) and Stage 2.0/2.1/2.3-run are committed:

```
a9524f5 perf(ingest): bound tag generation (num_predict, repeat_penalty, fail-fast on timeout)
f6ee393 fix(ingest): set_payload uses points= not points_selector= (tag propagation)
e713804 feat(ingest): unblock Phase 13 enrichment (TAG_MODEL=qwen3.5:9b, think=false)
159c38e fix(ingest): make enrichment propagation durable across Qdrant restarts      # 2.0
828d456 fix(retrieval): RRF-fuse the catalog arm …                                    # 1.6
ee41c9c fix(ingest): re-chunk the 50 poisoned files; verify stores reconcile          # 1.4
6301f87 feat(ingest): reindex_file.py …                                               # 1.3
0759f44 fix(ingest): drop degenerate ASR-loop chunks; subdivide oversize …            # 1.2
b75e119 fix(chunker): danda-aware sentence split …                                    # 1.1
83b0374 chore(eval): reproduce retrieval failure; commit chunk-quality + score audits # Stage 0
```

Plan commit messages **still unused** (work not committed): the abstention commit
(`fix(rag): real abstention path behind RAG_ALLOW_ABSTAIN …`), 2.4, and all of Stage 3.

---

## 2. CRITICAL — the working tree holds THREE independent uncommitted streams

A new session opening this repo sees a dirty tree spanning three unrelated initiatives. **Do not
`git add -A`. Do not bundle them.** Commit each stream separately, to its own plan message.

### Stream A — Stage 2.3 correctness fix (THIS session; ready to commit)
Files: `ingestion/utils/tag_schema.py`, `ingestion/enrich_content_tags.py`,
`tests/unit/test_enrich_content_tags.py`.
- Replaced `format:"json"` with a **grammar-constrained JSON Schema** (`TAG_FORMAT_SCHEMA`):
  `maxItems` + `maxLength` + the two enums are now structurally unviolable. Both are enforced by
  llama.cpp's grammar; **`uniqueItems` is NOT** (measured), so a code-side `dedupe_arrays()` strips
  duplicate padding before Postgres and the Qdrant payload fan-out.
- `NUM_PREDICT` 2048→4096 (worst real response was 2,860 tokens); `CHARS_PER_TOKEN` 4→1.7
  (Devanagari density, measured against Ollama `prompt_eval_count`); `DEFAULT_MAX_TOKENS_SINGLE_PASS`
  28k→27k so prompt+overhead+output fits `NUM_CTX`. This finally routes long files to **map-reduce**,
  which had never once executed; map-reduce now unions fragment entities back into the result.
- A `TAG_FORMAT_SCHEMA_BOUNDED` fallback (retried once, on parse-failure only) whose worst-case
  output provably fits `num_predict` — for degenerate ASR chant loops.
- `validate_tags` now also rejects arrays over cap (belt to the grammar's braces).
- **Result: 171 dead-letters → 0.** 9,153 → 9,324 tagged. +1,286 distinct names. 33 tests added
  (55 in the enrichment file). Full forensic trail in `doc.md` (four entries dated 2026-07-10).
- **No plan commit message exists for a 2.3-only fix** — ask the user, or propose e.g.
  `fix(ingest): grammar-constrained tag decoding + dedupe; recover 171 dead-letters`.

### Stream B — "Phase 17" text-normalization + gazetteer (in progress, NOT in the Foundation plan)
New: `ingestion/normalize_text.py` (+ `tests/unit/test_normalize_text.py`),
`ingestion/gazetteer.py` (+ `tests/unit/test_gazetteer.py`).
Modified: `ingestion/chunker_cleaned.py`, `ingestion/chunker_json.py`, `ingestion/chunker_text.py`,
`ingestion/utils/path_parser.py` (+ their tests).
- `normalize_text.py` — deterministic Hindi text repair (intra-word script mixes, Devanagari
  misspellings of loanwords, YouTube artifacts) applied before chunking. Pure functions.
- `gazetteer.py` — corpus-mined entity gazetteer: cluster surface forms by phonetic key,
  majority-vote the canonical spelling, write `data/gazetteer_review.md` for human approval, then
  stop. Never rewrites on its own.
- This references **PRD Phase 17**, a separate/newer initiative than plan_foundation.md's Stages
  0-3. `path_parser.py` may also carry optional **1.7** (location/event_id recovery) — inspect the
  diff before assuming. **Confirm scope with the user before committing any of this.** It is not on
  the Foundation critical path and must not ride along with Stream A or C.

### Stream C — Stage 1.5 abstention (partial, uncommitted)
Files: `rag_api/config.py` (`allow_abstain` at :145, `RAG_ALLOW_ABSTAIN` at `.env.example:105`),
`rag_api/synthesis.py`, `rag_api/app.py`, `tests/unit/test_rag_api_synthesis.py`.
- The config knob and env var exist; the `synthesis.py` filter path is modified but the plan's
  `fix(rag): real abstention path …` commit is **absent from git log**. Treat as unverified.
- Gated on the **0.2 score histogram** (which IS committed). Keep default `RAG_ALLOW_ABSTAIN=false`
  until the histogram proves a citation floor that doesn't refuse legitimate cross-script hits.
- Finish per `plan_foundation.md §1.5` (drop the `or results[:1]` safety net at `synthesis.py:194`
  under `allow_abstain`; update docstring), add the negative-query test, commit, or explicitly
  defer. `puran singh` must still answer; a genuine invented negative must return `NO_CONTEXT_MESSAGE`
  with 0 citations.

---

## 3. Remaining tasks by stage (the actual "what's left")

### Stage 2 — two loose ends to fully close
1. **Commit Stream A** (above). Waiting on a commit message.
2. **281-file duplicate-array backfill.** 281 files still carry 792 exact-duplicate array entries
   from the ORIGINAL enrichment run (Stream A only cleaned the 173 it re-tagged). Fix is **pure SQL**
   (`array_agg(DISTINCT …)` order-preserving, or apply `dedupe_arrays` semantics) — **no model
   calls** — then **re-propagate the 6-field subset to Qdrant payloads** (a store write → single
   writer, single lane). Verify PG array = Qdrant payload length per file afterward.
3. **2.4 — Build the summary index.** `ingestion/build_summary_index.py` already exists (no new
   code). Its predicate is stricter than 2.3's producer: `tagged_at IS NOT NULL` AND ≥1 summary
   present (~9,324 eligible). Running it creates the `transcript_summaries` collection and turns on
   `scope=summaries` + `scope=two_stage` in `retrieval.py` (:813/:841), both silent no-ops today.
   **Acceptance:** `transcript_summaries` point count > 0; `/api/health` reflects it; `scope=two_stage`
   returns results where it returned `[]`. **Commit:** `feat(ingest): build summary index; enable
   scope=summaries/two_stage`. This is the natural next move — zero new code, unblocks two scopes.

### Stage 1.5 — abstention (Stream C) — finish or defer before Stage 3.

### Stage 1.7 — OPTIONAL, cheap, off critical path.
Loosen `_EVENT_RE` in `ingestion/utils/path_parser.py:122-129` (it forbids digits in the location
group and requires a day range), then re-run the `59fbd3d` backfill (re-parse identity key,
`set_payload`, no re-chunk/re-embed). Lifts `location`/`event_id` above the 66% ceiling. **Note the
`path_parser.py` diff already in the tree may be this** — check before doing it twice.

### Stage 3 — the whole measurement layer, untouched. NONE started.
- **3.1 Rebuild `eval/run_eval.py`.** The committed file can't run — its only import
  (`open_webui_functions/`) is absent from HEAD (deletion half-applied). Rebuild the file from
  `git show 59fbd3d^:eval/run_eval.py` — **NOT from HEAD** (the plan text is wrong on this point;
  HEAD's copy is the broken one). Retarget onto `rag_api.retrieval.Retriever.search`. Keep
  Hit@1/5/10 + MRR, the per-type `bm25_weight` sweep, the CI gate (quote Hit@5 < 0.80 → exit 1).
  **Add abstention precision/recall as first-class metrics** (Hit@k can't see the false-abstention
  regression 1.5 introduces).
- **3.2 Corpus-grounded golden set — `eval/golden_queries_corpus.yaml`. Gates 3.3.** The existing
  `eval/golden_queries.yaml` (30 queries) measures a mock quarterly-review + a CAP-theorem essay —
  **zero corpus overlap**, full rewrite needed. Seed positives from `conversation_history` (**64
  rows** — decides how much is free); augment with doc2query-style synthetic queries (generation,
  permitted; not fine-tuning). **Must include a negative set** (invented people/places/events) to
  score abstention. `पुरन सिंह` (215 chunks / 443 occurrences) is a **positive**, never a negative.
- **3.3 Baseline + grid-search — `eval/bench_backends.py` (never run).** `recall_at_k` is 0 by
  construction until 3.2 lands. Grid-search `bm25_weight`, `candidates_per_source`, `cite_min_score`,
  RRF `k`, plus the untouched Qdrant `params.quantization.oversampling` + `rescore` + `hnsw_ef` (free
  recall from the existing int8 index). Set `include_catalog` explicitly (defaults True — a confound).
  **Record the baseline before any Stage 4 change.** Adopt only settings that beat baseline.

### Out of scope here
- **Stage 4** (advanced query-side retrieval) is a separate doc:
  `docs/plan_stage4_advanced_retrieval.md`. Do not start it.
- **Phase 17** (Stream B) is a separate initiative from the Foundation plan.

---

## 4. Suggested order for the next session
1. Get a commit message from the user and **commit Stream A** (2.3 fix) alone.
2. **2.4** summary index (zero new code, high value).
3. Decide Stream C (abstention): finish + commit, or stash/defer.
4. Clarify Stream B (Phase 17) scope with the user — commit or set aside; keep it off A/C.
5. Then Stage 3: 3.1 → 3.2 → 3.3 in order (3.2 gates 3.3).
6. Optional 1.7 anytime (check the existing path_parser diff first).

**Do not** run two store-writers concurrently. **Do not** advance a phase without a "go".
