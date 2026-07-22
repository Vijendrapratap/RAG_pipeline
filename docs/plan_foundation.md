# Implementation Plan — Foundation (Stage 0 → 3)

> Scope: everything needed to make the system **truthful, populated, and measurable**.
> Stage 4 (advanced query-side technique) is a **separate document**: `docs/plan_stage4_advanced_retrieval.md`.
> Supersedes the reindex framing in `.claude/plans/toasty-squishing-wall.md`, which was written
> before commit `59fbd3d` and assumed a full-corpus metadata reindex that is **already done**.

---

## Corrected state — measured live 2026-07-09 (not from memory)

All numbers below were read from the running stack (rag-api on :8081, Qdrant, Postgres
`openwebui`), not extrapolated. The stack is up and healthy: `tantivy_docs=24567`,
`embed_model=bge-m3`, `chat_model=qwen3.5:9b`, `reranker=true`, `pageindex_trees=3`.

| Area | Live measurement | Verdict |
|---|---|---|
| `session_date` (Qdrant / file_meta) | **99.4% / 98.8%** (9,227/9,335; 108 genuinely undated) | ✅ **DONE** by backfill `59fbd3d` — no reindex needed |
| `season` | 99.8% | ✅ done |
| `track_type` | 100% (defaulted) | ✅ |
| `location` / `event_id` | **65.8%** | ⚠️ structural ceiling; ~cheap partial recovery possible (see 1.7) |
| Enrichment (`people_named`, `topics`, `summary_*`, `tagged_at`) | **0.0%** everywhere | ❌ **never run** — the real Stage 2 |
| Oversize chunks (`chunk_meta`) | 138 chunks > 6k chars; **124 are ASR loops** (uniq-ratio < 0.15, 3.76 M chars of garbage); **14 legit-long**; max **586,331 chars / 42 unique words** | ❌ across only **50 files** |
| `TAG_MODEL` env | unset → falls back to code default `qwen2.5:7b`, **which is not installed** (`gemma4:31b`, `qwen3.5:27b`, `qwen3.5:9b`, `bge-m3` are) | ❌ blocks enrichment |
| `RAG_ALLOW_ABSTAIN` | does not exist | — to be added |
| `cite_min_score` | `0.75` (unfalsified against real score distribution) | ⚠️ calibrate in 0.2 |

**Consequence for this plan:** the critical path is **not** a metadata reindex. It is
(a) a 50-file chunk-quality repair, (b) the enrichment run that has never happened, and
(c) the measurement harness that lets us prove any of it worked. The oversize repair still
lands **before** enrichment, because enrichment reconstructs each file's transcript from
`chunk_meta.text` and a 90,212-word ASR loop is pure poison to the tagging model.

---

## Constraints (verified — do not violate)

| | |
|---|---|
| Open-source only | No paid APIs. `CHAT_PROVIDER=openrouter` is the sole gated exception (opt-in, per-use). |
| Banned | Fine-tuning / LoRA incl. on the reranker (`PRD.md:55`). |
| Locked — ask first | Qdrant; bge-m3 **via Ollama**; int8 quantization; TARGET/MAX/OVERLAP chunk params; reranker choice (`PRD §3`). Danda fix is a **bug fix**, not a param change. |
| Hardware | RTX 5090 (32 GB VRAM), **~31 GB system RAM** — CPU spill is catastrophic; no co-resident second large model. |
| `format=json` gotcha | `format=json` + a thinking model ⇒ **empty `response`** (output goes to `thinking`). `qwen3.5:9b` is a thinking model → every structured call needs `"think": false`. |
| Process | One phase = one commit, exact message. Show acceptance output before advancing. Never swallow an exception. Append a timestamped entry to `doc.md` after every meaningful action. |
| Never run two writers | Only ONE process may hold the Tantivy `IndexWriter` lock or write Qdrant/Postgres at a time. `set_payload`-by-`source_file` (enrichment) races point delete+upsert (re-chunk). |

---

## Execution DAG

```
0.2 score histogram ─────────────────────────► 1.5 abstention (gated on 0.2)
0.3 chunk audit (DONE — formalize) ─┐
1.1 danda split ────────────────────┤
1.2 chunker guards + desync fix ─────┼─► 1.4 targeted re-chunk (50 files) ─► 1.4v verify
1.3 reindex_file.py (5 deletes) ─────┘                                          │
                                                                               ▼
2.0 durability ─┐                                              (clean chunk_meta.text)
2.1 unblock  ───┼─► 2.2 calibrate 100 ─► 2.3 ENRICH 9,335 (1.5–4 d GPU) ─► 2.4 summary index
                                                                               │
1.6 catalog fusion (rag_api, independent) ─────────────────────────────────────┤
                                                                               ▼
3.1 eval harness ─► 3.2 golden set + negatives ─► 3.3 grid-search + baseline
```

**Hard ordering:** `1.1 → 1.2 → 1.4` · `1.3 gates 1.4` · `0.2 gates 1.5` · **`1.4 before 2.3`**
(clean the 50 poisoned files first) · `2.1 + 2.0 gate 2.3` · `2.3 → 2.4` · `3.2 gates 3.3`.
`1.6` (retrieval-side) and Stage 3 harness code are **independent** and can be built in
parallel sessions (see "Parallelization" at the end).

**Estimate:** ~5–6 working days of build + the enrichment GPU wall (measured in 2.2; treat
1.5–4 d as a ceiling). The metadata reindex that dominated the old estimate is gone.

---

## Stage 0 — Reproduce & measure  *(mostly done; formalize + commit)*

Nothing here changes behavior. It converts guesses into committed, re-runnable artifacts.

### 0.1 Reproduce the retrieval failure
`POST /api/query {"query":"search some discourse about puran singh"}` against :8081 (auth).
Capture the retrieved chunks (full text), reranker scores, and the answer. Classify:
- real `पुरन सिंह` passages + fabricated answer → **grounding bug** (prompt/citation, not abstention);
- passages that never mention him → **reranker calibration** (score floor is the lever);
- model refused → noise-floor bug; abstention makes it **worse**.
Use `पुरन सिंह` (215 chunks / 443 occurrences) as a **positive** in 3.2, never a negative.

### 0.2 Reranker score histogram — **BLOCKS 1.5**
New committed `eval/score_histogram.py`: run ~30 hand-picked known-good queries (Devanagari +
Latin; quote / name / thematic) through `Retriever.search`, dump the reranker score of the top
hit and of hit #4. **The citation floor is set from this distribution, not from 0.75.** If
legitimate cross-script hits cluster low (`prob_faced.md §3.1` claims 0.01–0.18) and Puran Singh
reranked 0.88, both cannot be typical — this measurement decides whether abstention ships at all.

### 0.3 Chunk-quality audit — **DONE, needs to become a committed script**
Already measured live: **124 ASR-loop chunks** (uniq-ratio < 0.15), **14 legit-long**, across
**50 files**; worst is 586,331 chars / 42 unique words. Formalize as committed
`ingestion/audit_chunks.py` (full scan of `chunk_meta`, reporting per-chunk uniq-word ratio and
real bge-m3 token count — **not** `words×1.3`, which under-counts Devanagari). Emit the 50-file
work-list that 1.4 consumes.

**Acceptance:** reproduction transcript; committed histogram script + its output; committed audit
script + the 50-file work-list. Do not start 1.5 or 1.4 until these exist.

---

## Stage 1 — Truth

### 1.1 Danda-aware sentence split — `ingestion/chunker_text.py:33`
Current: `SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")` — no Devanagari terminators, so Hindi
prose collapses to one "sentence." Change to:
```python
SENTENCE_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")
```
Load-bearing: `split_sentences` is the **only** subdivision tool 1.2 has. Its production importer
is `ingestion/catalog/index_catalog.py` → **rebuild the catalog collection** after.
Also fix docs pointing operators at the wrong chunker (`README.md`, `docs/getting_started.md`,
`scripts/start_pipeline.sh`, `scripts/preflight.sh`, `PRD.md`) and the stale `RAW_TRANSCRIPTS_DIR`
in `.env.example` (production tree is `/mnt/d/Transcription whisperx/Output`).
**Test:** `pytest tests/unit/test_catalog_index.py`; the previously-unsplit `detail_contents` now split.

### 1.2 Chunker guards — `ingestion/chunker_json.py` + `ingestion/bulk_ingest_hardened.py`
Root cause of oversize is `chunker_json.py:150-189`: `MAX_TOKENS=700` is a **soft target**, never a
cap — a single giant segment is appended unconditionally at `:177` and emitted alone by the `:183`
"pathological single segment" branch. Fixes:
1. **Degenerate guard (the 124 loops).** In `chunk_segments`, drop a chunk whose unique-word ratio
   < 0.15 (ASR loop) and record a **chunk-level dead-letter** (`source_file`, offset, first 200
   chars, reason) — never silently. These are upstream Whisper failures; embedding a 42-word loop
   is pure noise. (Upstream re-transcription of the 50 files is a separate, deferred task.)
2. **Subdivide the 14 legit-long.** When a single segment exceeds `MAX_TOKENS`, split it with
   `chunker_text.split_sentences` (danda-aware after 1.1) and pack sentences into `MAX_TOKENS`-bounded
   chunks. Raise loudly if a *single sentence* still exceeds the cap — never truncate silently.
3. **Fix the Tantivy/points desync** — `bulk_ingest_hardened.py:511` iterates `zip(ids, batch)`, so a
   chunk dropped by `vec_is_finite` at `:491` is still indexed into BM25. Iterate the filtered
   `points` instead, and add a chunk-level dead-letter where `vec_is_finite` currently does
   `log.error; continue`.
4. **Add `len(vec) == 1024`** next to the `vec_is_finite` guard.
   *(Filter AFTER `ids` are assigned at `:473`, exactly like `vec_is_finite` — `chunk_uuid` keys off
   the global index, so filtering earlier shifts every downstream uuid and breaks resume + verify.)*

### 1.3 Reindex infrastructure — new `scripts/reindex_file.py`
Re-chunking is a no-op today (progress-DB skip) and, once unblocked, orphans every store because
**no delete path exists anywhere**. Build the five operations, per `source_file`:
- Qdrant filtered delete (`delete` with a `source_file` `must` filter);
- `tantivy_writer.delete_documents("source_file", src)` — **works today**: tantivy 0.26.0 exposes it
  and `source_file` is declared with the `raw` tokenizer (the `doc.md:1255` "append-only" note is wrong);
- `DELETE FROM chunk_meta WHERE source_file = …`;
- `UPDATE file_meta SET tagged_at = NULL WHERE source_file = …` (so a re-chunked file re-enriches);
- `progress.reset_to_pending(name)`.
Also fix, in `bulk_ingest_hardened.py`:
- the progress skip at `:596-599` (`status=='ok'` → silent `continue`) must honor a `--no-skip-existing`
  / reindex path;
- the `mark_ok`-before-commit race: `:614` commits `ok` before the periodic
  `tantivy_writer.commit()` at `:651-653` (every `TANTIVY_COMMIT_EVERY` files). A crash between them
  loses up to that many files' BM25 docs while they read `ok`. Commit Tantivy before `mark_ok`, or
  make the two atomic per batch.

### 1.4 Targeted re-chunk of the 50 files — **before enrichment**
Run `reindex_file.py` over the 0.3 work-list (~50 files): delete → re-chunk (with 1.2 guards) →
re-embed → re-upsert. This removes 3.76 M chars of garbage from `chunk_meta.text` so the enrichment
model isn't fed 90k-word loops.
**Acceptance (1.4v — bidirectional verify):** re-run `audit_chunks.py` → 0 chunks with uniq-ratio
< 0.15, 0 chunks > real 8192 tokens; Qdrant `points_count` and `tantivy_doc_count()` reconcile with
`SELECT sum(chunk_count) FROM file_meta`; assert **total characters of the 50 files are preserved
minus the quarantined loops** (re-chunk must not lose legit prose); BM25 returns no `chunk_id` absent
from Qdrant. `verify_ingestion` is one-directional — do not trust it alone here.

### 1.5 Abstention — score floor only  *(gated on 0.2)*
- Add `allow_abstain: bool = False` to `Settings` (`rag_api/config.py`, near `cite_min_score:136`) +
  `.env.example` (`RAG_ALLOW_ABSTAIN`).
- `filter_min_score` (`synthesis.py:174-194`): when `allow_abstain`, `return kept` (drop `or results[:1]`
  at `:194`). Update the docstring — it currently documents the safety net as intentional.
- `trim_by_relevance`'s net at `:171` is **unreachable** (`score >= score*ratio` for `ratio ≤ 1`) — leave
  it or drop it for symmetry, but do not claim it changes behavior.
- **Caller-safe (verified):** `filter_min_score` has one production caller (`app.py:474`); both
  `Synthesizer.generate` and `.stream` route empty → `_no_context` → `NO_CONTEXT_MESSAGE`;
  `app.py:501` (`count`) handles `[]`.
- **Set `cite_min_score` from 0.2's histogram**, not 0.75. Keep default `RAG_ALLOW_ABSTAIN=false` until
  the histogram proves a floor that does not refuse legitimate cross-script hits.
- **Explicitly NOT doing** a lexical/BM25 existence gate — proven unimplementable (OR-default BM25 →
  never fires; overloaded `[]` → abstains on 100% when `TANTIVY_DIR` is misconfigured; virama-splitting
  tokenizer makes even a Devanagari existence check unsound).

### 1.6 Catalog fusion — `rag_api/retrieval.py:648-663`  *(independent, rag-api only)*
Bug: catalog rows are concatenated at `:652` with **no re-sort**, then truncated positionally at `:660`
(`fused[:candidates_per_source*2]`). Survivors = `|dense_ids ∩ bm25_ids|` → **zero when the arms are
disjoint**. RRF scores are bounded by `1/61 ≈ 0.0164`; a catalog row at cosine 0.92 loses its slot to a
transcript row at RRF 0.0002 — a ~50× scale mismatch. `rrf_fuse` (`:109-145`) can't take a third list as
written (two asymmetric loops `h["id"]` vs `h["chunk_id"]`, scalar `(1-w)`/`w` split, BM25-specific
payload fallback). **Fix:** generalize `rrf_fuse` to N weighted ranked lists and fuse the catalog arm by
rank (not raw cosine), or min-max normalize the catalog cosines into the RRF range before concat, then
**re-sort**. While here: `final_top_k` (`RAG_TOP_K=4`) is dead in the happy path (only live at the
reranker-down fallback `:565`) — wire it or delete it; do not grid-search it in 3.3.
**Acceptance:** a catalog row survives to the reranker when the transcript arms are disjoint.

### 1.7 (optional, cheap) Recover location/event_id — `ingestion/utils/path_parser.py:122-129`
`_EVENT_RE` forbids digits in the location group (`[A-Za-z][^\d]*?`) and requires a day range
(`d1 - d2`), so `SEC - 28 GROUND FARIDABAD` and single-day events fail — part of the 34% location null.
Loosen both, then **re-run the backfill** (`59fbd3d`'s tool: re-parse identity key, `set_payload`,
no re-chunk/re-embed). Not on the critical path; lifts `location`/`event_id` above 66%.

---

## Stage 2 — Enrichment  *(the real remaining gap — 0% today)*

### 2.0 Durability — `ingestion/enrich_content_tags.py:348-355`
Current: `write_tags` commits `tagged_at = NOW()` **before** `qdrant_set_payload_for_file`, and a
propagation failure is logged-and-swallowed. Resume is `WHERE tagged_at IS NULL` → a single transient
Qdrant blip strands that file's payloads **forever**. Over 9,335 files × 1.5–4 days a Qdrant restart is
not hypothetical. **Fix:** propagate-then-commit, or add the `--repropagate` flag the comment promises.
Also: dead-letter transport exceptions that escape the retry decorator (`:340-341`, currently returned as
a reason string with no artifact — violates CLAUDE.md rule 6); and fix the 120-char dead-letter filename
truncation (`:124`) that lets similar long paths overwrite each other.

### 2.1 Unblock — `ingestion/enrich_content_tags.py` (steps 1+2 MUST land together)
1. `:51` → `TAG_MODEL = os.environ.get("TAG_MODEL", "qwen3.5:9b")`; set `TAG_MODEL=qwen3.5:9b` in `.env`,
   `.env.example`, `docker-compose.yml`. Fix docstrings still naming "Qwen 2.5 7B" (`:1`, `:15`, `:368`,
   the `--model qwen2.5:14b` usage example).
2. **Add `"think": false`** to `ollama_generate_json` (`:100-116`). `qwen3.5:9b` is a thinking model;
   `format=json` + thinking ⇒ empty `response` ⇒ `parse_model_json` → `(None, "empty response")` ⇒
   **every file dead-letters**. `rag_api/pageindex.py:245` has this fix; `ingestion/` never got it.
   `CHAT_THINK=off` cannot save you — only `rag_api/synthesis.py` reads it.
3. Re-check `NUM_CTX = 32_768` (`:74`) for a 9B on this box; watch for CPU spill (~31 GB RAM).

### 2.2 Calibrate on 100 files
`scripts/06_enrich_tags.sh --limit 100`. Record seconds/file → × 9,335 = the real 2.3 wall. Hand-inspect
5 rows: `people_named` / `scriptures_referenced` populated, Devanagari, not hallucinated, no
empty-response dead-letters. **Run it twice (N=1 and N≥2 parallel) to measure whether GPU concurrency
helps — see Parallelization.**

### 2.3 Run it — `scripts/06_enrich_tags.sh`
All 9,335 files, resumable, failures to dead-letter. Monitor `enrich.log`. Writes
`file_meta.{people_named, topics, scriptures_referenced, summary_hindi, summary_english}` and propagates
the 6-field subset into Qdrant payloads.
**Acceptance:** `SELECT count(*) FROM file_meta WHERE tagged_at IS NOT NULL` → 9,335;
`... WHERE people_named <> '{}'` → non-zero; Qdrant `people_named` coverage jumps from 0%.

### 2.4 Build the summary index — `ingestion/build_summary_index.py` (no new code)
Predicate is stricter than 2.3's producer (`tagged_at IS NOT NULL` **AND** ≥1 summary present). Turns on
`scope=summaries` and `scope=two_stage` (`retrieval.py:813`/`:841`), both currently silent no-ops.
**Acceptance:** `transcript_summaries` point count > 0; `/api/health` reflects it; `scope=two_stage`
returns results where it previously returned `[]`.

---

## Stage 3 — Make progress measurable

### 3.1 Rebuild the eval harness — `eval/run_eval.py`
`git show HEAD:eval/run_eval.py` recovers a file that **cannot run** — its only import,
`open_webui_functions/`, is absent from HEAD's tree (deletion half-applied). Retarget onto
`rag_api.retrieval.Retriever` — mostly deletion: the six private-method pokes collapse into one
`Retriever.search(query, filters=None, top_k=…, bm25_weight=BM25_WEIGHT_BY_TYPE[qtype])` (`retrieval.py:619`);
`_score_analytics`'s prose-parsing hacks disappear because `rag_api/analytics.py` returns dicts. Keep
Hit@1/5/10 + MRR, the per-type `bm25_weight` sweep, and the CI gate (`exit 1` if quote Hit@5 < 0.80).
**Add abstention precision/recall as first-class metrics** — Hit@k cannot detect the false-abstention
regression 1.5 introduces.

### 3.2 A corpus-grounded golden set — `eval/golden_queries_corpus.yaml`  *(gates 3.3)*
The 30 existing queries measure a mock quarterly-review meeting and a CAP-theorem essay — zero corpus
overlap. Rewrite: seed positives from `conversation_history` (`002_conversations.sql`, stores
`question` + `citations` JSONB — **count the rows first**, it decides how much of 3.2 is free); augment
with doc2query-style synthetic queries (generation, not fine-tuning — permitted). **Must include a
negative set** (invented people/places/events) to score abstention. `पुरन सिंह` is a positive.

### 3.3 Baseline + grid-search — `eval/bench_backends.py` (never run)
`recall_at_k` is 0.0 by construction until 3.2 lands; latency columns are trustworthy today. Grid-search
`bm25_weight`, `candidates_per_source`, `cite_min_score`, RRF `k` (PRD §10's sanctioned tuning loop), plus
Qdrant search-time `params.quantization.oversampling` + `rescore` + `hnsw_ef` (never set today — free recall
from the existing int8 index). Set `include_catalog` explicitly (it defaults `True` — a confound).
**Record the baseline before any Stage 4 change.** Adopt only settings that beat baseline.

---

## Verification (show output before advancing)

| Step | Acceptance |
|---|---|
| 0.1 | Reproduction transcript: chunks + scores + answer; failure classified. |
| 0.2 | Committed `score_histogram.py` + output over ~30 queries; a defensible `cite_min_score`. |
| 0.3 | Committed `audit_chunks.py`; 50-file work-list; real degenerate + oversize counts, tokenizer named. |
| 1.1 | `pytest tests/unit/test_catalog_index.py`; previously-unsplit `detail_contents` now split; catalog rebuilt. |
| 1.2 | Unit tests: a 90k-word loop is dead-lettered not embedded; a legit-long segment subdivides to ≤ MAX_TOKENS. |
| 1.4 | Re-audit: 0 loops, 0 chunks > 8192 real tokens; store counts reconcile; chars preserved minus loops; BM25 ⊆ Qdrant. |
| 1.5 | `RAG_ALLOW_ABSTAIN=true`: a genuine negative → `NO_CONTEXT_MESSAGE`, 0 citations; `puran singh` still answers. New tests in `tests/unit/test_rag_api_synthesis.py`. |
| 1.6 | A catalog row survives to the reranker when the transcript arms are disjoint. |
| 2.0 | Kill Qdrant mid-run; the affected file re-tags/re-propagates on resume. |
| 2.1 | 100-file run: seconds/file recorded; 5 rows hand-inspected; no empty-response dead-letters. |
| 2.3 | `file_meta.tagged_at IS NOT NULL` → 9,335; `people_named <> '{}'` non-zero; Qdrant `people_named` > 0%. |
| 2.4 | `transcript_summaries` count > 0; `scope=two_stage` returns results where it returned `[]`. |
| 3.1/3.2 | `python -m eval.run_eval --queries eval/golden_queries_corpus.yaml` writes results; Hit@5, MRR, **abstention P/R**. Baseline. |
| 3.3 | Grid-search table; adopt only settings beating baseline. |

---

## Commit sequence (one per step)

```
chore(eval): reproduce retrieval failure; commit chunk-quality + score-distribution audits
fix(chunker): danda-aware sentence split; correct stale chunker docs
fix(ingest): drop degenerate ASR-loop chunks; subdivide oversize; fix tantivy/points desync
feat(ingest): reindex_file.py — per-file delete across Qdrant/Tantivy/Postgres; fix progress skip + commit race
fix(ingest): re-chunk the 50 poisoned files; verify stores reconcile
fix(rag): real abstention path behind RAG_ALLOW_ABSTAIN; calibrate citation floor from histogram
fix(retrieval): RRF-fuse the catalog arm; catalog candidates no longer dropped before rerank
fix(ingest): make enrichment propagation durable across Qdrant restarts
feat(ingest): unblock (TAG_MODEL=qwen3.5:9b, think=false) and run Phase 13 enrichment
feat(ingest): build summary index; enable scope=summaries/two_stage
feat(eval): corpus-grounded golden set with negatives; retarget run_eval onto Retriever
feat(eval): baseline + knob grid-search; record before Stage 4
```

Append a timestamped entry to `doc.md` after each.

---

## Parallelization (independent sessions)

Disjoint file ownership so merges are trivial; only ONE lane may touch the live stores.

| Lane | Owns | Tasks | Live stores? |
|---|---|---|---|
| **A — Index Owner** | `chunker_json.py`, `bulk_ingest_hardened.py`, `path_parser.py`, new `reindex_file.py`, `enrich_content_tags.py` | 0.3, 1.1→1.4, 1.7, 2.0→2.4 | **Yes, exclusively** |
| **B — Retrieval** | `rag_api/synthesis.py`, `config.py`, `retrieval.py` | 0.2, 1.5, 1.6 | read-only |
| **C — Eval** | `eval/*` | 3.1, 3.2 | read-only |

Lane A is the critical path and the only store-writer. B and C build inside A's enrichment
GPU wall. Danda (1.1) must land in A before 1.2. **GPU-concurrency knob:** in 2.2, run the
calibration at N=1 and N≥2 (`OLLAMA_NUM_PARALLEL` + a `--shard i/N` predicate on
`hashtext(source_file) % N`) and read seconds/file off both — potentially halves the 2.3 wall
for a quarter-day of work. Watch VRAM: 9B q4 + N×32k KV alongside bge-m3 + reranker; spill to
the ~31 GB system RAM makes it *slower*.

**Worktree gotchas:** `data/` and `.env` are gitignored → a fresh worktree has neither. B/C
point `QDRANT_URL`/`PG_DSN`/`TANTIVY_DIR` at the absolute live paths (read-only) and get their
own `.env`. `doc.md` is gitignored and every lane appends to it — use `doc.d/lane-{a,b,c}.md`
and concatenate, or three-quarters of the build log is lost.
