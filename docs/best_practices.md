# RAG Quality Best Practices — transcript-rag

A practical, opinionated guide for getting the highest retrieval accuracy
out of this pipeline. Companion to [PRD.md](../PRD.md),
[runbook.md](runbook.md), and [architecture.md](architecture.md).

This document covers what to do (and what not to do) at each stage —
**transcription → chunking → embedding → indexing → querying → evaluation**
— so that a real-world query like *"what did Swami ji say on a monsoon day
about barsat?"* lands on the right passage instead of the wrong one.

The advice here is concrete. Every numbered item is either already
implemented, controlled by an `.env` knob, or a clearly-marked
recommendation for the upstream ASR pipeline.

---

## 1. The accuracy ladder

Read this first. Retrieval accuracy is bounded top-down — fixing a lower
rung doesn't help if a higher rung is broken.

1. **Transcription quality.** If Whisper mis-hears "Anush" as "honest",
   no retrieval system on Earth will find Anush. Fix transcription before
   you fix retrieval.
2. **Chunking that preserves meaning.** A chunk split mid-thought makes
   both embeddings and BM25 worse. Chunk on natural boundaries
   (sentences, speaker turns, segments), not byte counts.
3. **Header enrichment.** The text the embedder sees should include the
   metadata you want the embedding to encode (event, date, track type,
   season). This is one of the highest-leverage improvements available.
4. **Hard filters on indexed fields.** Once metadata is in the payload,
   use it. A `season="monsoon"` filter beats any amount of fuzzy matching.
5. **Hybrid retrieval (dense + BM25).** Each catches what the other misses.
6. **Reranking.** Cross-encoder on the top-N gives you another precision
   bump for almost free at our scale (50-ish candidates).
7. **Query rewriting (HyDE / multi-query).** When the user's phrasing
   doesn't match the corpus phrasing, generate intermediate text that does.

Order matters. Don't fine-tune the reranker (#6) before you've fixed
chunking (#2).

---

## 2. Transcription (upstream of this repo)

The transcripts come in via Whisper / WhisperX. The biggest win for
retrieval quality is **prompt-biasing Whisper with your domain vocabulary**.
Without it, uncommon names ("Anush"), Sanskrit/Hindi technical terms
("pravachan", "sambodhan", "barsat"), place names ("Rishikesh"), and
honorifics ("Swami ji") get butchered.

**Do:**

- Maintain a glossary file (`docs/whisper_glossary.txt`, one phrase per
  line) of names, places, key terms, song titles you expect to encounter.
  Update it whenever you see a recurring mis-transcription.
- Pass the glossary as Whisper's `initial_prompt`:
  ```bash
  whisper input.wav --initial_prompt "$(cat docs/whisper_glossary.txt | tr '\n' ' ')" ...
  ```
- Use **`whisper-large-v3`** (or whisper.cpp's `ggml-large-v3-q5_0.bin`)
  for production runs. The accuracy gap vs `medium` on multilingual
  Indic content is meaningful and the cost is one-time.
- Prefer **WhisperX** output (`*.json` with diarization + word-level
  timestamps) over plain Whisper text. The JSON chunker (`chunker_json.py`)
  preserves more structure than the text chunker, even when every
  speaker is the same person.
- Set `language="hi"` or `language="en"` explicitly when you know — auto-
  detection sometimes mis-classifies code-mixed Hindi/English content.

**Don't:**

- Don't run distilled Whisper variants in production. The 4× speedup
  isn't worth the accuracy regression on names.
- Don't post-process transcripts with regex "fixes" that touch content —
  it's hard to keep those rules consistent over thousands of files.
  Fix the source instead (better glossary, larger model).

---

## 3. Chunking

Already implemented per PRD §6 Phase 3. The knobs that matter:

| Setting | Value | Why |
|---|---|---|
| `TARGET_TOKENS` | 450 | Sweet spot for bge-m3 retrieval: enough context for the embedding to be specific, not so much that signal dilutes |
| `MAX_TOKENS` | 700 | Hard cap; below bge-m3's 8K limit, leaves headroom for header text |
| `MIN_TOKENS` | 200 (JSON only) | Avoids single-sentence chunks that have no context |
| `OVERLAP_SENTENCES` | 2 (text only) | Catches answers that straddle a chunk boundary |

**Do:**

- Always chunk on **sentence or segment boundaries**, never on word/byte
  counts. Mid-thought splits hurt both BM25 (broken keyword spans) and
  embeddings (incomplete semantics).
- For diarized JSON: respect **speaker boundaries** when at/past target.
  A chunk that mixes Swami ji and a questioner is two different things
  in one embedding — bad for retrieval.
- For plain text: keep the **2-sentence overlap** between consecutive
  chunks. This is what catches answers like "X. And the reason is Y." when
  the chunk boundary falls between the two sentences.
- After chunking a sample, **read 10 random chunks** to spot issues
  (truncation, encoding artifacts, header noise). If they look ugly to
  you, they look ugly to the embedder.

**Don't:**

- Don't shrink chunks below ~200 tokens "because they'll be more precise."
  Sub-paragraph chunks lose context and retrieval quality collapses.
- Don't grow chunks past ~700 tokens to "get more context per chunk."
  bge-m3 produces a single fixed-size vector regardless — bigger chunks
  give you a blurrier vector, not a richer one.

---

## 4. Header enrichment (Phase 12, biggest accuracy lever)

Every chunk starts with a header line that the embedding model **sees
verbatim**. After Phase 12, this header includes the parsed path metadata:

```
[Event: 01 NOIDA 7 - 10 JAN 2010 | Date: 2010-01-07 | Time: 18:00 | Track: PRAVACHAN | Type: discourse | Season: winter]
[Source: 04 PRAVACHAN.json | 00:00:00 -> 00:05:23 | Speakers: Swami ji]
<body text>
```

Why this matters: a query like "what did Swami ji say in Noida about
barsat" now matches semantically even *without* the filter, because
the chunk literally contains "NOIDA" and "Swami ji" near the topic body.
Filters then sharpen it further.

**Do:**

- Always invoke the chunkers with `--base-dir` (or set
  `RAW_TRANSCRIPTS_BASE_DIR`). Without it, header enrichment is a no-op
  and you lose the precision boost.
- Keep the header **compact**. The header eats from the same token budget
  as the body — current format is ~30 tokens, which is fine. Don't bloat
  it with every available field.
- If you add a new metadata field worth retrieving on, add it to **both**
  the header (so embeddings see it) **and** the payload (so filters can
  enforce it).

**Don't:**

- Don't add fields to the payload only — without header inclusion the
  field works as a hard filter but contributes nothing to ranking when
  the filter isn't set.
- Don't include fields that are noisy or unreliable. A wrong header is
  worse than no header (pollutes the embedding).

---

## 5. Embedding

We use **bge-m3** via Ollama (1024-dim, multilingual, 8K context, hybrid-
friendly). It's the right choice — don't substitute without re-running
the eval suite.

**Do:**

- **Pin the model.** Use the exact tag (`bge-m3` at its current Ollama
  digest) and record the digest in `doc.md` when ingestion begins.
  If you re-ingest later with a different version, vectors are not
  comparable — old and new chunks become functionally separate corpora.
- **Use the same model for queries and documents.** bge-m3 is symmetric,
  so no separate "query" model. Don't add prefixes ("query:", "passage:")
  the way E5 wants — bge-m3 doesn't.
- **Batch embeddings during ingestion.** Default batch size is 32 — bump
  to 64 if VRAM allows; that's the single largest ingestion-speed knob.
- **Verify embedding finiteness** before upsert (already done in
  `bulk_ingest_hardened.py` via `vec_is_finite()`). NaN/Inf in even one
  dimension breaks the entire HNSW index.

**Don't:**

- Don't quantize embeddings yourself; let Qdrant do it (`int8` scalar
  quantization, configured in `qdrant_setup.py`). You get ~4× storage
  savings with negligible recall loss when query vectors stay full-precision.
- Don't normalize embeddings manually — bge-m3 outputs are already
  L2-normalized and cosine distance is what Qdrant uses.

---

## 6. Indexing (Qdrant + Tantivy)

Both indexes are populated by the same ingestion run. Keep them in sync.

**Do:**

- Run `qdrant_setup.py` once before ingestion to create payload indexes.
  Querying without an index for a filter field works but is much slower
  on large collections — and Qdrant won't add the index retroactively
  without re-touching every point.
- Run `analytics_schema.sql` **before** ingestion so the chunk_meta /
  file_meta rows accumulate from the start. The `PostgresWriter` is
  self-disabling if the schema isn't there, but you want the data.
- After ingestion, run `verify_ingestion.py` — sample 1,000 chunks and
  confirm they round-trip cleanly through both Qdrant and Tantivy.
- **Commit Tantivy every 50 files** (already done; don't lower this — too
  many commits hurt throughput) and on graceful shutdown.

**Don't:**

- Don't write to Tantivy from more than one process at a time. It uses a
  filesystem lock; concurrent writers crash with `LockBusy`.
- Don't change `always_ram=True` in Qdrant config unless RAM is genuinely
  insufficient — query latency goes from ~300ms to ~1.5s when filter +
  quantized vectors live on disk.

---

## 7. Querying

This is where Phase 12 pays off most. The LLM-callable filter args on
`search_transcripts()` are the precision lever.

### What the LLM should learn (system prompt cues)

These belong in your model's system prompt (per
[docs/model_config.md](model_config.md)):

- *"If the user mentions a season ('monsoon', 'barsat', 'winter'), pass
  the `season` arg."*
- *"If the user mentions a city or place ('Noida', 'Rishikesh'), pass
  `location`."*
- *"If the user is asking about Swami ji's teachings, default
  `track_type=['discourse', 'address']` to exclude bhajans and music."*
- *"If the user asks about a date or year, pass `date_range`."*
- *"Combine filters — they AND together."*

### Default filter strategy for teaching queries

For most user questions about *what Swami ji said* on topics, the
appropriate default is:

```python
search_transcripts(query, track_type=["discourse", "address"])
```

This excludes:
- Bhajan tracks (lyrics, not teachings)
- Meditation tracks (silence + ambient)
- Music tracks (ENTRY MUSIC, RETURN MUSIC — pure noise)
- Invocations (very short, repetitive content)

You'd drop the filter only if the user explicitly asks for songs
("what bhajans were sung") or wants the full session.

### Query rewriting techniques worth adding

These are not yet implemented but are the next obvious upgrades:

1. **HyDE (Hypothetical Document Embeddings).** Have the chat LLM
   write a *plausible answer* to the question, embed that, and search
   by the answer's embedding. Hugely improves recall when the question
   uses different phrasing than the document.
2. **Multi-query.** Generate 3–5 paraphrases of the user's question
   (different vocabularies — formal, casual, with synonyms), retrieve
   for each, union the candidates, rerank as one set.
3. **Filter extraction.** Run a small LLM pass to extract structured
   filters from the question before searching:
   - "What did Swami ji say in Rishikesh in 2015?" → `location="RISHIKESH"`,
     `date_range=("2015-01-01", "2015-12-31")`.

Add these as separate function tools (`search_with_hyde`,
`search_multi_query`) so the LLM can choose when to use them.

### What to *not* do at query time

- Don't over-filter. If filters reduce the candidate pool below ~10,
  results get sparse and the reranker has less signal. Better to use
  filters as soft preferences via header enrichment.
- Don't pass `top_k > 20`. The reranker is the precision step; asking
  for 50 final results means you're showing the LLM noise.
- Don't bypass the reranker even if it's slow. Disabling it (or letting
  the fallback path fire frequently) drops Hit@5 noticeably.

---

## 8. Evaluation discipline

You can't improve what you don't measure. The eval harness (Phase 9) is
the feedback loop — use it.

**Do:**

- **Grow the golden-query set.** Every time you observe a real user query
  whose result was bad, add it to `eval/golden_queries.yaml` with the
  correct expected source file / chunk. Aim for 200+ queries by the time
  the corpus is fully ingested.
- **Re-run the eval after every change to the retrieval path.** Even a
  one-line change to `bm25_weight` can move Hit@5 by several points.
- **Spot-check sample chunks visually.** Pull 20 random chunks from
  Qdrant, read them. If something looks weird (truncated header, mojibake,
  empty body), trace it back to the chunker / encoding.
- **Track per-type metrics separately.** Quote-finding, single-transcript
  Q&A, cross-corpus, and analytics queries have different ceilings (see
  PRD §10). Don't average them into a single number — diagnose by type.

**Don't:**

- Don't tune `bm25_weight` against a single example. Tune against the
  golden set with at least 30 queries per type.
- Don't ignore retrieval failures because "the LLM still gave a good
  answer." Hallucinated-but-correct is not the same as retrieved-and-cited;
  it'll fail on the next adjacent query.

---

## 9. Corpus-specific quirks (this project)

Specific to the Swami ji transcripts and Phase 12 path metadata:

- **Every audio file has Swami ji's voice.** `primary_speaker="Swami ji"`
  is set on every chunk; speaker-filter UX is mostly moot for this corpus.
  Useful when you eventually add second-speaker tracks (interviews, Q&A
  with guests by name).
- **Music/meditation/invocation tracks add noise.** Their Whisper output
  is either silence, repetitive chants, or random hallucinations from
  ambient sound. The default `track_type=["discourse","address"]` filter
  on teaching queries is *not optional* for good precision.
- **Bhajan lyrics are valuable but separate.** A query like "find the
  bhajan that goes 'na hara hai ishq'" should pass `track_type="bhajan"`
  explicitly. Don't lump them in with discourses.
- **Hindi + English code-mixing.** bge-m3 handles it; don't translate.
  Maintain glossary entries in both scripts/transliterations
  ("barsat" + "बरसात") for queries that use either form.
- **Date math goes through `session_date`, not the filename.** Filenames
  like `04 PRAVACHAN.wav` don't carry dates — the date comes from the
  parent folder. The path parser handles this; if you ever flatten the
  folder hierarchy, you lose the dates.

---

## 10. Anti-patterns (things that look helpful but aren't)

- ❌ **Adding more chunking strategies "to be safe."** Multiple chunk
   sizes means multiple embeddings per source — explodes index size,
   gives marginal recall improvement, complicates dedup.
- ❌ **Crawling for fancier embedding models.** bge-m3 is at the top of
   the open-source multilingual leaderboard for retrieval. Switching to
   a "newer" model that hasn't been benchmarked on retrieval-specific
   tasks usually regresses.
- ❌ **Fine-tuning the embedder on your transcripts.** This is a
   months-long project for a few percentage points. Spend the time on
   ASR quality and chunking instead.
- ❌ **Disabling the reranker because "dense + BM25 was already pretty
   good."** The reranker is the single biggest precision contributor
   after retrieval — keep it.
- ❌ **Storing full transcripts in Postgres instead of `chunk_meta.text`.**
   You'd duplicate ~400 GB to get a fuzzier search than Tantivy provides.
- ❌ **Running ingestion and chat at the same time.** Both hit the GPU.
   bge-m3 + qwen 2.5 7B + reranker pushed to 12 GB will OOM and crash
   one of them. Ingest first, query second.

---

## 11. Quick troubleshooting decision tree

> *"My query isn't returning what I expect."*

1. Is the expected chunk in the index at all?
   - `python -m ingestion.verify_ingestion`
   - If no → ingestion problem, check `dead_letter/` and `ingest_progress.sqlite`.

2. Is it returned by **dense** alone? (Bypass BM25.)
   - Add temporary `bm25_weight=0` to the function tool valves and re-query.
   - If yes here but not in combined → BM25 is dominating; lower
     `bm25_weight` or drop fewer Tantivy hits.

3. Is it returned by **BM25** alone? (`bm25_weight=1`.)
   - If yes here but not in combined → dense is dominating; check the
     embedding (encoding artifacts? truncated header?).

4. Is the **filter** excluding it?
   - Try the query with no filters. If suddenly it appears → check the
     payload of the missing chunk in Qdrant. The filter field probably
     isn't populated (path parser may have failed for that file —
     look in `ingest.log` for `path_parse` warnings).

5. Is the **reranker** demoting it?
   - Pull the top-40 RRF results (before rerank) and look for it manually.
     If it was in top-10 RRF but not top-8 after rerank → that's the
     reranker doing its job; either query phrasing isn't close enough,
     or the chunk genuinely isn't the best answer.

6. Still missing?
   - Was the speech mis-transcribed? Search for the *expected* phrase
     in the source `.json`/`.txt` files via `grep`. If absent → ASR
     problem; add a glossary entry and re-transcribe that file.

---

## 11.5. Content-based tagging (Phase 13)

Phase 13 adds a *content-derived* metadata layer that lives next to the
path-derived Phase 12 layer. They answer different questions and you
should reach for the right one:

| Question type | Layer | Field |
|---|---|---|
| "When was this recorded?" | Path | `session_date`, `season`, `year` |
| "Where was this recorded?" | Path | `location`, `event_id` |
| "What kind of session was it (per folder)?" | Path | `track_type` |
| "What kind of session does it sound like?" | Content | `event_type` |
| "What language was spoken?" | Content | `primary_language` |
| "What was it about?" | Content | `topics`, `summary_hindi`, `summary_english` |
| "Who did Guruji mention?" | Content | `people_named` |
| "What scriptures were cited?" | Content | `scriptures_referenced` |
| "Verbatim hints at time/place inside the audio?" | Content | `timing_clues`, `location_clues` |

Phase 12 metadata is free, deterministic, and instant — always available
the moment ingest finishes. Phase 13 metadata is generated by Qwen 2.5 7B
reading the whole transcript and is slow (~20–60 sec per file on GPU,
impractical on CPU). It runs as a **separate resumable pass** after
ingest, so search keeps working even mid-enrichment — content-tag filters
just become available file-by-file as tagging completes.

**When to use each filter.**

- Prefer **path filters** when the question is *intrinsic* to the
  recording: "discourses from the Noida camp" → `location="NOIDA",
  track_type="discourse"`. Free and 100% accurate.
- Use **content filters** when the question is about *what was said*:
  "where did Guruji talk about the Bhagavad Gita?" →
  `scriptures_referenced=["Bhagavad Gita"]`. Subject to model
  recall — see prompt-tuning notes below.
- **Combine them** for precision: "monsoon-day satsangs that mention
  karma yoga" → `season="monsoon", event_type="satsang",
  topics=["karma-yoga"]`. The intersection is usually small enough
  to feed straight into reranking.

**Prompt-tuning checklist for content tags.**

1. *Verbatim quote enforcement.* `timing_clues` and `location_clues` are
   supposed to be exact phrases from the transcript. The prompt says so
   explicitly. If you find paraphrases sneaking in, tighten the prompt
   with one negative example: "BAD: 'morning practice'. GOOD: 'आज सुबह
   ध्यान के समय'."
2. *Hindi script integrity.* `summary_hindi` must be Devanagari. If the
   model returns transliterated Latin script ("guru ji ne karma yoga par
   pravachan diya"), update the prompt to add: "Output MUST use देवनागरी
   characters, not Latin transliteration."
3. *Topic cardinality.* Prompt says 3–5 topics, lowercase, hyphenated.
   Too many topics dilutes filtering. If the model returns 10+, lower
   the cap explicitly: "EXACTLY 3, NEVER 4 or more."
4. *Enum drift.* The validator rejects unknown `event_type` /
   `primary_language` values. If you see frequent dead-letter hits for
   `"event_type": "lecture"` or similar, decide whether to expand the
   enum or hold the line. Default: hold the line — drift defeats the
   whole point of an enum.

**Cost expectations.**

| Hardware | Per file | 1k files | 100k files | 1M files |
|---|---|---|---|---|
| RTX 4090 single GPU | ~20s | ~5.5 hr | ~23 days | ~7.6 mo |
| RTX 3090 single GPU | ~30s | ~8.3 hr | ~35 days | ~11 mo |
| 8× RTX 3090 | ~30s/GPU | ~1 hr | ~4.4 days | ~6 weeks |
| CPU only (impractical) | ~8–15 min | weeks | years | decades |

For the test slice (a few GB, hundreds of files): plan for **a few
hours**, then evaluate whether the tag quality is worth scaling to the
full corpus. If the prompt needs tuning, you've only thrown away a few
hours — not weeks.

**Failure modes to watch.**

- Truncated JSON output → bumped `num_ctx`, increased
  `--max-tokens-single-pass`, or shorten summaries in the prompt.
- Bad JSON → dead-lettered under `dead_letter/tag_failures/`. Inspect
  the raw output; if a model formatting quirk recurs, tweak
  `parse_model_json()` to handle it.
- `set_payload` failures → file_meta is already updated; re-run the
  enrichment script with a custom filter (or a small one-off script) to
  re-propagate. Not catastrophic — search degrades to "content filter
  doesn't match this file" rather than wrong results.

---

## 12. References

- [PRD.md](../PRD.md) — full spec, locked decisions (§3), accuracy ceilings (§10)
- [docs/runbook.md](runbook.md) — operational procedures
- [docs/troubleshooting.md](troubleshooting.md) — symptom → fix map
- [docs/architecture.md](architecture.md) — system layout
- [bge-m3 paper](https://arxiv.org/abs/2402.03216) — embedding model
- [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — reranker
- [Qdrant filtering](https://qdrant.tech/documentation/concepts/filtering/) — payload-filter reference

---

*Last updated: Phase 13 (2026-05-18). Update when retrieval-path code,
default filters, or the content-tag schema change.*
