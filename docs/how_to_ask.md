# How to ask the AI — a phrasing cookbook

A practical guide to getting good answers out of the transcript-RAG
dashboard. The retrieval layer is precise (Qdrant + Tantivy + reranker
routinely hit 0.99+ on exact-phrase queries), so when an answer feels
wrong it is almost always the **way the question was phrased**, not a
data problem. This file collects the patterns that work, the
anti-patterns that don't, and the dashboard toggles to pair with each.

For deployment / install, see [user_guide.md](user_guide.md). For the
endpoints behind the dashboard, see [dashboard.md](dashboard.md).

---

## TL;DR rules

1. **Ask a full question, never just a name or a phrase.** A title alone
   ("AA HIMMAT KA EK KADAM") gives the LLM nothing to answer — it will
   summarise the *content* of the matched chunk instead of telling you
   *when / where / who*.
2. **Use the right endpoint for the question type.**
   `/api/query` (= the dashboard's *answer pane*) is for content
   questions. `/api/search` (= the dashboard's *result cards*) is for
   metadata lookup and "where is this from" questions.
3. **Toggle `Find quote` only when hunting an exact phrase.**
   Default mode (dense + BM25 fusion) is right for everything else.
   `Find quote` shifts weighting toward BM25 — perfect for exact-text
   hunts, but it does **not** change how the LLM writes its answer.
4. **Look at the result cards under the answer.** Every chunk's header
   line carries the full metadata (`[Event: … | Date: … | Time: … |
   Track: … | Type: … | Season: … | Speakers: …]`). When the LLM's
   prose doesn't surface a field, the cards always will.

---

## The four question types

### Type 1 — Content questions ("what does Swami ji teach about X?")

This is what `/api/query` is built for. Default mode handles it.

**Phrasing template** — full sentence, normal mode:

```text
What is meditation?
What does Swami ji say about karma yoga?
How should one approach individual meditation?
```

**Working example from this session:**

> Q: *"What is meditation?"*
> A: *"Meditation is an active process that can lead to mental peace
> and good health [1][2]. It is described as a mohabbat (a form of love)
> towards oneself, involving a direct experience of one's own
> existence [2]…"*

Three citations, prose answer, citations clickable in the UI.

---

### Type 2 — Metadata lookup ("when / where / who / what time")

Ask a **full sentence** with the question word. Leave `Find quote`
**OFF** — the title is usually a rare-enough token that BM25 dominates
the fusion naturally; tipping further toward BM25 doesn't help and
sometimes makes the LLM ignore the question.

**Phrasing template:**

```text
When and where was {TRACK NAME} played, and at what time?
Which event was {TRACK NAME} recorded in?
What was the date and time of the {TRACK NAME} discourse?
```

**Working example from this session:**

> Q: *"When and where was AA HIMMAT KA EK KADAM played and at what time?"*
> A: *"AA HIMMAT KA EK KADAM was played on 12–14 January 2010 at
> PITAMPURA DELHI during winter season [N3]. The specific timing
> mentioned is 18:00 [N3]."*

If the prose answer ever skips a field you wanted (Type / Speakers /
Season), **look at the cited card below the answer** — those fields are
always in the chunk header.

Works equally well in Hindi:

> Q: *"आ हिम्मत का एक कदम कब और कहाँ बजाया गया"*
> A: *"आ हिम्मत का एक कदम पिताम्पुरा दिल्ली में 12 - 14 जनवरी 2010 को बजाया गया।"*

---

### Type 3 — Paste-and-locate ("where is this line from?")

**Do not use the answer pane for this.** The LLM will either echo the
quote back at you or reply "this is from [1]" — neither is useful. The
right tool is the **Search tab itself**: paste the line, toggle
`Find quote` ON, hit search, and read the metadata header on the
top result card.

**Working example from this session** — pasted the line
*"meditation is an active vast medicine for mental peace and good
health"* with `find_quote: true` against `/api/search`:

```text
score: 0.9962
[Event: 02 PITAMPURA DELHI 12 - 14 JAN 2010 | Date: 2010-01-12 |
 Time: 10:30 | Track: SAMBODHAN | Type: address | Speakers: Swami ji]
[Source: 04 SAMBODHAN.json | 00:00:07 -> 00:02:34]
```

Everything you need is in those two lines: file, event, date, time,
track, speaker, timestamp range. Scores above 0.95 = near-verbatim
match; 0.80–0.95 = very close paraphrase; below 0.70 = the corpus may
not have that exact line.

**Curl equivalent** (no LLM in the loop):

```bash
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"the exact line you pasted","find_quote":true,"top_k":3}'
```

---

### Type 4 — Cross-corpus / topic questions ("which tracks talk about X?")

These need `scope: "summaries"` or `scope: "two_stage"` — they search
one vector per *file* instead of one vector per *chunk*, so one talk
about karma yoga ranks above the single chunk that happens to mention
the word in passing.

**These don't work in this setup yet** — the `transcript_summaries`
Qdrant collection has not been built. Until you run
`python -m ingestion.build_summary_index`, summary-scoped queries
return empty. Use chunk scope and accept that long talks may rank below
short chunks for now.

When the summary index is built, the phrasing is the same as Type 1,
just with the scope switch:

```bash
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"which talks cover karma yoga","scope":"summaries"}'
```

---

## Anti-patterns (what NOT to do)

| Don't do this | Why it fails | Do this instead |
|---|---|---|
| Paste just a title: `"AA HIMMAT KA EK KADAM"` | LLM has no question to answer; it summarises the lyrics | `"When and where was AA HIMMAT KA EK KADAM played?"` |
| Ask a metadata question with `Find quote` on | Find quote shifts retrieval, not synthesis; LLM still tries to interpret the lyrics rhetorically | Leave Find quote OFF for metadata questions |
| Ask `/api/query` to tell you where a pasted line is from | LLM answers *"this is from [1]"* — citation number, no metadata | Use `/api/search` (or the dashboard Search tab) with `find_quote: true` and read the card header |
| Set `scope: "summaries"` right now | No summary index built — silently returns 0 results | Stay on default `chunks` scope until `build_summary_index` runs |
| Add `auto_filters: false` then forget to add filters yourself | Year/date signals in the query are no longer auto-detected; retrieval may miss the right year | Keep `auto_filters: true` (default) unless you know it's misfiring |

---

## Dashboard toggle cheat sheet

| Toggle | What it does | When to flip |
|---|---|---|
| `Find quote` | Retrieval weighting → BM25-heavy (0.85 vs 0.65) | Exact-phrase hunt only |
| `scope: chunks` (default) | Hybrid search across all chunks | Almost everything |
| `scope: summaries` | One result per *file*, ranked by summary match | Cross-corpus topic questions **(needs summary index)** |
| `scope: two_stage` | Summary picks files → chunks within them | Topic questions where you also want exact passages **(needs summary index)** |
| `auto_filters` (default ON) | Pulls year / date / season / place / topic signals from the query and applies strong ones automatically | Leave on. Detection is shown in `detected_filters` so nothing is silently applied |
| `expand_query` (HyDE) | LLM writes a hypothetical passage; its embedding is mean-pooled with the query vector before dense search. Costs one extra LLM call (~1–3 s) | Vague / under-specified queries where dense retrieval is the weak link |
| `answer_language: auto` (default) | Detects Hindi vs English from the query (Devanagari ratio) | Leave on. Force `hindi` / `english` only if you want to override |
| `stream: true` | SSE stream — `meta` event then `token` deltas | Long answers; lets you start reading before generation finishes |

---

## Working query examples (copy-paste tested against this stack)

All of these returned good answers in the session that produced this
file. They cover all four question types.

```text
# Type 1 — content
What is meditation?
What is karma yoga?
How does one practice individual meditation?

# Type 2 — metadata
When and where was AA HIMMAT KA EK KADAM played?
What is the track type and speaker for SAMBODHAN?
At what time was the meditation track on 8 January 2010 held?

# Type 2 — Hindi
आ हिम्मत का एक कदम कब और कहाँ बजाया गया?
मेडिटेशन ट्रैक कौन से इवेंट में हुआ था?

# Type 3 — paste-and-locate (use Search tab + Find quote)
Meditation is an active vast medicine for mental peace and good health
आहिम्मत का एक कदम बढ़ा, प्रभु बाहों में तुझी को लेलेंगे
```

---

## Why the synthesis layer occasionally misses metadata

The system prompt for `/api/query` (see
[`rag_api/synthesis.py`](../rag_api/synthesis.py)) is general-purpose:
*"answer the user's question using only the retrieved passages, cite as
`[N]`"*. It is optimised for content questions ("what does Swami ji
teach about X?"), not for structural extraction ("when, where, who").
The retrieved chunks always carry the metadata in the header line, but
qwen2.5:7b at temperature 0.2 sometimes prefers to discuss the *content*
of a match instead of the *metadata wrapper around* it.

Two real ways to make this more reliable:

1. **Use the right endpoint per question type** — the patterns above.
   `/api/search` is unconditionally reliable for metadata; it has no
   LLM involved.
2. **Swap the chat model.** The fine-tuned 26B model in the PRD plan
   handles instruction-following on extraction questions much better
   than 7B. One line in `.env` (`CHAT_MODEL=...`), no code change.

A future improvement would be to specialise the synthesis prompt when
the query contains question words like *when / where / who* — but
that's a deliberate change to `rag_api/synthesis.py`, not something to
work around here.

---

## What you still cannot do (set expectations)

| Capability | Status | Unblocks when |
|---|---|---|
| `scope: summaries` / `two_stage` | Returns empty | `python -m ingestion.build_summary_index` is run after Phase 13 enrichment populates summaries in `file_meta` |
| Hindi full-text analytics (`/api/analytics/*`) at speed | Works, but does a sequential scan | `psql -f infra/postgres/migrations/001_hindi_fts.sql` is run once |
| Questions about content outside the 3 Delhi 2010 events | Returns 0 results (correctly) | More files are ingested |
| 26B fine-tuned answers | Falls back to 7B | The 26B model is fine-tuned and pulled into Ollama |
