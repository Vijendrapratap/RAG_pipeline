# Plan — four dashboard/RAG improvements (2026-07-15)

User-reported issues and asks, verbatim intent:

1. **Citation clicks land on the wrong turn** in a multi-turn chat, and the
   card shown is sometimes a *summary* when the user expected transcript text.
2. **"hi hello what you can do" should be answered by the normal LLM**, not by
   the RAG pipeline (and today, not even by an LLM — a canned string).
3. **"Summary" requests should return Swami ji's exact words** — verbatim
   bullet lines lifted from the transcript, not LLM-paraphrased prose.
4. **"Best sitting" requests should rank sittings** by (a) frequency of
   query-related words with synonym/bilingual bridging (rain = barish =
   बारिश), (b) longer sittings promoted, (c) how often the sitting has been
   played.

---

## A. Citation anchors are global — clicks scroll to the FIRST turn

**Diagnosis.** `ResultCard` sets `id="cite-{index}"`
(frontend/src/components/ResultCard.tsx:167) and `AnswerPane.onCite` resolves
clicks with `document.getElementById("cite-"+n)`
(frontend/src/components/AnswerPane.tsx:34). Every turn in a thread renders
cards numbered 1..N, so ids collide across turns and `getElementById` always
returns the **first** turn's card. That also explains "shows summary": if turn
1 was a list-of-sittings turn (summary citations), clicking [2] in a later
transcript-grounded turn scrolls up to turn 1's summary card.

**Fix.** Scope anchors per pane: `useId()` in `AnswerPane` → pass
`anchorPrefix` to `ResultCard`; card id becomes `{prefix}-cite-{n}` and
`onCite` looks up the same. `ResultList` (search mode, no [N] markers) passes
no prefix → no id at all → zero collisions. No backend change.

List-intent turns keep summary cards deliberately — they enumerate whole
sittings; the card's `summary` badge says so. Ordinary questions already cite
transcript chunks; after this fix the click lands on them.

## B. Chitchat answered by the actual chat model

**Diagnosis.** The router's `chitchat` class short-circuits to a **fixed
string** (`CHITCHAT_MESSAGE` in rag_api/synthesis.py). Detection is
exact-match against a phrase set, so a combined "hi hello what you can do"
falls through to full RAG.

**Fix.**
* Broaden `is_chitchat` (rag_api/route.py) with a second rule: ≤ 8 words and
  **every** token in a small non-content chitchat vocabulary (greetings +
  meta/capability words + pronouns/fillers, bilingual). Any content word
  (rain, ध्यान, kabir…) breaks the match, so corpus queries can't be swallowed.
* New `RAG_CHITCHAT_LLM` (default **1**): the chitchat handler calls the chat
  model directly — no retrieval, no citations, `think=false` (an 8-second
  think chain on "hi" is absurd) — with a small system prompt describing what
  the assistant can do (search discourses bilingually, list sittings,
  summarize, quote) and the last few history turns for continuity. Streaming
  and non-streaming both supported. **Fail-open:** any model error falls back
  to the existing fixed reply.

## C. Verbatim summary mode — his words, not ours

**Design.** Deterministic intent detection (no LLM needed), same pattern as
`detect_list_intent`:

* `detect_verbatim_intent(query)` — summary/summarize/saransh/सारांश/मुख्य
  बातें/key points/main points/word to word/exact words…
* New plan intent `verbatim` (planner precedence: **best > list > verbatim**;
  deterministic always outranks the LLM plan).
* Retrieval: normal chunk pipeline (planner variants still apply) but
  `top_k = max(req.top_k, RAG_VERBATIM_TOP_K=10)` so the bullets cover more
  of the sitting.
* Synthesis: `build_verbatim_question` replaces the raw query — output ONLY
  bullet points; each bullet an **exact, word-for-word quote** copied from a
  passage TEXT, in its **original language** (never translated, never
  paraphrased), ending with its [N] cite; pick the 5–12 most substantive
  lines; skip irrelevant passages.

## D. Best-sitting ranking

**Data available.** `chunk_meta` has full text + a `'simple'` FTS GIN index
(term frequency per file is one indexed GROUP BY); `file_meta.duration_sec`
exists. **Play counts do not exist** → new migration
`infra/postgres/migrations/004_play_events.sql`:
`play_events(id BIGSERIAL, source_file TEXT NOT NULL, played_at TIMESTAMPTZ
DEFAULT now())` + index. `POST /api/track/played` records one row; the
dashboard's `TrackPanel` fires it once per listen (`onPlay`, deduped per
mount). Fire-and-forget: a failed beacon never breaks playback.

**Ranking (new `rag_api/best.py`).**
1. Candidates: `search_summaries_multi(variants)` — the planner's bilingual
   variants do the synonym bridging (rain/barish/बारिश/वर्षा), the summary
   vectors bridge the rest semantically. Respects detected filters.
2. Stats for candidates only (one query each, `source_file = ANY(...)`):
   * mentions: FTS count of each variant per file, summed (the user's
     "most frequently used word" criterion);
   * length: `file_meta.duration_sec`;
   * plays: `COUNT(*)` from `play_events`.
3. Pure `score_candidates`: components min-max normalized (counts log-scaled),
   composite = **0.50·semantic + 0.25·mentions + 0.15·duration + 0.10·plays**.
   Weights are constants, documented — tune later with evidence, not knobs.
4. Fail-open: Postgres down → stats are 0, semantic order survives, logged.
5. Results carry `mention_count / duration_sec / play_count` in metadata;
   `build_context_block` gains Mentions/Length/Plays lines so the model can
   say *why* a sitting ranked where it did; `build_best_question` asks for a
   best→worst numbered list citing those stats.

**Detection.** `detect_best_intent(query)` — best/top/greatest/sabse
accha/badhiya/श्रेष्ठ/बेहतरीन + a sitting word; optional count (default 10,
same clamps as list intent). New plan intent `best_sittings`.

## Config (three touches, always)

| Flag | Default | Meaning |
|---|---|---|
| `RAG_CHITCHAT_LLM` | 1 | chitchat answered by the chat model (fallback: fixed reply) |
| `RAG_VERBATIM_TOP_K` | 10 | chunk budget for verbatim-summary turns |

Both added to `rag_api/config.py`, `.env`, `.env.example`, **and the
docker-compose environment allowlist** (the RAG_PLANNER lesson). Best/verbatim
intents ride the existing `RAG_PLANNER` flag — they are planner intents.

## Acceptance probes

1. Two-turn chat (list turn, then thematic turn) → clicking [N] in turn 2
   scrolls to turn 2's card (manual browser check; ids verified in built JS).
2. `"hi hello what you can do"` → `query_class=chitchat`, 0 citations, answer
   is model-generated (varies run to run), fast.
3. `"summary of the sitting on rain"` → plan intent `verbatim`, bullets are
   verbatim substrings of the cited passages, in original language.
4. `"best sittings on rain"` → plan intent `best_sittings`, ranked list with
   mention counts/duration/plays visible in citations' metadata.
5. Non-regression battery: "hi"→chitchat, gibberish→abstain,
   "puran singh ke baare mein"→name, "बारिश पर 5 प्रवचन बताओ"→5-entry list,
   ध्यान thematic→grounded Hindi answer.

## Out of scope

Transliteration + Tantivy reindex (own phase); groundedness calibration;
weight tuning for the best-sitting composite (needs eval data, not guesses).
