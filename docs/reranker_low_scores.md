# Why reranker scores read ~0.01 and the order "feels wrong"

A technical explainer for the case where the answer is **good** but the citation
scores look alarmingly low (e.g. `0.014`, `0.013`, … `0.010`) and the passage the
model actually used (`[8]`, score `0.010`) is ranked **below** a passage it didn't
really need (`[1]`, score `0.014`).

**Short version: this is not a bug.** The number is the raw cross-encoder
relevance score, and it is low because the *query* was romanized Hindi + English
scored against a *Devanagari* corpus. In that regime the scores collapse into a
narrow near-zero band, and inside that band the ordering is noise — so don't read
`0.014 > 0.010` as "[1] is more relevant than [8]."

---

## 1. The exact case

**Query (as typed):**
> `Man ko kaise control kre` / `ye kaha and kab bola gaya tha aur swami ji ne kya kya bola tha is bare mai`

**Result list (rerank scores):**

| # | Track | Type | Score |
|---|---|---|---|
| 1 | NA MAIN BANDA NA KHUDA THA | bhajan | 0.014 |
| 2 | NA MAIN BANDA NA KHUDA THA | bhajan | 0.013 |
| 3 | NA MAIN BANDA NA KHUDA THA | bhajan | 0.013 |
| 4 | MAN PANCHI TU UR JA VATAN MEIN | bhajan | 0.012 |
| 5 | MEDITATION (WITHOUT OM) | bhajan | 0.011 |
| 6 | MEDITATION (WITHOUT OM) | bhajan | 0.011 |
| 7 | SAMBODHAN & PRAVACHAN | bhajan | 0.010 |
| 8 | **MEDITATION** | **meditation** | 0.010 |

The model's answer was correctly grounded in **[7] and [8]** — the meditation /
pravachan passages that actually teach "you are a seer, not the mind; watch it,
don't do mental exercise." Those are the **lowest-scored** results.

So two things look wrong:
- **(a)** every score is ~0.01 (looks like "nothing matched"), and
- **(b)** the most useful passage is ranked last.

Both have the same root.

---

## 2. How the score is produced (code-grounded)

The displayed score is the **raw** `relevance_score` from the
`bge-reranker-v2-m3` cross-encoder (served by Infinity), passed straight through
with no normalization:

- [`retrieval.py:551-569`](../rag_api/retrieval.py#L551-L569) — POSTs
  `{query, documents}` to the reranker and reads `relevance_score` verbatim
  (`score = float(item["relevance_score"])`).
- [`retrieval.py:555`](../rag_api/retrieval.py#L555) — the **raw user query** is
  what's scored: `"query": query`. There is no transliteration or cleanup before
  this call.
- The UI shows that exact float.

`bge-reranker-v2-m3` emits a logit; Infinity applies a **sigmoid**, so the score
is a relevance probability in `[0, 1]`. Reading it backwards:

```
score 0.010  ->  sigmoid(x) = 0.010  ->  x ≈ -4.6   (a confident "weak match")
score 0.50   ->  neutral
score 0.90+  ->  a strong match
```

So `~0.01` is **not** "the reranker is broken" — it's the cross-encoder saying,
fairly confidently, *"as this query is phrased, none of these passages is a strong
match."* The question is **why** it's so sure they're weak.

---

## 3. Root causes

### 3.1 Cross-script mismatch — the dominant cause
The query is **romanized Hindi + English** (`Man ko kaise control kre … aur swami
ji ne kya kya bola …`). The documents are **Devanagari** (`मन … देखना … you are a
seer`). `bge-reranker-v2-m3` is multilingual, but a Latin-script query against a
Devanagari document is a *weak* pairing — the model never sees the surface
overlap it relies on, so every pair scores near the floor. This alone pushes all
eight scores into the 0.01 band.

Why it isn't caught today: the only Devanagari-aware logic is **quote detection**
([`query_parse.py:184-222`](../rag_api/query_parse.py#L184-L222)), which is
deliberately conservative and fires only on a *mostly-Devanagari* pasted passage.
A romanized question has ~0 Devanagari ratio, so it is **not** transliterated or
cleaned — the full romanized string goes to retrieval and rerank as-is.

### 3.2 Code-mixed, compound, meta-query
The query bundles three intents in two scripts:
1. the actual question — "how to control the mind",
2. a provenance ask — "where and when was it said",
3. a content ask — "what all did Swami ji say".

A cross-encoder scores *one* `(query, passage)` pair. Handed a long mixed-intent
string, it can't anchor on the core question, which further depresses and flattens
the scores. (Contrast: a clean `"मन को कैसे नियंत्रित करें"` would score far higher.)

### 3.3 Noisy ASR + bhajan dilution (explains the *order*)
The corpus is Whisper ASR of live discourses. Two effects:
- **Dilution:** each chunk is long and padded with repetitive filler
  (`आध्यात्मिक प्रवचन और भजन …` repeated, devotional/poetic lines, ASR garble).
  The few instructional sentences are a small fraction of the text, so even the
  genuinely-teaching passage `[8]` looks "mostly off-topic" to the cross-encoder.
- **Lexical density fooling rank:** the bhajan `NA MAIN BANDA NA KHUDA THA` `[1]`
  is *saturated* with mind-words (`माइंड`, `मन`, `माइंड के हाथ`, `माइंड की दुनिया`).
  It's a *song about the singer's realization*, not a how-to — but its sheer
  density of mind-tokens nudges it a hair above the instructional pravachan.

### 3.4 Score compression → the ordering is noise
This is the key to "[1] 0.014 ranked over [8] 0.010 feels wrong." When every
result lands in `0.010–0.014` (a **0.004** spread on a `[0,1]` scale), the
differences are **within the model's noise floor**. The reranker is effectively
saying *"all weakly relevant, can't tell them apart."* Treat the order inside such
a compressed band as **unreliable** — `0.014` is not meaningfully "better" than
`0.010` here.

### 3.5 Display amplifies the confusion
Showing the raw `0.010` to a user reads as "broken / irrelevant," when it actually
means "weak match *as phrased*." The number is honest but not intuitive for this
corpus/query type. (Same family of issue as the earlier catalog-row "score 0.03"
problem, which was fixed by giving the cross-encoder better *text*; here the lever
is better *query*.)

---

## 4. Why the answer was still correct

Low/мisordered rerank scores did **not** break the answer, because:
1. Retrieval still pulled the right passages into the candidate set (dense bge-m3
   handles cross-script far better than the cross-encoder's surface scoring).
2. **All** top-k passages are sent to the synthesis LLM, not just `[1]`.
3. The relevance-trim (`SYNTH_MIN_SCORE_RATIO=0.2`,
   [`synthesis.trim_by_relevance`](../rag_api/synthesis.py)) did **not** bite:
   threshold = `0.2 × 0.014 = 0.0028`, and every passage (0.010–0.014) clears it,
   so all eight were kept. The trim only fires when one hit *dominates* — not in a
   compressed band like this. Good interaction: it kept `[7]`/`[8]` instead of
   discarding the actual answer.
4. `qwen3.5:9b` then did the semantic selection the cross-encoder couldn't, and
   grounded the answer in `[7]`/`[8]`.

So: **retrieval surfaced the right content; the reranker just couldn't *order* it
confidently; the LLM recovered.**

---

## 5. What the low score does NOT mean
- ❌ "The data is wrong / irrelevant." — It's relevant; the *query↔doc scoring* is weak.
- ❌ "The reranker is broken." — It's behaving correctly for a hard cross-script input.
- ❌ "0.014 is genuinely a better passage than 0.010." — That gap is noise.

---

## 6. Fixes / levers (ranked by impact)

### Lever 1 — Transliterate the romanized query to Devanagari before retrieval + rerank ⭐
The single highest-impact change, and it targets the dominant cause (§3.1).
Detect a romanized/Latin query and transliterate to Devanagari
(`Man ko kaise control kre` → `मन को कैसे कंट्रोल करे`) before building the dense
vector **and** before the rerank call. Expected effect: scores **lift and spread**
(same-script pairs), and the meditation/pravachan passages separate above the
bhajans. Use an OSS transliterator (e.g. `indic-transliteration`, MIT) per the
open-source-only rule. This is exactly the "query translation/normalization before
rerank" lever already flagged in the project's cross-lingual notes.

### Lever 2 — Reduce the query to its core question
Strip the provenance/meta wrapper for *retrieval* (keep it for the *answer*), so
the cross-encoder scores `"मन को कैसे नियंत्रित करें"` rather than the full
mixed-intent string. Mirrors the existing quote tail-strip, generalized to
questions.

### Lever 3 — Make the displayed score intuitive
Don't show a bare `0.010`. Options: a **relevance band** (strong/medium/weak),
a **min-max-normalized** score within the result set, or hide the number when the
band is compressed (max−min below a threshold) and show "similar relevance"
instead. Honest *and* legible. (Pure UX; no retrieval change.)

### Lever 4 — `track_type`-aware prior for "how-to / teaching" queries
For instructional questions, gently boost `pravachan` / `meditation` over `bhajan`
(or at least surface `track_type` in the UI so it's obvious `[1]` is a *song* and
`[8]` is a *discourse*). Prevents lexical-density songs from topping teaching
questions.

### Lever 5 — Upstream ASR / chunk hygiene
Strip the repetitive `आध्यात्मिक प्रवचन और भजन …` filler and tighten chunks so the
instructional signal isn't diluted (§3.3). Bigger effort, upstream of retrieval.

---

## 7. Recommended next step
Implement **Lever 1 (query transliteration)**. It directly attacks the dominant
cause, is small and localized (one normalization step before the dense-vector and
rerank calls), and should both raise the absolute scores and fix the
bhajan-over-pravachan ordering.

### Quick experiment to confirm the diagnosis
Re-run the **same** question typed in Devanagari, e.g.:
> `मन को कैसे नियंत्रित करें — यह कहाँ और कब कहा गया, और स्वामी जी ने इस बारे में क्या कहा`

Expectation: the meditation/pravachan passages (`[7]`/`[8]`) rise toward the top
and the score band widens well above `0.01`. If they do, §3.1 is confirmed as the
primary cause and Lever 1 is the fix.
