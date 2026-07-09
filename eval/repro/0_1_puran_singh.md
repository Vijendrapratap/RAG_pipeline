# 0.1 — Reproduction: "puran singh" retrieval

**Plan ref:** `docs/plan_foundation.md` §0.1. **Run:** 2026-07-09 against the live
stack (rag-api :8081, `embed_model=bge-m3`, `chat_model=qwen3.5:9b`, `reranker=true`).
**Method:** `POST /api/search` (scope=chunks, backend=hybrid, top_k=10) for the
chunk-level reranker scores; `POST /api/query` for the synthesized answer. Auth via
`X-Dashboard-Password`. Reproduce with `eval/repro/run_0_1.py`.

## Query

> `search some discourse about puran singh`

## Retrieved chunks (reranker `relevance_score`, best first)

`retrieval_ms ≈ 604`, `count=10`, `backend=hybrid`.

| rank | score | mentions Puran Singh? | source_file |
|---:|---:|:--|:--|
| 0 | **0.0706** | ✅ `वारेपूर्ण सिंग` / `वारी पूरन सिंग` | `…16 HOTEL KEYS LUDHIANA…/18 SEP…1130 AM/04 NA GARZ MUJHEY HARAM SE.json` |
| 1 | 0.0264 | ❌ homograph `पुरन`=*complete* (a MEDITATION track) | `…15 DWARKA DELHI…/02 MEDITATION.json` |
| 2 | 0.0241 | ❌ homograph `विश्राम पुरन` | `…16 HOTEL KEYS…/13 QUES - 6.json` |
| 3 | 0.0088 | ✅ `पुरण सिंग यही गाय … वारे वा पुरण सिंग` (densest, on-topic bhajan) | `…16 GURUPURNIMA DELHI 2014…/03 LA PILA RAAT DIN SAKIYA.json` |
| 4–9 | 0.0073–0.0081 | ❌ | `Dagshai 2005/…/02 PRAVACHAN.json` (×6) |

Key spelling note: Whisper renders the name **`पूर्ण/पूरन सिंग`**, not the query's
transliteration `पुरन`. A naïve `पुरन` substring test therefore *misses the top hit*
and false-hits the homograph `पुरन` ("complete") — a trap for any lexical existence gate
(cf. plan §1.5's rejection of a BM25 existence check).

## Answer (`/api/query`, grounded, citation `[1]` → chunk #0)

> Swami ji mentions "Vare Pun Singh" (Puran Singh) … "Masti ke Badshah" (Emperor of
> intoxication) … "Tune loot liya mujko. Narg! Mujhe haram se…" … Puran Singh's master
> became his "Surahi" (water flask) … This account is from "16 HOTEL KEYS LUDHIANA 17 -
> 19 SEP 2010" on September 18, 2010, at 11:30 AM [1].

**Grounding verified** — every quoted fragment is present verbatim in chunk #0:
`मुझे हराम से`, `तूने लूट लिया मुझको`, `मस्ती के बादशाह`, `सुराही/स्राही`, `वारेपूर्ण सिंग`.
Minor narrative embellishment ("surrendered his life entirely"), but no fabrication.

## Classification (per §0.1 rubric)

Neither of the failure buckets the plan anticipated:

- **Not a retrieval miss** — real Puran Singh passages are surfaced (ranks 0 and 3).
- **Not a grounding/citation bug** — the answer is quote-faithful to the cited chunk.
- **Not a refusal** — the model answered.

**On this query the pipeline succeeds today.** The plan's "retrieval failure" framing is
**stale** (predates the `59fbd3d` metadata backfill).

## The live risk this exposes — score calibration (confirms §0.2 gates §1.5)

The whole reranker distribution for a legitimate cross-script query sits at **0.007–0.07**
(the `prob_faced.md §3.1` "0.01–0.18" claim is right; the "Puran Singh reranked 0.88" claim
is **not** reproduced). Yet `cite_min_score = 0.75` (`rag_api/config.py:136`) is ~10× the
*entire* realistic range.

Consequence: `filter_min_score` (`synthesis.py:174-194`) drops **all** results and today
only answers via its `or results[:1]` safety net at `:194`. If §1.5 removes that net behind
`RAG_ALLOW_ABSTAIN` **without** recalibrating the floor from §0.2's histogram, the system
will abstain on **100 %** of queries — including this working one.

Two secondary observations for later stages (not §0 work):
1. **Reranker homograph confusion** — the `पुरन`=*complete* meditation chunk (0.0264)
   outranks the on-topic Puran Singh bhajan (#3, 0.0088). A reranker-quality issue.
2. `पुरन सिंग` is a **positive** for §3.2's golden set (never a negative), as the plan states.
