# §0.2 — Reranker score histogram & the citation floor

**Plan ref:** `docs/plan_foundation.md` §0.2 (BLOCKS §1.5). **Run:** 2026-07-09, live
stack, `reranker=bge-reranker-v2-m3`. **Reproduce:** `python -m eval.score_histogram`
over the 30 known-good queries in `eval/histogram_queries.yaml`. Raw report:
`eval/results/score_histogram_*.json`.

## What was measured

For each of 30 hand-picked **known-good** queries (name / quote / thematic, Devanagari +
Latin), the reranker `relevance_score` of the **top hit** and of **hit #4**. `/api/search`
returns the post-rerank score and applies **no** `filter_min_score`, so this is the raw,
unfiltered reranker distribution — the exact quantity `cite_min_score` gates on.

## Top-hit distribution (n=30)

| min | p05 | p10 | p25 | median | p75 | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.040 | 0.167 | 0.307 | 0.700 | **0.909** | 0.979 | 0.992 | 0.997 |

By script: **Devanagari median 0.926** (min 0.239, n=20) · **Latin median 0.704**
(min 0.040, n=10). By type: quote 0.982 · name 0.921 · thematic 0.866.

The reranker separates **well** for most legitimate queries (median 0.91) — but there is a
**long low tail**, entirely from cross-script mismatch:

| query | script | top-hit |
|---|---|---:|
| `surat shabd yoga` | latin | **0.040** |
| `what is maya illusion` | latin | 0.108 |
| `सुरत शब्द योग क्या है` | deva | 0.239 |
| `what is naam simran` | latin | 0.315 |
| `bhakti and divine love` | latin | 0.375 |
| `puran singh discourse` | latin | 0.622 |

This resolves the §0.1 puzzle: the reranker isn't globally weak. The 0.07 score there came
from the **verbose English phrasing** "search some discourse about puran singh"; the plain
`who is puran singh` scores **0.96** and `पूरन सिंग का प्रसंग` scores **0.94**.

## The citation floor — findings

**`cite_min_score = 0.75` is badly miscalibrated.** It retains only **21/30 (70%)** of
known-good **top hits** — it would drop the single best chunk for 9 legitimate queries. Today
that regression is invisible only because `filter_min_score` falls back to `results[:1]`
(`synthesis.py:194`). Remove that net in §1.5 without lowering the floor ⇒ **abstain on ~30%
of good queries.**

Floor retention (keep the legit top hit; trim the weak tail):

| floor | top-hit kept | hit#4 kept |
|---:|---:|---:|
| 0.02 | 30/30 (100%) | 30/30 (100%) |
| **0.03** | **30/30 (100%)** | 29/30 (97%) |
| 0.05 | 29/30 (97%) | 28/30 (93%) |
| 0.10 | 29/30 (97%) | 24/30 (80%) |
| 0.75 (current) | 21/30 (70%) | 13/30 (43%) |

## Recommendation

1. **Lower `cite_min_score` from `0.75` to `0.03`.** This keeps 100% of known-good top hits
   (lowest legitimate cross-script top hit = 0.040) while still trimming the very weakest tail.
   This alone fixes the live false-drop the §0.1 net is currently masking.

2. **A single reranker-score floor is NOT a sound abstention gate on its own.** The
   legitimate cross-script tail (0.04–0.32) overlaps the noise floor, so *no* single threshold
   both keeps cross-script hits and abstains meaningfully. Per the plan, this measurement
   "decides whether abstention ships" — the verdict is: **keep `RAG_ALLOW_ABSTAIN=false` by
   default**, and do not certify a floor for abstention until §3.2's **negative set** places it
   on the false-answer vs false-abstain trade-off. §0.2 sets the floor to stop dropping good
   hits; §3.2 decides whether abstention is safe to enable.
