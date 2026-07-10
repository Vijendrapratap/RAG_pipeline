# Stage 3 baseline — retrieval, grid-search, backend comparison

Recorded **before any Stage 4 change**, against the live stack (rag-api :8081,
Qdrant `transcripts`=24,414 / `transcript_summaries`=9,324 / `catalog`=26,423,
reranker up), over `eval/golden_queries_corpus.yaml` (30 queries: 8 entity,
8 quote, 10 negative, 4 analytics). Reproduce with the commands at the bottom.

## 1. Retrieval baseline (`run_eval.py`, defaults: cite_min_score=0.03, catalog off)

| type | n | hit@1 | hit@5 | hit@10 | mrr |
|------|--:|------:|------:|-------:|----:|
| quote (doc2query verbatim) | 8 | 100% | 100% | 100% | 1.000 |
| topic (entity, match=any)  | 8 | 37.5% | 62.5% | 62.5% | 0.103–0.487\* |

\* topic MRR varies with bm25 weight (see grid). **CI gate (quote hit@5 ≥ 0.80) passes.**

- analytics pass_rate = 100% (4/4).
- **abstention** @0.03: precision 100%, recall 40%, false-abstention 0%
  (TP=4 FP=0 FN=6 TN=16).

The entity topic ceiling (62.5% hit@10) is the **cross-script gap** (Stage 0.1
puran-singh failure): 3 of 8 entities never surface a genuinely-relevant file.
Two are simply too sparse to retrieve (नीलिमा = 2 chunks, चेतन विश्वास = 1 chunk).

## 2. Knob grid-search (`grid_search.py`)

Retrieval metrics are flat across bm25/catalog — only MRR moves:

| bm25 | catalog | quote h@5 | topic h@10 | topic mrr |
|------|---------|----------:|-----------:|----------:|
| per-type (0.65/0.85) | off | 100% | 62% | **0.487** |
| per-type | on | 100% | 62% | 0.328 |
| dense 0.50 | off | 100% | 62% | 0.417 |
| bm25 0.85 | off | 100% | 62% | 0.487 |

- **`include_catalog=True` regresses** topic MRR (0.487 → 0.328): catalog rows
  displace the correct transcript files. Confirms the plan's "confound" note —
  keep the default **off**.
- **dense-heavy (0.50) is worse** than the per-type default for entity queries
  (names are lexical); 0.85 is no better than the 0.65 default. **No change.**

### cite_min_score sweep (baseline config) — the deferred-abstention menu

| floor | precision | recall | false-abstention |
|-------|----------:|-------:|-----------------:|
| 0.03–0.10 | 100% | 40% | 0% |
| 0.20–0.30 | 86% | 60% | 6% |
| 0.50 | 89% | 80% | 6% |

Only relevant **if** abstention ships (`RAG_ALLOW_ABSTAIN`, Stream C, currently
deferred). At today's 0.03 default nothing false-abstains. Raising the floor
buys negative-recall at the cost of false-abstaining ~6% of legit (low-score,
cross-script) positives — the exact tension Stage 1.5 must weigh.

## 3. Backend comparison (`bench_backends.py`, live HTTP, top_k=10)

| backend | median ms | mean ms | recall@10 | errors |
|---------|----------:|--------:|----------:|-------:|
| hybrid | 319 | 316 | **68%** | 0/16 |
| pageindex | 1025 | 1140 | 2% | 0/16 |

Hybrid is **3.2× faster and ~34× better recall**. PageIndex has only 3 section
trees built, so it returns nothing for almost every query. **Hybrid stays the
default backend.**

## Verdict

**No wired knob beats the baseline; `include_catalog=on` regresses. Adopt no
change.** The open levers are Stage 4 work: the cross-script entity gap (query
translation / expansion) and the Qdrant int8 search params
(`oversampling`/`rescore`/`hnsw_ef`) that are not wired into the search body
today. Record this table before touching either.

## Reproduce

```bash
set -a; source .env; set +a
python -m eval.run_eval     --queries eval/golden_queries_corpus.yaml
python -m eval.grid_search  --queries eval/golden_queries_corpus.yaml
python -m eval.bench_backends --queries eval/golden_queries_corpus.yaml --top-k 10 --repeat 2
```
