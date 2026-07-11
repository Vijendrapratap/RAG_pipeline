# Stage 4.1 — deterministic query router A/B

Recorded against the live stack (rag-api :8081, Qdrant `transcripts`=24,414,
reranker up) over `eval/golden_queries_corpus.yaml` (30 queries) and
`eval/router_queries.yaml` (37 labeled queries). Reproduce with the commands at
the bottom. Timestamped JSON reports are gitignored; this `.md` is the record
(same convention as `baseline_stage3.md`).

## 1. Router class accuracy (`router_queries.yaml`, 37 queries)

**36/37 = 97.3%.** Per-class: quote 5/5, thematic 10/10, analytic 6/6,
followup 6/6, **name 9/10**. The single miss is `name08` (नीलिमा — a bare
single-word devotee name with no honorific / suffix / seed entry), which falls
through to `thematic`. That is the documented, expected gap: `people_named` is
mostly romanized and covers ~1/8 of the golden Devanagari subjects, so a
vocab-match would miss it too (see `docs/stage4_kickoff.md` §4.1.b). It is a
*retrieval* concern, not a routing blocker — every retrieval class routes
catalog off, so a name↔thematic slip still captures the headline win.

Deterministic classifier cost: **7.4 µs/query** (pure regex + string ops, no
I/O, no LLM) — i.e. ~0 ms, well inside the "deterministic path adds ~0 ms" bar.

## 2. Retrieval A/B (`golden_queries_corpus.yaml`, top_k=40, cite floor 0.03)

| config | topic hit@1 | topic hit@5 | topic mrr | quote hit@5 | neg pass | abstain P / R |
|---|--:|--:|--:|--:|--:|--:|
| **A — live `/api/query` default** (catalog **on**, bm25 0.65) | 12.5% | 62.5% | 0.046 | 100% | 30% | 100% / **30%** |
| **B — recorded baseline** (catalog off, per-type bm25) | 37.5% | 62.5% | 0.103 | 100% | 40% | 100% / 40% |
| **C — router** (`--route`; per-class settings, no type hint) | **37.5%** | 62.5% | **0.103** | 100% | 40% | 100% / **40%** |

- **The router (C) reproduces the best-known config (B) automatically** — with
  no per-query type hint, purely from `route.classify` — and does **not** regress
  any class: quote hit@5 stays 100% (the CI gate holds under quote→find_quote
  routing), analytics 100%, topic hit@5 62.5%.
- **Against what production actually serves (A), the router is a clear win:**
  topic hit@1 12.5%→37.5% (**+25 pp**), topic MRR 0.046→0.103 (**2.2×**), and
  negative-abstention recall 30%→40% (**+10 pp**). Catalog-on displaces the
  correct transcript files *and* gives negatives spurious high scores (fewer
  abstain); routing catalog off per-class fixes both — the "free accuracy the
  baseline already proved," now applied automatically.
- **Hit@5 ties the baseline** because the entity ceiling (62.5%) is the
  cross-script retrieval gap (3/8 sparse entities never surface), which the
  router does not attack — that is transliteration/expansion work (Deferred /
  4.2). No wired knob moved hit@5 in Stage 3 either.

## 3. Verdict

The deterministic router **ships flag-gated, default off** (`RAG_ROUTER=off`).
It ties the recorded offline baseline and strictly improves on the live
catalog-on default at ~0 ms cost with no regression. Its standing value:

1. It operationalizes catalog-off per-class (recovering the live default's
   −25 pp hit@1 / −10 pp abstention-recall) without a manual flag per request.
2. It is the class-inference substrate 4.2 (HyDE gated to `thematic`) and 4.3
   (follow-up rewrite) build on — those need the class, which this provides at
   97.3% accuracy.

Enabling it by default (or, equivalently, fixing the request-level
`include_catalog` default) is an operator decision backed by this A/B; the code
lands default-off per the Stage 4 governance ("default off until an A/B proves
the win").

## Reproduce

```bash
set -a; source .env; set +a
# A — live default (what /api/query serves today)
python -m eval.run_eval --queries eval/golden_queries_corpus.yaml --include-catalog --bm25-weight 0.65
# B — recorded Stage 3 baseline
python -m eval.run_eval --queries eval/golden_queries_corpus.yaml
# C — router
python -m eval.run_eval --queries eval/golden_queries_corpus.yaml --route
# class accuracy
python - <<'PY'
import yaml; from rag_api.route import classify
rows = yaml.safe_load(open('eval/router_queries.yaml', encoding='utf-8'))
ok = sum(classify(r['query'], bool(r.get('history_present'))) == r['expected_class'] for r in rows)
print(f'router accuracy: {ok}/{len(rows)} = {ok/len(rows):.1%}')
PY
```
