"""Stage 3.1 eval harness — runs golden queries through the retrieval pipeline
(bypassing the LLM) and emits Hit@k + MRR + abstention precision/recall, plus a
JSON report under eval/results/<timestamp>.json.

Retargeted onto `rag_api.retrieval.Retriever` (the old copy imported the deleted
`open_webui_functions/` package and could not run). The six private-method pokes
of the old harness collapse into one `Retriever.search(...)` call; analytics
scoring reads dicts from `rag_api.analytics.Analytics` instead of parsing prose.

Query types:
  - quote / qa : a specific chunk is expected (source_file + substring).
  - topic      : every expected source_file must appear in the top-K.
  - analytics  : a Postgres analytics call with numeric / membership expectations.
  - negative   : an invented person/place/event that has no real answer — the
                 system SHOULD abstain. Scores abstention recall.

Abstention (positive class = "abstained") is derived from the citation floor:
a query abstains when its top reranked score < cite_min_score, or nothing was
retrieved. Positives (quote/qa/topic) that abstain are false abstentions — the
regression 1.5 risks and Hit@k cannot see. Reported as first-class P/R.

CLI:
    python -m eval.run_eval --queries eval/golden_queries_corpus.yaml
    python -m eval.run_eval --queries ... --cite-min-score 0.20 --include-catalog
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from rag_api.analytics import Analytics
from rag_api.config import get_settings
from rag_api.retrieval import Retriever

HIT_KS = (1, 5, 10)
# Quote-finding skews BM25 heavier (find_quote spec); qa/topic use the blend.
BM25_WEIGHT_BY_TYPE = {"quote": 0.85, "qa": 0.65, "topic": 0.65, "negative": 0.65}
RETRIEVAL_TYPES = frozenset({"quote", "qa", "topic"})


def _pg_dsn() -> str:
    """Analytics runs on the host; fall back to localhost composed from
    POSTGRES_PASSWORD (same pattern as ingestion)."""
    return os.environ.get("PG_DSN") or (
        f"postgresql://owui:{os.environ.get('POSTGRES_PASSWORD', '')}"
        "@localhost:5432/openwebui"
    )


# ---- retrieval ----------------------------------------------------------


def _retrieve(
    retriever: Retriever,
    query: str,
    qtype: str,
    top_k: int,
    include_catalog: bool,
    bm25_weight_override: float | None,
) -> list[dict[str, Any]]:
    """One embed -> dense+bm25 -> RRF -> rerank pass, best-first result dicts."""
    weight = (
        bm25_weight_override if bm25_weight_override is not None
        else BM25_WEIGHT_BY_TYPE.get(qtype, 0.65)
    )
    return retriever.search(
        query, filters=None, top_k=top_k,
        bm25_weight=weight, include_catalog=include_catalog,
    )


def _abstained(results: list[dict[str, Any]], cite_min_score: float) -> bool:
    """The system abstains when nothing clears the citation floor. Results are
    best-first, so results[0] carries the max reranked score."""
    if not results:
        return True
    return float(results[0].get("score") or 0.0) < cite_min_score


# ---- per-type scoring ---------------------------------------------------


def _match_quote_or_qa(
    results: list[dict[str, Any]], expected_source_file: str,
    expected_chunk_contains: str,
) -> int | None:
    """1-indexed rank of the first chunk matching both source_file and the
    substring; None if absent."""
    needle = expected_chunk_contains.lower()
    for rank, r in enumerate(results, start=1):
        if r.get("source_file") != expected_source_file:
            continue
        if needle in (r.get("text") or "").lower():
            return rank
    return None


def _match_topic(
    results: list[dict[str, Any]], expected_source_files: list[str], k: int,
    mode: str = "all",
) -> tuple[bool, list[str]]:
    """Topic match over the top-K.

    mode="all" (default): every expected file must appear — a precise multi-doc
    test. mode="any": at least one expected file must appear — the right test
    for entity questions whose ground truth is *a set of genuinely-relevant
    files* (Postgres-verified), where retrieval surfacing any one is a hit.
    """
    seen = {r.get("source_file") for r in results[:k]}
    present = [f for f in expected_source_files if f in seen]
    missing = [f for f in expected_source_files if f not in seen]
    passed = bool(present) if mode == "any" else (not missing)
    return passed, missing


def _score_analytics(
    an: Analytics, q: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """(passed, reason, extra). Reads dicts — no prose parsing."""
    name = q["tool"]
    args = q.get("args", {})
    if name == "count_mentions":
        out = an.count_mentions(**args)
        n = int(out["chunk_count"])
        lo, hi = q.get("expected_count_min"), q.get("expected_count_max")
        if lo is not None and n < lo:
            return False, f"count {n} < min {lo}", {"count": n}
        if hi is not None and n > hi:
            return False, f"count {n} > max {hi}", {"count": n}
        return True, f"count {n} within [{lo}, {hi}]", {"count": n}
    if name == "list_transcripts":
        out = an.list_transcripts(**args)
        found = {t.get("source_file") for t in out["transcripts"]}
        expected = set(q.get("expected_source_files") or [])
        missing = expected - found
        if missing:
            return False, f"missing {sorted(missing)}", {"found": sorted(found)}
        return True, f"all present ({sorted(found)})", {"found": sorted(found)}
    if name == "top_speakers":
        out = an.top_speakers(**args)
        names = [s.get("speaker") for s in out["speakers"]]
        exp = q.get("expected_speaker_in_top")
        if exp is None:
            return False, "missing expected_speaker_in_top", {"speakers": names}
        if exp in names:
            return True, f"speaker {exp!r} present", {"speakers": names}
        return False, f"speaker {exp!r} not in {names}", {"speakers": names}
    return False, f"unknown analytics tool: {name!r}", {}


# ---- aggregation --------------------------------------------------------


def _aggregate(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    for q in per_query:
        t = q["type"]
        bt = by_type.setdefault(
            t, {"n": 0, "hits": {k: 0 for k in HIT_KS}, "mrr_sum": 0.0, "pass": 0},
        )
        bt["n"] += 1
        if t in RETRIEVAL_TYPES:
            for k in HIT_KS:
                if q.get(f"pass_at_{k}"):
                    bt["hits"][k] += 1
            bt["mrr_sum"] += float(q.get("reciprocal_rank") or 0.0)
        elif q.get("passed"):
            bt["pass"] += 1

    summary: dict[str, Any] = {}
    for t, bt in by_type.items():
        n = bt["n"]
        if t in RETRIEVAL_TYPES:
            summary[t] = {
                "n": n,
                **{f"hit_at_{k}": (bt["hits"][k] / n) if n else 0.0 for k in HIT_KS},
                "mrr": (bt["mrr_sum"] / n) if n else 0.0,
            }
        else:
            summary[t] = {"n": n, "pass_rate": (bt["pass"] / n) if n else 0.0}
    return summary


def _abstention_metrics(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    """Abstention P/R. Positive class = "the system abstained".

    TP: negative query, abstained (correct).      FP: positive query, abstained
    (false abstention — the 1.5 regression). FN: negative, answered anyway.
    TN: positive, answered. Precision = TP/(TP+FP); recall = TP/(TP+FN).
    """
    tp = fp = fn = tn = 0
    for q in per_query:
        t = q["type"]
        if t == "negative":
            if q.get("abstained"):
                tp += 1
            else:
                fn += 1
        elif t in RETRIEVAL_TYPES:
            if q.get("abstained"):
                fp += 1
            else:
                tn += 1
    n_pos, n_neg = tp + fn, fp + tn  # negatives, positives (by class)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_negative": n_pos, "n_positive": n_neg,
        "precision": precision, "recall": recall,
        "false_abstention_rate": (fp / n_neg) if n_neg else None,
    }


# ---- main ---------------------------------------------------------------


def run(
    queries_path: Path,
    out_dir: Path,
    *,
    top_k: int,
    cite_min_score: float,
    include_catalog: bool,
    bm25_weight_override: float | None,
) -> tuple[dict[str, Any], Path]:
    queries = yaml.safe_load(queries_path.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{queries_path}: expected a non-empty YAML list")

    settings = get_settings()
    retriever = Retriever(settings)
    analytics = Analytics(_pg_dsn())

    per_query: list[dict[str, Any]] = []
    for q in queries:
        qid, qtype = q["id"], q["type"]
        record: dict[str, Any] = {"id": qid, "type": qtype}
        try:
            if qtype in ("quote", "qa"):
                results = _retrieve(retriever, q["query"], qtype, top_k,
                                    include_catalog, bm25_weight_override)
                rank = _match_quote_or_qa(
                    results, q["expected_source_file"],
                    q["expected_chunk_contains"],
                )
                record["n_results"] = len(results)
                record["first_hit_rank"] = rank
                record["reciprocal_rank"] = 1.0 / rank if rank else 0.0
                for k in HIT_KS:
                    record[f"pass_at_{k}"] = rank is not None and rank <= k
                record["abstained"] = _abstained(results, cite_min_score)
                record["top_score"] = (
                    round(float(results[0]["score"]), 4) if results else 0.0
                )
                record["reason"] = (
                    f"matched at rank {rank}" if rank is not None
                    else f"no match in top {len(results)}"
                )
            elif qtype == "topic":
                results = _retrieve(retriever, q["query"], qtype, top_k,
                                    include_catalog, bm25_weight_override)
                expected = q["expected_source_files"]
                mode = q.get("match", "all")
                record["n_results"] = len(results)
                record["match_mode"] = mode
                for k in HIT_KS:
                    passed, missing = _match_topic(results, expected, k, mode)
                    record[f"pass_at_{k}"] = passed
                    if k == 10:
                        record["reason"] = (
                            f"{mode}: passed in top 10" if passed
                            else f"{mode}: missing in top 10: {missing}"
                        )
                # Soft MRR: reciprocal of the worst rank among expected files,
                # only when every expected file was found.
                first_ranks: list[int] = []
                for f in expected:
                    for r_idx, r in enumerate(results, start=1):
                        if r.get("source_file") == f:
                            first_ranks.append(r_idx)
                            break
                record["reciprocal_rank"] = (
                    1.0 / max(first_ranks)
                    if first_ranks and len(first_ranks) == len(expected) else 0.0
                )
                record["abstained"] = _abstained(results, cite_min_score)
                record["top_score"] = (
                    round(float(results[0]["score"]), 4) if results else 0.0
                )
            elif qtype == "negative":
                results = _retrieve(retriever, q["query"], "negative", top_k,
                                    include_catalog, bm25_weight_override)
                abstained = _abstained(results, cite_min_score)
                record["n_results"] = len(results)
                record["abstained"] = abstained
                record["passed"] = abstained
                record["top_score"] = (
                    round(float(results[0]["score"]), 4) if results else 0.0
                )
                record["reason"] = (
                    "correctly abstained" if abstained
                    else f"answered (top score {record['top_score']} "
                         f">= floor {cite_min_score})"
                )
            elif qtype == "analytics":
                passed, reason, extra = _score_analytics(analytics, q)
                record["passed"] = passed
                record["reason"] = reason
                record.update(extra)
            else:
                record["passed"] = False
                record["reason"] = f"unknown type {qtype!r}"
        except Exception as e:
            record["passed"] = False
            record["error"] = f"{type(e).__name__}: {e}"
            record["reason"] = "exception during evaluation"
            for k in HIT_KS:
                record.setdefault(f"pass_at_{k}", False)
            record.setdefault("reciprocal_rank", 0.0)
        per_query.append(record)

    report = {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "queries_path": str(queries_path),
        "collection": getattr(settings, "qdrant_collection", "transcripts"),
        "params": {
            "top_k": top_k,
            "cite_min_score": cite_min_score,
            "include_catalog": include_catalog,
            "bm25_weight_override": bm25_weight_override,
            "bm25_weight_by_type": BM25_WEIGHT_BY_TYPE,
        },
        "total_queries": len(per_query),
        "by_type": _aggregate(per_query),
        "abstention": _abstention_metrics(per_query),
        "queries": per_query,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return report, out_path


def _print_summary(report: dict[str, Any], out_path: Path) -> None:
    print(f"\nEval report: {out_path}")
    print(f"Total queries: {report['total_queries']}  "
          f"(cite_min_score={report['params']['cite_min_score']}, "
          f"include_catalog={report['params']['include_catalog']})")
    for t, m in report["by_type"].items():
        if t in RETRIEVAL_TYPES:
            print(f"  {t:9s} n={m['n']:2d}  "
                  f"hit@1={m['hit_at_1']:.2%} hit@5={m['hit_at_5']:.2%} "
                  f"hit@10={m['hit_at_10']:.2%}  mrr={m['mrr']:.3f}")
        else:
            print(f"  {t:9s} n={m['n']:2d}  pass_rate={m['pass_rate']:.2%}")
    ab = report["abstention"]
    p = "n/a" if ab["precision"] is None else f"{ab['precision']:.2%}"
    r = "n/a" if ab["recall"] is None else f"{ab['recall']:.2%}"
    far = ("n/a" if ab["false_abstention_rate"] is None
           else f"{ab['false_abstention_rate']:.2%}")
    print(f"  abstention  neg={ab['n_negative']} pos={ab['n_positive']}  "
          f"precision={p} recall={r}  false_abstention_rate={far} "
          f"(TP={ab['tp']} FP={ab['fp']} FN={ab['fn']} TN={ab['tn']})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 3.1 retrieval eval harness.")
    p.add_argument("--queries", required=True, help="Path to a golden YAML set.")
    p.add_argument("--out-dir", default="eval/results",
                   help="Directory for JSON reports (default eval/results).")
    p.add_argument("--top-k", type=int, default=40,
                   help="Reranked results to score within (default 40 = "
                        "candidates_per_source; must be >= max Hit@k).")
    p.add_argument("--cite-min-score", type=float, default=None,
                   help="Citation floor for abstention (default settings value).")
    p.add_argument("--include-catalog", action="store_true",
                   help="Fold curated-catalog points into the candidate pool "
                        "(defaults off — set explicitly per plan 3.3).")
    p.add_argument("--bm25-weight", type=float, default=None,
                   help="Override the per-type BM25 weight for all query types.")
    p.add_argument("--quote-hit5-threshold", type=float, default=0.80,
                   help="CI gate: quote-type Hit@5 must be >= this (PRD 0.80).")
    args = p.parse_args(argv)

    cite_min_score = (
        args.cite_min_score if args.cite_min_score is not None
        else get_settings().cite_min_score
    )
    report, out_path = run(
        Path(args.queries), Path(args.out_dir),
        top_k=args.top_k, cite_min_score=cite_min_score,
        include_catalog=args.include_catalog,
        bm25_weight_override=args.bm25_weight,
    )
    _print_summary(report, out_path)

    quote_metrics = report["by_type"].get("quote")
    if quote_metrics and quote_metrics["hit_at_5"] < args.quote_hit5_threshold:
        print(
            f"\nFAIL: quote-type hit@5 = {quote_metrics['hit_at_5']:.2%} "
            f"< threshold {args.quote_hit5_threshold:.0%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
