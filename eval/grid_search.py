"""Stage 3.3 knob grid-search — records the retrieval baseline and sweeps the
tunable knobs against the corpus golden set, so any adopted setting is one that
provably beats baseline (PRD §10's sanctioned tuning loop).

Efficient by construction: retrieval is run ONCE per (bm25_weight, include_catalog)
config; the cite_min_score sweep is computed analytically from the recorded top
scores (abstention is a post-retrieval threshold, so it needs no re-retrieval).

Knobs swept here (already wired through Retriever.search / Settings):
  - bm25_weight       : dense/BM25 blend (per-type default vs global overrides)
  - include_catalog   : fold curated-catalog points into the pool (defaults off)
  - cite_min_score    : abstention citation floor (analytic sweep)

Knobs deliberately NOT swept (would need new plumbing, not just a value change —
that is Stage 4 territory, flagged here rather than silently omitted):
  - candidates_per_source, RRF k  : constructor/constant, not per-call args
  - Qdrant params.quantization.oversampling / rescore / hnsw_ef : never wired into
    the search body today (free int8 recall left on the table — a Stage 4 lever)

CLI:
    python -m eval.grid_search --queries eval/golden_queries_corpus.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import yaml

from rag_api.config import get_settings
from rag_api.retrieval import Retriever

from eval.run_eval import (
    BM25_WEIGHT_BY_TYPE,
    HIT_KS,
    RETRIEVAL_TYPES,
    _match_quote_or_qa,
    _match_topic,
    _retrieve,
)

# (label, bm25_weight_override) — None means "use the per-type default".
BM25_CONFIGS: list[tuple[str, float | None]] = [
    ("per-type", None), ("dense0.50", 0.50), ("bm25-0.85", 0.85),
]
CATALOG_CONFIGS = [False, True]
CITE_FLOORS = [0.03, 0.05, 0.10, 0.20, 0.30, 0.50]


def _score_one(retriever: Retriever, q: dict[str, Any], top_k: int,
               include_catalog: bool, bm25_override: float | None) -> dict[str, Any]:
    """Retrieve once and record everything both retrieval and abstention need."""
    qtype = q["type"]
    results = _retrieve(retriever, q["query"], qtype, top_k, include_catalog,
                        bm25_override)
    top_score = float(results[0]["score"]) if results else 0.0
    rec: dict[str, Any] = {"type": qtype, "top_score": top_score}
    if qtype in ("quote", "qa"):
        rank = _match_quote_or_qa(results, q["expected_source_file"],
                                  q["expected_chunk_contains"])
        rec["rr"] = 1.0 / rank if rank else 0.0
        for k in HIT_KS:
            rec[f"hit_{k}"] = bool(rank and rank <= k)
    elif qtype == "topic":
        mode = q.get("match", "all")
        for k in HIT_KS:
            passed, _ = _match_topic(results, q["expected_source_files"], k, mode)
            rec[f"hit_{k}"] = passed
        ranks = []
        for f in q["expected_source_files"]:
            for i, r in enumerate(results, 1):
                if r.get("source_file") == f:
                    ranks.append(i)
                    break
        rec["rr"] = (1.0 / min(ranks)) if ranks else 0.0  # best expected file
    return rec


def _retrieval_metrics(recs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in ("quote", "topic"):
        sub = [r for r in recs if r["type"] == t]
        if not sub:
            continue
        out[t] = {
            "n": len(sub),
            **{f"hit_at_{k}": sum(r[f"hit_{k}"] for r in sub) / len(sub) for k in HIT_KS},
            "mrr": sum(r["rr"] for r in sub) / len(sub),
        }
    return out


def _abstention_at(recs: list[dict[str, Any]], floor: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for r in recs:
        abstained = r["top_score"] < floor
        if r["type"] == "negative":
            tp += abstained
            fn += not abstained
        elif r["type"] in RETRIEVAL_TYPES:
            fp += abstained
            tn += not abstained
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "floor": floor, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "false_abstention_rate": (fp / (fp + tn)) if (fp + tn) else None,
    }


def run(queries_path: Path, out_dir: Path, top_k: int) -> dict[str, Any]:
    queries = [q for q in yaml.safe_load(queries_path.read_text(encoding="utf-8"))
               if q.get("type") in (RETRIEVAL_TYPES | {"negative"})]
    retriever = Retriever(get_settings())

    configs: list[dict[str, Any]] = []
    for bm_label, bm_w in BM25_CONFIGS:
        for cat in CATALOG_CONFIGS:
            recs = [_score_one(retriever, q, top_k, cat, bm_w) for q in queries]
            configs.append({
                "bm25": bm_label, "include_catalog": cat,
                "retrieval": _retrieval_metrics(recs),
                "abstention_sweep": [_abstention_at(recs, f) for f in CITE_FLOORS],
            })

    report = {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "queries_path": str(queries_path), "top_k": top_k,
        "bm25_weight_by_type": BM25_WEIGHT_BY_TYPE,
        "cite_floors": CITE_FLOORS, "configs": configs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"grid_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    report["_out_path"] = str(out_path)
    return report


def _print(report: dict[str, Any]) -> None:
    print("\n=== Retrieval grid (quote hit@5 · topic hit@10 · topic mrr) ===")
    print("| bm25 | catalog | quote h@5 | topic h@10 | topic mrr |")
    print("|------|---------|----------:|-----------:|----------:|")
    baseline = None
    for c in report["configs"]:
        q = c["retrieval"].get("quote", {})
        t = c["retrieval"].get("topic", {})
        row = (f"| {c['bm25']} | {c['include_catalog']} | "
               f"{q.get('hit_at_5', 0):.0%} | {t.get('hit_at_10', 0):.0%} | "
               f"{t.get('mrr', 0):.3f} |")
        print(row)
        if c["bm25"] == "per-type" and not c["include_catalog"]:
            baseline = c
    print("\n=== Abstention vs cite_min_score (baseline config: per-type, catalog off) ===")
    print("| floor | precision | recall | false-abstention |")
    print("|-------|----------:|-------:|-----------------:|")
    for a in (baseline or report["configs"][0])["abstention_sweep"]:
        p = "n/a" if a["precision"] is None else f"{a['precision']:.0%}"
        r = "n/a" if a["recall"] is None else f"{a['recall']:.0%}"
        far = "n/a" if a["false_abstention_rate"] is None else f"{a['false_abstention_rate']:.0%}"
        print(f"| {a['floor']:.2f} | {p} | {r} | {far} |")
    print(f"\nFull report: {report['_out_path']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 3.3 knob grid-search.")
    p.add_argument("--queries", default="eval/golden_queries_corpus.yaml")
    p.add_argument("--out-dir", default="eval/results")
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args(argv)
    _print(run(Path(args.queries), Path(args.out_dir), args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
