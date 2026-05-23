"""Phase 9 eval harness — runs golden queries through the retrieval
pipeline (bypassing the LLM) and emits Hit@k + MRR metrics plus a JSON
report under eval/results/<timestamp>.json.

Per PRD §6 Phase 9 acceptance:
  - Runs without error
  - Hit@5 >= 80% for quote-finding queries on the fixture-only test set
  - Report is well-formed JSON with per-query results

CLI:
    python -m eval.run_eval --queries eval/golden_queries.yaml
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

from open_webui_functions.analytics import Tools as AnalyticsTools
from open_webui_functions.search_transcripts import Tools as SearchTools


HIT_KS = (1, 5, 10)
# Quote-finding skews BM25 heavier per PRD §6 Phase 7 find_quote spec; qa
# and topic use the default search blend.
BM25_WEIGHT_BY_TYPE = {"quote": 0.85, "qa": 0.65, "topic": 0.65}


# ---- search tool factory -------------------------------------------------


def _make_search_tool(collection: str | None = None) -> SearchTools:
    t = SearchTools()
    t.valves.qdrant_url = os.environ.get("QDRANT_URL", t.valves.qdrant_url)
    t.valves.qdrant_key = os.environ.get("QDRANT_API_KEY", t.valves.qdrant_key)
    t.valves.qdrant_collection = (
        collection or os.environ.get("QDRANT_COLLECTION") or t.valves.qdrant_collection
    )
    t.valves.ollama_url = os.environ.get("OLLAMA_URL", t.valves.ollama_url)
    t.valves.reranker_url = os.environ.get("RERANKER_URL", t.valves.reranker_url)
    t.valves.tantivy_proxy_url = os.environ.get(
        "TANTIVY_URL", t.valves.tantivy_proxy_url
    )
    return t


def _make_analytics_tool() -> AnalyticsTools:
    """The Open WebUI tool's default DSN uses the docker-network name
    'postgres'; eval runs on the host, so fall back to localhost composed
    from POSTGRES_PASSWORD (same pattern as ingestion/bulk_ingest_hardened.py
    and the Phase 7 integration tests)."""
    t = AnalyticsTools()
    pg_dsn = os.environ.get("PG_DSN") or (
        f"postgresql://owui:{os.environ.get('POSTGRES_PASSWORD', '')}"
        "@localhost:5432/openwebui"
    )
    t.valves.pg_dsn = pg_dsn
    return t


# ---- retrieval per query type --------------------------------------------


def _retrieve_ranked(
    tool: SearchTools, query: str, qtype: str
) -> list[tuple[str, float, dict[str, Any]]]:
    """Run embed -> dense+bm25 -> RRF -> rerank and return the reranked
    [(chunk_id, score, payload)] list, bypassing _format and the public
    string-returning methods."""
    prior = tool.valves.bm25_weight
    try:
        tool.valves.bm25_weight = BM25_WEIGHT_BY_TYPE.get(qtype, 0.65)
        vec = tool._embed(query)
        dense = tool._dense(vec, speaker=None, source_file=None)
        bm25 = tool._bm25(query)
        fused = tool._rrf(dense, bm25, tool.valves.bm25_weight)
        # Rerank up to candidates_per_source for stable Hit@10 measurement.
        reranked = tool._rerank(query, fused[: tool.valves.candidates_per_source])
    finally:
        tool.valves.bm25_weight = prior
    return reranked


# ---- per-type scoring ----------------------------------------------------


def _match_quote_or_qa(
    ranked: list[tuple[str, float, dict[str, Any]]],
    expected_source_file: str,
    expected_chunk_contains: str,
) -> int | None:
    """Return 1-indexed rank of the first chunk matching both
    expected_source_file and substring-contains; None if no match."""
    needle = expected_chunk_contains.lower()
    for rank, (_cid, _score, pl) in enumerate(ranked, start=1):
        if pl.get("source_file") != expected_source_file:
            continue
        text = (pl.get("text") or "").lower()
        if needle in text:
            return rank
    return None


def _match_topic(
    ranked: list[tuple[str, float, dict[str, Any]]],
    expected_source_files: list[str],
    k: int,
) -> tuple[bool, list[str]]:
    """A topic query passes at K iff every expected source file appears at
    least once in the top-K. Returns (passed, missing_files)."""
    seen = {pl.get("source_file") for _, _, pl in ranked[:k]}
    missing = [f for f in expected_source_files if f not in seen]
    return (not missing), missing


def _score_analytics(
    tool: AnalyticsTools, q: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Returns (passed, reason, raw_result_payload)."""
    name = q["tool"]
    args = q.get("args", {})
    if name == "count_mentions":
        out = tool.count_mentions(**args)
        try:
            n = int(out.split("mentioned in ")[1].split(" chunks")[0])
        except (IndexError, ValueError):
            return False, f"could not parse count from: {out!r}", {"raw": out}
        lo = q.get("expected_count_min")
        hi = q.get("expected_count_max")
        if lo is not None and n < lo:
            return False, f"count {n} < min {lo}", {"count": n, "raw": out}
        if hi is not None and n > hi:
            return False, f"count {n} > max {hi}", {"count": n, "raw": out}
        return True, f"count {n} within [{lo}, {hi}]", {"count": n, "raw": out}
    if name == "list_transcripts_mentioning":
        out = tool.list_transcripts_mentioning(**args)
        expected = set(q.get("expected_source_files") or [])
        # Output lines look like '  sample_whisperx.json (3)'; harvest names.
        found = set()
        for line in out.splitlines()[1:]:  # skip the header line
            line = line.strip()
            if not line:
                continue
            # Strip trailing ' (N)' count.
            name_part = line.rsplit(" (", 1)[0]
            found.add(name_part)
        missing = expected - found
        if missing:
            return False, f"missing expected files: {sorted(missing)}", {
                "found": sorted(found), "raw": out,
            }
        return True, f"all expected files present (got {sorted(found)})", {
            "found": sorted(found), "raw": out,
        }
    if name == "top_speakers_for_topic":
        out = tool.top_speakers_for_topic(**args)
        expected_speaker = q.get("expected_speaker_in_top")
        if expected_speaker is None:
            return False, "missing expected_speaker_in_top", {"raw": out}
        if expected_speaker in out:
            return True, f"speaker {expected_speaker!r} present", {"raw": out}
        return False, f"speaker {expected_speaker!r} not in: {out!r}", {"raw": out}
    return False, f"unknown analytics tool: {name!r}", {}


# ---- aggregation --------------------------------------------------------


def _aggregate(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    for q in per_query:
        t = q["type"]
        bt = by_type.setdefault(t, {"n": 0, "hits": {k: 0 for k in HIT_KS},
                                    "mrr_sum": 0.0, "pass": 0})
        bt["n"] += 1
        if t == "analytics":
            if q["passed"]:
                bt["pass"] += 1
        else:
            for k in HIT_KS:
                if q.get(f"pass_at_{k}"):
                    bt["hits"][k] += 1
            bt["mrr_sum"] += float(q.get("reciprocal_rank") or 0.0)

    summary: dict[str, Any] = {}
    for t, bt in by_type.items():
        n = bt["n"]
        if t == "analytics":
            summary[t] = {"n": n, "pass_rate": (bt["pass"] / n) if n else 0.0}
        else:
            summary[t] = {
                "n": n,
                **{f"hit_at_{k}": (bt["hits"][k] / n) if n else 0.0 for k in HIT_KS},
                "mrr": (bt["mrr_sum"] / n) if n else 0.0,
            }
    return summary


# ---- main ---------------------------------------------------------------


def run(
    queries_path: Path,
    out_dir: Path,
    collection: str | None,
) -> tuple[dict[str, Any], Path]:
    queries = yaml.safe_load(queries_path.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{queries_path}: expected non-empty YAML list")

    search_tool = _make_search_tool(collection)
    analytics_tool = _make_analytics_tool()

    per_query: list[dict[str, Any]] = []
    for q in queries:
        qid = q["id"]
        qtype = q["type"]
        record: dict[str, Any] = {"id": qid, "type": qtype}
        try:
            if qtype in ("quote", "qa"):
                ranked = _retrieve_ranked(search_tool, q["query"], qtype)
                rank = _match_quote_or_qa(
                    ranked, q["expected_source_file"],
                    q["expected_chunk_contains"],
                )
                record["n_results"] = len(ranked)
                record["first_hit_rank"] = rank
                record["reciprocal_rank"] = 1.0 / rank if rank else 0.0
                for k in HIT_KS:
                    record[f"pass_at_{k}"] = rank is not None and rank <= k
                if rank is None:
                    record["reason"] = (
                        f"no chunk matching source={q['expected_source_file']!r} + "
                        f"contains={q['expected_chunk_contains']!r} in top "
                        f"{len(ranked)}"
                    )
                else:
                    record["reason"] = f"matched at rank {rank}"
            elif qtype == "topic":
                ranked = _retrieve_ranked(search_tool, q["query"], qtype)
                record["n_results"] = len(ranked)
                for k in HIT_KS:
                    passed, missing = _match_topic(
                        ranked, q["expected_source_files"], k,
                    )
                    record[f"pass_at_{k}"] = passed
                    if k == 10:
                        record["reason"] = ("all expected files present in top 10"
                                            if passed else f"missing in top 10: {missing}")
                # No single "first_hit_rank" for topic — use min rank across all
                # expected files as a soft MRR proxy.
                first_ranks = []
                for f in q["expected_source_files"]:
                    for r, (_, _, pl) in enumerate(ranked, start=1):
                        if pl.get("source_file") == f:
                            first_ranks.append(r)
                            break
                record["reciprocal_rank"] = (
                    1.0 / max(first_ranks) if first_ranks
                    and len(first_ranks) == len(q["expected_source_files"]) else 0.0
                )
            elif qtype == "analytics":
                passed, reason, extra = _score_analytics(analytics_tool, q)
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

    by_type = _aggregate(per_query)
    report = {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "queries_path": str(queries_path),
        "collection": search_tool.valves.qdrant_collection,
        "total_queries": len(per_query),
        "by_type": by_type,
        "queries": per_query,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, out_path


def _print_summary(report: dict[str, Any], out_path: Path) -> None:
    print(f"\nEval report: {out_path}")
    print(f"Total queries: {report['total_queries']}")
    for t, m in report["by_type"].items():
        if t == "analytics":
            print(f"  {t:9s} n={m['n']:2d}  pass_rate={m['pass_rate']:.2%}")
        else:
            print(
                f"  {t:9s} n={m['n']:2d}  "
                f"hit@1={m['hit_at_1']:.2%} hit@5={m['hit_at_5']:.2%} "
                f"hit@10={m['hit_at_10']:.2%}  mrr={m['mrr']:.3f}"
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 9 eval harness.")
    p.add_argument("--queries", required=True,
                   help="Path to golden_queries.yaml")
    p.add_argument("--out-dir", default="eval/results",
                   help="Directory for JSON reports (default eval/results)")
    p.add_argument("--collection", default=None,
                   help="Qdrant collection name to evaluate against. "
                        "Defaults to env QDRANT_COLLECTION or 'transcripts'.")
    p.add_argument("--quote-hit5-threshold", type=float, default=0.80,
                   help="Pass threshold for quote-type Hit@5 (PRD: 0.80).")
    args = p.parse_args(argv)

    report, out_path = run(
        Path(args.queries), Path(args.out_dir), args.collection,
    )
    _print_summary(report, out_path)

    # PRD acceptance gate: quote-type Hit@5 must be >= threshold.
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
