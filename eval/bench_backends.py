"""Side-by-side retrieval benchmark: hybrid vs PageIndex.

Fires a fixed query set at the *live* rag-api `/api/search` endpoint once per
backend and reports the speed and (where ground truth exists) the quality gap,
so the hybrid-vs-PageIndex trade-off is reproducible instead of eyeballed.

Unlike `eval.run_eval` (which drives the in-process retrieval tools), this hits
the running HTTP service — exactly the path the dashboard uses — so the reported
`retrieval_ms` and `backend` routing are the real thing. It does NOT need the
PageIndex trees to exist: if none are built, PageIndex simply returns 0 results
and that shows up plainly in the table (a useful signal in itself).

For each query and each backend it records server-reported `retrieval_ms`
(median over --repeat runs), result count, and the top results. When a query
carries `expected_source_file(s)` (reused from golden_queries.yaml), it also
computes recall@k — does the engine surface the expected transcript in top-k —
which is comparable across both backends because both return `source_file`.

CLI:
    python -m eval.bench_backends \\
        [--queries eval/golden_queries.yaml] [--base http://localhost:8081] \\
        [--top-k 8] [--repeat 3] [--details] [--out-dir eval/results]

Auth: set DASHBOARD_PASSWORD in the environment if the API requires it.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml

BACKENDS = ("hybrid", "pageindex")
# Query types from golden_queries.yaml that exercise retrieval (skip analytics).
RETRIEVAL_TYPES = {"quote", "qa", "topic"}


# ---- query loading -------------------------------------------------------


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Load retrieval queries. Accepts golden_queries.yaml (a list of dicts
    with id/type/query + optional expected_source_file[s]) or a plain-text file
    of one query per line."""
    if path.suffix.lower() in (".yaml", ".yml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{path}: expected a YAML list")
        out: list[dict[str, Any]] = []
        for q in raw:
            if q.get("type") not in RETRIEVAL_TYPES:
                continue
            expected = q.get("expected_source_files")
            if expected is None and q.get("expected_source_file"):
                expected = [q["expected_source_file"]]
            out.append({
                "id": q.get("id", q["query"][:24]),
                "type": q.get("type", "qa"),
                "query": q["query"],
                "expected_source_files": expected or [],
            })
        return out
    # plain text: one query per non-empty, non-comment line
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if s and not s.startswith("#"):
            out.append({"id": f"L{i}", "type": "qa", "query": s,
                        "expected_source_files": []})
    return out


# ---- HTTP client ---------------------------------------------------------


def _headers() -> dict[str, str]:
    pw = os.environ.get("DASHBOARD_PASSWORD", "")
    h = {"Content-Type": "application/json"}
    if pw:
        h["X-Dashboard-Password"] = pw
    return h


def search_once(
    base: str, query: str, backend: str, top_k: int, timeout: float,
) -> dict[str, Any]:
    """One POST /api/search. Returns the parsed response, or a record with an
    `error` key on any HTTP/transport failure — a down service or a bad backend
    should mark that cell, not abort the whole benchmark."""
    body = {
        "query": query, "find_quote": False, "scope": "chunks",
        "backend": backend, "top_k": top_k,
        "filters": {}, "auto_filters": True, "expand_query": False,
    }
    try:
        r = requests.post(f"{base.rstrip('/')}/api/search",
                          json=body, headers=_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        detail = ""
        if isinstance(e, requests.HTTPError) and e.response is not None:
            detail = f" — {e.response.text[:200]}"
        return {"error": f"{type(e).__name__}: {e}{detail}"}


# ---- per-query benchmarking ----------------------------------------------


def recall_at_k(results: list[dict[str, Any]], expected: list[str], k: int) -> float | None:
    """Fraction of expected source files present in the top-k results, or None
    when the query has no ground truth."""
    if not expected:
        return None
    seen = {r.get("source_file") for r in results[:k]}
    return sum(1 for f in expected if f in seen) / len(expected)


def bench_query(
    base: str, q: dict[str, Any], top_k: int, repeat: int, timeout: float,
) -> dict[str, Any]:
    """Run one query through every backend `repeat` times. Latency is the
    server-reported `retrieval_ms` (excludes network), summarised as the
    median so a single GC/cold-cache spike doesn't dominate."""
    record: dict[str, Any] = {"id": q["id"], "type": q["type"], "query": q["query"],
                              "expected_source_files": q["expected_source_files"],
                              "backends": {}}
    for backend in BACKENDS:
        server_ms: list[float] = []
        wall_ms: list[float] = []
        last: dict[str, Any] = {}
        for _ in range(repeat):
            t0 = time.monotonic()
            resp = search_once(base, q["query"], backend, top_k, timeout)
            wall_ms.append((time.monotonic() - t0) * 1000.0)
            last = resp
            if "error" not in resp and resp.get("retrieval_ms") is not None:
                server_ms.append(float(resp["retrieval_ms"]))
        if "error" in last:
            record["backends"][backend] = {"error": last["error"]}
            continue
        results = last.get("results", [])
        record["backends"][backend] = {
            "server_ms_median": round(statistics.median(server_ms), 1) if server_ms else None,
            "wall_ms_median": round(statistics.median(wall_ms), 1),
            "count": last.get("count", len(results)),
            "recall_at_k": recall_at_k(results, q["expected_source_files"], top_k),
            "top": [
                {
                    "source_file": r.get("source_file"),
                    "start_sec": r.get("start_sec"),
                    "score": r.get("score"),
                    "snippet": (r.get("text") or "")[:120].replace("\n", " "),
                }
                for r in results[:3]
            ],
        }
    return record


# ---- reporting -----------------------------------------------------------


def _fmt_ms(v: float | None) -> str:
    return f"{v:.0f}" if isinstance(v, (int, float)) else "—"


def _fmt_recall(v: float | None) -> str:
    return f"{v:.0%}" if isinstance(v, (int, float)) else "—"


def print_table(records: list[dict[str, Any]]) -> None:
    """A compact per-query comparison table to stdout (Markdown-friendly)."""
    print()
    print("| id | type | hybrid ms | pageindex ms | speedup | "
          "hybrid n | pi n | hybrid r@k | pi r@k |")
    print("|----|------|----------:|-------------:|--------:|"
          "---------:|-----:|-----------:|-------:|")
    for rec in records:
        h = rec["backends"].get("hybrid", {})
        p = rec["backends"].get("pageindex", {})
        h_ms, p_ms = h.get("server_ms_median"), p.get("server_ms_median")
        speedup = (
            f"{p_ms / h_ms:.1f}x" if isinstance(h_ms, (int, float))
            and isinstance(p_ms, (int, float)) and h_ms else "—"
        )
        h_cell = h.get("error", _fmt_ms(h_ms))[:18]
        p_cell = p.get("error", _fmt_ms(p_ms))[:18]
        print(
            f"| {rec['id']} | {rec['type']} | {h_cell} | {p_cell} | {speedup} | "
            f"{h.get('count', '—')} | {p.get('count', '—')} | "
            f"{_fmt_recall(h.get('recall_at_k'))} | {_fmt_recall(p.get('recall_at_k'))} |"
        )


def print_details(records: list[dict[str, Any]]) -> None:
    for rec in records:
        print(f"\n### {rec['id']} ({rec['type']}) — {rec['query']!r}")
        if rec["expected_source_files"]:
            print(f"  expected: {rec['expected_source_files']}")
        for backend in BACKENDS:
            b = rec["backends"].get(backend, {})
            if "error" in b:
                print(f"  [{backend}] ERROR: {b['error']}")
                continue
            print(f"  [{backend}] {_fmt_ms(b.get('server_ms_median'))} ms · "
                  f"{b.get('count')} results")
            for t in b.get("top", []):
                ss = f"@{t['start_sec']:.0f}s" if t.get("start_sec") is not None else ""
                print(f"      {t['score']}  {t['source_file']} {ss}  {t['snippet']!r}")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate medians + means across all queries, per backend."""
    out: dict[str, Any] = {}
    for backend in BACKENDS:
        latencies = [
            rec["backends"][backend]["server_ms_median"]
            for rec in records
            if backend in rec["backends"]
            and rec["backends"][backend].get("server_ms_median") is not None
        ]
        recalls = [
            rec["backends"][backend]["recall_at_k"]
            for rec in records
            if backend in rec["backends"]
            and rec["backends"][backend].get("recall_at_k") is not None
        ]
        errors = sum(
            1 for rec in records
            if "error" in rec["backends"].get(backend, {})
        )
        out[backend] = {
            "queries": len(records),
            "errors": errors,
            "median_ms": round(statistics.median(latencies), 1) if latencies else None,
            "mean_ms": round(statistics.mean(latencies), 1) if latencies else None,
            "mean_recall_at_k": round(statistics.mean(recalls), 3) if recalls else None,
            "recall_n": len(recalls),
        }
    return out


def _print_summary(summary: dict[str, Any], top_k: int) -> None:
    print("\n=== Summary (server-side retrieval_ms; recall@%d where ground "
          "truth exists) ===" % top_k)
    for backend in BACKENDS:
        s = summary[backend]
        recall = (f"{s['mean_recall_at_k']:.0%} (n={s['recall_n']})"
                  if s["mean_recall_at_k"] is not None else "n/a")
        print(f"  {backend:10s}  median={_fmt_ms(s['median_ms'])}ms  "
              f"mean={_fmt_ms(s['mean_ms'])}ms  recall@{top_k}={recall}  "
              f"errors={s['errors']}/{s['queries']}")
    h, p = summary["hybrid"]["median_ms"], summary["pageindex"]["median_ms"]
    if isinstance(h, (int, float)) and isinstance(p, (int, float)) and h:
        print(f"\n  PageIndex median latency is {p / h:.1f}x hybrid "
              f"({_fmt_ms(p)}ms vs {_fmt_ms(h)}ms).")


# ---- main ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark hybrid vs PageIndex retrieval.")
    p.add_argument("--queries", default="eval/golden_queries.yaml",
                   help="golden_queries.yaml or a plain-text file (one query/line).")
    p.add_argument("--base", default=os.environ.get("RAG_API_BASE",
                                                     "http://localhost:8081"),
                   help="rag-api base URL (default env RAG_API_BASE or :8081).")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--repeat", type=int, default=3,
                   help="Runs per query per backend; latency is the median.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N queries (quick smoke run).")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request timeout (PageIndex reasoning can be slow).")
    p.add_argument("--details", action="store_true",
                   help="Print the top results per backend per query.")
    p.add_argument("--out-dir", default="eval/results")
    args = p.parse_args(argv)

    queries = load_queries(Path(args.queries))
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("No retrieval queries found.", file=sys.stderr)
        return 2

    print(f"Benchmarking {len(queries)} queries x {len(BACKENDS)} backends "
          f"x {args.repeat} repeats against {args.base} (top_k={args.top_k})")

    records: list[dict[str, Any]] = []
    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q['id']}: {q['query'][:60]}", flush=True)
        records.append(bench_query(args.base, q, args.top_k, args.repeat, args.timeout))

    print_table(records)
    if args.details:
        print_details(records)
    summary = summarize(records)
    _print_summary(summary, args.top_k)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Timestamp comes from the OS clock here (not in a workflow), so wall-clock
    # naming is fine and keeps reports sortable.
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = out_dir / f"bench_{ts}.json"
    out_path.write_text(json.dumps({
        "base": args.base, "top_k": args.top_k, "repeat": args.repeat,
        "queries_path": str(args.queries),
        "summary": summary, "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
