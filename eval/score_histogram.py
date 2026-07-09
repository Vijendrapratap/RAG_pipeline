"""§0.2 — Reranker score histogram over known-good queries.  BLOCKS §1.5.

Runs a hand-picked set of known-good queries (Devanagari + Latin; name / quote /
thematic — see `eval/histogram_queries.yaml`) through the live `/api/search`
endpoint and records the bge-reranker-v2-m3 `relevance_score` of the **top hit**
and of **hit #4** for each. The point: `cite_min_score` (the citation floor that
`rag_api.synthesis.filter_min_score` enforces, and that §1.5's abstention path
depends on) must be set from *this* distribution — not the current 0.75, which
§0.1 showed is ~10x above the entire realistic reranker range.

Why top-hit vs hit #4: the top hit is the best legitimate match (a floor must not
reject it); hit #4 approximates the weak tail a floor *should* trim. The gap
between the two distributions is the calibration headroom.

The endpoint returns the post-rerank `score` (retrieval.py:238 -> _rerank ->
Infinity `relevance_score`) and does NOT apply `filter_min_score` (that runs only
in /api/query), so this measures the raw, unfiltered reranker distribution.

Usage:
    python -m eval.score_histogram [--queries eval/histogram_queries.yaml]
        [--base http://localhost:8081] [--out-dir eval/results]

Auth: DASHBOARD_PASSWORD from the environment or `.env`.
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

ROOT = Path(__file__).resolve().parents[1]
# Candidate floors to report retention against — spans the range §0.1 observed.
CANDIDATE_FLOORS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.75]


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def percentile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0,100]); None on empty input.
    Kept dependency-free so the eval harness needs only requests+yaml."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def dist(xs: list[float]) -> dict[str, Any]:
    return {
        "n": len(xs),
        "min": round(min(xs), 4) if xs else None,
        "p05": round(percentile(xs, 5), 4) if xs else None,
        "p10": round(percentile(xs, 10), 4) if xs else None,
        "p25": round(percentile(xs, 25), 4) if xs else None,
        "median": round(statistics.median(xs), 4) if xs else None,
        "p75": round(percentile(xs, 75), 4) if xs else None,
        "p90": round(percentile(xs, 90), 4) if xs else None,
        "max": round(max(xs), 4) if xs else None,
    }


def search(base: str, headers: dict[str, str], query: str, top_k: int,
           timeout: float) -> dict[str, Any]:
    body = {"query": query, "find_quote": False, "scope": "chunks",
            "backend": "hybrid", "top_k": top_k, "filters": {},
            "auto_filters": True, "expand_query": False}
    r = requests.post(f"{base}/api/search", json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reranker score histogram (§0.2).")
    p.add_argument("--queries", default=str(ROOT / "eval" / "histogram_queries.yaml"))
    p.add_argument("--base", default=os.environ.get("RAG_API_BASE", "http://localhost:8081"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = p.parse_args(argv)

    env = load_env(ROOT / ".env")
    password = os.environ.get("DASHBOARD_PASSWORD") or env.get("DASHBOARD_PASSWORD", "")
    headers = {"Content-Type": "application/json", "X-Dashboard-Password": password}
    base = args.base.rstrip("/")

    queries = yaml.safe_load(Path(args.queries).read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        print(f"No queries in {args.queries}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    print(f"§0.2 histogram: {len(queries)} known-good queries -> {base}\n")
    print(f"{'id':22} {'type':9} {'scr':5} {'top':>7} {'#4':>7}  count")
    for q in queries:
        resp = search(base, headers, q["query"], args.top_k, args.timeout)
        results = resp.get("results", [])
        top = float(results[0]["score"]) if len(results) >= 1 else None
        h4 = float(results[3]["score"]) if len(results) >= 4 else None
        rows.append({"id": q.get("id"), "type": q.get("type"), "script": q.get("script"),
                     "query": q["query"], "top": top, "h4": h4, "count": len(results)})
        print(f"{str(q.get('id'))[:22]:22} {str(q.get('type'))[:9]:9} "
              f"{str(q.get('script'))[:5]:5} "
              f"{('%.4f' % top) if top is not None else '   -  ':>7} "
              f"{('%.4f' % h4) if h4 is not None else '   -  ':>7}  {len(results)}")

    tops = [r["top"] for r in rows if r["top"] is not None]
    h4s = [r["h4"] for r in rows if r["h4"] is not None]

    # Retention of the known-good top hit vs the weak tail (hit #4) at each floor.
    floor_table = []
    for f in CANDIDATE_FLOORS:
        top_keep = sum(1 for x in tops if x >= f)
        h4_keep = sum(1 for x in h4s if x >= f)
        floor_table.append({
            "floor": f,
            "tophit_retained": f"{top_keep}/{len(tops)}",
            "tophit_pct": round(100 * top_keep / len(tops), 1) if tops else None,
            "hit4_retained": f"{h4_keep}/{len(h4s)}",
            "hit4_pct": round(100 * h4_keep / len(h4s), 1) if h4s else None,
        })

    # Recommendation: largest floor that still keeps EVERY known-good top hit,
    # snapped just below the observed min so a slightly-worse legit hit survives.
    min_top = min(tops) if tops else 0.0
    recommended = round(max(0.0, min_top * 0.8), 4)

    summary = {
        "base": base, "queries_path": str(args.queries), "n_queries": len(queries),
        "top_hit_dist": dist(tops), "hit4_dist": dist(h4s),
        "by_type": {t: dist([r["top"] for r in rows if r["type"] == t and r["top"] is not None])
                    for t in sorted({r["type"] for r in rows})},
        "by_script": {s: dist([r["top"] for r in rows if r["script"] == s and r["top"] is not None])
                      for s in sorted({r["script"] for r in rows})},
        "floor_retention": floor_table,
        "min_top_hit": round(min_top, 4),
        "recommended_cite_min_score": recommended,
        "current_cite_min_score": float(os.environ.get("RAG_CITE_MIN_SCORE", "0.75")),
    }

    print("\n=== top-hit score distribution (known-good) ===")
    print(json.dumps(summary["top_hit_dist"], indent=2))
    print("=== hit #4 (weak tail) distribution ===")
    print(json.dumps(summary["hit4_dist"], indent=2))
    print("\n=== floor retention: keep top hit, trim tail ===")
    print(f"{'floor':>7}  {'top-hit kept':>14}  {'hit#4 kept':>12}")
    for r in floor_table:
        print(f"{r['floor']:>7}  {r['tophit_retained']:>8} ({r['tophit_pct']:>5}%)  "
              f"{r['hit4_retained']:>6} ({r['hit4_pct']:>5}%)")
    print(f"\nmin known-good top-hit score : {summary['min_top_hit']}")
    print(f"current cite_min_score       : {summary['current_cite_min_score']}")
    print(f"RECOMMENDED cite_min_score   : {summary['recommended_cite_min_score']}  "
          f"(keeps every known-good top hit)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = out_dir / f"score_histogram_{ts}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
