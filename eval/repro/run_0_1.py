"""Reproduce the §0.1 "puran singh" retrieval, re-runnably.

Hits the live rag-api: `/api/search` for chunk-level reranker scores and
`/api/query` for the synthesized answer, then prints a compact table and writes
the full JSON (chunks + answer + citations) for inspection. The dashboard
password is read from the environment or `.env` and never printed.

Usage:
    python -m eval.repro.run_0_1 [--query "..."] [--base http://localhost:8081]
                                 [--out eval/repro/0_1_out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERY = "search some discourse about puran singh"


def load_env(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser so this runs standalone from a host shell where
    docker hasn't injected the environment. Real config still wins via os.environ."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reproduce the §0.1 puran-singh retrieval.")
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--base", default=os.environ.get("RAG_API_BASE", "http://localhost:8081"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", default=str(ROOT / "eval" / "repro" / "0_1_out.json"))
    args = p.parse_args(argv)

    env = load_env(ROOT / ".env")
    password = os.environ.get("DASHBOARD_PASSWORD") or env.get("DASHBOARD_PASSWORD", "")
    headers = {"Content-Type": "application/json", "X-Dashboard-Password": password}
    base = args.base.rstrip("/")

    out: dict[str, Any] = {"query": args.query, "base": base}

    sbody = {"query": args.query, "find_quote": False, "scope": "chunks",
             "backend": "hybrid", "top_k": args.top_k, "filters": {},
             "auto_filters": True, "expand_query": False}
    r = requests.post(f"{base}/api/search", json=sbody, headers=headers, timeout=120)
    r.raise_for_status()
    sresp = r.json()
    results = sresp.get("results", [])
    out["search_count"] = sresp.get("count")
    out["retrieval_ms"] = sresp.get("retrieval_ms")
    out["chunks"] = [
        {"rank": i, "score": c.get("score"), "source_file": c.get("source_file"),
         "text": c.get("text") or ""}
        for i, c in enumerate(results)
    ]

    qbody = {**sbody, "top_k": 8, "stream": False}
    r = requests.post(f"{base}/api/query", json=qbody, headers=headers, timeout=180)
    out["query_status"] = r.status_code
    try:
        qresp = r.json()
        out["answer"] = qresp.get("answer")
        out["citations"] = qresp.get("citations")
    except ValueError as e:
        out["query_error"] = f"{type(e).__name__}: {e}"
        out["query_body_head"] = r.text[:500]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"query   : {args.query}")
    print(f"search  : count={out['search_count']} retrieval_ms={out['retrieval_ms']}")
    print(f"query   : status={out['query_status']}")
    print("rank  score   source_file")
    for c in out["chunks"]:
        print(f"  {c['rank']:>2}  {c['score']:<7} {c['source_file']}")
    print(f"\nfull JSON (chunks + answer + citations): {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
