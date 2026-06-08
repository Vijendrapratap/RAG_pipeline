"""Side-by-side A/B of chat models over the *real* RAG pipeline.

`eval/run_eval.py` scores retrieval (model-independent). This script does the
complementary job: it runs the same questions through the full
`POST /api/query` path once per chat model, so you can compare the thing the
chat model actually controls — **answer quality** and **generation speed** —
holding retrieval constant.

For each (model, query) it records the answer, retrieval_ms (returned by the
API), measured generation time, and an approximate tokens/sec. Results are
written as both a human-readable Markdown report and a JSON file under
`eval/compare_results/`.

Speed note: switching chat models forces Ollama to load the new weights. To
keep the timing about *generation* and not *loading*, each model is warmed up
(one tiny direct Ollama call) before its timed queries.

Usage (host side; rag-api on :8081 per the local override):
    python -m eval.compare_models \
        --api http://localhost:8081 \
        --ollama http://localhost:11434 \
        --password "$DASHBOARD_PASSWORD" \
        --queries eval/compare_queries.yaml \
        --models qwen3.5:9b qwen3:14b gemma4:12b

Omit --models to compare every Ollama model the dashboard currently offers
(`GET /api/models`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
import yaml


def _discover_models(api: str, password: str) -> list[str]:
    """All Ollama-provider models the dashboard currently advertises."""
    r = requests.get(f"{api}/api/models", timeout=30)
    r.raise_for_status()
    return [
        m["model"]
        for m in r.json().get("models", [])
        if m.get("provider") == "ollama"
    ]


def _unload(ollama: str, model: str, timeout: float) -> None:
    """Evict a model from VRAM (keep_alive=0, no prompt → unload, no generate).

    Without this, switching between large models leaves the previous one
    resident; the next model then gets a shrunken VRAM budget and spills to
    CPU, which on a low-system-RAM box collapses to ~2 tok/s. Unloading first
    gives every model a clean, comparable budget.
    """
    try:
        requests.post(
            f"{ollama}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=min(timeout, 60.0),
        )
    except requests.RequestException:
        pass


def _warmup(ollama: str, model: str, timeout: float) -> None:
    """Load the model into VRAM so the first timed query isn't paying for it."""
    try:
        requests.post(
            f"{ollama}/api/generate",
            json={"model": model, "prompt": "hi", "stream": False,
                  "keep_alive": "30m"},
            timeout=timeout,
        ).raise_for_status()
    except requests.RequestException as e:
        print(f"  ! warmup failed for {model}: {e}")


def _approx_tokens(text: str) -> int:
    """Cheap token estimate: ~4 chars/token is a decent average for mixed
    Hindi/English. Only used for a relative tokens/sec figure, not billing."""
    return max(1, round(len(text) / 4))


def _ask(
    api: str, password: str, query: str, model: str,
    answer_language: str, timeout: float,
) -> dict[str, Any]:
    """One non-streaming /api/query turn with a per-request model override."""
    headers = {"Content-Type": "application/json"}
    if password:
        headers["X-Dashboard-Password"] = password
    body = {
        "query": query,
        "provider": "ollama",
        "model": model,
        "stream": False,
        "answer_language": answer_language,
    }
    t0 = time.monotonic()
    r = requests.post(f"{api}/api/query", headers=headers, json=body,
                      timeout=timeout)
    total_ms = round((time.monotonic() - t0) * 1000.0, 1)
    r.raise_for_status()
    data = r.json()
    answer = data.get("answer") or ""
    retrieval_ms = float(data.get("retrieval_ms") or 0.0)
    gen_ms = max(0.0, total_ms - retrieval_ms)
    toks = _approx_tokens(answer)
    return {
        "answer": answer,
        "retrieval_ms": retrieval_ms,
        "total_ms": total_ms,
        "gen_ms": round(gen_ms, 1),
        "approx_tokens": toks,
        "approx_tok_per_s": round(toks / (gen_ms / 1000.0), 1) if gen_ms else 0.0,
        "citations": len(data.get("citations") or []),
        "answer_language": data.get("answer_language"),
    }


def run(
    api: str, ollama: str, password: str, queries: list[dict[str, Any]],
    models: list[str], timeout: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for model in models:
        print(f"\n=== model: {model} ===")
        # Evict the others so this model gets the full VRAM budget (avoids the
        # partial-CPU-offload thrash when switching between large models).
        for other in models:
            if other != model:
                _unload(ollama, other, timeout)
        _warmup(ollama, model, timeout)
        for q in queries:
            qid = q.get("id") or q["query"][:40]
            lang = q.get("answer_language", "auto")
            print(f"  - {qid} ...", end="", flush=True)
            try:
                res = _ask(api, password, q["query"], model, lang, timeout)
                print(f" {res['gen_ms']:.0f}ms gen, "
                      f"~{res['approx_tok_per_s']} tok/s, "
                      f"{res['citations']} cites")
            except Exception as e:  # noqa: BLE001 - record, don't abort the matrix
                res = {"error": f"{type(e).__name__}: {e}"}
                print(f" ERROR: {e}")
            records.append({"model": model, "query_id": qid,
                            "query": q["query"], **res})
    return {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "api": api,
        "models": models,
        "n_queries": len(queries),
        "records": records,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    """Group by query so each question's answers sit side by side."""
    by_query: dict[str, list[dict[str, Any]]] = {}
    for rec in report["records"]:
        by_query.setdefault(rec["query"], []).append(rec)

    lines = [
        f"# Model A/B comparison — {report['timestamp']}",
        "",
        f"Models: {', '.join(report['models'])}",
        "",
        "Speed = generation only (retrieval time subtracted). tok/s is an "
        "approximate 4-chars/token estimate for relative comparison.",
        "",
    ]
    for query, recs in by_query.items():
        lines.append(f"## {query}")
        lines.append("")
        lines.append("| Model | gen ms | ~tok/s | cites | lang |")
        lines.append("|---|--:|--:|--:|---|")
        for r in recs:
            if "error" in r:
                lines.append(f"| {r['model']} | — | — | — | ERROR |")
            else:
                lines.append(
                    f"| {r['model']} | {r['gen_ms']:.0f} | "
                    f"{r['approx_tok_per_s']} | {r['citations']} | "
                    f"{r.get('answer_language')} |"
                )
        lines.append("")
        for r in recs:
            lines.append(f"### {r['model']}")
            if "error" in r:
                lines.append(f"> ERROR: {r['error']}")
            else:
                lines.append(r["answer"].strip() or "_(empty answer)_")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="A/B chat models over the RAG pipeline.")
    p.add_argument("--api", default=os.environ.get("RAG_API_BASE",
                                                    "http://localhost:8081"))
    p.add_argument("--ollama", default=os.environ.get("OLLAMA_URL",
                                                       "http://localhost:11434"))
    p.add_argument("--password", default=os.environ.get("DASHBOARD_PASSWORD", ""))
    p.add_argument("--queries", default="eval/compare_queries.yaml")
    p.add_argument("--models", nargs="*", default=None,
                   help="Ollama model ids. Default: all from GET /api/models.")
    p.add_argument("--out-dir", default="eval/compare_results")
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args(argv)

    queries = yaml.safe_load(Path(args.queries).read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        raise SystemExit(f"{args.queries}: expected a non-empty YAML list")

    models = args.models or _discover_models(args.api, args.password)
    if not models:
        raise SystemExit("no models to compare (none passed, none discovered)")

    report = run(args.api, args.ollama, args.password, queries, models,
                 args.timeout)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"{ts}.json").write_text(json.dumps(report, indent=2,
                                                   ensure_ascii=False),
                                        encoding="utf-8")
    md_path = out_dir / f"{ts}.md"
    _write_markdown(report, md_path)
    print(f"\nReport: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
