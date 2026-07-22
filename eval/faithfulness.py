"""Faithfulness eval — do generated answers stay inside their passages?

Hit@k / MRR (eval/run_eval.py) score RETRIEVAL; nothing so far scores the
ANSWER. This harness closes that: it runs golden queries through the full
retrieve → cite-floor → synthesize pipeline in-process, then uses the
groundedness judge (rag_api.ground) to score each answer sentence against the
passages it cited. Metrics:

  * faithfulness  — supported sentences / total sentences, per answered query;
                    aggregated as the mean over answered positives.
  * abstain rate  — negatives that correctly abstained (floor + allow_abstain),
                    and positives that falsely abstained.
  * leak fidelity — for negatives that DID answer, how unsupported the answer
                    was (the "hallucinated a made-up entity" signature).

Judge failures (unreachable model, unusable verdict) score the query as
`judge_failed` and are excluded from the mean — logged, never hidden.

Needs the live stack (Qdrant + reranker + Ollama). Run from the repo root:
    python -m eval.faithfulness --queries eval/golden_queries_corpus.yaml
    python -m eval.faithfulness --queries ... --limit 10 --no-route
Report: stdout table + eval/results/faithfulness_<timestamp>.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any

import requests
import yaml

from rag_api.config import get_settings
from rag_api.ground import _GROUND_SYSTEM, build_ground_prompt, parse_verdict, split_sentences
from rag_api.lang import resolve_language
from rag_api.ollama_chat import chat_text
from rag_api.retrieval import Retriever
from rag_api.route import route as route_query
from rag_api.synthesis import NO_CONTEXT_MESSAGE, Synthesizer, build_context_block, filter_min_score

log = logging.getLogger("eval.faithfulness")

# Answer-bearing golden types. Analytics rows are Postgres checks, not answers.
ANSWER_TYPES = frozenset({"quote", "qa", "topic", "negative"})


def _retrieve(retriever: Retriever, query: str, top_k: int, use_route: bool):
    if use_route:
        d = route_query(query, retriever.settings)
        if not d.retrieve:
            return []  # chitchat never reaches retrieval; not in golden sets
        if d.find_quote:
            return retriever.find_quote(d.query, top_k)
        return retriever.search(query, filters=None, top_k=top_k,
                                bm25_weight=d.bm25_weight,
                                include_catalog=d.include_catalog)
    return retriever.search(query, filters=None, top_k=top_k,
                            bm25_weight=None, include_catalog=False)


def _judge(session: requests.Session, settings, answer: str,
           results: list[dict[str, Any]]) -> tuple[float | None, int, int]:
    """Score `answer` against `results`: (faithfulness, supported, total).

    faithfulness None = judge failed (excluded from means, counted separately).
    """
    sentences = split_sentences(answer)
    if not sentences:
        return None, 0, 0
    prompt = build_ground_prompt(sentences, build_context_block(results))
    reply = chat_text(
        session, settings.ollama_url, settings.chat_model,
        _GROUND_SYSTEM, prompt,
        timeout=settings.chat_timeout_s, temperature=0.0,
        num_ctx=settings.chat_num_ctx, num_predict=128,
        log=log, label="faithfulness judge", subject=answer[:80],
    )
    unsupported = parse_verdict(reply, len(sentences)) if reply else None
    if unsupported is None:
        return None, 0, len(sentences)
    supported = len(sentences) - len(unsupported)
    return supported / len(sentences), supported, len(sentences)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queries", default="eval/golden_queries_corpus.yaml")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--cite-min-score", type=float, default=None,
                    help="override the settings floor for this run")
    ap.add_argument("--no-route", action="store_true",
                    help="bypass the Stage 4.1 router (flat pipeline)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    floor = args.cite_min_score if args.cite_min_score is not None else settings.cite_min_score
    retriever = Retriever(settings)
    synthesizer = Synthesizer(settings)
    session = requests.Session()

    rows_in = [q for q in yaml.safe_load(Path(args.queries).read_text(encoding="utf-8"))
               if q.get("type") in ANSWER_TYPES]
    if args.limit:
        rows_in = rows_in[: args.limit]

    no_context = set(NO_CONTEXT_MESSAGE.values())
    rows: list[dict[str, Any]] = []
    for q in rows_in:
        query, qtype = q["query"], q["type"]
        try:
            results = _retrieve(retriever, query, args.top_k, not args.no_route)
        except Exception as e:  # noqa: BLE001 - per-query isolation, logged
            log.error("retrieval failed for %s %r: %s", q.get("id"), query, e)
            rows.append({"id": q.get("id"), "type": qtype, "error": str(e)})
            continue
        results = filter_min_score(results, floor, allow_abstain=True)
        if not results:
            rows.append({"id": q.get("id"), "type": qtype, "abstained": True})
            continue
        lang = resolve_language("auto", query)
        try:
            answer = synthesizer.generate(query, results, lang)
        except Exception as e:  # noqa: BLE001 - per-query isolation, logged
            log.error("synthesis failed for %s %r: %s", q.get("id"), query, e)
            rows.append({"id": q.get("id"), "type": qtype, "error": str(e)})
            continue
        if not answer or answer in no_context:
            rows.append({"id": q.get("id"), "type": qtype, "abstained": True})
            continue
        faith, supported, total = _judge(session, settings, answer, results)
        rows.append({
            "id": q.get("id"), "type": qtype, "abstained": False,
            "faithfulness": faith, "supported": supported, "sentences": total,
            "top_score": float(results[0].get("score") or 0.0),
            "answer": answer,
        })
        tag = "judge_failed" if faith is None else f"{faith:.2f}"
        print(f"  {q.get('id'):8} {qtype:8} faith={tag:12} "
              f"({supported}/{total} sentences)")

    pos = [r for r in rows if r["type"] != "negative" and "error" not in r]
    neg = [r for r in rows if r["type"] == "negative" and "error" not in r]
    pos_scored = [r["faithfulness"] for r in pos
                  if r.get("faithfulness") is not None]
    neg_answered = [r for r in neg if not r.get("abstained")]
    summary = {
        "queries": len(rows),
        "errors": sum(1 for r in rows if "error" in r),
        "judge_failed": sum(1 for r in rows if not r.get("abstained", True)
                            and r.get("faithfulness") is None),
        "cite_min_score": floor,
        "routed": not args.no_route,
        "positive_faithfulness_mean": (
            round(sum(pos_scored) / len(pos_scored), 3) if pos_scored else None),
        "positive_false_abstain": sum(1 for r in pos if r.get("abstained")),
        "negative_total": len(neg),
        "negative_abstained": sum(1 for r in neg if r.get("abstained")),
        "negative_answered": len(neg_answered),
        "negative_leak_faithfulness": [r.get("faithfulness") for r in neg_answered],
    }
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"faithfulness_{stamp}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
