# Stage 4.3 — conversational follow-up rewriting A/B

Recorded against the live stack. A context-dependent follow-up ("उसके बाद क्या
हुआ", "उन्होंने और कहाँ इसका ज़िक्र किया") carries no entity for the retriever —
the referent lives in the previous turn. `rag_api.followup.FollowupRewriter`
rewrites it to a standalone query (one `think=false` chat call) using the
request `history`, *before* retrieval. Gated by `RAG_FOLLOWUP_REWRITE` (default
off) + `router_enabled`, and fires only when the router tags the query
`followup` (history present + anaphoric shape).

## Metric

Hit@5 against the golden entity files is the wrong lens here: it inherits the
cross-script entity gap and the golden set lists only a *subset* of genuinely
relevant files. The discriminating signal for a follow-up rewrite is whether it
restores **entity-relevant retrieval** — how many of the top-5 chunks actually
mention the referent. Raw anaphoric follow-up → the entity is gone → ~0;
rewritten → the entity is back.

## Result (entity-relevant chunks in top-5, bm25_weight=0.75)

| referent | raw follow-up | rewritten | rewrite produced |
|---|--:|--:|---|
| पुरन सिंह  | **0** | **3** | "...पुरन सिंह के भक्ति का वर्णन... कितने प्रवचनों में इसका ज़िक्र" |
| विवेकानंद | **0** | **5** | "स्वामी विवेकानंद के बारे में और कहाँ विस्तार से कहा गया है" |
| नामदेव    | **0** | **2** | "नामदेव की भक्ति की कथा और कहाँ आती है" |

The raw follow-up surfaces zero entity-relevant chunks in every case; the
rewrite restores 2–5 by resolving the anaphora and naming the entity. This is
the ship criterion met (standalone-rewrite retrieval > raw-follow-up retrieval),
with the round-trip firing only when `history` is non-empty and the class is
`followup`.

## Verdict

Ships flag-gated, **default off** (`RAG_FOLLOWUP_REWRITE`). The rewrite demonstrably
recovers retrieval the raw follow-up loses, at one ~1 s chat round-trip that
fires only for genuine follow-ups. It stays off by default because (a) a
multi-turn golden set for a full regression A/B does not exist yet, and (b) the
turn is already synthesis-bound past the 2 s budget — enabling it is an operator
call backed by this demonstration. Note: this improves *retrieval* for
follow-ups; giving the *answer* model the conversation context is a separate,
out-of-scope concern.

## Reproduce

```bash
set -a; source .env; set +a
# (script in the Stage 4.3 build-log entry / this file's git history)
python -m pytest tests/unit/test_rag_api_followup.py -q
```
