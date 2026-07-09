# §1.4 — Targeted re-chunk of the 50 poisoned files + bidirectional verify

**Plan ref:** `docs/plan_foundation.md` §1.4 / §1.4v. **Run:** 2026-07-09, live stack.
**Scope:** the 50 files carrying an oversize (>6k-char or >8192-token) chunk — the
enrichment-poison set (`ingestion/chunk_audit_worklist.json`, `--oversize-only`).

## What ran

1. **Re-chunk (staging, non-destructive)** — `chunker_cleaned.process_file` over the 50
   matched `*.raw.json` (base `/mnt/d/Transcription whisperx/Output`) into a staging dir.
   All 50 ok, 0 failed; **108 degenerate ASR-loop chunks dropped**, 0 single-sentence
   oversize raises; 4 files correctly fell back to raw text (cleaned was lossy).
   487 old chunks → **334 new chunks**.
2. **Delete** — `python -m ingestion.reindex_file --worklist … --oversize-only`:
   deleted **487 Qdrant points / 487 chunk_meta rows** across all stores; NULLed
   `file_meta.tagged_at`; reset progress.
3. **Re-ingest** — `bulk_ingest_hardened --chunks-dir <staging> --no-skip-existing`:
   50 files, 334 chunks, **0 failed**, 22.9 s. `tantivy commit + mark-ok at file 50`
   confirms the deferred-`mark_ok` commit-race fix fired.

Safety: the source `raw.json` transcripts are never touched (operation is recoverable by
re-ingest); a backup of the 50 files' pre-delete `chunk_meta` rows was taken first.

## §1.4v acceptance — all green

| Check | Result |
|---|---|
| **Store reconciliation** — chunk_meta == file_meta Σchunk_count == Qdrant == Tantivy | **24,414 == 24,414 == 24,414 == 24,414** ✅ |
| ASR-loop chunks (uniq < 0.15, ≥50 words) among the 50 | **0** (was 487→334 chunks) ✅ |
| Chunks > real **8192** bge-m3 tokens among the 50 | **0** (max 6,754) ✅ |
| Per-file reconcile chunk_meta == Qdrant == Tantivy (all 50) | **0 mismatches** ✅ |
| **BM25 ⊆ Qdrant** — Tantivy chunk_ids absent from Qdrant | **0 orphans** (334 == 334) ✅ |
| **Char preservation** — new body / baseline non-loop body (uniq on body, as the guard checks) | **0.9944** ✅ |

The pre-header-strip ratio (0.962) and body-only ratio vs the *full-text*-classified baseline
(0.9335) both understate preservation: 14 baseline chunks were "non-loop" only because header
words lifted their full-text uniq above 0.15, while their bodies are genuine loops (50,456 chars)
that the body-level guard correctly drops. Against a body-level-classified baseline, **99.44%**
of legitimate prose is preserved — the residual 0.56% is cleaned-vs-raw / repacking drift.

## Residual (out of §1.4 scope)

§1.4 cleaned the 50 oversize-poison files. The corpus still holds ~450 sub-6k ASR-loop chunks
across ~160 other files (audit `loop_chunk_count=574`, of which 124 were in this oversize set).
They are smaller and off the enrichment critical path; the 1.2 guards mean any future re-chunk
of those files drops them too. An optional `reindex_file --worklist` (no `--oversize-only`) sweep
would clear them before Stage 2 enrichment.
