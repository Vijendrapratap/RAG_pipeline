# §0.3 — Chunk-quality audit summary

**Plan ref:** `docs/plan_foundation.md` §0.3. **Run:** 2026-07-09, full scan of
`chunk_meta` (24,567 chunks / 9,327 files) on the live Postgres. **Reproduce:**
`python -m ingestion.audit_chunks`. Work-list + per-flagged-chunk detail:
`ingestion/chunk_audit_worklist.json`.

**Tokenizer:** real **`BAAI/bge-m3`** (XLM-RoBERTa SentencePiece, the same tokenizer
Ollama serves), loaded offline — **not** `words × 1.3`, which under-counts Devanagari
(a 7-word Hindi phrase is 16 real tokens, not 9).

**Thresholds:** ASR loop = unique-word ratio `< 0.15` with `≥ 50` words; oversize =
`> 6000` chars (framing) and `> 8192` real tokens (bge-m3's hard context limit).

## Results — reconciled with the plan's live numbers

The plan's figures were scoped to oversize (>6k-char) chunks. At that scope this audit
reproduces them **exactly**; the full-corpus scan then shows the larger picture.

| metric | this audit | plan | note |
|---|---:|---:|---|
| chunks > 6000 chars | **138** | 138 | ✅ |
| — of which ASR loops | **124** | 124 | ✅ |
| — legit-long (healthy, non-loop) | **14** | 14 | ✅ subdivide in §1.2 |
| worst chunk chars | **586,331** | 586,331 | ✅ |
| worst chunk (words / unique) | 101,633 / 46 (ratio 0.0005) | 42 unique | ~ |
| **files with a >6k-char chunk** | **50** | 50 | ✅ **plan's work-list** |
| ASR loops **corpus-wide** (any size) | **574** | — | 450 are sub-6k |
| chunks > real **8192 tokens** | **42** (only **3** healthy) | — | 22 files |
| files with **any** loop chunk | **200** | — | 160 only sub-6k loops |
| **work-list files total** (any defect) | **210** | — | |

Worst offenders are `MEDITATION`, `COMPLETE SITTING (ZOOM RECORDER)`, and
`OM GURUVE NAMAH` tracks — Whisper looping a single phrase for tens of thousands of
words (up to 169,412 real tokens in one chunk).

## Scope decision this hands to §1.4 (Stage 1, not §0)

- **Plan's stated scope — 50 files:** every file with a >6k-char chunk. This is the
  enrichment-poison set (§2.3 reconstructs each transcript from `chunk_meta.text`; a
  90k-word loop is what poisons the tagger). Re-chunking these is the critical path.
- **Full clean — 210 files:** also removes the 450 sub-6k loops across ~160 more files.
  Higher retrieval-noise cleanup, but off the enrichment critical path.

Both work-lists are in `chunk_audit_worklist.json` (each file row carries
`loop_chunks` / `over_char_chunks` / `over_token_chunks`), so §1.4 can select either the
50-file oversize set or the full 210. **Recommendation:** §1.4 does the 50 oversize files
first (unblocks §2.3), then optionally sweeps the remaining loop-only files.
