# Curated catalog enrichment (`ingestion/catalog/`)

Ingests the client's **`Camp Record Export.xlsx`** — a 34-year (1980–2018),
~29.5k-row human-curated index of the same discourse corpus the pipeline
transcribes — and uses it to make retrieval more accurate: real performer
credits, canonical (Devanagari) song titles, release references, and per-sitting
hand transcriptions, all filterable.

Everything here is **additive**. Loading the catalog never rewrites the running
`chunk_meta` / `file_meta` tables; the only mutation of the live search index is
the explicit, opt-in `backfill --commit` step.

## How it fits together

```
 Camp Record Export.xlsx
        │
        │  normalize.py   (pure: clean text, parse Sitting, durations,
        ▼                  performers; reuse path_parser for season/track_type)
   CatalogTrack records ──────────────► join_key  =  YYYY-MM-DD | SEQ | TRACKNO
        │                                          (identical to the key
        │                                           path_parser derives from a
        │                                           folder — that's the sync)
        ├── load.py     ──► catalog_tracks.csv + catalog_sittings.csv
        │                   + catalog_report.json   (+ optional --audio-root
        │                                             sync-coverage report)
        │
        └── backfill.py ──► Postgres: catalog_sitting / catalog_track tables
                            Qdrant:   set_payload performers / catalog_title /
                                      location onto MATCHED chunk points,
                                      selected by the join-key FILTER
                                      (session_date + session_seq + track_no)
```

### Why the key is date-anchored, not location-anchored

A camp happens on specific dates, so `date + seq + track_no` identifies a track
**globally**. Location is deliberately *not* in the key: audio folders carry a
neighbourhood (`PITAMPURA DELHI`) while the catalog carries the city (`Delhi`,
with spelling drift like `Chhattarpur`). Measured on the accessible slice:
location-keyed matched 30/64 audio tracks; date-keyed matched **63/63**.
`normalize_location()` folds location to a clean city facet for *filtering only*.

### Why backfill matches by filter, not filename

Track filenames repeat across every sitting (`06 SAMBODHAN.json` exists in
hundreds of camps). Matching on the bare filename would cross-contaminate. The
backfill instead pushes each catalog track's payload to the Qdrant points whose
own `session_date + session_seq + track_no` equal the join key — unambiguous,
and a no-op for tracks whose audio isn't ingested yet.

## Files

| File | Role |
|---|---|
| `normalize.py` | Pure normalisation + join-key logic. No I/O. Unit-tested. |
| `load.py`      | CLI: spreadsheet → clean CSVs + a sync-coverage report. Read-only wrt the stack. |
| `backfill.py`  | **Merge** path: catalog → Postgres tables + Qdrant payload enrichment of existing chunks. Dry-run by default. |
| `index_catalog.py` | **Separate-source** path (chosen): catalog → its own `catalog` Qdrant collection (detail_contents chunks + titles). Dry-run by default. Paired with `open_webui_functions/search_catalog.py` + `infra/qdrant/qdrant_catalog_setup.py`. |

## Two ways to expose it (pick one)

- **Separate source (current):** index into a standalone `catalog` collection and
  search it with the parallel `search_catalog` tool. Zero risk to the running
  index, clean Whisper-vs-catalog accuracy comparison, trivially reversible. Cost:
  result-level duplication (dedup by `join_key` later). See
  [docs/catalog_enrichment.md](../../docs/catalog_enrichment.md) §2.5.
- **Merge (`backfill.py`):** enrich existing transcript chunks in place. One
  unified result, no duplication, but mutates the live index and needs the
  transcripts ingested first. On the shelf for later.

## Running it

Produce clean tables + a coverage report (writes only to `data/catalog/`, which
is gitignored):

```bash
python -m ingestion.catalog.load \
    --xlsx "/mnt/c/Users/Pc/Downloads/Camp Record Export.xlsx" \
    --out-dir data/catalog \
    --audio-root "/mnt/d/Transcription whisperx/Output" \
    --audio-glob "**/*.cleaned.json"
```

Preview the live enrichment without writing anything:

```bash
python -m ingestion.catalog.backfill \
    --xlsx "/mnt/c/Users/Pc/Downloads/Camp Record Export.xlsx" \
    --qdrant http://localhost:6333 --postgres "$POSTGRES_DSN" \
    --audio-root "/mnt/d/Transcription whisperx/Output"
```

Apply it (after `infra/postgres/analytics_schema.sql` and
`infra/qdrant/qdrant_setup.py` have been run, and the transcripts are ingested):

```bash
python -m ingestion.catalog.backfill --xlsx ... \
    --postgres "$POSTGRES_DSN" --qdrant http://localhost:6333 \
    --collection transcripts --audio-root ... --commit
```

The new `performers` filter then flows through `rag_api` retrieval and the Open
WebUI `search_transcripts` tool automatically (e.g. *"bhajans sung by Abhipsa"*).

## Tests

`tests/unit/test_catalog_normalize.py` — 42 cases, including
`test_join_key_parity_with_path_parser` (catalog row ↔ folder path agree) and
`test_join_key_is_location_independent` (neighbourhood vs city still align).
