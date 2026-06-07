-- transcript-rag analytics schema. Per PRD §6 Phase 6 + Phase 12.
-- Idempotent: safe to re-run. Phase 12 columns appended via
-- ALTER ... ADD COLUMN IF NOT EXISTS so a stale DB upgrades in place.

CREATE TABLE IF NOT EXISTS chunk_meta (
    chunk_id     UUID PRIMARY KEY,
    source_file  TEXT NOT NULL,
    speakers     TEXT[] NOT NULL DEFAULT '{}',
    start_sec    FLOAT,
    end_sec      FLOAT,
    word_count   INTEGER,
    text         TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_speakers ON chunk_meta USING GIN (speakers);
CREATE INDEX IF NOT EXISTS idx_chunk_source ON chunk_meta(source_file);
-- Full-text index for analytics. The corpus is Hindi, so the config is
-- 'simple' (lowercase + tokenise, no English stemming or stopwords). Analytics
-- queries in rag_api/analytics.py MUST use the same config to hit this index.
-- A database created before this change keeps an 'english' index — rebuild it
-- with infra/postgres/migrations/001_hindi_fts.sql.
CREATE INDEX IF NOT EXISTS idx_chunk_text_fts ON chunk_meta USING GIN (to_tsvector('simple', text));

CREATE TABLE IF NOT EXISTS file_meta (
    source_file   TEXT PRIMARY KEY,
    duration_sec  FLOAT,
    speakers      TEXT[],
    chunk_count   INTEGER,
    ingested_at   TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- Phase 12 — path-based metadata columns + indexes.
-- ============================================================================

ALTER TABLE chunk_meta ADD COLUMN IF NOT EXISTS session_date DATE;
ALTER TABLE chunk_meta ADD COLUMN IF NOT EXISTS track_type   TEXT;
ALTER TABLE chunk_meta ADD COLUMN IF NOT EXISTS track_title  TEXT;
ALTER TABLE chunk_meta ADD COLUMN IF NOT EXISTS location     TEXT;
ALTER TABLE chunk_meta ADD COLUMN IF NOT EXISTS event_id     TEXT;
ALTER TABLE chunk_meta ADD COLUMN IF NOT EXISTS season       TEXT;

CREATE INDEX IF NOT EXISTS idx_chunk_session_date ON chunk_meta(session_date);
CREATE INDEX IF NOT EXISTS idx_chunk_track_type   ON chunk_meta(track_type);
CREATE INDEX IF NOT EXISTS idx_chunk_location     ON chunk_meta(location);
CREATE INDEX IF NOT EXISTS idx_chunk_event_id     ON chunk_meta(event_id);
CREATE INDEX IF NOT EXISTS idx_chunk_season       ON chunk_meta(season);

ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS session_date DATE;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS track_type   TEXT;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS location     TEXT;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS event_id     TEXT;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS season       TEXT;

CREATE INDEX IF NOT EXISTS idx_file_session_date ON file_meta(session_date);
CREATE INDEX IF NOT EXISTS idx_file_track_type   ON file_meta(track_type);
CREATE INDEX IF NOT EXISTS idx_file_location     ON file_meta(location);
CREATE INDEX IF NOT EXISTS idx_file_event_id     ON file_meta(event_id);
CREATE INDEX IF NOT EXISTS idx_file_season       ON file_meta(season);

-- ============================================================================
-- Phase 13 — content-based tags (per-file LLM enrichment).
-- ============================================================================
-- These columns hold per-file metadata extracted by Qwen 2.5 7B reading the
-- whole transcript. Path metadata (Phase 12) says when/where it was recorded;
-- these columns say what the audio is *about*. tagged_at NULL means "not yet
-- tagged" — the enrichment script is resumable on this predicate.

ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS event_type            TEXT;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS primary_language      TEXT;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS topics                TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS people_named          TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS places_named          TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS scriptures_referenced TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS timing_clues          TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS location_clues        TEXT[];
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS summary_hindi         TEXT;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS summary_english       TEXT;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS tagged_at             TIMESTAMP;
ALTER TABLE file_meta ADD COLUMN IF NOT EXISTS tag_model             TEXT;

CREATE INDEX IF NOT EXISTS idx_file_event_type       ON file_meta(event_type);
CREATE INDEX IF NOT EXISTS idx_file_primary_language ON file_meta(primary_language);
CREATE INDEX IF NOT EXISTS idx_file_tagged_at        ON file_meta(tagged_at);
CREATE INDEX IF NOT EXISTS idx_file_topics           ON file_meta USING GIN (topics);
CREATE INDEX IF NOT EXISTS idx_file_people_named     ON file_meta USING GIN (people_named);
CREATE INDEX IF NOT EXISTS idx_file_places_named     ON file_meta USING GIN (places_named);
CREATE INDEX IF NOT EXISTS idx_file_scriptures_ref   ON file_meta USING GIN (scriptures_referenced);

-- ============================================================================
-- Phase 14 — curated catalog enrichment ('Camp Record Export' spreadsheet).
-- ============================================================================
-- A 34-year human-curated index of the same discourse corpus. These tables
-- hold the catalog verbatim (loaded by ingestion/catalog/backfill.py); they
-- are the authoritative source for performer credits, canonical (Devanagari)
-- song titles, release references, and per-sitting hand transcriptions.
-- Rows align to transcripts on the date-anchored join key
-- ('YYYY-MM-DD|SEQ|TRACKNO') — the same key path_parser derives from folders.
-- These tables are STANDALONE: loading them never touches chunk_meta /
-- file_meta, so the running retrieval path is unaffected until backfill
-- explicitly enriches Qdrant payloads.

CREATE TABLE IF NOT EXISTS catalog_sitting (
    sitting_key     TEXT PRIMARY KEY,          -- 'YYYY-MM-DD|SEQ|'
    location        TEXT,                      -- normalized city facet
    camp_place      TEXT,                      -- raw place, for display
    camp_year       INTEGER,
    year_reliable   BOOLEAN,
    session_date    DATE,
    session_seq     INTEGER,
    session_time    TEXT,
    season          TEXT,
    track_count     INTEGER,
    performers      TEXT[] DEFAULT '{}',
    release_ref     TEXT,
    venue           TEXT,
    detail_contents TEXT,                      -- per-sitting hand transcription
    loaded_at       TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cat_sit_date       ON catalog_sitting(session_date);
CREATE INDEX IF NOT EXISTS idx_cat_sit_location   ON catalog_sitting(location);
CREATE INDEX IF NOT EXISTS idx_cat_sit_season     ON catalog_sitting(season);
CREATE INDEX IF NOT EXISTS idx_cat_sit_performers ON catalog_sitting USING GIN (performers);

CREATE TABLE IF NOT EXISTS catalog_track (
    join_key            TEXT PRIMARY KEY,       -- 'YYYY-MM-DD|SEQ|TRACKNO'
    sitting_key         TEXT REFERENCES catalog_sitting(sitting_key),
    location            TEXT,
    session_date        DATE,
    session_seq         INTEGER,
    track_no            INTEGER,
    track_title         TEXT,                   -- canonical song/bhajan title
    track_type          TEXT,
    duration_sec        INTEGER,
    performers          TEXT[] DEFAULT '{}',
    matched_source_file TEXT,                   -- set by backfill on alignment
    loaded_at           TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cat_trk_sitting  ON catalog_track(sitting_key);
CREATE INDEX IF NOT EXISTS idx_cat_trk_date     ON catalog_track(session_date);
CREATE INDEX IF NOT EXISTS idx_cat_trk_matched  ON catalog_track(matched_source_file);
