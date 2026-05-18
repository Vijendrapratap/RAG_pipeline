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
CREATE INDEX IF NOT EXISTS idx_chunk_text_fts ON chunk_meta USING GIN (to_tsvector('english', text));

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
