-- Migration 004 — play_events: one row per listen of a recording.
--
-- The dashboard's track player (frontend TrackPanel) POSTs
-- /api/track/played once per listen; the best-sitting ranking
-- (rag_api/best.py) reads COUNT(*) per source_file as its popularity
-- signal. Append-only; no updates, no deletes.
--
-- WHEN TO RUN: once, before enabling best-sitting ranking. Until then the
-- ranking treats every play count as 0 (logged, fail-open).
--
--   psql "$POSTGRES_DSN" -f infra/postgres/migrations/004_play_events.sql

CREATE TABLE IF NOT EXISTS play_events (
    id          BIGSERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    played_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_play_events_source ON play_events(source_file);
