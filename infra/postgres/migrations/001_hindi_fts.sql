-- Migration 001 — Hindi-correct full-text search.
--
-- The analytics full-text index was created with the 'english' text-search
-- config: `to_tsvector('english', text)`. The corpus is Hindi (Devanagari),
-- so the English stemmer and stopword list mangle the lexemes — full-text
-- counts miss matches and over-match on stemmed noise.
--
-- This migration rebuilds the index with the 'simple' config (lowercase +
-- tokenise, no stemming, no stopwords), which is correct for Hindi. The
-- analytics queries in rag_api/analytics.py use 'simple' to match; Postgres
-- only uses a functional index when the query expression is textually
-- identical, so the index and the queries must move together.
--
-- WHEN TO RUN: once, on an existing database whose idx_chunk_text_fts was
-- built before this change. A database freshly created from
-- analytics_schema.sql already has the 'simple' index — skip this.
--
-- COST: rebuilding a GIN index over hundreds of millions of chunks is a
-- long, I/O-heavy operation. The statements below use CONCURRENTLY so writes
-- (ingestion) are not blocked. CONCURRENTLY cannot run inside a transaction
-- block — run this file with `psql -f`, NOT wrapped in BEGIN/COMMIT.
--
--   psql "$POSTGRES_DSN" -f infra/postgres/migrations/001_hindi_fts.sql
--
-- During the rebuild window analytics full-text queries fall back to a
-- sequential scan (slow but correct). Verify afterwards with:
--   \d+ chunk_meta   -- idx_chunk_text_fts should show to_tsvector('simple', text)

DROP INDEX CONCURRENTLY IF EXISTS idx_chunk_text_fts;

CREATE INDEX CONCURRENTLY idx_chunk_text_fts
    ON chunk_meta USING GIN (to_tsvector('simple', text));
