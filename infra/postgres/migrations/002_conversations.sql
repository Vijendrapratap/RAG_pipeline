-- Migration 002 — Conversation history for the dashboard.
--
-- Stores one row per completed dashboard Q&A turn so the UI can show a
-- Claude-style "Recent" list in the sidebar and let users revisit past
-- answers in read-only mode. The current dashboard is single-tenant (shared
-- password), so there is no user_id — every saved row is visible to anyone
-- who can sign in. Add a user_id column if multi-user auth ever lands.
--
-- WHEN TO RUN: once, on the same Postgres database that backs analytics.
--
--   psql "$POSTGRES_DSN" -f infra/postgres/migrations/002_conversations.sql

CREATE TABLE IF NOT EXISTS conversation_history (
    id                UUID PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    title             TEXT        NOT NULL,
    question          TEXT        NOT NULL,
    answer            TEXT        NOT NULL DEFAULT '',
    mode              TEXT        NOT NULL,          -- 'answer' | 'search'
    scope             TEXT        NOT NULL,          -- 'chunks' | 'summaries' | 'two_stage'
    answer_language   TEXT,
    find_quote        BOOLEAN     NOT NULL DEFAULT FALSE,
    expanded          BOOLEAN     NOT NULL DEFAULT FALSE,
    top_k             INTEGER     NOT NULL,
    filters           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    applied_filters   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    detected_filters  JSONB       NOT NULL DEFAULT '[]'::jsonb,
    citations         JSONB       NOT NULL DEFAULT '[]'::jsonb
);

-- The sidebar always queries by recency, so the descending index pays off.
CREATE INDEX IF NOT EXISTS idx_conversation_history_created_at
    ON conversation_history (created_at DESC);
