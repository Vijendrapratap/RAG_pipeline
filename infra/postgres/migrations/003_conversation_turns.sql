-- Migration 003 — Group Q&A turns into multi-turn conversations.
--
-- The dashboard was single-turn: one conversation_history row == one sidebar
-- entry, and asking again started a brand-new immutable row. To support
-- ChatGPT/Claude-style linked chats (many turns in one thread, reopen and
-- continue), we group rows into a conversation.
--
-- This is ADDITIVE, not a rebuild:
--   * `id` stays the per-turn primary key.
--   * `conversation_id` groups the turns of one thread.
--   * `turn_index` orders turns within a thread (0-based).
-- Every existing single-turn row backfills to conversation_id = id, so it
-- becomes a valid one-turn conversation — zero data loss.
--
-- WHEN TO RUN: once, after 002_conversations.sql, on the same Postgres database.
--
--   psql "$POSTGRES_DSN" -f infra/postgres/migrations/003_conversation_turns.sql

ALTER TABLE conversation_history
    ADD COLUMN IF NOT EXISTS conversation_id UUID,
    ADD COLUMN IF NOT EXISTS turn_index       INTEGER NOT NULL DEFAULT 0;

-- Backfill: each pre-existing row is its own single-turn conversation.
UPDATE conversation_history
    SET conversation_id = id
    WHERE conversation_id IS NULL;

-- Default a fresh id so a writer that does NOT set conversation_id (e.g. the
-- previous single-turn rag-api still running during a rolling redeploy) keeps
-- working — each such insert lands as its own one-turn conversation. The new
-- code always sets conversation_id explicitly, so this default stays dormant
-- for it. gen_random_uuid() is core in Postgres 13+.
ALTER TABLE conversation_history
    ALTER COLUMN conversation_id SET DEFAULT gen_random_uuid();

ALTER TABLE conversation_history
    ALTER COLUMN conversation_id SET NOT NULL;

-- The reader fetches one thread's turns in order; the sidebar groups by thread.
CREATE INDEX IF NOT EXISTS idx_conversation_history_conversation
    ON conversation_history (conversation_id, turn_index);
