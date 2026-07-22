/**
 * TypeScript mirror of the rag-api JSON contract (rag_api/app.py,
 * rag_api/retrieval.py). Keep these in sync with the
 * backend — they are the only guard against silent contract drift.
 */

// ---- /api/health ---------------------------------------------------------

export interface Health {
  ok: boolean;
  services: { ollama: boolean; qdrant: boolean; reranker: boolean };
  bm25_enabled: boolean;
  tantivy_docs: number;
  retrieval_backend: Backend;
  pageindex_trees: number;
  auth_required: boolean;
  chat_provider: ChatProvider;
  chat_model: string;
  embed_model: string;
}

// ---- /api/models ---------------------------------------------------------

export interface ChatModel {
  provider: ChatProvider;
  model: string;
  label: string;
  is_default: boolean;
}

export interface ModelsResponse {
  models: ChatModel[];
}

// ---- request shapes ------------------------------------------------------

export type Scope = "chunks" | "summaries" | "two_stage" | "catalog";
export type AnswerLanguage = "auto" | "hindi" | "english";

/**
 * Retrieval engine. "hybrid" is the default Qdrant+BM25+rerank path;
 * "pageindex" is the opt-in LLM tree-reasoning backend for A/B comparison.
 * Mirror of rag_api.app.RetrievalBackend.
 */
export type Backend = "hybrid" | "pageindex";

/** Optional metadata filters. Mirror of rag_api.app.FilterModel. */
export interface Filters {
  speaker?: string | null;
  source_file?: string | null;
  season?: string | null;
  track_type?: string[] | null;
  location?: string | null;
  event_id?: string | null;
  date_range?: [string, string] | null;
  event_type?: string | null;
  primary_language?: string | null;
  topics?: string[] | null;
  people_named?: string[] | null;
  scriptures_referenced?: string[] | null;
  /** Phase 14: curated-catalog performer credits (singers/chorus). */
  performers?: string[] | null;
  /** Phase 14: camp year (sheet column CampYear). */
  year?: string | null;
}

export interface SearchBody {
  query: string;
  find_quote: boolean;
  scope: Scope;
  backend: Backend;
  top_k: number;
  filters: Filters;
  auto_filters: boolean;
  expand_query: boolean;
}

export type ChatProvider = "ollama" | "openrouter";

/** Per-request model override. Both fields must be set together. */
export interface ModelChoice {
  provider: ChatProvider;
  model: string;
}

export interface QueryBody extends SearchBody {
  answer_language: AnswerLanguage;
  stream: boolean;
  /** Optional override; backend validates against the env allowlist. */
  provider?: ChatProvider;
  model?: string;
  /**
   * Prior turns of the open thread, oldest first. When present and the query
   * looks like a follow-up, the backend rewrites it to a standalone query
   * before retrieval (RAG_ROUTER + RAG_FOLLOWUP_REWRITE). Empty = no rewrite.
   */
  history?: { question: string; answer: string }[];
}

// ---- /api/filters --------------------------------------------------------

/** Distinct file_meta values, keyed as rag_api.db.get_filter_options emits. */
export interface FilterOptions {
  seasons: string[];
  locations: string[];
  event_ids: string[];
  track_types: string[];
  event_types: string[];
  primary_languages: string[];
  topics: string[];
  scriptures_referenced: string[];
  /** Phase 14: performer credits from the curated catalog. */
  performers: string[];
  /** Phase 14: camp years from the curated catalog (newest first). */
  years: string[];
}

export interface FilterOptionsResponse {
  db_ok: boolean;
  options: Partial<FilterOptions>;
}

// ---- detections + results ------------------------------------------------

/** One signal pulled from the query text by rag_api.query_parse. */
export interface Detection {
  field: string;
  value: string | string[] | [string, string];
  matched: string;
  confidence: "strong" | "soft";
}

/** A chunk or summary hit — uniform shape from to_result / to_summary_result. */
export interface RetrievalResult {
  result_type: "chunk" | "summary" | "catalog";
  /** Which store the hit came from (Phase 14). */
  source?: "transcript" | "catalog";
  chunk_id: string;
  score: number;
  text: string;
  source_file: string | null;
  start_sec: number | null;
  end_sec: number | null;
  speakers: string[];
  summary_english?: string | null;
  summary_hindi?: string | null;
  metadata: Record<string, unknown>;
}

// ---- /api/search ---------------------------------------------------------

export interface SearchResponse {
  query: string;
  find_quote: boolean;
  scope: string;
  // Optional: present on a live /api/search response, absent on a turn
  // hydrated from saved history (which doesn't persist engine/timing).
  backend?: Backend;
  retrieval_ms?: number;
  count: number;
  results: RetrievalResult[];
  detected_filters: Detection[];
  applied_filters: Record<string, unknown>;
  expanded: boolean;
}

// ---- /api/query (streaming meta event) -----------------------------------

export interface QueryMeta {
  query: string;
  find_quote: boolean;
  scope: string;
  // Optional: present on live /api/query meta events, absent on metas
  // hydrated from saved history (which doesn't persist engine/timing).
  backend?: Backend;
  retrieval_ms?: number;
  answer_language: string;
  count: number;
  citations: RetrievalResult[];
  detected_filters: Detection[];
  applied_filters: Record<string, unknown>;
  expanded: boolean;
}

// ---- /api/history --------------------------------------------------------

/**
 * Sidebar listing — one entry per conversation (thread), light payload with no
 * answer/citation bodies. `id` is the conversation_id; `title` is the first
 * turn's; `created_at` is the thread's latest activity (so continuing an old
 * chat floats it to the top).
 */
export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;   // ISO timestamp
  mode: "answer" | "search";
  scope: string;
  /** Number of Q&A turns in the thread. */
  turn_count: number;
}

export interface ConversationListResponse {
  items: ConversationSummary[];
  count: number;
}

/** One persisted Q&A turn — what POST /api/history returns, and the elements
 * of a conversation's `turns`. `conversation_id` groups turns into a thread. */
export interface ConversationTurn {
  id: string;
  conversation_id: string;
  turn_index: number;
  title: string;
  created_at: string;
  question: string;
  answer: string;
  mode: "answer" | "search";
  scope: string;
  answer_language: string | null;
  find_quote: boolean;
  expanded: boolean;
  top_k: number;
  filters: Filters;
  applied_filters: Record<string, unknown>;
  detected_filters: Detection[];
  citations: RetrievalResult[];
}

/** Full conversation returned by GET /api/history/{id} — all turns, oldest
 * first, under one thread. */
export interface ConversationRecord {
  id: string;
  title: string;
  created_at: string;
  mode: "answer" | "search";
  scope: string;
  turns: ConversationTurn[];
}

/** Body for POST /api/history. */
export interface HistorySaveBody {
  question: string;
  answer: string;
  mode: "answer" | "search";
  scope: Scope;
  top_k: number;
  find_quote: boolean;
  expanded: boolean;
  answer_language: string | null;
  filters: Filters;
  applied_filters: Record<string, unknown>;
  detected_filters: Detection[];
  citations: RetrievalResult[];
  /** Append to this thread; omit/null to start a new conversation. */
  conversation_id?: string | null;
}
