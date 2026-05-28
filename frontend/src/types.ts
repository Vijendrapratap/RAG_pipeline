/**
 * TypeScript mirror of the rag-api JSON contract (rag_api/app.py,
 * rag_api/retrieval.py, rag_api/analytics.py). Keep these in sync with the
 * backend — they are the only guard against silent contract drift.
 */

// ---- /api/health ---------------------------------------------------------

export interface Health {
  ok: boolean;
  services: { ollama: boolean; qdrant: boolean; reranker: boolean };
  bm25_enabled: boolean;
  tantivy_docs: number;
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

export type Scope = "chunks" | "summaries" | "two_stage";
export type AnswerLanguage = "auto" | "hindi" | "english";

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
}

export interface SearchBody {
  query: string;
  find_quote: boolean;
  scope: Scope;
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
  result_type: "chunk" | "summary";
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
  answer_language: string;
  count: number;
  citations: RetrievalResult[];
  detected_filters: Detection[];
  applied_filters: Record<string, unknown>;
  expanded: boolean;
}

// ---- /api/analytics/* ----------------------------------------------------

export interface MentionsResponse {
  term: string;
  speaker: string | null;
  chunk_count: number;
}

export interface SpeakerCount {
  speaker: string;
  chunk_count: number;
}

export interface SpeakersResponse {
  term: string;
  speakers: SpeakerCount[];
}

export interface TranscriptCount {
  source_file: string;
  chunk_count: number;
}

export interface TranscriptsResponse {
  term: string;
  transcripts: TranscriptCount[];
}

// ---- /api/history --------------------------------------------------------

/** Sidebar listing — light payload, no answer/citation bodies. */
export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;   // ISO timestamp
  mode: "answer" | "search";
  scope: string;
}

export interface ConversationListResponse {
  items: ConversationSummary[];
  count: number;
}

/** Full record returned by GET /api/history/{id} and POST /api/history. */
export interface ConversationRecord extends ConversationSummary {
  question: string;
  answer: string;
  answer_language: string | null;
  find_quote: boolean;
  expanded: boolean;
  top_k: number;
  filters: Filters;
  applied_filters: Record<string, unknown>;
  detected_filters: Detection[];
  citations: RetrievalResult[];
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
}
