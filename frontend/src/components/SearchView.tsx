import { useEffect, useRef, useState } from "react";

import {
  ApiError,
  getFilters,
  getHistoryItem,
  listModels,
  saveHistory,
  search,
  streamQuery,
} from "../api";
import { cleanFilters } from "../filters";
import { BrandMark } from "./BrandMark";
import type {
  AnswerLanguage,
  Backend,
  ChatModel,
  ChatProvider,
  ConversationTurn,
  Filters,
  FilterOptions,
  QueryMeta,
  RetrievalResult,
  Scope,
  SearchResponse,
} from "../types";

import { AnswerPane } from "./AnswerPane";
import { DetectedFilters } from "./DetectedFilters";
import { FilterPanel } from "./FilterPanel";
import { QueryBar, type Mode } from "./QueryBar";
import { ResultList } from "./ResultList";

interface Props {
  onAuthFail: () => void;
  activeConversationId: string | null;
  /** Called after a turn has been persisted to history. */
  onConversationSaved: () => void;
}

/**
 * One completed Q&A turn in the open thread. `answer`/`meta` hold the chat
 * response; `searchResp` holds a retrieval-only turn. Exactly one is populated
 * per turn depending on `mode`.
 */
interface CompletedTurn {
  question: string;
  mode: Mode;
  answer: string;
  meta: QueryMeta | null;
  searchResp: SearchResponse | null;
  error: string | null;
}

/**
 * A fresh conversation id, minted client-side so every turn of a new chat
 * shares it from the first submit (no dependency on the save round-trip).
 * `crypto.randomUUID` needs a secure context, which a plain-http LAN origin is
 * not — hence the RFC-4122 v4 fallback. Grouping ids need no crypto strength.
 */
function newConversationId(): string {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (ch) => {
    const r = (Math.random() * 16) | 0;
    const v = ch === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/** Rebuild a turn's response state from a persisted history turn. */
function turnFromRecord(t: ConversationTurn): CompletedTurn {
  const isSearch = t.mode === "search";
  const meta: QueryMeta | null = isSearch
    ? null
    : {
        query: t.question,
        find_quote: t.find_quote,
        scope: t.scope,
        answer_language: t.answer_language ?? "auto",
        count: t.citations.length,
        citations: t.citations,
        detected_filters: t.detected_filters,
        applied_filters: t.applied_filters,
        expanded: t.expanded,
      };
  const searchResp: SearchResponse | null = isSearch
    ? {
        query: t.question,
        find_quote: t.find_quote,
        scope: t.scope,
        count: t.citations.length,
        results: t.citations,
        detected_filters: t.detected_filters,
        applied_filters: t.applied_filters,
        expanded: t.expanded,
      }
    : null;
  return {
    question: t.question,
    mode: t.mode,
    answer: t.answer ?? "",
    meta,
    searchResp,
    error: null,
  };
}

/**
 * Chat tab. Owns a multi-turn conversation: a transcript of completed turns
 * (`turns`) plus the in-flight turn currently streaming. Asking a question
 * appends to the open thread instead of replacing it, and every turn of the
 * session is persisted under one `conversationId`. Reopening a saved chat
 * hydrates its turns and stays continuable — new questions append to it.
 */
export function SearchView({
  onAuthFail,
  activeConversationId,
  onConversationSaved,
}: Props) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("answer");
  const [scope, setScope] = useState<Scope>("chunks");
  const [backend, setBackend] = useState<Backend>("hybrid");
  const [topK, setTopK] = useState(4);
  const [findQuote, setFindQuote] = useState(false);
  const [autoFilters, setAutoFilters] = useState(true);
  const [expandQuery, setExpandQuery] = useState(false);
  const [answerLanguage, setAnswerLanguage] = useState<AnswerLanguage>("auto");
  const [filters, setFilters] = useState<Filters>({});
  const [filterDrawerOpen, setFilterDrawerOpen] = useState(false);

  const [options, setOptions] = useState<Partial<FilterOptions> | null>(null);
  const [dbOk, setDbOk] = useState(true);
  const [models, setModels] = useState<ChatModel[]>([]);
  // Composite "provider/model" key. null = use whatever the backend picked
  // as default. Persisted across submits but reset by "New chat" via remount.
  const [modelKey, setModelKey] = useState<string | null>(null);

  // The open thread: completed turns + the id they persist under (null until
  // the first turn saves, or until a saved chat is hydrated).
  const [turns, setTurns] = useState<CompletedTurn[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);

  // The in-flight (or just-submitted) turn — cleared into `turns` on completion.
  const [running, setRunning] = useState(false);
  const [meta, setMeta] = useState<QueryMeta | null>(null);
  const [answer, setAnswer] = useState("");
  const [searchResp, setSearchResp] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastSubmitted, setLastSubmitted] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await getFilters();
        if (cancelled) return;
        setOptions(r.options);
        setDbOk(r.db_ok);
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 401) onAuthFail();
        else setDbOk(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [onAuthFail]);

  // Selectable models — /api/models is unauthenticated so we can hit it
  // before login too, but only one tab actually needs it.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await listModels();
        if (cancelled) return;
        setModels(r.models);
      } catch {
        // Non-fatal: dropdown just hides itself when the list is empty.
        if (!cancelled) setModels([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Hydrate the whole thread from a saved conversation when the active id is
  // non-null. The opposite transition (id → null) is owned by App via a remount
  // key, so we never reset state from inside this effect.
  useEffect(() => {
    let cancelled = false;
    if (!activeConversationId) return;
    (async () => {
      try {
        const record = await getHistoryItem(activeConversationId);
        if (cancelled) return;
        const loaded = record.turns.map(turnFromRecord);
        setTurns(loaded);
        setConversationId(record.id);
        // Clear the in-flight slots — the thread now lives entirely in `turns`.
        setQuery("");
        setLastSubmitted(null);
        setAnswer("");
        setMeta(null);
        setSearchResp(null);
        setError(null);
        // Seed the composer from the last turn so continuing keeps its settings.
        const last = record.turns[record.turns.length - 1];
        if (last) {
          setMode(last.mode);
          setScope(last.scope as Scope);
          setTopK(last.top_k);
          setFindQuote(last.find_quote);
          setExpandQuery(last.expanded);
          setFilters(last.filters);
        }
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 401) onAuthFail();
        else setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeConversationId, onAuthFail]);

  // Keep the conversation pinned to the bottom while streaming.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [answer, searchResp, lastSubmitted, turns]);

  // Cancel any in-flight stream when the component unmounts (the App swaps
  // out this instance on "New chat" via a key prop).
  useEffect(() => () => abortRef.current?.abort(), []);

  async function onSubmit() {
    const text = query.trim();
    const cleaned = cleanFilters(filters);
    const hasFilters = Object.keys(cleaned).length > 0;
    // Valid with query text OR active filters (the latter is a filter-only
    // catalog browse). A query-less browse always uses the search endpoint —
    // there is nothing to answer without a question.
    if ((!text && !hasFilters) || running) return;
    const browse = !text;

    // Clear the in-flight slots for the new turn (the previous turn is already
    // committed to `turns`). We deliberately do NOT touch `turns` — asking
    // appends to the open thread.
    setMeta(null);
    setAnswer("");
    setSearchResp(null);
    setError(null);
    setLastSubmitted(text || "Browse by filters");
    setQuery("");
    setRunning(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    // Per-request model override. Send only when the user explicitly picked
    // a non-default; null = let the backend use its env default.
    const picked = modelKey
      ? models.find((m) => `${m.provider}/${m.model}` === modelKey)
      : undefined;
    const modelOverride: { provider?: ChatProvider; model?: string } = picked
      ? { provider: picked.provider, model: picked.model }
      : {};
    const body = {
      query: text,
      find_quote: findQuote,
      scope,
      backend,
      top_k: topK,
      filters: cleaned,
      auto_filters: autoFilters,
      expand_query: expandQuery,
      ...modelOverride,
    };

    // Prior turns of the open thread — lets the backend rewrite a follow-up
    // into a standalone query before retrieval (answer mode only).
    const history = turns.map((t) => ({ question: t.question, answer: t.answer }));

    // The thread id — reuse the open one, or mint it now so this turn and every
    // later one persist under a single conversation (avoids a save-race split).
    const convId = conversationId ?? newConversationId();
    if (!conversationId) setConversationId(convId);

    let finalAnswer = "";
    let finalMeta = null as QueryMeta | null;
    let turnError: string | null = null;
    let liveSearch = null as SearchResponse | null;
    let aborted = false;

    try {
      if (mode === "answer" && !browse) {
        await streamQuery(
          { ...body, answer_language: answerLanguage, history },
          {
            onMeta: (m) => { finalMeta = m; setMeta(m); },
            onToken: (t) => {
              finalAnswer += t;
              setAnswer((prev) => prev + t);
            },
            onDone: () => {},
            onError: (detail) => { turnError = detail; setError(detail); },
          },
          ctrl.signal,
        );
      } else {
        const r = await search(body);
        liveSearch = r;
        setSearchResp(r);
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        // user-initiated stop — leave the partial turn visible but uncommitted
        aborted = true;
      } else if (e instanceof ApiError && e.status === 401) {
        onAuthFail();
        setRunning(false);
        abortRef.current = null;
        return;
      } else {
        turnError = e instanceof Error ? e.message : String(e);
        setError(turnError);
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }

    if (aborted) return;

    // Commit the finished turn to the transcript and clear the in-flight slots
    // (both in one tick, so the turn moves seamlessly into `turns`).
    const committed: CompletedTurn = {
      question: text || "Browse by filters",
      mode,
      answer: finalAnswer,
      meta: finalMeta,
      searchResp: liveSearch,
      error: turnError,
    };
    setTurns((prev) => [...prev, committed]);
    setLastSubmitted(null);
    setMeta(null);
    setAnswer("");
    setSearchResp(null);
    setError(null);

    // Persist the completed turn. Failures here don't block the user — the
    // answer is already on screen — but they do leave a console warning so
    // an absent /api/history (e.g. migration not run) is visible.
    if (turnError) return;
    try {
      await saveHistory({
        question: text,
        answer: finalAnswer,
        mode,
        scope,
        top_k: topK,
        find_quote: findQuote,
        expanded: !!(finalMeta?.expanded ?? liveSearch?.expanded),
        answer_language:
          finalMeta?.answer_language ?? (mode === "answer" ? answerLanguage : null),
        filters: cleaned,
        applied_filters:
          finalMeta?.applied_filters ?? liveSearch?.applied_filters ?? {},
        detected_filters:
          finalMeta?.detected_filters ?? liveSearch?.detected_filters ?? [],
        citations:
          finalMeta?.citations ?? liveSearch?.results ?? [],
        conversation_id: convId,
      });
      // Refresh the sidebar so the new / updated thread appears. We deliberately
      // don't flip `activeConversationId` — that would re-hydrate and interrupt
      // the user mid-conversation.
      onConversationSaved();
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        onAuthFail();
        return;
      }
      // Non-fatal — log and continue.
      console.warn("history save failed:", e);
    }
  }

  function onStop() {
    abortRef.current?.abort();
  }

  const isEmpty = turns.length === 0 && !lastSubmitted && !running;

  return (
    <div className="search">
      <div className="chat-scroll" ref={scrollRef}>
        <div className="chat-inner">
          {isEmpty ? (
            <Welcome />
          ) : (
            <>
              {turns.map((t, i) => (
                <TurnView
                  key={i}
                  question={t.question}
                  mode={t.mode}
                  answer={t.answer}
                  meta={t.meta}
                  searchResp={t.searchResp}
                  error={t.error}
                  running={false}
                  filters={filters}
                  onAddFilter={setFilters}
                />
              ))}
              {(lastSubmitted || running) && (
                <TurnView
                  question={lastSubmitted ?? ""}
                  mode={mode}
                  answer={answer}
                  meta={meta}
                  searchResp={searchResp}
                  error={error}
                  running={running}
                  filters={filters}
                  onAddFilter={setFilters}
                />
              )}
            </>
          )}
        </div>
      </div>

      <div className="composer-wrap">
        <QueryBar
          query={query} onQuery={setQuery}
          mode={mode} onMode={setMode}
          scope={scope} onScope={setScope}
          backend={backend} onBackend={setBackend}
          topK={topK} onTopK={setTopK}
          findQuote={findQuote} onFindQuote={setFindQuote}
          autoFilters={autoFilters} onAutoFilters={setAutoFilters}
          expandQuery={expandQuery} onExpandQuery={setExpandQuery}
          answerLanguage={answerLanguage} onAnswerLanguage={setAnswerLanguage}
          models={models} modelKey={modelKey} onModelKey={setModelKey}
          running={running} onSubmit={onSubmit} onStop={onStop}
          filters={filters}
          onOpenFilters={() => setFilterDrawerOpen(true)}
        />
      </div>

      {filterDrawerOpen && (
        <FilterPanel
          filters={filters}
          options={options}
          dbOk={dbOk}
          onChange={setFilters}
          onClose={() => setFilterDrawerOpen(false)}
        />
      )}
    </div>
  );
}

interface TurnViewProps {
  question: string;
  mode: Mode;
  answer: string;
  meta: QueryMeta | null;
  searchResp: SearchResponse | null;
  error: string | null;
  running: boolean;
  filters: Filters;
  onAddFilter: (next: Filters) => void;
}

/** One turn: the user's question bubble + the assistant's answer / results. */
function TurnView({
  question, mode, answer, meta, searchResp, error, running, filters, onAddFilter,
}: TurnViewProps) {
  const detections = mode === "answer"
    ? meta?.detected_filters ?? []
    : searchResp?.detected_filters ?? [];
  const applied = mode === "answer"
    ? meta?.applied_filters ?? {}
    : searchResp?.applied_filters ?? {};
  const expanded = mode === "answer"
    ? !!meta?.expanded
    : !!searchResp?.expanded;
  // Engine + retrieval latency, surfaced so the hybrid-vs-PageIndex speed
  // difference is visible per turn. Guarded with != null so turns hydrated
  // from older saved chats (no backend/timing fields) simply hide the badge.
  const badgeSrc = mode === "answer" ? meta : searchResp;
  const engineBadge =
    badgeSrc && badgeSrc.retrieval_ms != null
      ? { backend: badgeSrc.backend, ms: badgeSrc.retrieval_ms }
      : null;
  const liveResults: RetrievalResult[] = searchResp?.results ?? [];

  return (
    <>
      <div className="bubble-user">{question}</div>
      <div className="bubble-assistant">
        <BrandMark className="bubble-avatar bubble-avatar--logo" decorative />
        <div className="bubble-content">
          <DetectedFilters
            detections={detections}
            applied={applied}
            filters={filters}
            onAddFilter={onAddFilter}
            expanded={expanded}
          />
          {engineBadge && (
            <div
              className={"engine-badge engine-badge--" + engineBadge.backend}
              title="Retrieval engine and time spent fetching passages"
            >
              {engineBadge.backend === "pageindex"
                ? "PageIndex · LLM reasoning"
                : "Hybrid · vector + BM25"}
              {" · "}
              {Math.round(engineBadge.ms ?? 0)} ms
            </div>
          )}
          {mode === "answer" ? (
            <AnswerPane
              answer={answer}
              meta={meta}
              running={running}
              error={error}
            />
          ) : (
            <>
              {error && <div className="answer-error">{error}</div>}
              {searchResp && (
                <ResultList
                  results={liveResults}
                  count={searchResp.count}
                  query={searchResp.query}
                />
              )}
              {!searchResp && running && <span className="cursor" />}
            </>
          )}
        </div>
      </div>
    </>
  );
}

function Welcome() {
  return (
    <div className="welcome">
      <BrandMark className="welcome-mark welcome-mark--logo" />
      <span className="welcome-eyebrow">Vishvas Foundation · Discourse Archive</span>
      <h2>Ask the archive</h2>
      <p>
        Search through every recorded discourse in Hindi or English. Every
        answer is grounded in passages from the source recording, with
        citations you can open to read or listen.
      </p>
    </div>
  );
}
