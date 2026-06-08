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
import type {
  AnswerLanguage,
  Backend,
  ChatModel,
  ChatProvider,
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
  onActiveConversationChange: (id: string | null) => void;
  /** Called after a turn has been persisted to history. */
  onConversationSaved: () => void;
}

interface Example {
  text: string;
  hint: string;
}

const EXAMPLES: Example[] = [
  { text: "कर्म योग क्या है",                              hint: "Concept · Hindi" },
  { text: "What did Swami ji say about meditation?",        hint: "Theme · English" },
  { text: "In which event was the SAMBODHAN track recorded?", hint: "Find by source" },
  { text: "Top discourses on dharma from 2015",             hint: "Filtered search" },
];

/**
 * Chat tab. Owns the conversation state for the current turn (or the
 * read-only state hydrated from a past conversation when `activeConversationId`
 * is set). Asking a new question always starts a brand-new turn — past
 * conversations are immutable.
 */
export function SearchView({
  onAuthFail,
  activeConversationId,
  onActiveConversationChange,
  onConversationSaved,
}: Props) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("answer");
  const [scope, setScope] = useState<Scope>("chunks");
  const [backend, setBackend] = useState<Backend>("hybrid");
  const [topK, setTopK] = useState(8);
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

  const [running, setRunning] = useState(false);
  const [meta, setMeta] = useState<QueryMeta | null>(null);
  const [answer, setAnswer] = useState("");
  const [searchResp, setSearchResp] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastSubmitted, setLastSubmitted] = useState<string | null>(null);
  const [viewingHistory, setViewingHistory] = useState(false);

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

  // Hydrate from a saved conversation when the active id is non-null.
  // The opposite transition (id → null) is owned by App via a remount key,
  // so we never reset state from inside this effect — that race ate the
  // submit's setState calls before.
  useEffect(() => {
    let cancelled = false;
    if (!activeConversationId) return;
    (async () => {
      try {
        const record = await getHistoryItem(activeConversationId);
        if (cancelled) return;
        setQuery("");
        setLastSubmitted(record.question);
        setAnswer(record.answer ?? "");
        setMeta({
          query: record.question,
          find_quote: record.find_quote,
          scope: record.scope,
          answer_language: record.answer_language ?? "auto",
          count: record.citations.length,
          citations: record.citations,
          detected_filters: record.detected_filters,
          applied_filters: record.applied_filters,
          expanded: record.expanded,
        });
        setSearchResp(null);
        setError(null);
        setMode(record.mode);
        setScope(record.scope as Scope);
        setTopK(record.top_k);
        setFindQuote(record.find_quote);
        setExpandQuery(record.expanded);
        setFilters(record.filters);
        setViewingHistory(true);
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
  }, [answer, searchResp, lastSubmitted]);

  // Cancel any in-flight stream when the component unmounts (the App swaps
  // out this instance on "New chat" via a key prop).
  useEffect(() => () => abortRef.current?.abort(), []);

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
  // difference is visible per turn. Guarded with != null so metas hydrated
  // from older saved chats (no backend/timing fields) simply hide the badge.
  const badgeSrc = mode === "answer" ? meta : searchResp;
  const engineBadge =
    badgeSrc && badgeSrc.retrieval_ms != null
      ? { backend: badgeSrc.backend, ms: badgeSrc.retrieval_ms }
      : null;

  function resetResponse() {
    setMeta(null);
    setAnswer("");
    setSearchResp(null);
    setError(null);
  }

  async function onSubmit() {
    const text = query.trim();
    if (!text || running) return;

    // Submitting a question while viewing a past chat starts a fresh turn.
    if (viewingHistory || activeConversationId) {
      onActiveConversationChange(null);
      setViewingHistory(false);
    }

    resetResponse();
    setLastSubmitted(text);
    setQuery("");
    setRunning(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const cleaned = cleanFilters(filters);
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

    let finalAnswer = "";
    let finalMeta = null as QueryMeta | null;
    let turnError: string | null = null;
    let liveSearch = null as SearchResponse | null;

    try {
      if (mode === "answer") {
        await streamQuery(
          { ...body, answer_language: answerLanguage },
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
        // user-initiated stop — do not save partial turns
        setRunning(false);
        abortRef.current = null;
        return;
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
      });
      // Refresh the sidebar so the new entry appears. We deliberately don't
      // flip `activeConversationId` to the saved id — that would re-fetch and
      // toggle the view into read-only mode immediately after the user asked.
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

  function onExample(text: string) {
    setQuery(text);
  }

  const liveResults: RetrievalResult[] = searchResp?.results ?? [];
  const isEmpty = !lastSubmitted && !running;

  return (
    <div className="search">
      <div className="chat-scroll" ref={scrollRef}>
        <div className="chat-inner">
          {isEmpty ? (
            <Welcome onExample={onExample} />
          ) : (
            <>
              {viewingHistory && (
                <div className="history-banner">
                  Viewing a saved chat · ask a new question to start a fresh
                  one.{" "}
                  <button
                    className="link-btn"
                    onClick={() => onActiveConversationChange(null)}
                  >
                    New chat
                  </button>
                </div>
              )}
              {lastSubmitted && (
                <div className="bubble-user">{lastSubmitted}</div>
              )}

              <div className="bubble-assistant">
                <img className="bubble-avatar bubble-avatar--logo" src="/logo.svg" alt="" aria-hidden="true" />
                <div className="bubble-content">
                  <DetectedFilters
                    detections={detections}
                    applied={applied}
                    filters={filters}
                    onAddFilter={setFilters}
                    expanded={expanded}
                  />
                  {engineBadge && (
                    <div
                      className={
                        "engine-badge engine-badge--" + engineBadge.backend
                      }
                      title="Retrieval engine and time spent fetching passages"
                    >
                      {engineBadge.backend === "pageindex"
                        ? "PageIndex · LLM reasoning"
                        : "Hybrid · vector + BM25"}
                      {" · "}
                      {Math.round(engineBadge.ms)} ms
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
                        <ResultList results={liveResults} count={searchResp.count} />
                      )}
                      {!searchResp && running && <span className="cursor" />}
                    </>
                  )}
                </div>
              </div>
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

function Welcome({ onExample }: { onExample: (text: string) => void }) {
  return (
    <div className="welcome">
      <img className="welcome-mark welcome-mark--logo" src="/logo.svg" alt="Vishvas Foundation" />
      <span className="welcome-eyebrow">Vishvas Foundation · Discourse Archive</span>
      <h2>Ask the archive</h2>
      <p>
        Search through every recorded discourse in Hindi or English. Every
        answer is grounded in passages from the source recording, with
        citations you can open to read or listen.
      </p>
      <div className="welcome-divider" aria-hidden="true">
        <span className="welcome-divider-mark" />
      </div>
      <div className="welcome-examples-head">Try asking</div>
      <div className="welcome-examples">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.text}
            className="welcome-example"
            onClick={() => onExample(ex.text)}
            title={ex.hint}
          >
            <span className="welcome-example-text">{ex.text}</span>
            <span className="welcome-example-hint">{ex.hint}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
