import { useEffect, useRef, useState } from "react";

import { ApiError, getFilters, search, streamQuery } from "../api";
import { cleanFilters } from "../filters";
import type {
  AnswerLanguage,
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
}

/**
 * The Search tab: query bar + filter sidebar + answer / results pane.
 *
 * In "answer" mode the request goes to /api/query as a streaming SSE turn
 * (meta → tokens → done). In "search" mode it goes to /api/search and the
 * answer pane is hidden. Both share the same filter state + chip surface.
 */
export function SearchView({ onAuthFail }: Props) {
  // Query + control state
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("answer");
  const [scope, setScope] = useState<Scope>("chunks");
  const [topK, setTopK] = useState(8);
  const [findQuote, setFindQuote] = useState(false);
  const [autoFilters, setAutoFilters] = useState(true);
  const [expandQuery, setExpandQuery] = useState(false);
  const [answerLanguage, setAnswerLanguage] = useState<AnswerLanguage>("auto");
  const [filters, setFilters] = useState<Filters>({});

  // Server-side filter vocabulary for the dropdowns.
  const [options, setOptions] = useState<Partial<FilterOptions> | null>(null);
  const [dbOk, setDbOk] = useState(true);

  // Response state
  const [running, setRunning] = useState(false);
  const [meta, setMeta] = useState<QueryMeta | null>(null);
  const [answer, setAnswer] = useState("");
  const [searchResp, setSearchResp] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

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

  // Detections + applied filters live on the most recent response of either
  // mode — read whichever is current.
  const detections = mode === "answer"
    ? meta?.detected_filters ?? []
    : searchResp?.detected_filters ?? [];
  const applied = mode === "answer"
    ? meta?.applied_filters ?? {}
    : searchResp?.applied_filters ?? {};
  const expanded = mode === "answer"
    ? !!meta?.expanded
    : !!searchResp?.expanded;

  function resetResponse() {
    setMeta(null);
    setAnswer("");
    setSearchResp(null);
    setError(null);
  }

  async function onSubmit() {
    if (!query.trim() || running) return;
    resetResponse();
    setRunning(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const body = {
      query: query.trim(),
      find_quote: findQuote,
      scope,
      top_k: topK,
      filters: cleanFilters(filters),
      auto_filters: autoFilters,
      expand_query: expandQuery,
    };

    try {
      if (mode === "answer") {
        await streamQuery(
          { ...body, answer_language: answerLanguage },
          {
            onMeta: (m) => setMeta(m),
            onToken: (t) => setAnswer((prev) => prev + t),
            onDone: () => {},
            onError: (detail) => setError(detail),
          },
          ctrl.signal,
        );
      } else {
        const r = await search(body);
        setSearchResp(r);
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        // user-initiated stop — nothing to report
      } else if (e instanceof ApiError && e.status === 401) {
        onAuthFail();
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }

  function onStop() {
    abortRef.current?.abort();
  }

  const liveResults: RetrievalResult[] = searchResp?.results ?? [];

  return (
    <div className="search">
      <FilterPanel
        filters={filters}
        options={options}
        dbOk={dbOk}
        onChange={setFilters}
      />
      <div className="search-main">
        <QueryBar
          query={query} onQuery={setQuery}
          mode={mode} onMode={setMode}
          scope={scope} onScope={setScope}
          topK={topK} onTopK={setTopK}
          findQuote={findQuote} onFindQuote={setFindQuote}
          autoFilters={autoFilters} onAutoFilters={setAutoFilters}
          expandQuery={expandQuery} onExpandQuery={setExpandQuery}
          answerLanguage={answerLanguage} onAnswerLanguage={setAnswerLanguage}
          running={running} onSubmit={onSubmit} onStop={onStop}
        />

        <DetectedFilters
          detections={detections}
          applied={applied}
          filters={filters}
          onAddFilter={setFilters}
          expanded={expanded}
        />

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
          </>
        )}
      </div>
    </div>
  );
}
