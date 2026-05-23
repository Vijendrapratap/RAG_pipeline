import type { KeyboardEvent as ReactKeyboardEvent } from "react";

import type { AnswerLanguage, Scope } from "../types";

export type Mode = "answer" | "search";

interface Props {
  query: string;
  onQuery: (q: string) => void;
  mode: Mode;
  onMode: (m: Mode) => void;
  scope: Scope;
  onScope: (s: Scope) => void;
  topK: number;
  onTopK: (n: number) => void;
  findQuote: boolean;
  onFindQuote: (b: boolean) => void;
  autoFilters: boolean;
  onAutoFilters: (b: boolean) => void;
  expandQuery: boolean;
  onExpandQuery: (b: boolean) => void;
  answerLanguage: AnswerLanguage;
  onAnswerLanguage: (l: AnswerLanguage) => void;
  running: boolean;
  onSubmit: () => void;
  onStop: () => void;
}

/**
 * Query input + the full row of retrieval / synthesis switches.
 *
 * `find_quote` is an exact-phrase BM25 hunt; the backend ignores scope, auto-
 * filters and HyDE when it is set, so those controls disable to match.
 * `answer_language` is only meaningful in answer mode.
 */
export function QueryBar(props: Props) {
  const {
    query, onQuery, mode, onMode, scope, onScope, topK, onTopK,
    findQuote, onFindQuote, autoFilters, onAutoFilters,
    expandQuery, onExpandQuery, answerLanguage, onAnswerLanguage,
    running, onSubmit, onStop,
  } = props;

  function onKeyDown(e: ReactKeyboardEvent<HTMLTextAreaElement>) {
    // Ctrl/Cmd+Enter submits, matching common chat-box conventions.
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (!running && query.trim()) onSubmit();
    }
  }

  const semanticDisabled = findQuote;

  return (
    <div className="querybar">
      <textarea
        className="querybar-input"
        rows={3}
        placeholder="Ask in Hindi or English — कर्म योग क्या है"
        value={query}
        onChange={(e) => onQuery(e.target.value)}
        onKeyDown={onKeyDown}
      />
      <div className="querybar-row">
        <label className="field-inline">
          <span>Mode</span>
          <select
            value={mode}
            onChange={(e) => onMode(e.target.value as Mode)}
            disabled={running}
          >
            <option value="answer">Answer</option>
            <option value="search">Search only</option>
          </select>
        </label>

        <label className="field-inline">
          <span>Scope</span>
          <select
            value={scope}
            onChange={(e) => onScope(e.target.value as Scope)}
            disabled={running || semanticDisabled}
            title={semanticDisabled ? "ignored for quote hunts" : ""}
          >
            <option value="chunks">Chunks</option>
            <option value="summaries">Summaries</option>
            <option value="two_stage">Two-stage</option>
          </select>
        </label>

        <label className="field-inline">
          <span>Top K</span>
          <input
            type="number"
            min={1}
            max={40}
            value={topK}
            onChange={(e) =>
              onTopK(Math.max(1, Math.min(40, Number(e.target.value) || 1)))
            }
            disabled={running}
          />
        </label>

        {mode === "answer" && (
          <label className="field-inline">
            <span>Answer in</span>
            <select
              value={answerLanguage}
              onChange={(e) =>
                onAnswerLanguage(e.target.value as AnswerLanguage)
              }
              disabled={running}
            >
              <option value="auto">Auto</option>
              <option value="hindi">Hindi</option>
              <option value="english">English</option>
            </select>
          </label>
        )}

        <label className="check">
          <input
            type="checkbox"
            checked={findQuote}
            onChange={(e) => onFindQuote(e.target.checked)}
            disabled={running}
          />
          <span>Find quote</span>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={autoFilters}
            onChange={(e) => onAutoFilters(e.target.checked)}
            disabled={running || semanticDisabled}
          />
          <span>Auto filters</span>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={expandQuery}
            onChange={(e) => onExpandQuery(e.target.checked)}
            disabled={running || semanticDisabled}
            title="HyDE expansion — one extra LLM call (slower, sometimes better)"
          />
          <span>HyDE</span>
        </label>

        <div className="querybar-spacer" />

        {running ? (
          <button className="btn btn--stop" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button
            className="btn btn--primary"
            onClick={onSubmit}
            disabled={!query.trim()}
          >
            {mode === "answer" ? "Ask" : "Search"}
          </button>
        )}
      </div>
    </div>
  );
}
