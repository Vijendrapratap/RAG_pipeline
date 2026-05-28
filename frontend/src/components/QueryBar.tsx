import { useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";

import type { AnswerLanguage, ChatModel, Filters, Scope } from "../types";
import { countActive } from "../filters";

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
  /** Models offered by /api/models. Empty list / single entry => no dropdown. */
  models: ChatModel[];
  /** `provider/model` key of the selected option, or null = use default. */
  modelKey: string | null;
  onModelKey: (k: string | null) => void;
  running: boolean;
  onSubmit: () => void;
  onStop: () => void;
  filters: Filters;
  onOpenFilters: () => void;
}

/** Stable string id for a ChatModel. Mirrors how SearchView serialises picks. */
export function modelKeyOf(m: ChatModel): string {
  return `${m.provider}/${m.model}`;
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="12" y1="19" x2="12" y2="5" />
      <polyline points="5 12 12 5 19 12" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}

function SlidersIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" />
      <line x1="12" y1="21" x2="12" y2="12" /><line x1="12" y1="8" x2="12" y2="3" />
      <line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" />
      <line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" />
      <line x1="17" y1="16" x2="23" y2="16" />
    </svg>
  );
}

/**
 * Composer: chat-style input at the bottom of the conversation.
 * Power controls (mode/scope/topK/HyDE/quote/auto-filters/language) live in a
 * collapsible "Advanced" disclosure so non-tech users see only the question
 * box. Filter drawer opens via the funnel button.
 */
export function QueryBar(props: Props) {
  const {
    query, onQuery, mode, onMode, scope, onScope, topK, onTopK,
    findQuote, onFindQuote, autoFilters, onAutoFilters,
    expandQuery, onExpandQuery, answerLanguage, onAnswerLanguage,
    models, modelKey, onModelKey,
    running, onSubmit, onStop, filters, onOpenFilters,
  } = props;

  // Single-model deployments don't need a dropdown — keep the UI quiet.
  const showModelPicker = models.length > 1;
  const defaultModelEntry =
    models.find((m) => m.is_default) ?? models[0];
  const defaultModelKey = defaultModelEntry ? modelKeyOf(defaultModelEntry) : "";

  const [showAdvanced, setShowAdvanced] = useState(false);

  function onKeyDown(e: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!running && query.trim()) onSubmit();
    }
  }

  const semanticDisabled = findQuote;
  const activeFilters = countActive(filters);

  return (
    <div className={"composer" + (showAdvanced ? " composer--advanced" : "")}>
      <textarea
        className="composer-input"
        rows={1}
        placeholder="Ask in Hindi or English — कर्म योग क्या है"
        value={query}
        onChange={(e) => onQuery(e.target.value)}
        onKeyDown={onKeyDown}
      />

      <div className="composer-bar">
        <button
          type="button"
          className={"composer-filter-btn" + (activeFilters > 0 ? " composer-filter-btn--active" : "")}
          onClick={onOpenFilters}
          title="Filters"
        >
          <SlidersIcon />
          <span>Filters</span>
          {activeFilters > 0 && (
            <span className="composer-filter-count">{activeFilters}</span>
          )}
        </button>

        <button
          type="button"
          className="btn btn--ghost"
          onClick={() => setShowAdvanced((v) => !v)}
          title="Advanced options"
        >
          <span className={"advanced-chevron" + (showAdvanced ? " advanced-chevron--open" : "")}
                style={{ display: "inline-block", transform: showAdvanced ? "rotate(90deg)" : undefined,
                         transition: "transform 150ms ease", marginRight: 4 }}>
            ▸
          </span>
          Advanced
        </button>

        <span className="composer-hint">Press Enter to send · Shift + Enter for a new line</span>

        <div className="composer-spacer" />

        {running ? (
          <button className="btn btn--stop btn--icon" onClick={onStop} title="Stop">
            <StopIcon />
          </button>
        ) : (
          <button
            className="btn btn--primary btn--icon"
            onClick={onSubmit}
            disabled={!query.trim()}
            title="Send"
          >
            <SendIcon />
          </button>
        )}
      </div>

      {showAdvanced && (
        <div className="advanced advanced--open">
          <div className="advanced-grid">
            <label className="field-inline">
              <span>Response</span>
              <select
                value={mode}
                onChange={(e) => onMode(e.target.value as Mode)}
                disabled={running}
              >
                <option value="answer">Answer with citations</option>
                <option value="search">Show passages only</option>
              </select>
            </label>

            <label className="field-inline">
              <span>Search depth</span>
              <select
                value={scope}
                onChange={(e) => onScope(e.target.value as Scope)}
                disabled={running || semanticDisabled}
                title={semanticDisabled ? "ignored for quote hunts" : ""}
              >
                <option value="chunks">Passages (default)</option>
                <option value="summaries">Whole transcripts</option>
                <option value="two_stage">Best transcripts → passages</option>
              </select>
            </label>

            <label className="field-inline">
              <span># of passages</span>
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
                  onChange={(e) => onAnswerLanguage(e.target.value as AnswerLanguage)}
                  disabled={running}
                >
                  <option value="auto">Auto (match question)</option>
                  <option value="hindi">Hindi</option>
                  <option value="english">English</option>
                </select>
              </label>
            )}

            {mode === "answer" && showModelPicker && (
              <label className="field-inline" title="Which chat model writes the answer">
                <span>Model</span>
                <select
                  value={modelKey ?? defaultModelKey}
                  onChange={(e) => onModelKey(e.target.value)}
                  disabled={running}
                >
                  {models.map((m) => (
                    <option key={modelKeyOf(m)} value={modelKeyOf(m)}>
                      {m.label}
                      {m.is_default ? " · default" : ""}
                    </option>
                  ))}
                </select>
              </label>
            )}

            <label className="check" title="Auto-detect dates / years / locations from your question">
              <input
                type="checkbox"
                checked={autoFilters}
                onChange={(e) => onAutoFilters(e.target.checked)}
                disabled={running || semanticDisabled}
              />
              <span>Auto-detect filters</span>
            </label>
            <label className="check" title="BM25-heavy exact-phrase hunt">
              <input
                type="checkbox"
                checked={findQuote}
                onChange={(e) => onFindQuote(e.target.checked)}
                disabled={running}
              />
              <span>Find exact quote</span>
            </label>
            <label className="check" title="HyDE — one extra LLM call (slower, sometimes better)">
              <input
                type="checkbox"
                checked={expandQuery}
                onChange={(e) => onExpandQuery(e.target.checked)}
                disabled={running || semanticDisabled}
              />
              <span>Smart expand (HyDE)</span>
            </label>
          </div>
        </div>
      )}
    </div>
  );
}
