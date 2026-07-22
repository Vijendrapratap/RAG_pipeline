import { useCallback, useEffect, useId, useState } from "react";
import type { ReactNode } from "react";

import type { QueryMeta } from "../types";
import { ResultCard } from "./ResultCard";

/** Citations stay collapsed until the user opens them, keeping the focus on
 *  the answer. They auto-reveal nothing — the answer text is the headline. */
const SHOW_CITATIONS_COLLAPSED_BY_DEFAULT = true;

interface Props {
  answer: string;
  meta: QueryMeta | null;
  running: boolean;
  error: string | null;
}

const CITATION_RE = /\[(\d+)\]/g;

/**
 * Render the streaming answer with clickable `[N]` markers that scroll the
 * matching citation card into view. Citations are listed below the answer,
 * each numbered 1..N to match the `[N]` references.
 */
export function AnswerPane({ answer, meta, running, error }: Props) {
  const [flashed, setFlashed] = useState<number | null>(null);
  const [showSources, setShowSources] = useState(!SHOW_CITATIONS_COLLAPSED_BY_DEFAULT);
  // Per-pane anchor prefix. Every turn in a thread numbers its citations 1..N,
  // so a bare `cite-N` id collides across turns and getElementById would always
  // scroll to the FIRST turn's card — this keeps each turn's [N] clicks inside
  // that turn's own sources.
  const uid = useId();

  const onCite = useCallback((n: number) => {
    // Opening the panel is harmless if it was already open.
    setShowSources(true);
    // Defer the scroll so the citation cards have a frame to mount in.
    window.requestAnimationFrame(() => {
      const el = document.getElementById(`${uid}-cite-${n}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        setFlashed(n);
      }
    });
  }, [uid]);

  // Clear the highlight pulse so the same citation can flash again later.
  useEffect(() => {
    if (flashed == null) return;
    const t = window.setTimeout(() => setFlashed(null), 1200);
    return () => window.clearTimeout(t);
  }, [flashed]);

  // A new question resets the panel to its default collapsed state.
  useEffect(() => {
    if (running) setShowSources(!SHOW_CITATIONS_COLLAPSED_BY_DEFAULT);
  }, [running]);

  const segments = renderAnswer(answer, meta?.count ?? 0, onCite);
  const sourcesReady = !running && !!meta && meta.citations.length > 0;

  return (
    <>
      <div className="answer-body">
        {segments}
        {running && <span className="cursor" />}
      </div>
      {error && <div className="answer-error">{error}</div>}
      {sourcesReady && (
        <div className="sources">
          <button
            className="sources-toggle"
            onClick={() => setShowSources((v) => !v)}
            aria-expanded={showSources}
          >
            <span
              className="advanced-chevron"
              style={{
                display: "inline-block",
                transform: showSources ? "rotate(90deg)" : undefined,
                transition: "transform 150ms ease",
                marginRight: 6,
              }}
            >
              ▸
            </span>
            {showSources ? "Hide sources" : "Show sources"}{" "}
            <span className="sources-count">{meta!.count}</span>
            <span className="sources-meta">
              · {prettyScope(meta!.scope)} · {meta!.answer_language}
            </span>
          </button>
          {showSources && (
            <div className="citations">
              {meta!.citations.map((c, i) => (
                <ResultCard
                  key={c.chunk_id}
                  result={c}
                  index={i + 1}
                  anchorPrefix={uid}
                  highlight={flashed === i + 1}
                  query={meta!.query}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </>
  );
}

function prettyScope(scope: string): string {
  switch (scope) {
    case "chunks": return "passages";
    case "summaries": return "transcripts";
    case "two_stage": return "transcripts → passages";
    default: return scope;
  }
}

/** Split the answer text on `[N]` markers and return inline-renderable nodes. */
function renderAnswer(
  text: string,
  citationCount: number,
  onCite: (n: number) => void,
): ReactNode[] {
  if (!text) return [];
  const out: ReactNode[] = [];
  let last = 0;
  let key = 0;
  for (const match of text.matchAll(CITATION_RE)) {
    const idx = match.index ?? 0;
    if (idx > last) {
      out.push(<span key={key++}>{text.slice(last, idx)}</span>);
    }
    const n = Number(match[1]);
    const valid = n >= 1 && n <= citationCount;
    out.push(
      <button
        key={key++}
        className={"cite-ref" + (valid ? "" : " cite-ref--bad")}
        onClick={() => valid && onCite(n)}
        disabled={!valid}
        title={valid ? `Show citation ${n}` : "no such citation"}
      >
        [{n}]
      </button>,
    );
    last = idx + match[0].length;
  }
  if (last < text.length) {
    out.push(<span key={key++}>{text.slice(last)}</span>);
  }
  return out;
}
