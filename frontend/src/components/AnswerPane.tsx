import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";

import type { QueryMeta } from "../types";
import { ResultCard } from "./ResultCard";

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

  const onCite = useCallback((n: number) => {
    const el = document.getElementById(`cite-${n}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      setFlashed(n);
    }
  }, []);

  // Clear the highlight pulse so the same citation can flash again later.
  useEffect(() => {
    if (flashed == null) return;
    const t = window.setTimeout(() => setFlashed(null), 1200);
    return () => window.clearTimeout(t);
  }, [flashed]);

  const segments = renderAnswer(answer, meta?.count ?? 0, onCite);

  return (
    <section className="answer">
      {meta && (
        <div className="answer-meta">
          <span>Language: {meta.answer_language}</span>
          <span>Scope: {meta.scope}</span>
          <span>{meta.count} citations</span>
        </div>
      )}
      <div className="answer-body">
        {segments}
        {running && <span className="cursor" />}
      </div>
      {error && <div className="answer-error">{error}</div>}
      {meta && meta.citations.length > 0 && (
        <div className="citations">
          <h3>Citations</h3>
          {meta.citations.map((c, i) => (
            <ResultCard
              key={c.chunk_id}
              result={c}
              index={i + 1}
              highlight={flashed === i + 1}
            />
          ))}
        </div>
      )}
    </section>
  );
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
