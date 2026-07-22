import type { RetrievalResult } from "../types";
import { ResultCard } from "./ResultCard";

interface Props {
  results: RetrievalResult[];
  count: number;
  /** The user's query, forwarded so passages highlight matched terms. */
  query?: string;
}

/** Render a /api/search result set — no LLM answer, just the ranked hits. */
export function ResultList({ results, count, query }: Props) {
  if (results.length === 0) {
    return <div className="empty">No matches.</div>;
  }
  return (
    <section className="results">
      <div className="answer-meta">
        <span>{count} results</span>
      </div>
      {results.map((r, i) => (
        <ResultCard key={r.chunk_id} result={r} index={i + 1} query={query} />
      ))}
    </section>
  );
}
