import { useState } from "react";

import {
  ApiError,
  analyticsMentions,
  analyticsTranscripts,
} from "../api";
import type {
  MentionsResponse,
  TranscriptsResponse,
} from "../types";

interface Props {
  onAuthFail: () => void;
}

type Result =
  | { kind: "mentions"; data: MentionsResponse }
  | { kind: "transcripts"; data: TranscriptsResponse };

/**
 * Postgres-backed corpus stats: how many times the corpus mentions a term and
 * which transcripts hold it. One simple form per query — each call hits a
 * distinct /api/analytics/* endpoint. (No speaker breakdown: the corpus is a
 * single voice, Guruji.)
 */
export function AnalyticsView({ onAuthFail }: Props) {
  const [term, setTerm] = useState("");
  const [limit, setLimit] = useState(10);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run<T extends Result>(
    fn: () => Promise<T["data"]>,
    kind: T["kind"],
  ) {
    if (!term.trim() || running) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const data = await fn();
      setResult({ kind, data } as Result);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onAuthFail();
      else setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="analytics">
      <h2>Analytics</h2>
      <p className="muted">
        Hindi-correct full-text counts over <code>chunk_meta</code>. Uses the
        Postgres <code>'simple'</code> FTS config.
      </p>

      <div className="analytics-form">
        <label className="field">
          <span>Term</span>
          <input
            type="text"
            value={term}
            placeholder="e.g. dharma  /  कर्म"
            onChange={(e) => setTerm(e.target.value)}
          />
        </label>
        <label className="field">
          <span>Limit</span>
          <input
            type="number"
            min={1}
            max={100}
            value={limit}
            onChange={(e) =>
              setLimit(Math.max(1, Math.min(100, Number(e.target.value) || 1)))
            }
          />
        </label>
      </div>

      <div className="analytics-buttons">
        <button
          className="btn btn--primary"
          disabled={running || !term.trim()}
          onClick={() => run(() => analyticsMentions(term.trim()), "mentions")}
        >
          Mention count
        </button>
        <button
          className="btn"
          disabled={running || !term.trim()}
          onClick={() =>
            run(() => analyticsTranscripts(term.trim(), limit), "transcripts")
          }
        >
          Transcripts
        </button>
      </div>

      {error && <div className="answer-error">{error}</div>}
      {result && <Outcome result={result} />}
    </div>
  );
}

function Outcome({ result }: { result: Result }) {
  if (result.kind === "mentions") {
    const d = result.data;
    return (
      <div className="card">
        <header className="card-head">
          <span className="card-source">Mention count</span>
        </header>
        <p className="card-text">
          <strong>{d.chunk_count.toLocaleString()}</strong> chunk(s) mention{" "}
          <code>{d.term}</code>.
        </p>
      </div>
    );
  }
  const d = result.data;
  return (
    <table className="table">
      <thead>
        <tr><th>#</th><th>Transcript</th><th>Chunks</th></tr>
      </thead>
      <tbody>
        {d.transcripts.map((t, i) => (
          <tr key={t.source_file}>
            <td>{i + 1}</td>
            <td>{t.source_file}</td>
            <td>{t.chunk_count.toLocaleString()}</td>
          </tr>
        ))}
        {d.transcripts.length === 0 && (
          <tr><td colSpan={3} className="muted">no matches</td></tr>
        )}
      </tbody>
    </table>
  );
}
