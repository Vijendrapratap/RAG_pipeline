import { useState } from "react";

import {
  ApiError,
  analyticsMentions,
  analyticsSpeakers,
  analyticsTranscripts,
} from "../api";
import type {
  MentionsResponse,
  SpeakersResponse,
  TranscriptsResponse,
} from "../types";

interface Props {
  onAuthFail: () => void;
}

type Result =
  | { kind: "mentions"; data: MentionsResponse }
  | { kind: "speakers"; data: SpeakersResponse }
  | { kind: "transcripts"; data: TranscriptsResponse };

/**
 * Postgres-backed corpus stats: how many times the corpus mentions a term,
 * who talks about it, which transcripts hold it. One simple form per query —
 * each call hits a distinct /api/analytics/* endpoint.
 */
export function AnalyticsView({ onAuthFail }: Props) {
  const [term, setTerm] = useState("");
  const [speaker, setSpeaker] = useState("");
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
          <span>Speaker (optional)</span>
          <input
            type="text"
            value={speaker}
            placeholder="for 'mentions' only"
            onChange={(e) => setSpeaker(e.target.value)}
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
          onClick={() =>
            run(
              () => analyticsMentions(term.trim(), speaker.trim() || undefined),
              "mentions",
            )
          }
        >
          Mention count
        </button>
        <button
          className="btn"
          disabled={running || !term.trim()}
          onClick={() => run(() => analyticsSpeakers(term.trim(), limit), "speakers")}
        >
          Top speakers
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
          <code>{d.term}</code>
          {d.speaker ? <> for speaker <code>{d.speaker}</code></> : null}.
        </p>
      </div>
    );
  }
  if (result.kind === "speakers") {
    const d = result.data;
    return (
      <table className="table">
        <thead>
          <tr><th>#</th><th>Speaker</th><th>Chunks</th></tr>
        </thead>
        <tbody>
          {d.speakers.map((s, i) => (
            <tr key={s.speaker}>
              <td>{i + 1}</td>
              <td>{s.speaker}</td>
              <td>{s.chunk_count.toLocaleString()}</td>
            </tr>
          ))}
          {d.speakers.length === 0 && (
            <tr><td colSpan={3} className="muted">no matches</td></tr>
          )}
        </tbody>
      </table>
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
