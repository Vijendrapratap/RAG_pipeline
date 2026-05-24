import type { RetrievalResult } from "../types";

interface Props {
  result: RetrievalResult;
  /** 1-based citation number. Sets the DOM id `cite-N` for scroll-to. */
  index?: number;
  /** Highlight pulse, e.g. when a [N] citation marker was clicked. */
  highlight?: boolean;
}

function fmtTime(sec: number | null): string | null {
  if (sec == null) return null;
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

/** Render one chunk- or summary-level result. Reused for /api/search hits
 *  and for /api/query citations. */
export function ResultCard({ result, index, highlight }: Props) {
  const start = fmtTime(result.start_sec);
  const end = fmtTime(result.end_sec);
  const tsRange = start && end ? `${start}–${end}` : start ?? "";

  return (
    <article
      className={"card" + (highlight ? " card--flash" : "")}
      id={index !== undefined ? `cite-${index}` : undefined}
    >
      <header className="card-head">
        {index !== undefined && <span className="cite-num">[{index}]</span>}
        <span className="card-source">{result.source_file ?? "(unknown)"}</span>
        <span
          className={
            "badge " +
            (result.result_type === "summary" ? "badge--sum" : "badge--chunk")
          }
        >
          {result.result_type}
        </span>
        {tsRange && <span className="card-time">{tsRange}</span>}
        <span className="card-score" title="fused score">
          {result.score.toFixed(3)}
        </span>
      </header>

      <p className="card-text">{result.text}</p>

      {result.summary_hindi && result.summary_hindi !== result.text && (
        <p className="card-text card-text--alt">{result.summary_hindi}</p>
      )}

      <footer className="card-foot">
        {result.speakers.length > 0 && (
          <span className="card-speakers">
            {result.speakers.join(", ")}
          </span>
        )}
        {Object.entries(result.metadata).map(([k, v]) => (
          <span key={k} className="meta-pill">
            <em>{k}</em>{" "}
            {Array.isArray(v) ? v.join(", ") : String(v)}
          </span>
        ))}
      </footer>
    </article>
  );
}
