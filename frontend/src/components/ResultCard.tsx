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

  const isCatalog = result.source === "catalog" || result.result_type === "catalog";
  const meta = result.metadata as Record<string, unknown>;
  // For catalog rows the raw source_file is "catalog:KEY" — show the canonical
  // title / place instead so the sheet entry reads cleanly.
  const sourceLabel = isCatalog
    ? String(meta.track_title ?? meta.camp_place ?? meta.location ?? "Curated catalog")
    : (result.source_file ?? "(unknown)");
  const badgeClass = isCatalog
    ? "badge--catalog"
    : result.result_type === "summary" ? "badge--sum" : "badge--chunk";
  const badgeText = isCatalog ? "Catalog · sheet" : result.result_type;

  return (
    <article
      className={"card" + (isCatalog ? " card--catalog" : "") + (highlight ? " card--flash" : "")}
      id={index !== undefined ? `cite-${index}` : undefined}
    >
      <header className="card-head">
        {index !== undefined && <span className="cite-num">[{index}]</span>}
        <span className="card-source">{sourceLabel}</span>
        <span className={"badge " + badgeClass}>{badgeText}</span>
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
