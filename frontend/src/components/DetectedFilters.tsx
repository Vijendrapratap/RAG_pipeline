import { applyDetection, detectionActive, formatValue } from "../filters";
import type { Detection, Filters } from "../types";

interface Props {
  detections: Detection[];
  applied: Record<string, unknown>;
  filters: Filters;
  onAddFilter: (next: Filters) => void;
  expanded: boolean;
}

/**
 * Phase-D transparency panel.
 *
 * Strong signals (years, ISO dates) come back as `confidence: "strong"` — they
 * were already auto-applied to the retrieval and show as a non-interactive
 * "auto" badge. Soft signals (a season / topic word that *might* be a filter)
 * show as a "+ apply" chip — one click adds them to the active filter panel.
 *
 * `applied` is the backend's view of what actually filtered retrieval. The
 * "expanded" badge appears when HyDE was used for this turn.
 */
export function DetectedFilters({
  detections,
  applied,
  filters,
  onAddFilter,
  expanded,
}: Props) {
  if (detections.length === 0 && Object.keys(applied).length === 0 && !expanded) {
    return null;
  }
  return (
    <section className="detected">
      {detections.length > 0 && (
        <div className="detected-row">
          <span className="detected-label">Detected in query:</span>
          {detections.map((d, i) => {
            const isActive = detectionActive(filters, d);
            const cls =
              "chip" +
              (d.confidence === "strong" ? " chip--strong" : " chip--soft") +
              (isActive ? " chip--active" : "");
            const text = `${d.field}: ${formatValue(d.value)}`;
            if (d.confidence === "strong") {
              return (
                <span key={i} className={cls} title={`matched: "${d.matched}"`}>
                  {text} <em>auto</em>
                </span>
              );
            }
            return (
              <button
                key={i}
                className={cls}
                title={`matched: "${d.matched}" — click to add`}
                disabled={isActive}
                onClick={() => onAddFilter(applyDetection(filters, d))}
              >
                {text} {isActive ? <em>added</em> : <em>+ apply</em>}
              </button>
            );
          })}
        </div>
      )}
      {Object.keys(applied).length > 0 && (
        <div className="detected-row">
          <span className="detected-label">Applied to retrieval:</span>
          {Object.entries(applied).map(([k, v]) => (
            <span key={k} className="chip chip--applied">
              {k}: {formatValue(v)}
            </span>
          ))}
        </div>
      )}
      {expanded && (
        <div className="detected-row">
          <span className="chip chip--hyde">HyDE expansion used</span>
        </div>
      )}
    </section>
  );
}
