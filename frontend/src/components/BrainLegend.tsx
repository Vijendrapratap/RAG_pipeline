import type { CState } from "../corpus";
import { STATE_COLOR, STATE_LABEL, STATE_MEANING } from "../corpus";

const ORDER: CState[] = ["remembered", "written", "failed"];

/**
 * What the colours mean, always on screen.
 *
 * The last paragraph is not filler. This map draws the recordings that made it
 * into the search index; a recording that was never transcribed has no row in
 * the database and therefore no dot. Without that sentence a person reads an
 * almost-full disc and concludes the archive is almost fully processed.
 */
export function BrainLegend({ counts }: { counts: Record<CState, number> | null }) {
  return (
    <div className="brain-legend">
      <div className="brain-legend-title">What the lights mean</div>
      <ul className="brain-legend-list">
        {ORDER.map((s) => (
          <li key={s} className="brain-legend-row">
            <span className="brain-swatch" style={{ background: STATE_COLOR[s] }} aria-hidden />
            <div>
              <div className="brain-legend-label">
                {STATE_LABEL[s]}
                {counts && (
                  <span className="brain-legend-count">{counts[s].toLocaleString()}</span>
                )}
              </div>
              <div className="brain-legend-meaning">{STATE_MEANING[s]}</div>
            </div>
          </li>
        ))}
      </ul>
      <p className="brain-legend-note">
        Bigger dot, more was said. A cluster glows brighter the more of it we can
        place in time.
      </p>
      <p className="brain-legend-note brain-legend-note--caveat">
        Only recordings already added to search appear here. Audio that has not
        been transcribed yet has no dot — this map cannot see the folders on
        disk, only what has been read into the archive.
      </p>
    </div>
  );
}
