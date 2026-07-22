import { memo } from "react";

import type { CorpusNode } from "../corpus";
import {
  STATE_COLOR, STATE_LABEL, formatCount, formatDuration, levelName, nodeState,
} from "../corpus";
import { TrackPanel } from "./TrackPanel";

interface Props {
  node: CorpusNode | null;
  /** Direct children of `node`, when it is a folder. Must be referentially stable
   *  across renders, or `memo` below buys nothing. */
  childNodes: CorpusNode[];
  loading: boolean;
  onOpen: (node: CorpusNode) => void;
  onAuthFail: () => void;
}

export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="brain-stat">
      <div className="brain-stat-value">{value}</div>
      <div className="brain-stat-label">{label}</div>
    </div>
  );
}

/** Plain-language date. "2002-03-26" helps nobody read a shelf of tapes. */
export function humanDate(iso: string | null): string {
  if (!iso) return "Date unknown";
  const d = new Date(iso + "T00:00:00");
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });
}

/**
 * The panel beside the map. Everything here is a fact from `file_meta` restated
 * in plain language — no derived scores, no estimates, nothing the database
 * cannot back up.
 *
 * `memo`, and not as a micro-optimisation: this subtree contains `TrackPanel`,
 * which renders one `<li>` per transcript segment — a few thousand for a two-hour
 * pravachan. A spacing slider or a panel resize re-renders `BrainView` at pointer
 * rate, and without this every one of those list items is reconciled each time.
 *
 * The archive totals live in `ArchiveHome`, as a sibling. They used to be here,
 * behind `node === null`, which meant they vanished the moment you clicked
 * anything.
 */
export const CorpusDetail = memo(function CorpusDetail(
  { node, childNodes, loading, onOpen, onAuthFail }: Props,
) {
  if (!node) return null;

  const state = nodeState(node);
  const parts = node.path.split("/");
  const trail = parts.slice(0, -1);
  const recordings = childNodes.filter((c) => c.is_leaf);

  return (
    <aside className="brain-detail">
      {trail.length > 0 && (
        <div className="brain-breadcrumb" title={node.path}>
          {trail.map((p, i) => <span key={i}>{p}</span>)}
        </div>
      )}
      <div className="brain-detail-kicker">{levelName(node)}</div>
      <h2 className="brain-detail-title">{node.name.replace(/\.json$/, "")}</h2>

      <div className="brain-state-pill" style={{ borderColor: STATE_COLOR[state] }}>
        <span className="brain-swatch" style={{ background: STATE_COLOR[state] }} aria-hidden />
        {STATE_LABEL[state]}
      </div>

      {node.is_leaf ? (
        <>
          <div className="brain-stats">
            <Stat label="passages" value={formatCount(node.n_chunks)} />
            <Stat label="long" value={formatDuration(node.duration_sec)} />
          </div>
          <p className="brain-detail-line">
            Recorded <strong>{humanDate(node.session_date)}</strong>.
          </p>
          {node.n_chunks === 0 && (
            <p className="brain-detail-line muted">
              No searchable text. The transcript for this recording came back empty.
            </p>
          )}
          {/* Keyed on the path: switching recordings must tear the old player
              down, or the previous track keeps streaming behind the new one. */}
          <TrackPanel key={node.path} sourceFile={node.path} onAuthFail={onAuthFail} />
        </>
      ) : (
        <>
          <div className="brain-stats">
            <Stat label="recordings" value={formatCount(node.n_files)} />
            <Stat label="passages" value={formatCount(node.n_chunks)} />
            <Stat label="of audio" value={formatDuration(node.duration_sec)} />
          </div>
          <p className="brain-detail-line">
            <strong>{formatCount(node.remembered)}</strong> can be placed in time
            {node.written > 0 && <>, <strong>{formatCount(node.written)}</strong> cannot</>}.
          </p>

          {loading && <p className="brain-detail-hint">Opening…</p>}

          {/* The recordings themselves, once we have them. A sitting holds two or
              three; naming them here saves hunting for a three-pixel dot. */}
          {recordings.length > 0 && (
            <ul className="brain-child-list">
              {recordings.map((r) => (
                <li key={r.path}>
                  <button onClick={() => onOpen(r)}>
                    <span className="brain-child-dot"
                          style={{ background: STATE_COLOR[nodeState(r)] }} aria-hidden />
                    <span className="brain-child-name">{r.name.replace(/\.json$/, "")}</span>
                    <span className="brain-child-meta">{formatDuration(r.duration_sec)}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </aside>
  );
});
