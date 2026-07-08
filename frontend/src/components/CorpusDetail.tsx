import type { CorpusNode, CorpusSummary } from "../corpus";
import {
  STATE_COLOR, STATE_LABEL, formatCount, formatDuration, levelName, nodeState,
} from "../corpus";
import { TrackPanel } from "./TrackPanel";

interface Props {
  node: CorpusNode | null;
  summary: CorpusSummary | null;
  /** Direct children of `node`, when it is a folder. */
  childNodes: CorpusNode[];
  loading: boolean;
  onOpen: (node: CorpusNode) => void;
  onAuthFail: () => void;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="brain-stat">
      <div className="brain-stat-value">{value}</div>
      <div className="brain-stat-label">{label}</div>
    </div>
  );
}

/** Plain-language date. "2002-03-26" helps nobody read a shelf of tapes. */
function humanDate(iso: string | null): string {
  if (!iso) return "Date unknown";
  const d = new Date(iso + "T00:00:00");
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });
}

function ArchiveOverview({ summary }: { summary: CorpusSummary }) {
  const undated = summary.n_written;
  return (
    <>
      <div className="brain-detail-kicker">The whole archive</div>
      <h2 className="brain-detail-title">Everything added to search</h2>
      <div className="brain-stats">
        <Stat label="recordings" value={formatCount(summary.n_files)} />
        <Stat label="passages" value={formatCount(summary.n_chunks)} />
        <Stat label="of audio" value={formatDuration(summary.duration_sec)} />
      </div>
      <p className="brain-detail-line">
        Spanning <strong>{humanDate(summary.min_session_date)}</strong> to{" "}
        <strong>{humanDate(summary.max_session_date)}</strong>.
      </p>
      {undated > 0 && (
        <p className="brain-detail-line muted">
          {formatCount(undated)} recordings are searchable but carry no date — their
          folder never said when they were recorded.
        </p>
      )}
      {summary.n_failed > 0 && (
        <div className="brain-alert">
          <div className="brain-alert-title">
            {summary.n_failed} {summary.n_failed === 1 ? "recording is" : "recordings are"} broken
          </div>
          <p>
            These were indexed under a bare filename with no folder, so nothing can
            say which camp or day they belong to.
          </p>
          <ul className="brain-alert-list">
            {summary.degenerate_files.map((f) => (
              <li key={f}><code>{f}</code></li>
            ))}
          </ul>
        </div>
      )}
      <p className="brain-detail-hint">
        Click a cluster to open it and see inside. Click a single recording to
        play it and read what was said. Scroll to zoom, drag to move.
      </p>
    </>
  );
}

/**
 * The panel beside the map. Everything here is a fact from `file_meta` restated
 * in plain language — no derived scores, no estimates, nothing the database
 * cannot back up.
 */
export function CorpusDetail({ node, summary, childNodes, loading, onOpen, onAuthFail }: Props) {
  if (!node) {
    return (
      <aside className="brain-detail">
        {summary
          ? <ArchiveOverview summary={summary} />
          : <div className="brain-detail-kicker">Loading the archive…</div>}
      </aside>
    );
  }

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
}
