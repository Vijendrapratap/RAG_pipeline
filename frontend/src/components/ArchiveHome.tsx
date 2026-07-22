import type { CorpusSummary } from "../corpus";
import { formatCount, formatDuration } from "../corpus";
import { Stat, humanDate } from "./CorpusDetail";

interface Props {
  summary: CorpusSummary | null;
  collapsed: boolean;
  onToggle: () => void;
}

/**
 * What the whole archive is, in five numbers — pinned to the top of the panel.
 *
 * It used to live inside `CorpusDetail`, gated on `node === null`, so it vanished
 * the instant you clicked anything. The totals are the one piece of context that
 * stays true no matter where you are in the tree, which is exactly the thing that
 * should not disappear when you go looking around.
 *
 * It collapses rather than scrolling away, because it competes with the
 * transcript for the same 340 px column and the transcript is why you are here.
 */
export function ArchiveHome({ summary, collapsed, onToggle }: Props) {
  if (!summary) {
    return <div className="brain-home"><div className="brain-detail-kicker">Loading the archive…</div></div>;
  }

  if (collapsed) {
    return (
      <button className="brain-home brain-home-strip" onClick={onToggle}
              aria-expanded={false} aria-controls="brain-home-body">
        <span>{formatCount(summary.n_files)}</span>
        <span className="brain-home-sep">·</span>
        <span>{formatCount(summary.n_chunks)}</span>
        <span className="brain-home-sep">·</span>
        <span>{formatDuration(summary.duration_sec)}</span>
        <span className="brain-home-chevron" aria-hidden>▾</span>
      </button>
    );
  }

  const undated = summary.n_written;
  return (
    <div className="brain-home" id="brain-home-body">
      <div className="brain-home-head">
        <div>
          <div className="brain-detail-kicker">The whole archive</div>
          <h2 className="brain-detail-title">Everything added to search</h2>
        </div>
        <button className="brain-home-collapse" onClick={onToggle}
                aria-expanded aria-controls="brain-home-body"
                title="Collapse the archive totals">▴</button>
      </div>

      <div className="brain-stats">
        <Stat label="recordings" value={formatCount(summary.n_files)} />
        <Stat label="passages" value={formatCount(summary.n_chunks)} />
        <Stat label="of audio" value={formatDuration(summary.duration_sec)} />
      </div>
      <p className="brain-detail-line">
        Spanning <strong>{humanDate(summary.min_session_date)}</strong> to{" "}
        <strong>{humanDate(summary.max_session_date)}</strong>.
      </p>
      {/* PRD §6: without this, a nearly-full disc reads as a nearly-finished archive. */}
      <p className="brain-detail-line muted">
        These are the <strong>{formatCount(summary.n_files)}</strong> recordings already
        added to search. Audio that has never been transcribed has no dot on this map —
        it cannot see the folders on disk, only what has been read into the archive.
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
        Each dot is a collection. Click one to open it — only that branch opens, so
        the map stays legible — then keep clicking inward, one level at a time.
        <kbd>Esc</kbd> steps back out. Click a single recording to play it and read
        what was said. Drag a dot anywhere and it carries what's inside it; drag the
        background to pan; scroll to zoom.
      </p>
    </div>
  );
}
