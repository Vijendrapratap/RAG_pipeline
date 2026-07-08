import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getTrackTranscript } from "../api";
import type { TrackTranscript } from "../corpus";

interface Props {
  sourceFile: string;
  onAuthFail: () => void;
}

/** mm:ss. Nothing here runs past an hour, and hh:mm:ss would be noise. */
function stamp(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/**
 * The active segment for a playhead position.
 *
 * Binary search, not `.findIndex()`: `timeupdate` fires ~4×/s and a long
 * pravachan runs to a few thousand segments. It is also the reason segments
 * must stay sorted by `start` — WhisperX emits them that way.
 *
 * Returns the last segment that has started, which is what you want in the gaps
 * between segments: the line you just heard stays lit rather than flickering off.
 */
function activeIndex(segments: { start: number }[], t: number): number {
  let lo = 0;
  let hi = segments.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (segments[mid].start <= t) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

/**
 * A recording: play its isolated vocals, read what was said, click a line to
 * jump there.
 *
 * The audio is 32-bit PCM WAV, ~350 KB/s and often several hundred MB. Two
 * things keep that sane: `preload="metadata"` (the browser fetches the header,
 * not the file) and the server answering Range requests, which `FileResponse`
 * does natively. Never fetch this into a blob.
 */
export function TrackPanel({ sourceFile, onAuthFail }: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const listRef = useRef<HTMLOListElement>(null);

  const [data, setData] = useState<TrackTranscript | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(-1);
  const [follow, setFollow] = useState(true);
  const [audioBroken, setAudioBroken] = useState(false);
  const [showCleaned, setShowCleaned] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    setActive(-1);
    setAudioBroken(false);
    (async () => {
      try {
        const t = await getTrackTranscript(sourceFile);
        if (!cancelled) setData(t);
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 401) { onAuthFail(); return; }
        setError(
          e instanceof ApiError && e.status === 404
            ? "This recording is in the search index, but its transcript file is not on disk."
            : "The transcript could not be loaded.",
        );
      }
    })();
    return () => { cancelled = true; };
  }, [sourceFile, onAuthFail]);

  const onTimeUpdate = useCallback(() => {
    const el = audioRef.current;
    if (!el || !data) return;
    const i = activeIndex(data.segments, el.currentTime);
    setActive((prev) => (prev === i ? prev : i));
  }, [data]);

  // Keep the spoken line on screen — unless the reader has scrolled away, in
  // which case yanking the list back under their cursor every quarter-second is
  // hostile. `follow` turns itself off on manual scroll and back on via the pill.
  useEffect(() => {
    if (!follow || active < 0) return;
    // No `behavior: "smooth"` here — that option overrides CSS, and the
    // `prefers-reduced-motion` block could not switch it off. `.track-lines`
    // sets `scroll-behavior` instead, which the media query does reach.
    listRef.current?.querySelector<HTMLElement>(`[data-i="${active}"]`)
      ?.scrollIntoView({ block: "center" });
  }, [active, follow]);

  const seek = useCallback((t: number) => {
    const el = audioRef.current;
    if (!el) return;
    el.currentTime = t;
    setFollow(true);
    void el.play().catch(() => { /* a play() rejected by autoplay policy is not an error */ });
  }, []);

  if (error) return <p className="brain-detail-line muted">{error}</p>;
  if (!data) return <p className="brain-detail-line muted">Opening the recording…</p>;

  return (
    <div className="track-panel">
      {data.audio_url && !audioBroken ? (
        <audio
          ref={audioRef}
          className="track-audio"
          src={data.audio_url}
          controls
          preload="metadata"
          onTimeUpdate={onTimeUpdate}
          onSeeked={onTimeUpdate}
          onError={() => setAudioBroken(true)}
        />
      ) : (
        <p className="brain-detail-line muted">
          {data.audio_url
            // 32-bit integer PCM. Chromium decodes it; Firefox's WAV decoder
            // handles 8/16/24-bit int and 32-bit float, and gives up here.
            ? "This browser cannot play the isolated audio (32-bit WAV). The desktop app can."
            : "No isolated audio was found for this recording."}
        </p>
      )}

      <div className="track-tabs" role="tablist" aria-label="Transcript view">
        <button
          role="tab" aria-selected={!showCleaned}
          className={!showCleaned ? "is-active" : ""}
          onClick={() => setShowCleaned(false)}
        >
          Timed lines
        </button>
        <button
          role="tab" aria-selected={showCleaned}
          className={showCleaned ? "is-active" : ""}
          onClick={() => setShowCleaned(true)}
          disabled={!data.cleaned_text}
        >
          Cleaned text
        </button>
        {!showCleaned && active >= 0 && !follow && (
          <button className="track-follow" onClick={() => setFollow(true)}>
            Follow the audio
          </button>
        )}
      </div>

      {data.clean_note && (
        <p className="brain-detail-line muted">
          The cleanup pass was discarded for this recording ({data.clean_note}); what you
          see is the raw transcription.
        </p>
      )}

      {showCleaned ? (
        <div className="track-cleaned">{data.cleaned_text}</div>
      ) : data.segments.length === 0 ? (
        <p className="brain-detail-line muted">
          The transcript for this recording came back empty. Nothing was heard.
        </p>
      ) : (
        <ol
          className="track-lines"
          ref={listRef}
          // Wheel and pointer, never `onScroll`: the `scrollIntoView` above
          // fires `scroll` too, so following the audio would switch following
          // off on its very first line. These two are unambiguously the reader.
          onWheel={() => setFollow(false)}
          onPointerDown={() => setFollow(false)}
        >
          {data.segments.map((s, i) => (
            <li key={i} data-i={i} className={i === active ? "is-active" : ""}>
              <button onClick={() => seek(s.start)} title={`Jump to ${stamp(s.start)}`}>
                <span className="track-time">{stamp(s.start)}</span>
                <span className="track-text">{s.text}</span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
