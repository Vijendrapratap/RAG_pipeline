import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getCorpusState, getCorpusSummary } from "../api";
import type { CState, CorpusNode, CorpusSummary } from "../corpus";
import { SKELETON_DEPTH, hasHiddenChildren, parentPath } from "../corpus";
import { SpatialGrid } from "../viz/grid";
import type { Placed } from "../viz/radialTree";
import { buildTree, radialTree, rootNode } from "../viz/radialTree";
import { relaxAngles } from "../viz/relax";
import type { Viewport } from "../viz/draw";
import { draw, toLayout } from "../viz/draw";
import { BrainLegend } from "./BrainLegend";
import { CorpusDetail } from "./CorpusDetail";

const POLL_MS = 10_000;
/** Backoff ceiling. Stage C restarts rag-api mid-ingest; the map must ride it out. */
const MAX_BACKOFF_MS = 60_000;

const MIN_SCALE = 0.06;
const MAX_SCALE = 4;

/** How long the camera takes to reach a cluster you opened. */
const CAM_MS = 420;
/** Breathing room, in layout units, around a framed cluster. */
const CAM_PAD = 90;

interface Props {
  onAuthFail: () => void;
}

/** A camera flight in progress. `null` means the view is at rest. */
interface CamFlight {
  fx: number; fy: number; fs: number;
  tx: number; ty: number; ts: number;
  t0: number;
  dur: number;
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/** Counters the acceptance tests read. Never used by the UI itself. */
declare global {
  interface Window {
    __brainPerf?: { frames: number; slowFrames: number; lastFrameMs: number; fetches: number };
  }
}

function perf() {
  if (!window.__brainPerf) {
    window.__brainPerf = { frames: 0, slowFrames: 0, lastFrameMs: 0, fetches: 0 };
  }
  return window.__brainPerf;
}

function clampScale(s: number): number {
  return Math.max(MIN_SCALE, Math.min(MAX_SCALE, s));
}

export function BrainView({ onAuthFail }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const [summary, setSummary] = useState<CorpusSummary | null>(null);
  const [nodes, setNodes] = useState<Map<string, CorpusNode>>(new Map());
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  /** The opened cluster. Drives the camera and the dimming; null = whole archive. */
  const [focusPath, setFocusPath] = useState<string | null>(null);
  const [hoverPath, setHoverPath] = useState<string | null>(null);
  const [loadingPath, setLoadingPath] = useState<string | null>(null);
  const [stale, setStale] = useState(false);
  const [fatal, setFatal] = useState<string | null>(null);

  // Live, not latched at mount: the OS setting can be toggled while the tab is
  // open, and the previous `useMemo(..., [])` would keep animating until remount.
  const [reduceMotion, setReduceMotion] = useState(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
  );
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!mq) return;
    const onChange = (e: MediaQueryListEvent) => setReduceMotion(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  // Mutable render state. Kept out of React: these change every frame during a
  // camera flight, and a setState per frame would re-render the detail panel 60
  // times a second.
  const viewRef = useRef<Viewport>({ x: 0, y: 0, scale: 0.3, width: 1, height: 1 });
  const placedRef = useRef<Placed[]>([]);
  const gridRef = useRef(new SpatialGrid());
  const hoverRef = useRef<Placed | null>(null);
  const selectedRef = useRef<Placed | null>(null);
  const expandedRef = useRef(expanded);
  const focusRef = useRef<string | null>(null);
  const dirtyRef = useRef(true);
  const framedRef = useRef(false);
  const camRef = useRef<CamFlight | null>(null);
  const reduceMotionRef = useRef(reduceMotion);
  /** `${focusPath}|${childCount}` of the last flight. Stops the 10 s poll — which
   *  hands back a fresh `nodes` Map every tick — from yanking the camera back. */
  const framedForRef = useRef<string | null>(null);

  useEffect(() => { reduceMotionRef.current = reduceMotion; }, [reduceMotion]);
  useEffect(() => { focusRef.current = focusPath; }, [focusPath]);

  const onError = useCallback((e: unknown) => {
    if (e instanceof ApiError && e.status === 401) { onAuthFail(); return true; }
    return false;
  }, [onAuthFail]);

  // ---- data ---------------------------------------------------------------

  const mergeNodes = useCallback((incoming: CorpusNode[]) => {
    setNodes((prev) => {
      const next = new Map(prev);
      for (const n of incoming) next.set(n.path, n);
      return next;
    });
  }, []);

  /** Refetch the skeleton plus every currently-open cluster. Never the whole
   *  tree: an open session costs one request, a closed one costs nothing. */
  const refetch = useCallback(async (openPaths: string[]) => {
    const p = perf();
    p.fetches++;
    const skeleton = await getCorpusState("", SKELETON_DEPTH);
    const opened = await Promise.all(openPaths.map((path) => {
      p.fetches++;
      return getCorpusState(path, 1);
    }));
    mergeNodes([...skeleton.children, ...opened.flatMap((r) => r.children)]);
  }, [mergeNodes]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [s] = await Promise.all([getCorpusSummary(), refetch([])]);
        if (!cancelled) { setSummary(s); setStale(false); }
      } catch (e) {
        if (cancelled || onError(e)) return;
        setFatal(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { cancelled = true; };
  }, [refetch, onError]);

  // Live refresh. Polls only while the tab is visible; on a `version` change it
  // refetches. On failure it holds the last good picture and backs off — during
  // Stage C's `docker compose restart rag-api` this endpoint is gone for 5-10 s,
  // and a map that blanks itself mid-ingest is worse than one that waits.
  useEffect(() => {
    let timer: number | undefined;
    let backoff = POLL_MS;
    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      if (document.visibilityState !== "visible") {
        timer = window.setTimeout(tick, POLL_MS);
        return;
      }
      try {
        const s = await getCorpusSummary();
        if (cancelled) return;
        if (s.version !== summary?.version) {
          await refetch([...expandedRef.current]);
          if (cancelled) return;
          setSummary(s);
        }
        setStale(false);
        backoff = POLL_MS;
      } catch (e) {
        if (cancelled || onError(e)) return;
        setStale(true);
        backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
      }
      timer = window.setTimeout(tick, backoff);
    };

    timer = window.setTimeout(tick, POLL_MS);
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [summary?.version, refetch, onError]);

  /**
   * Open a node: select it, and — if it is a folder — fly the camera into it,
   * fetching its children first when they are not already on the map.
   *
   * This used to be `toggleExpand`, gated on `canExpand(node)` (which really
   * meant "are this node's children un-fetched?"). A collection's children come
   * down with the skeleton, so clicking `Dagshai 2001` failed that gate and did
   * *nothing at all* — not even move the view. Fetching and opening are separate
   * questions, and only the first one has anything to do with depth.
   */
  const focusNode = useCallback(async (node: CorpusNode) => {
    setSelectedPath(node.path);
    // A recording is a destination, not a container: it opens the track panel,
    // and pulling the camera off its siblings would only lose the reader's place.
    if (node.is_leaf) return;

    setFocusPath(node.path);
    if (!hasHiddenChildren(node) || expandedRef.current.has(node.path)) return;

    setLoadingPath(node.path);
    try {
      perf().fetches++;
      const res = await getCorpusState(node.path, 1);
      mergeNodes(res.children);
      setExpanded((prev) => new Set(prev).add(node.path));
    } catch (e) {
      if (!onError(e)) setStale(true);
    } finally {
      setLoadingPath(null);
    }
  }, [mergeNodes, onError]);

  /** Step out one level. The subtree stays loaded — re-entering is free. */
  const focusUp = useCallback(() => {
    setSelectedPath(null);
    setFocusPath((prev) => (prev ? parentPath(prev) || null : null));
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") focusUp();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [focusUp]);

  // ---- layout -------------------------------------------------------------

  useEffect(() => { expandedRef.current = expanded; }, [expanded]);

  useEffect(() => {
    if (nodes.size === 0) return;
    const root = rootNode(summary?.n_files ?? 0, summary?.n_chunks ?? 0);
    const placed = radialTree(buildTree(root, [...nodes.values()]));
    relaxAngles(placed);
    placedRef.current = placed;
    // The only rebuild there is. Nothing moves a node after this — the drift is
    // gone, and the camera transforms at draw time, not in layout space.
    gridRef.current.rebuild(placed);
    selectedRef.current = placed.find((p) => p.node.path === selectedPath) ?? null;

    const v = viewRef.current;
    // Frame the whole archive on the first layout. Ring radii are derived from
    // how much each ring must hold, so the outermost radius is not knowable in
    // advance — a hardcoded starting zoom would land somewhere arbitrary.
    if (!framedRef.current && v.width > 1 && v.height > 1) {
      const extent = Math.max(64, ...placed.map((p) => p.dist + p.r));
      v.scale = clampScale(Math.min(v.width, v.height) / (extent * 2.16));
      v.x = 0;
      v.y = 0;
      framedRef.current = true;
      framedForRef.current = "null|0";
    }
    dirtyRef.current = true;
  }, [nodes, summary?.n_files, summary?.n_chunks, selectedPath]);

  /** Where the camera should sit to show `path` and its children. */
  const frameFor = useCallback((path: string | null): CamFlight | null => {
    const v = viewRef.current;
    const placed = placedRef.current;
    if (v.width <= 1 || placed.length === 0) return null;

    let pts: Placed[];
    if (path === null) {
      pts = placed;
    } else {
      const parent = placed.find((p) => p.node.path === path);
      if (!parent) return null;
      const kids = placed.filter((p) => p.parent?.node.path === path);
      // A folder whose children have not arrived yet still frames — on itself,
      // at a readable zoom. Otherwise the click reads as "nothing happened",
      // which is precisely the bug being fixed.
      pts = kids.length > 0 ? [parent, ...kids] : [parent];
    }

    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const w = maxX - minX + CAM_PAD;
    const h = maxY - minY + CAM_PAD;

    return {
      fx: v.x, fy: v.y, fs: v.scale,
      tx: (minX + maxX) / 2,
      ty: (minY + maxY) / 2,
      ts: clampScale(Math.min(v.width / w, v.height / h)),
      t0: performance.now(),
      dur: reduceMotionRef.current ? 0 : CAM_MS,
    };
  }, []);

  // Retarget when the opened cluster changes, or when it gains children. Keyed
  // on both so the 10 s poll — which replaces the `nodes` Map every tick even
  // when nothing changed — cannot drag the camera out from under the reader.
  useEffect(() => {
    if (!framedRef.current) return;
    const kidCount = focusPath === null
      ? 0
      : [...nodes.keys()].filter((k) => k.startsWith(focusPath + "/")).length;
    const key = `${focusPath}|${kidCount}`;
    if (framedForRef.current === key) return;

    const flight = frameFor(focusPath);
    if (!flight) return;
    framedForRef.current = key;
    camRef.current = flight;
    dirtyRef.current = true;
  }, [focusPath, nodes, frameFor]);

  // ---- render loop --------------------------------------------------------

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    let raf = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);

    const resize = () => {
      const r = wrap.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      viewRef.current.width = r.width;
      viewRef.current.height = r.height;
      canvas.width = Math.round(r.width * dpr);
      canvas.height = Math.round(r.height * dpr);
      canvas.style.width = `${r.width}px`;
      canvas.style.height = `${r.height}px`;
      dirtyRef.current = true;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const frame = () => {
      raf = requestAnimationFrame(frame);
      if (document.visibilityState !== "visible") return;

      // The only thing that moves. When no cluster is being opened, `camRef` is
      // null, nothing marks the canvas dirty, and this loop costs one branch per
      // frame — the map is genuinely still. That stillness is the signal: a
      // glance tells you whether anything is happening.
      const cam = camRef.current;
      if (cam) {
        const k = cam.dur <= 0 ? 1 : Math.min(1, (performance.now() - cam.t0) / cam.dur);
        const e = easeInOutCubic(k);
        const v = viewRef.current;
        v.x = cam.fx + (cam.tx - cam.fx) * e;
        v.y = cam.fy + (cam.ty - cam.fy) * e;
        // Scale interpolates geometrically. Lerping it makes a zoom-out appear to
        // accelerate, because scale is a ratio and the eye reads its logarithm.
        v.scale = cam.fs * Math.pow(cam.ts / cam.fs, e);
        if (k >= 1) camRef.current = null;
        dirtyRef.current = true;
      }

      if (!dirtyRef.current) return;
      dirtyRef.current = false;

      const start = performance.now();
      draw(ctx, {
        placed: placedRef.current,
        view: viewRef.current,
        hover: hoverRef.current,
        selected: selectedRef.current,
        expanded: expandedRef.current,
        focus: focusRef.current,
        dark: true,
      }, dpr);

      const p = perf();
      p.lastFrameMs = performance.now() - start;
      p.frames++;
      if (p.lastFrameMs > 16) p.slowFrames++;
    };
    raf = requestAnimationFrame(frame);

    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  // ---- interaction --------------------------------------------------------

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pt = toLayout(e.clientX - rect.left, e.clientY - rect.top, viewRef.current);
    const hit = gridRef.current.pick(pt.x, pt.y, 5 / viewRef.current.scale);
    if (hit?.node.path !== hoverRef.current?.node.path) {
      hoverRef.current = hit;
      dirtyRef.current = true;
      setHoverPath(hit?.node.path ?? null);
    }
  }, []);

  const onWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    // Any hand on the controls cancels the flight, or the camera fights the user.
    camRef.current = null;
    const v = viewRef.current;
    const rect = e.currentTarget.getBoundingClientRect();
    const before = toLayout(e.clientX - rect.left, e.clientY - rect.top, v);
    v.scale = clampScale(v.scale * Math.exp(-e.deltaY * 0.0014));
    const after = toLayout(e.clientX - rect.left, e.clientY - rect.top, v);
    // Keep the point under the cursor pinned while the scale changes.
    v.x += before.x - after.x;
    v.y += before.y - after.y;
    dirtyRef.current = true;
  }, []);

  const dragRef = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    camRef.current = null;
    dragRef.current = { x: e.clientX, y: e.clientY, moved: false };
  }, []);

  const onMouseDrag = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const d = dragRef.current;
    if (!d) { onMouseMove(e); return; }
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    const v = viewRef.current;
    v.x -= dx / v.scale;
    v.y -= dy / v.scale;
    d.x = e.clientX;
    d.y = e.clientY;
    dirtyRef.current = true;
  }, [onMouseMove]);

  const onMouseUp = useCallback(() => {
    const d = dragRef.current;
    dragRef.current = null;
    if (d?.moved) return; // a drag is not a click

    const hit = hoverRef.current;
    selectedRef.current = hit;
    dirtyRef.current = true;
    if (!hit) {
      // Empty space: drop the selection, keep the cluster open. Escape leaves it.
      setSelectedPath(null);
      return;
    }
    void focusNode(hit.node);
  }, [focusNode]);

  // ---- render -------------------------------------------------------------

  const selected = selectedPath ? nodes.get(selectedPath) ?? null : null;
  /** Direct children of the selected folder — the "2 recordings" list. */
  const childNodes = selected && !selected.is_leaf
    ? [...nodes.values()]
        .filter((n) => n.path.startsWith(selected.path + "/")
          && n.depth === selected.depth + 1)
        .sort((a, b) => a.name.localeCompare(b.name))
    : [];
  const counts: Record<CState, number> | null = summary
    ? { remembered: summary.n_remembered, written: summary.n_written, failed: summary.n_failed }
    : null;

  if (fatal) {
    return (
      <div className="brain-view" data-theme="dark">
        <div className="brain-fatal">
          <h2>The archive map cannot load.</h2>
          <p>The database that stores what has been indexed did not answer.</p>
          <details><summary>Show technical details</summary><code>{fatal}</code></details>
        </div>
      </div>
    );
  }

  return (
    <div className="brain-view" data-theme="dark">
      <div className="brain-canvas-wrap" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          className="brain-canvas"
          onMouseMove={onMouseDrag}
          onMouseDown={onMouseDown}
          onMouseUp={onMouseUp}
          onMouseLeave={() => { dragRef.current = null; hoverRef.current = null; setHoverPath(null); dirtyRef.current = true; }}
          onWheel={onWheel}
          role="img"
          aria-label={
            summary
              ? `Archive map: ${summary.n_files.toLocaleString()} recordings, ` +
                `${summary.n_remembered.toLocaleString()} of them dated.`
              : "Archive map, loading"
          }
        />
        {nodes.size === 0 && !fatal && (
          <div className="brain-loading">Reading the archive…</div>
        )}
        {stale && (
          <div className="brain-stale" role="status">
            <span className="brain-stale-dot" aria-hidden />
            Reconnecting — showing the last picture we had
          </div>
        )}

        {/* Where you are, and the way back. Each crumb is a real ancestor path,
            so clicking one flies straight there without unloading anything. */}
        <nav className="brain-trail" aria-label="Archive location">
          <button onClick={() => { setFocusPath(null); setSelectedPath(null); }}
                  disabled={focusPath === null}>
            Whole archive
          </button>
          {focusPath?.split("/").map((name, i, all) => {
            const path = all.slice(0, i + 1).join("/");
            return (
              <button key={path} onClick={() => { setFocusPath(path); setSelectedPath(path); }}
                      disabled={i === all.length - 1}>
                {name}
              </button>
            );
          })}
          {focusPath && <span className="brain-trail-esc">Esc to go up</span>}
        </nav>

        {hoverPath && (
          <div className="brain-hint">
            Click to open · Scroll to zoom · Drag to move
          </div>
        )}
      </div>

      <div className="brain-side">
        <CorpusDetail
          node={selected}
          summary={summary}
          childNodes={childNodes}
          loading={!!selected && loadingPath === selected.path}
          onOpen={focusNode}
          onAuthFail={onAuthFail}
        />
        <BrainLegend counts={counts} />
      </div>
    </div>
  );
}
