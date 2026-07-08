import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, getCorpusState, getCorpusSummary } from "../api";
import type { CState, CorpusNode, CorpusSummary } from "../corpus";
import { SKELETON_DEPTH, canExpand } from "../corpus";
import { SpatialGrid } from "../viz/grid";
import type { Placed } from "../viz/radialTree";
import { buildTree, radialTree, rootNode } from "../viz/radialTree";
import { drift, relaxAngles } from "../viz/relax";
import type { Viewport } from "../viz/draw";
import { draw, toLayout } from "../viz/draw";
import { BrainLegend } from "./BrainLegend";
import { CorpusDetail } from "./CorpusDetail";

const POLL_MS = 10_000;
/** Backoff ceiling. Stage C restarts rag-api mid-ingest; the map must ride it out. */
const MAX_BACKOFF_MS = 60_000;

const MIN_SCALE = 0.06;
const MAX_SCALE = 4;

interface Props {
  onAuthFail: () => void;
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
  const [hoverPath, setHoverPath] = useState<string | null>(null);
  const [loadingPath, setLoadingPath] = useState<string | null>(null);
  const [stale, setStale] = useState(false);
  const [fatal, setFatal] = useState<string | null>(null);

  const reduceMotion = useMemo(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
    [],
  );

  // Mutable render state. Kept out of React: these change every frame, and a
  // setState per frame would re-render the detail panel 60 times a second.
  const viewRef = useRef<Viewport>({ x: 0, y: 0, scale: 0.3, width: 1, height: 1 });
  const placedRef = useRef<Placed[]>([]);
  const gridRef = useRef(new SpatialGrid());
  const hoverRef = useRef<Placed | null>(null);
  const selectedRef = useRef<Placed | null>(null);
  const expandedRef = useRef(expanded);
  const dirtyRef = useRef(true);
  const framedRef = useRef(false);
  /** Set when a cluster is opened, consumed by the next layout pass. */
  const focusRef = useRef<string | null>(null);

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

  const toggleExpand = useCallback(async (node: CorpusNode) => {
    if (!canExpand(node)) return;
    if (expanded.has(node.path)) {
      setExpanded((prev) => {
        const next = new Set(prev);
        next.delete(node.path);
        return next;
      });
      // Drop the subtree we loaded for it, but keep the skeleton depth intact.
      setNodes((prev) => {
        const next = new Map(prev);
        for (const key of next.keys()) {
          if (key.startsWith(node.path + "/") && next.get(key)!.depth > SKELETON_DEPTH) {
            next.delete(key);
          }
        }
        return next;
      });
      return;
    }
    setLoadingPath(node.path);
    try {
      perf().fetches++;
      const res = await getCorpusState(node.path, 1);
      focusRef.current = node.path;
      mergeNodes(res.children);
      setExpanded((prev) => new Set(prev).add(node.path));
    } catch (e) {
      if (!onError(e)) setStale(true);
    } finally {
      setLoadingPath(null);
    }
  }, [expanded, mergeNodes, onError]);

  // ---- layout -------------------------------------------------------------

  useEffect(() => { expandedRef.current = expanded; }, [expanded]);

  useEffect(() => {
    if (nodes.size === 0) return;
    const root = rootNode(summary?.n_files ?? 0, summary?.n_chunks ?? 0);
    const placed = radialTree(buildTree(root, [...nodes.values()]));
    relaxAngles(placed);
    placedRef.current = placed;
    gridRef.current.rebuild(placed);
    selectedRef.current = placed.find((p) => p.node.path === selectedPath) ?? null;

    const v = viewRef.current;
    const ready = v.width > 1 && v.height > 1;

    // Frame the whole archive on the first layout. Ring radii are derived from
    // how much each ring must hold, so the outermost radius is not knowable in
    // advance — a hardcoded starting zoom would land somewhere arbitrary.
    // Re-framing on every layout would yank the view out from under the user.
    if (!framedRef.current && ready) {
      const extent = Math.max(64, ...placed.map((p) => p.dist + p.r));
      v.scale = clampScale(Math.min(v.width, v.height) / (extent * 2.16));
      v.x = 0;
      v.y = 0;
      framedRef.current = true;
    }

    // A freshly opened cluster puts its tracks on a new outermost ring, inside
    // a wedge a fraction of a degree wide. At whole-archive zoom that is a few
    // sub-pixel dots at the rim — the operator double-clicks and nothing
    // visibly happens. So the view follows the thing they just opened.
    const focus = focusRef.current;
    if (focus && ready) {
      focusRef.current = null;
      const kids = placed.filter((p) => p.parent?.node.path === focus);
      const parent = placed.find((p) => p.node.path === focus);
      if (parent && kids.length > 0) {
        const pts = [parent, ...kids];
        const xs = pts.map((p) => p.x);
        const ys = pts.map((p) => p.y);
        const w = Math.max(...xs) - Math.min(...xs);
        const h = Math.max(...ys) - Math.min(...ys);
        const pad = 90;
        v.x = (Math.min(...xs) + Math.max(...xs)) / 2;
        v.y = (Math.min(...ys) + Math.max(...ys)) / 2;
        v.scale = clampScale(Math.min(v.width / (w + pad), v.height / (h + pad)));
      }
    }
    dirtyRef.current = true;
  }, [nodes, summary?.n_files, summary?.n_chunks, selectedPath]);

  // ---- render loop --------------------------------------------------------

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    let raf = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    const t0 = performance.now();

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
      const visible = document.visibilityState === "visible";

      // Motion is decoration, so it is the first thing sacrificed: reduced
      // motion, or a hidden tab, and we redraw only when something changed.
      const animating = visible && !reduceMotion && placedRef.current.length > 0;
      if (animating) {
        drift(placedRef.current, (performance.now() - t0) / 1000);
        gridRef.current.rebuild(placedRef.current);
        dirtyRef.current = true;
      }
      if (!dirtyRef.current || !visible) return;
      dirtyRef.current = false;

      const start = performance.now();
      draw(ctx, {
        placed: placedRef.current,
        view: viewRef.current,
        hover: hoverRef.current,
        selected: selectedRef.current,
        expanded: expandedRef.current,
        dark: true,
      }, dpr);

      const p = perf();
      p.lastFrameMs = performance.now() - start;
      p.frames++;
      if (p.lastFrameMs > 16) p.slowFrames++;
    };
    raf = requestAnimationFrame(frame);

    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, [reduceMotion]);

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
    setSelectedPath(hit?.node.path ?? null);
    dirtyRef.current = true;
  }, []);

  const onDoubleClick = useCallback(() => {
    const hit = hoverRef.current;
    if (hit) void toggleExpand(hit.node);
  }, [toggleExpand]);

  // ---- render -------------------------------------------------------------

  const selected = selectedPath ? nodes.get(selectedPath) ?? null : null;
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
          onDoubleClick={onDoubleClick}
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
        {hoverPath && <div className="brain-hint">Click to inspect · Double-click to open</div>}
      </div>

      <div className="brain-side">
        <CorpusDetail
          node={selected}
          summary={summary}
          expanded={selected ? expanded.has(selected.path) : false}
          loading={!!selected && loadingPath === selected.path}
          onToggleExpand={toggleExpand}
        />
        <BrainLegend counts={counts} />
      </div>
    </div>
  );
}
