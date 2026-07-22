import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, getCorpusState, getCorpusSummary } from "../api";
import type { CState, CorpusNode, CorpusSummary } from "../corpus";
import {
  MAX_NODE_RADIUS, PREFETCH_DEPTH, REVEAL_DEPTH, childrenUnfetched, parentPath, spineOf,
  visibleNodes,
} from "../corpus";
import type { Viewport } from "../viz/camera";
import { anchorPan, toLayout } from "../viz/camera";
import { draw } from "../viz/draw";
import { SpatialGrid } from "../viz/grid";
import type { Placed } from "../viz/radialTree";
import {
  applyDisplacements, buildTree, clampTarget, radialTree, rootNode, subtreeOf, translateSubtree,
} from "../viz/radialTree";
import { relaxAngles } from "../viz/relax";
import { ArchiveHome } from "./ArchiveHome";
import { BrainLegend } from "./BrainLegend";
import { CorpusDetail } from "./CorpusDetail";

const POLL_MS = 10_000;
/** Backoff ceiling. Stage C restarts rag-api mid-ingest; the map must ride it out. */
const MAX_BACKOFF_MS = 60_000;

/** `frameFor` clamps to this, so it is also the widest thing "Whole archive" can
 *  show. At 0.06 it silently clipped past an extent of ~6,600 — reachable today
 *  by opening the largest camp with Spread at 420, and trivially reachable by
 *  flinging a dot. 0.006 frames an extent of 92,000, which covers `MAX_DRAG_COORD`. */
const MIN_SCALE = 0.006;
const MAX_SCALE = 4;

/** Hover slop, in *layout* units, capped. `5 / scale` is 5 screen px, but at
 *  `MIN_SCALE` that is 833 layout units — a 19×19 sweep of the spatial grid on
 *  every pointermove. No dot is wider than this cap, so nothing is missed. */
const MAX_PICK_SLOP = 4 * MAX_NODE_RADIUS;

/** How long the camera takes to reach a cluster you opened. */
const CAM_MS = 420;
/** Breathing room, in layout units, around a framed cluster. */
const CAM_PAD = 90;

/** Spacing. `radialTree` floors `minRingGap` at 2·MAX_NODE_RADIUS = 44 whatever
 *  arrives here; 60 is where the picture stops being a solid disc. */
const SPREAD = { min: 60, max: 420, step: 5, def: 130 };
const BREATH = { min: 0, max: 24, step: 0.5, def: 2.5 };

/** A folder rename orphans an override. Cap the ledger so it cannot grow forever. */
const MAX_OVERRIDES = 500;

const SIDE = { min: 280, max: 640, def: 340, canvasMin: 360 };
const SIDE_NARROW = { min: 220, max: 900, def: 340, canvasMin: 260 };
const NARROW_QUERY = "(max-width: 1000px)";

const SPACING_KEY = "brain.spacing.v1";
/** v1 held bare angles. Reading one of those as a point would scatter the map, so
 *  the key changes with the shape and the old one is dropped on sight. */
const ARRANGE_KEY = "brain.arrange.v2";
const ARRANGE_KEY_V1 = "brain.arrange.v1";
const SIDE_KEY = "brain.side.v1";

interface Props {
  onAuthFail: () => void;
}

interface Spacing {
  minRingGap: number;
  padding: number;
}

/** A camera flight in progress. `null` means the view is at rest. */
interface CamFlight {
  fx: number; fy: number; fs: number;
  tx: number; ty: number; ts: number;
  t0: number;
  dur: number;
}

/** A press must travel this far, in screen pixels, before it stops being a click.
 *  Without it a one-pixel tremor on the way to a click counts as a drag, and the
 *  dot you meant to open silently moves instead. */
const DRAG_SLOP_PX = 3;

/** Panning the canvas, or carrying one node and everything inside it. Never both.
 *  `gx`/`gy` is where inside the dot the reader grabbed, in layout units — without
 *  it the node snaps its centre to the cursor on the first pixel of movement. */
type Drag =
  | { kind: "pan"; x: number; y: number; moved: boolean }
  | {
      kind: "node"; root: Placed; sub: Placed[];
      gx: number; gy: number; sx: number; sy: number; moved: boolean;
    };

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function clampScale(s: number): number {
  return clamp(s, MIN_SCALE, MAX_SCALE);
}

function num(v: unknown, fallback: number): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

/** Counters the acceptance tests read. Never used by the UI itself. */
declare global {
  interface Window {
    __brainPerf?: {
      frames: number; slowFrames: number; lastFrameMs: number; fetches: number;
      visibleNodes: number; layoutMs: number;
    };
  }
}

function perf() {
  if (!window.__brainPerf) {
    window.__brainPerf = {
      frames: 0, slowFrames: 0, lastFrameMs: 0, fetches: 0, visibleNodes: 0, layoutMs: 0,
    };
  }
  return window.__brainPerf;
}

// ---- preferences, all clamped on read ------------------------------------
// Every one of these came off disk and could say anything. A 900 px panel saved
// on a wide monitor must clamp on a laptop; a NaN spread must not silently
// become a blank canvas.

const DEFAULT_SPACING: Spacing = { minRingGap: SPREAD.def, padding: BREATH.def };

function loadSpacing(): Spacing {
  try {
    const raw = localStorage.getItem(SPACING_KEY);
    if (!raw) return DEFAULT_SPACING;
    const p = JSON.parse(raw) as Partial<Spacing>;
    return {
      minRingGap: clamp(num(p.minRingGap, SPREAD.def), SPREAD.min, SPREAD.max),
      padding: clamp(num(p.padding, BREATH.def), BREATH.min, BREATH.max),
    };
  } catch (e) {
    console.warn(`brain: ignoring unreadable "${SPACING_KEY}" preference`, e);
    return DEFAULT_SPACING;
  }
}

type Point = { x: number; y: number };

/** Where the reader put each node, as `[x, y]` in layout units. Points that came
 *  off disk are clamped by `applyDisplacements`; only their shape is checked here. */
function loadArrangement(): Map<string, Point> {
  const out = new Map<string, Point>();
  try {
    localStorage.removeItem(ARRANGE_KEY_V1);
    const raw = localStorage.getItem(ARRANGE_KEY);
    if (!raw) return out;
    for (const [path, v] of Object.entries(JSON.parse(raw) as Record<string, unknown>)) {
      if (out.size >= MAX_OVERRIDES) break;
      if (!Array.isArray(v) || v.length !== 2) continue;
      const [x, y] = v as unknown[];
      if (typeof x !== "number" || typeof y !== "number") continue;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      out.set(path, { x, y });
    }
  } catch (e) {
    console.warn(`brain: ignoring unreadable "${ARRANGE_KEY}" preference`, e);
  }
  return out;
}

/** Persist the arrangement, dropping paths the archive no longer has. A renamed
 *  folder would otherwise sit in the ledger forever, invisible and un-resettable. */
function saveArrangement(moves: Map<string, Point>, known: ReadonlySet<string>): void {
  for (const path of moves.keys()) if (known.size > 0 && !known.has(path)) moves.delete(path);
  const flat: Record<string, [number, number]> = {};
  for (const [path, p] of moves) flat[path] = [p.x, p.y];
  try {
    localStorage.setItem(ARRANGE_KEY, JSON.stringify(flat));
  } catch (e) {
    console.warn(`brain: could not save "${ARRANGE_KEY}"`, e);
  }
}

function loadSide(): { w: number; h: number } {
  try {
    const raw = localStorage.getItem(SIDE_KEY);
    if (!raw) return { w: SIDE.def, h: SIDE_NARROW.def };
    const p = JSON.parse(raw) as { w?: unknown; h?: unknown };
    return { w: num(p.w, SIDE.def), h: num(p.h, SIDE_NARROW.def) };
  } catch (e) {
    console.warn(`brain: ignoring unreadable "${SIDE_KEY}" preference`, e);
    return { w: SIDE.def, h: SIDE_NARROW.def };
  }
}

/** The node closest to the point the camera is centred on. It is the thing the
 *  reader is looking at, and therefore the thing a relayout must not move. */
function nearestTo(placed: Placed[], x: number, y: number): Placed | null {
  let best: Placed | null = null;
  let bestD = Infinity;
  for (const p of placed) {
    const d = (p.x - x) ** 2 + (p.y - y) ** 2;
    if (d < bestD) { bestD = d; best = p; }
  }
  return best;
}

export function BrainView({ onAuthFail }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const viewElRef = useRef<HTMLDivElement>(null);

  const [summary, setSummary] = useState<CorpusSummary | null>(null);
  const [nodes, setNodes] = useState<Map<string, CorpusNode>>(new Map());
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  /** The opened cluster. Drives the camera, the dimming, and — via `spineOf` —
   *  which nodes exist on the map at all. */
  const [focusPath, setFocusPath] = useState<string | null>(null);
  const [hoverPath, setHoverPath] = useState<string | null>(null);
  const [loadingPath, setLoadingPath] = useState<string | null>(null);
  const [stale, setStale] = useState(false);
  const [fatal, setFatal] = useState<string | null>(null);

  const [spacing, setSpacing] = useState<Spacing>(loadSpacing);
  const [controlsOpen, setControlsOpen] = useState(false);
  const [homeCollapsed, setHomeCollapsed] = useState(false);
  /** Bumped by Reset. The arrangement lives in a ref, so nothing else would notice. */
  const [arrangeTick, setArrangeTick] = useState(0);
  const [side, setSide] = useState(loadSide);
  const [narrow, setNarrow] = useState(
    () => window.matchMedia?.(NARROW_QUERY).matches ?? false,
  );

  /**
   * The open branch, derived — never stored.
   *
   * `focusNode`, `focusUp`, `Esc` and every breadcrumb already maintain
   * `focusPath` correctly, so collapse needs no second source of truth to keep in
   * step. Two consequences fall out for free: `Esc` *is* collapse, and
   * `selectedPath` can never point at a node the map has stopped drawing.
   */
  const open = useMemo(() => spineOf(focusPath), [focusPath]);

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

  useEffect(() => {
    const mq = window.matchMedia?.(NARROW_QUERY);
    if (!mq) return;
    const onChange = (e: MediaQueryListEvent) => setNarrow(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  // Mutable render state. Kept out of React: these change every frame during a
  // camera flight, and a setState per frame would re-render the detail panel 60
  // times a second.
  const viewRef = useRef<Viewport>({ x: 0, y: 0, scale: 0.3, width: 1, height: 1 });
  const placedRef = useRef<Placed[]>([]);
  const byPathRef = useRef<Map<string, Placed>>(new Map());
  const gridRef = useRef(new SpatialGrid());
  const hoverRef = useRef<Placed | null>(null);
  /** Mirrors `hoverPath`. Needed to re-resolve `hoverRef` after every relayout. */
  const hoverPathRef = useRef<string | null>(null);
  const selectedRef = useRef<Placed | null>(null);
  const openRef = useRef(open);
  const focusRef = useRef<string | null>(null);
  const dirtyRef = useRef(true);
  const framedRef = useRef(false);
  const camRef = useRef<CamFlight | null>(null);
  const reduceMotionRef = useRef(reduceMotion);
  const narrowRef = useRef(narrow);
  const spacingRef = useRef(spacing);
  const nodesRef = useRef(nodes);
  /** Absolute points the reader dragged nodes to, by path. A ref: it is written at
   *  pointer rate, and `applyDisplacements` reads it during layout. */
  const moveRef = useRef<Map<string, Point>>(loadArrangement());
  /** Paths whose children are already in `nodes`. Reset to the open branch on a
   *  version change — anything else may have changed underneath us. */
  const fetchedRef = useRef<Set<string>>(new Set());
  const dragRef = useRef<Drag | null>(null);
  /** `${focusPath}|${childCount}` of the last flight. Stops the 10 s poll — which
   *  hands back a fresh `nodes` Map every tick — from yanking the camera back. */
  const framedForRef = useRef<string | null>(null);

  useEffect(() => { reduceMotionRef.current = reduceMotion; }, [reduceMotion]);
  useEffect(() => { focusRef.current = focusPath; }, [focusPath]);
  useEffect(() => { openRef.current = open; }, [open]);
  useEffect(() => { narrowRef.current = narrow; }, [narrow]);
  useEffect(() => { spacingRef.current = spacing; }, [spacing]);
  useEffect(() => { nodesRef.current = nodes; }, [nodes]);

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

  /**
   * Refetch the skeleton plus the currently-open branch. Never the whole tree.
   *
   * `openPaths` is the spine — at most one path per level, five deep. It used to
   * be every folder ever opened, a set that only grew, so a long session's poll
   * tick fanned out one request per sitting the reader had ever glanced at.
   */
  const refetch = useCallback(async (openPaths: string[]) => {
    const p = perf();
    p.fetches++;
    const skeleton = await getCorpusState("", PREFETCH_DEPTH);
    const opened = await Promise.all(openPaths.map((path) => {
      p.fetches++;
      return getCorpusState(path, 1);
    }));
    mergeNodes([...skeleton.children, ...opened.flatMap((r) => r.children)]);
    fetchedRef.current = new Set(openPaths);
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
          await refetch([...openRef.current]);
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
   * Open a node: select it, reveal its children, and fly the camera into it —
   * fetching those children first when they are not already on the map.
   *
   * Opening, drawing and fetching are three different questions. This one only
   * answers the first two; `visibleNodes(nodes, spineOf(focusPath))` answers the
   * third for the renderer, and `childrenUnfetched` for the network.
   */
  const focusNode = useCallback(async (node: CorpusNode) => {
    setSelectedPath(node.path);
    // A recording is a destination, not a container: it opens the track panel,
    // and pulling the camera off its siblings would only lose the reader's place.
    if (node.is_leaf) return;

    setFocusPath(node.path);
    if (!childrenUnfetched(node, fetchedRef.current)) return;

    setLoadingPath(node.path);
    try {
      perf().fetches++;
      const res = await getCorpusState(node.path, 1);
      fetchedRef.current.add(node.path);
      mergeNodes(res.children);
    } catch (e) {
      if (!onError(e)) setStale(true);
    } finally {
      setLoadingPath(null);
    }
  }, [mergeNodes, onError]);

  /** Step out one level — which is exactly "collapse". The subtree stays in
   *  `nodes`, so re-entering costs nothing. */
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

  useEffect(() => {
    if (nodes.size === 0) return;
    const t0 = performance.now();
    const v = viewRef.current;

    // What the reader is looking at, before anything moves. Ring radii depend on
    // ring *demand*, so opening a cluster or nudging a slider moves every node in
    // layout space; without an anchor the picture lurches and takes the one thing
    // on screen the reader was holding on to with it.
    const anchor = framedRef.current ? nearestTo(placedRef.current, v.x, v.y) : null;
    const anchorAt = anchor ? { path: anchor.node.path, x: anchor.x, y: anchor.y } : null;

    const root = rootNode(summary?.n_files ?? 0, summary?.n_chunks ?? 0);
    const visible = visibleNodes(nodes.values(), open);
    const placed = radialTree(buildTree(root, visible), spacing);
    applyDisplacements(placed, moveRef.current);
    relaxAngles(placed, { padding: spacing.padding });

    placedRef.current = placed;
    // The only rebuild there is, outside a node drag. Nothing moves a node after
    // this — the drift is gone, and the camera transforms at draw time.
    gridRef.current.rebuild(placed);

    // One pass for everything downstream needs. This replaces four
    // `Math.max(64, ...placed.map(...))` spreads, which pass the whole array as
    // arguments and throw RangeError somewhere north of 65k nodes.
    let extent = 64;
    const byPath = new Map<string, Placed>();
    for (const p of placed) {
      byPath.set(p.node.path, p);
      const reach = p.dist + p.r;
      if (reach > extent) extent = reach;
    }
    byPathRef.current = byPath;

    // Both refs, not just the selection. `draw.ts` compares `Placed` by object
    // identity and `radialTree` reallocates every one of them, so a stale
    // `hoverRef` makes `neighbourhood()` return a set holding a single ghost —
    // and `dimmed()` is then true for every real node, dropping the entire map to
    // 10% opacity until the next mousemove. It used to fire only on the 10 s poll.
    // With collapse it would fire on every click.
    selectedRef.current = selectedPath ? byPath.get(selectedPath) ?? null : null;
    hoverRef.current = hoverPathRef.current ? byPath.get(hoverPathRef.current) ?? null : null;
    // A node drag holds `Placed` objects that no longer exist.
    if (dragRef.current?.kind === "node") dragRef.current = null;

    if (!framedRef.current && v.width > 1 && v.height > 1) {
      // Frame the whole archive on the first layout. Ring radii are derived from
      // how much each ring must hold, so the outermost radius is not knowable in
      // advance — a hardcoded starting zoom would land somewhere arbitrary.
      v.scale = clampScale(Math.min(v.width, v.height) / (extent * 2.16));
      v.x = 0;
      v.y = 0;
      framedRef.current = true;
      framedForRef.current = "null|0";
    } else if (anchorAt) {
      const now = byPath.get(anchorAt.path);
      if (now) Object.assign(v, anchorPan(v, anchorAt, now));
    }

    const p = perf();
    p.visibleNodes = placed.length;
    p.layoutMs = performance.now() - t0;
    dirtyRef.current = true;
    // `selectedPath` is deliberately *not* a dependency. It was, and clicking a
    // recording therefore ran `radialTree` plus 24 relaxation passes plus a full
    // `SpatialGrid.rebuild` to update one `.find()`. React always runs the newest
    // effect closure, so the read above still sees the current selection; it just
    // no longer re-lays the archive out to notice one.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, summary?.n_files, summary?.n_chunks, open, spacing, arrangeTick]);

  /** The cheap half of the split: a selection changes one reference, not a layout. */
  useEffect(() => {
    selectedRef.current = selectedPath ? byPathRef.current.get(selectedPath) ?? null : null;
    dirtyRef.current = true;
  }, [selectedPath]);

  useEffect(() => {
    try {
      localStorage.setItem(SPACING_KEY, JSON.stringify(spacing));
    } catch (e) {
      console.warn(`brain: could not save "${SPACING_KEY}"`, e);
    }
  }, [spacing]);

  /** Where the camera should sit to show `path` and its children. */
  const frameFor = useCallback((path: string | null): CamFlight | null => {
    const v = viewRef.current;
    const placed = placedRef.current;
    if (v.width <= 1 || placed.length === 0) return null;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, seen = 0;
    const take = (p: Placed) => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
      seen++;
    };

    if (path === null) {
      for (const p of placed) take(p);
    } else {
      const parent = byPathRef.current.get(path);
      if (!parent) return null;
      take(parent);
      // A folder whose children have not arrived yet still frames — on itself, at
      // a readable zoom. Otherwise the click reads as "nothing happened", which is
      // precisely the bug slice 1b was written to fix.
      for (const p of placed) if (p.parent === parent) take(p);
    }
    if (seen === 0) return null;

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

  // Retarget when the opened cluster changes, or when it gains children. Keyed on
  // both so the 10 s poll — which replaces the `nodes` Map every tick even when
  // nothing changed — cannot drag the camera out from under the reader. The
  // layout effect above is declared first and therefore runs first, so `placedRef`
  // is already post-expansion and the flight starts from the anchored camera.
  useEffect(() => {
    if (!framedRef.current) return;
    let kids = 0;
    if (focusPath !== null) {
      for (const p of placedRef.current) if (p.parent?.node.path === focusPath) kids++;
    }
    const key = `${focusPath}|${kids}`;
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
      // No compensating pan. `toScreen` is `(p - v)·s + width/2`, so `v` *is* the
      // layout point under the screen centre: when the width changes the content
      // re-centres itself. Nudging the pan by Δw/2 would be the bug, not the fix.
      dirtyRef.current = true;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const frame = () => {
      raf = requestAnimationFrame(frame);
      if (document.visibilityState !== "visible") return;

      // The only thing that moves on its own. When no cluster is being opened,
      // `camRef` is null, nothing marks the canvas dirty, and this loop costs one
      // branch per frame — the map is genuinely still. That stillness is the
      // signal: a glance tells you whether anything is happening.
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
        open: openRef.current,
        revealDepth: REVEAL_DEPTH,
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

  const pointAt = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return toLayout(e.clientX - rect.left, e.clientY - rect.top, viewRef.current);
  }, []);

  const pick = useCallback((pt: { x: number; y: number }) =>
    gridRef.current.pick(pt.x, pt.y, Math.min(5 / viewRef.current.scale, MAX_PICK_SLOP)), []);

  const setHover = useCallback((hit: Placed | null) => {
    const path = hit?.node.path ?? null;
    if (path === hoverPathRef.current) {
      hoverRef.current = hit; // same node, fresh object after a relayout
      return;
    }
    hoverRef.current = hit;
    hoverPathRef.current = path;
    dirtyRef.current = true;
    setHoverPath(path);
  }, []);

  /** Record where the reader let go, and let the frames that changed settle. */
  const commitNodeDrag = useCallback((d: Extract<Drag, { kind: "node" }>) => {
    // delete-then-set: a `Map` keeps insertion order, so this makes the most
    // recently arranged node the youngest and the eviction below the oldest.
    moveRef.current.delete(d.root.node.path);
    moveRef.current.set(d.root.node.path, { x: d.root.x, y: d.root.y });
    while (moveRef.current.size > MAX_OVERRIDES) {
      const oldest = moveRef.current.keys().next().value as string;
      moveRef.current.delete(oldest);
    }
    saveArrangement(moveRef.current, new Set(nodesRef.current.keys()));

    // Re-derive every `frame` from the ledger rather than patching the one the
    // live drag left behind. Costs one pointerup, and it makes the state after a
    // drag identical to the state after a reload — a whole class of bug closed.
    applyDisplacements(placedRef.current, moveRef.current);
    relaxAngles(placedRef.current, { padding: spacingRef.current.padding });
    gridRef.current.rebuild(placedRef.current);
    dirtyRef.current = true;
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    camRef.current = null;
    e.currentTarget.setPointerCapture(e.pointerId);

    const pt = pointAt(e);
    const hit = pick(pt);
    setHover(hit);

    // Drag a dot, the dot moves; drag the space between them, the map moves.
    // Nothing to learn, nothing to enable — which is the point: a hidden Shift
    // gesture behind a collapsed panel is a feature nobody has. Shift now forces
    // a pan, for the rare press that lands on a dot by accident. `hit.dist > 0`
    // excludes the root: it sits at the origin, and moving everything is a pan.
    if (hit && hit.dist > 0 && !e.shiftKey) {
      dragRef.current = {
        kind: "node", root: hit, sub: subtreeOf(hit, placedRef.current),
        gx: pt.x - hit.x, gy: pt.y - hit.y,
        sx: e.clientX, sy: e.clientY, moved: false,
      };
    } else {
      dragRef.current = { kind: "pan", x: e.clientX, y: e.clientY, moved: false };
    }
  }, [pick, pointAt, setHover]);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = dragRef.current;
    if (!d) { setHover(pick(pointAt(e))); return; }

    if (d.kind === "pan") {
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
      const v = viewRef.current;
      v.x -= dx / v.scale;
      v.y -= dy / v.scale;
      d.x = e.clientX;
      d.y = e.clientY;
      dirtyRef.current = true;
      return;
    }

    // Below the slop this press is still a click, and moving the node now would
    // steal it. Measured in screen pixels, not layout units: at MIN_SCALE a layout
    // unit is a thousandth of a pixel, and no press would ever survive as a click.
    if (!d.moved) {
      if (Math.abs(e.clientX - d.sx) + Math.abs(e.clientY - d.sy) <= DRAG_SLOP_PX) return;
      d.moved = true;
    }

    // Translate the already-collected subtree and re-bucket the grid — never
    // `radialTree` and 24 relaxation passes per pointermove. This is the one 60 Hz
    // path in the system, and `bench:viz` holds it to 16 ms.
    const pt = pointAt(e);
    const want = clampTarget(pt.x - d.gx, pt.y - d.gy);
    translateSubtree(d.sub, want.x - d.root.x, want.y - d.root.y);
    gridRef.current.rebuild(placedRef.current);
    dirtyRef.current = true;
  }, [pick, pointAt, setHover]);

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    if (!d) return;

    if (d.kind === "node") {
      // Under the slop it never became a drag, so it is still the click it started
      // as. This is what lets a dot be both a button and a handle.
      if (d.moved) commitNodeDrag(d);
      else void focusNode(d.root.node);
      return;
    }

    if (d.moved) return; // a drag is not a click

    const hit = hoverRef.current;
    selectedRef.current = hit;
    dirtyRef.current = true;
    if (!hit) {
      // Empty space: drop the selection, keep the cluster open. Escape leaves it.
      setSelectedPath(null);
      return;
    }
    void focusNode(hit.node);
  }, [commitNodeDrag, focusNode]);

  /** The pointer was taken away mid-drag (tab switch, a system gesture). Keep what
   *  the reader had already moved rather than silently abandoning it. */
  const onPointerCancel = useCallback(() => {
    const d = dragRef.current;
    dragRef.current = null;
    if (d?.kind === "node" && d.moved) commitNodeDrag(d);
  }, [commitNodeDrag]);

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

  const resetArrangement = useCallback(() => {
    moveRef.current.clear();
    saveArrangement(moveRef.current, new Set());
    setSpacing({ ...DEFAULT_SPACING });
    setArrangeTick((t) => t + 1);
  }, []);

  // ---- the resizable panel -------------------------------------------------

  const limits = useCallback(() => {
    const el = viewElRef.current;
    const cfg = narrowRef.current ? SIDE_NARROW : SIDE;
    const total = narrowRef.current ? (el?.clientHeight ?? 900) : (el?.clientWidth ?? 1200);
    return { min: cfg.min, max: Math.max(cfg.min, Math.min(cfg.max, total - cfg.canvasMin)) };
  }, []);

  const resizeRef = useRef<{ pending: number } | null>(null);

  /** Write straight to the DOM. `.brain-side` renders one `<li>` per transcript
   *  segment — thousands for a two-hour pravachan — and a `setState` per
   *  pointermove would reconcile every one of them. State is committed on release. */
  const applySide = useCallback((px: number) => {
    viewElRef.current?.style.setProperty(
      narrowRef.current ? "--brain-side-h" : "--brain-side-w", `${px}px`,
    );
  }, []);

  const commitSide = useCallback((px: number) => {
    setSide((s) => (narrowRef.current ? { ...s, h: px } : { ...s, w: px }));
  }, []);

  const onResizeDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    resizeRef.current = { pending: narrowRef.current ? side.h : side.w };
  }, [side.h, side.w]);

  const onResizeMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const r = resizeRef.current;
    const el = viewElRef.current;
    if (!r || !el) return;
    const box = el.getBoundingClientRect();
    const { min, max } = limits();
    const raw = narrowRef.current ? box.bottom - e.clientY : box.right - e.clientX;
    r.pending = Math.round(clamp(raw, min, max));
    applySide(r.pending);
  }, [applySide, limits]);

  const onResizeUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const r = resizeRef.current;
    resizeRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    if (r) commitSide(r.pending);
  }, [commitSide]);

  const onResizeKey = useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    const { min, max } = limits();
    const cur = narrowRef.current ? side.h : side.w;
    const step = e.shiftKey ? 64 : 16;
    let next: number | null = null;
    // Up and Left grow the panel: both move the separator away from it.
    if (e.key === "ArrowUp" || e.key === "ArrowLeft") next = cur + step;
    else if (e.key === "ArrowDown" || e.key === "ArrowRight") next = cur - step;
    else if (e.key === "Home") next = min;
    else if (e.key === "End") next = max;
    else if (e.key === "Enter" || e.key === " ") next = narrowRef.current ? SIDE_NARROW.def : SIDE.def;
    if (next === null) return;
    e.preventDefault();
    const px = Math.round(clamp(next, min, max));
    applySide(px);
    commitSide(px);
  }, [applySide, commitSide, limits, side.h, side.w]);

  const resetSide = useCallback(() => {
    const px = narrowRef.current ? SIDE_NARROW.def : SIDE.def;
    applySide(px);
    commitSide(px);
  }, [applySide, commitSide]);

  // Clamp on read *and* on every reflow: a 900 px panel saved on a wide monitor
  // must not eat a laptop's canvas, and rotating a tablet swaps which axis rules.
  useEffect(() => {
    const el = viewElRef.current;
    if (!el) return;
    const { min, max } = limits();
    const cur = narrow ? side.h : side.w;
    const px = Math.round(clamp(cur, min, max));
    el.style.setProperty("--brain-side-w", `${narrow ? side.w : px}px`);
    el.style.setProperty("--brain-side-h", `${narrow ? px : side.h}px`);
    if (px !== cur) commitSide(px);
    try {
      localStorage.setItem(SIDE_KEY, JSON.stringify(side));
    } catch (e) {
      console.warn(`brain: could not save "${SIDE_KEY}"`, e);
    }
  }, [side, narrow, limits, commitSide]);

  // ---- render -------------------------------------------------------------

  const selected = selectedPath ? nodes.get(selectedPath) ?? null : null;
  /** Direct children of the selected folder — the "2 recordings" list. Memoised:
   *  a fresh array every render would defeat `memo(CorpusDetail)`. */
  const childNodes = useMemo(() => (
    selected && !selected.is_leaf
      ? [...nodes.values()]
          .filter((n) => n.path.startsWith(selected.path + "/") && n.depth === selected.depth + 1)
          .sort((a, b) => a.name.localeCompare(b.name))
      : []
  ), [nodes, selected]);

  // Opening a recording is the one moment the totals are in the way: `TrackPanel`
  // wants every pixel of the column. Folders leave it alone — nothing appears
  // below them that the archive's own numbers are competing with.
  const openedLeaf = selected?.is_leaf ? selected.path : null;
  useEffect(() => { if (openedLeaf) setHomeCollapsed(true); }, [openedLeaf]);

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

  const sidePx = narrow ? side.h : side.w;
  const { min: sideMin, max: sideMax } = limits();

  return (
    <div className="brain-view" data-theme="dark" ref={viewElRef}>
      <div className="brain-canvas-wrap" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          className="brain-canvas"
          onPointerMove={onPointerMove}
          onPointerDown={onPointerDown}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerCancel}
          onPointerLeave={() => { if (!dragRef.current) setHover(null); }}
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
          {focusPath && <span className="brain-trail-esc">Esc to close it</span>}
        </nav>

        {/* Spacing. Both sliders are safe across their whole range — the ring-gap
            floor and the chord-exact relaxation are proved in check:viz. */}
        <div className="brain-controls">
          <button className="brain-controls-toggle" onClick={() => setControlsOpen((o) => !o)}
                  aria-expanded={controlsOpen} aria-controls="brain-controls-body">
            <span aria-hidden>{controlsOpen ? "▾" : "▸"}</span> Spacing &amp; layout
          </button>
          {controlsOpen && (
            <div className="brain-controls-body" id="brain-controls-body">
              <label>
                <span>Spread <em>{Math.round(spacing.minRingGap)}</em></span>
                <input type="range" min={SPREAD.min} max={SPREAD.max} step={SPREAD.step}
                       value={spacing.minRingGap}
                       onChange={(e) => setSpacing((s) => ({ ...s, minRingGap: Number(e.target.value) }))} />
              </label>
              <label>
                <span>Breathing room <em>{spacing.padding.toFixed(1)}</em></span>
                <input type="range" min={BREATH.min} max={BREATH.max} step={BREATH.step}
                       value={spacing.padding}
                       onChange={(e) => setSpacing((s) => ({ ...s, padding: Number(e.target.value) }))} />
              </label>
              <button className="brain-controls-reset" onClick={resetArrangement}>
                Reset spacing and arrangement
              </button>
            </div>
          )}
        </div>

        {hoverPath && (
          <div className="brain-hint">
            Drag a dot anywhere — it carries what's inside it · Click to open ·
            Scroll to zoom · Drag the background to pan
          </div>
        )}
      </div>

      <div
        className="brain-resizer"
        role="separator"
        tabIndex={0}
        aria-orientation={narrow ? "horizontal" : "vertical"}
        aria-controls="brain-side"
        aria-label="Resize the reading panel"
        aria-valuenow={sidePx}
        aria-valuemin={sideMin}
        aria-valuemax={sideMax}
        onPointerDown={onResizeDown}
        onPointerMove={onResizeMove}
        onPointerUp={onResizeUp}
        onPointerCancel={onResizeUp}
        onKeyDown={onResizeKey}
        onDoubleClick={resetSide}
      />

      <div className="brain-side" id="brain-side">
        <ArchiveHome
          summary={summary}
          collapsed={homeCollapsed}
          onToggle={() => setHomeCollapsed((c) => !c)}
        />
        <CorpusDetail
          node={selected}
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
