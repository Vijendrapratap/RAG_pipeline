/**
 * Types and the state→colour map for the archive map.
 *
 * The colours mean exactly what Postgres can prove, and nothing more. This map
 * shows the 9,335 tracks that are *indexed*, not the 27,806 that exist on disk;
 * a track that has never been transcribed has no row and therefore no node. The
 * legend says so. Do not add a "dark" state until the desktop overlay can see
 * the filesystem — an invented state is worse than a missing one.
 */

/**
 * Depth of the tree fetched up front: the collections *and* their groups. ~190
 * nodes in one ~40 KB request.
 *
 * Exactly one level deeper than `REVEAL_DEPTH`, on purpose. First paint draws
 * only the collections; the level below them is already in hand, so the first
 * click opens instantly instead of waiting on the network. Every click after
 * that costs one `depth=1` request.
 *
 * It was 3 (~1,950 nodes, ~430 KB) while the map drew every node it had ever
 * fetched.
 */
export const PREFETCH_DEPTH = 2;

/**
 * Nodes at or above this depth are always drawn. Below it, only what is open.
 *
 * 1: the whole archive is its collections, and nothing else. It was 2, which
 * put all ~190 groups on the disc at once — a solid outer smear — and made
 * clicking a collection a visual no-op, because its children were already on
 * screen and its grandchildren needed the *group* opened. One click, one level:
 * that is the only rule a reader should have to learn.
 */
export const REVEAL_DEPTH = 1;

/** Ceiling on `nodeRadius`. The ring-gap floor is derived from it: two dots on
 *  adjacent rings cannot touch while the rings are `2 * MAX_NODE_RADIUS` apart. */
export const MAX_NODE_RADIUS = 22;

/** A node's dominant state. Containers take the state of their majority. */
export type CState = "remembered" | "written" | "failed";

/** One node as the backend serves it. Mirrors `rag_api.corpus.build_nodes`. */
export interface CorpusNode {
  path: string;
  name: string;
  depth: number;
  is_leaf: boolean;
  n_files: number;
  n_chunks: number;
  duration_sec: number;
  remembered: number;
  written: number;
  failed: number;
  session_date: string | null;
}

export interface CorpusStateResponse {
  version: string;
  prefix: string;
  depth: number;
  children: CorpusNode[];
}

/** One transcript segment. `start`/`end` are seconds into the isolated audio. */
export interface TrackSegment {
  start: number;
  end: number;
  text: string;
}

/** Mirrors `GET /api/track/transcript`. */
export interface TrackTranscript {
  source_file: string;
  duration: number | null;
  language: string | null;
  model: string | null;
  segments: TrackSegment[];
  cleaned_text: string | null;
  /** Set when the Qwen polish was discarded (summarized, or a repetition loop). */
  clean_note: string | null;
  /** null when no isolated WAV is on disk. Already carries its access ticket. */
  audio_url: string | null;
}

export interface CorpusSummary {
  version: string;
  n_files: number;
  n_chunks: number;
  n_remembered: number;
  n_written: number;
  n_failed: number;
  degenerate_files: string[];
  duration_sec: number;
  min_session_date: string | null;
  max_session_date: string | null;
  max_ingested_at: string | null;
}

export const STATE_COLOR: Record<CState, string> = {
  remembered: "#f0a35e", // warm saffron — the archive's own accent, lit
  written: "#6f6552", //   dim clay — present, but the system has no date for it
  failed: "#d64541", //    red — broken identity key, cannot be located
};

export const STATE_LABEL: Record<CState, string> = {
  remembered: "Remembered",
  written: "Written down",
  failed: "Broken",
};

/** What each colour actually promises. Rendered verbatim in the legend. */
export const STATE_MEANING: Record<CState, string> = {
  remembered: "Searchable, and we know the day it was recorded.",
  written: "Searchable — but no date could be read from its folder.",
  failed: "Indexed under a name with no place in the archive.",
};

/** A node's state is its majority. Ties resolve toward the worse state, so a
 *  container never looks healthier than its contents. */
export function nodeState(n: CorpusNode): CState {
  if (n.failed > 0 && n.failed >= n.remembered && n.failed >= n.written) return "failed";
  return n.written >= n.remembered ? "written" : "remembered";
}

/** 0..1 — the share of a container that is fully remembered. Drives opacity, so
 *  a half-dated camp reads as half-lit rather than snapping to one colour. */
export function litRatio(n: CorpusNode): number {
  return n.n_files > 0 ? n.remembered / n.n_files : 0;
}

/** Node radius from magnitude. sqrt keeps *area* proportional to content —
 *  a linear radius would make a 1,000-track collection look 1,000× a track. */
export function nodeRadius(n: CorpusNode, isRoot = false): number {
  if (isRoot) return 14;
  const magnitude = n.n_chunks > 0 ? n.n_chunks : n.n_files;
  return Math.min(MAX_NODE_RADIUS, 2.2 + Math.sqrt(magnitude) * 0.7);
}

const HOUR = 3600;

export function formatDuration(sec: number): string {
  if (!sec || sec < 60) return `${Math.round(sec)}s`;
  if (sec < HOUR) return `${Math.round(sec / 60)} min`;
  const hours = sec / HOUR;
  if (hours < 100) return `${hours.toFixed(1)} hours`;
  return `${Math.round(hours).toLocaleString()} hours`;
}

export function formatCount(n: number): string {
  return n.toLocaleString();
}

/**
 * Whether this node's children still need fetching.
 *
 * A collection's children came down with the skeleton, so opening one costs a
 * request and reveals nothing new. This is *only* about the network — it used to
 * double as "can this node be opened at all", which is why clicking
 * `Dagshai 2001` did nothing whatsoever: depth 1 < the prefetch depth, so the
 * handler bailed before it ever moved the camera. Opening is `focusNode`;
 * fetching is this. They are different questions, and so is *drawing*, which is
 * `visibleNodes`.
 */
export function childrenUnfetched(n: CorpusNode, fetched: ReadonlySet<string>): boolean {
  return !n.is_leaf && n.depth >= PREFETCH_DEPTH && !fetched.has(n.path);
}

/** The parent path of a node, or "" at the archive root. */
export function parentPath(path: string): string {
  const i = path.lastIndexOf("/");
  return i < 0 ? "" : path.slice(0, i);
}

/** Ancestor paths, outermost first: "a/b/c" -> ["a", "a/b"]. */
export function ancestors(path: string): string[] {
  const parts = path.split("/");
  return parts.slice(0, -1).map((_, i) => parts.slice(0, i + 1).join("/"));
}

/**
 * The open branch: the focused node and every ancestor of it.
 *
 * Derived from `focusPath`, never stored. That is what makes "collapse" free —
 * there is no second source of truth to keep in step, `Esc` (which walks
 * `focusPath` up one level) *is* collapse, and `selectedPath` can never point at
 * a node the map has stopped drawing.
 */
export function spineOf(focusPath: string | null): Set<string> {
  const open = new Set<string>();
  if (!focusPath) return open;
  for (const a of ancestors(focusPath)) open.add(a);
  open.add(focusPath);
  return open;
}

/**
 * The nodes the map should draw: everything down to `revealDepth`, plus the
 * children of every open container below it.
 *
 * The renderer already knew which nodes were irrelevant — it dimmed them to 10%
 * opacity — and paid full layout, relaxation, spatial-grid and raster cost to
 * draw them invisibly. Progressive disclosure is deleting them instead.
 *
 * Keyed on `parentPath`, never `startsWith`: `Live Masters 2010` is a string
 * prefix of `Live Masters 2010 B`, and one rename would make that bug real.
 */
export function visibleNodes(
  all: Iterable<CorpusNode>, open: ReadonlySet<string>, revealDepth: number = REVEAL_DEPTH,
): CorpusNode[] {
  const sorted = [...all].sort((a, b) => a.depth - b.depth);
  const revealed = new Set<string>();
  const out: CorpusNode[] = [];
  for (const n of sorted) {
    if (n.depth > revealDepth && !revealed.has(parentPath(n.path))) continue;
    out.push(n);
    if (!n.is_leaf && (n.depth < revealDepth || open.has(n.path))) revealed.add(n.path);
  }
  return out;
}

/** The level a node sits at, named the way an archivist would name it.
 *  Derived from distance to the leaf, never from depth: the Dagshai tree
 *  inserts a month bucket that Live Masters has no equivalent of. */
export function levelName(n: CorpusNode): string {
  if (n.is_leaf) return "Recording";
  if (n.depth === 1) return "Collection";
  return "Group of sittings";
}
