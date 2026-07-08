/**
 * Radial hierarchy layout. Radius is tree depth; angle comes from cumulative
 * leaf count so siblings get arc proportional to what they contain.
 *
 * Why not a force simulation, which is what Obsidian's graph does: Obsidian's
 * vault is a sparse *web* of links over ~1,000 notes, and the force layout
 * discovers structure nobody entered by hand. This archive is a strict *tree* —
 * every edge is containment, and we already know the whole thing. Simulating it
 * would throw away the only relationship we know for certain and hand back a
 * hairball. So depth stays exact and only the angle relaxes (see relax.ts).
 *
 * The consequence: a node's distance from the centre always means the same
 * thing, so the picture can be read rather than merely admired.
 */
import type { CorpusNode } from "../corpus";
import { nodeRadius } from "../corpus";

export interface TreeNode {
  node: CorpusNode;
  children: TreeNode[];
}

export interface Placed {
  node: CorpusNode;
  /** Layout coordinates, centred on the origin. The renderer applies pan/zoom. */
  x: number;
  y: number;
  /** Drawn radius of the dot, in layout units. */
  r: number;
  /** Angle in radians — `relaxAngles` mutates this, then x/y are recomputed. */
  angle: number;
  /** Distance from origin. Set once from depth. Never mutated afterwards. */
  dist: number;
  parent: Placed | null;
}

/** The synthetic root. It has no row in `file_meta`, so it is built, not fetched. */
export function rootNode(nFiles: number, nChunks: number): CorpusNode {
  return {
    path: "", name: "The Archive", depth: 0, is_leaf: false,
    n_files: nFiles, n_chunks: nChunks, duration_sec: 0,
    remembered: 0, written: 0, failed: 0, session_date: null,
  };
}

/**
 * Assemble the flat node list the API returns into a tree.
 *
 * Nodes are keyed by `path`, and a child is attached to the deepest ancestor
 * that is present. That "deepest present ancestor" rule is load-bearing: when a
 * session's tracks are fetched but its siblings are not, the tracks must still
 * find their session — and when a depth-3 fetch stops above the tracks, nothing
 * should be orphaned into the root.
 */
export function buildTree(root: CorpusNode, nodes: CorpusNode[]): TreeNode {
  const byPath = new Map<string, TreeNode>();
  const tree: TreeNode = { node: root, children: [] };
  byPath.set("", tree);

  // Shallowest first, so a parent always exists before its children arrive.
  for (const n of [...nodes].sort((a, b) => a.depth - b.depth)) {
    if (byPath.has(n.path)) continue;
    const tn: TreeNode = { node: n, children: [] };
    byPath.set(n.path, tn);

    let parent = tree;
    for (let cut = n.path.lastIndexOf("/"); cut > 0; cut = n.path.lastIndexOf("/", cut - 1)) {
      const found = byPath.get(n.path.slice(0, cut));
      if (found) { parent = found; break; }
    }
    parent.children.push(tn);
  }
  return tree;
}

/** Leaves under `t`, minimum 1 — a childless folder still needs arc to sit in. */
function leafCount(t: TreeNode, memo: Map<TreeNode, number>): number {
  const hit = memo.get(t);
  if (hit !== undefined) return hit;
  let n = 0;
  if (t.children.length === 0) n = 1;
  else for (const c of t.children) n += leafCount(c, memo);
  memo.set(t, n);
  return n;
}

export interface LayoutOptions {
  /** Smallest allowed distance between consecutive rings. */
  minRingGap?: number;
  /** Radius of the innermost ring. */
  innerRadius?: number;
  /** Arc padding between adjacent dots, in layout units. */
  padding?: number;
}

/**
 * Ring radii, sized by what each ring must hold.
 *
 * A fixed gap per depth is the obvious choice and it is wrong here. Depth 3
 * holds 1,748 sittings; on a fixed-gap ring of radius 332 each gets 1.2 units of
 * arc while its dot needs about 7, so the ring is a solid smear and no amount of
 * angular relaxation can fix it — the space simply is not there.
 *
 * So each ring is pushed out until its circumference can seat its own contents:
 * `2πR >= Σ(2r + padding)`. Radius still encodes depth exactly — every node at a
 * given depth shares one radius — which is the invariant the whole picture rests
 * on. It just is not a *linear* encoding of it.
 */
function ringRadii(
  byDepth: Map<number, TreeNode[]>, innerRadius: number, minRingGap: number, padding: number,
): Map<number, number> {
  const radii = new Map<number, number>([[0, 0]]);
  const maxDepth = Math.max(0, ...byDepth.keys());
  let prev = 0;
  for (let d = 1; d <= maxDepth; d++) {
    const ring = byDepth.get(d) ?? [];
    let demand = 0;
    for (const t of ring) demand += 2 * nodeRadius(t.node) + padding;
    const needed = demand / (2 * Math.PI);
    const r = Math.max(prev + minRingGap, needed, d === 1 ? innerRadius : 0);
    radii.set(d, r);
    prev = r;
  }
  return radii;
}

/**
 * Place every node. Each subtree receives an angular wedge proportional to the
 * leaves it holds, and its children divide that wedge the same way, recursively.
 */
export function radialTree(tree: TreeNode, opts: LayoutOptions = {}): Placed[] {
  const minRingGap = opts.minRingGap ?? 130;
  const innerRadius = opts.innerRadius ?? 150;
  const padding = opts.padding ?? 2.5;

  const byDepth = new Map<number, TreeNode[]>();
  (function walk(t: TreeNode, d: number) {
    const at = byDepth.get(d);
    if (at) at.push(t);
    else byDepth.set(d, [t]);
    for (const c of t.children) walk(c, d + 1);
  })(tree, 0);
  const radii = ringRadii(byDepth, innerRadius, minRingGap, padding);

  const memo = new Map<TreeNode, number>();
  const out: Placed[] = [];

  const place = (
    t: TreeNode, depth: number, a0: number, a1: number, parent: Placed | null,
  ): void => {
    const angle = (a0 + a1) / 2;
    const dist = radii.get(depth) ?? 0;
    const p: Placed = {
      node: t.node,
      x: Math.cos(angle) * dist,
      y: Math.sin(angle) * dist,
      r: nodeRadius(t.node, depth === 0),
      angle, dist, parent,
    };
    out.push(p);

    const total = leafCount(t, memo);
    let cursor = a0;
    for (const c of t.children) {
      const share = (leafCount(c, memo) / total) * (a1 - a0);
      place(c, depth + 1, cursor, cursor + share, p);
      cursor += share;
    }
  };

  // Start at -90° so the first collection sits at the top, where the eye lands.
  place(tree, 0, -Math.PI / 2, Math.PI * 1.5, null);
  return out;
}

/** Recompute x/y after `relaxAngles` has moved angles. `dist` is never touched. */
export function applyAngles(placed: Placed[]): void {
  for (const p of placed) {
    p.x = Math.cos(p.angle) * p.dist;
    p.y = Math.sin(p.angle) * p.dist;
  }
}
