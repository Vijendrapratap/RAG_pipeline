/**
 * Radial hierarchy layout. Radius is tree depth; angle comes from cumulative
 * content so siblings get arc proportional to what they contain.
 *
 * Why not a force simulation, which is what Obsidian's graph does: Obsidian's
 * vault is a sparse *web* of links over ~1,000 notes, and the force layout
 * discovers structure nobody entered by hand. This archive is a strict *tree* —
 * every edge is containment, and we already know the whole thing. Simulating it
 * would throw away the only relationship we know for certain and hand back a
 * hairball. So depth stays exact and only the angle relaxes (see relax.ts).
 *
 * The consequence: a node's distance from the centre always means the same
 * thing — unless the reader picks it up and puts it somewhere else. The law is
 * not "radius is depth"; it is **the system never moves a node radially, only
 * the reader does**. `angle` and `dist` are what the layout chose and no drag
 * ever touches them; `x`/`y` are where the node is drawn, which is the layout's
 * choice plus whatever the reader has since said. See `applyDisplacements`.
 */
import type { CorpusNode } from "../corpus";
import { MAX_NODE_RADIUS, nodeRadius } from "../corpus";

const TAU = Math.PI * 2;

/** Two rings this far apart cannot have touching dots, whatever they hold. */
const MIN_RING_GAP = 2 * MAX_NODE_RADIUS;

/**
 * How far from the origin a node the reader has moved may sit.
 *
 * Derived, not chosen. `SpatialGrid.key()` packs a cell index (cell = 48) into
 * `(c + 32768) * 65536`, so a coordinate must stay inside ±32768·48 = 1,572,864
 * or two distant nodes silently share a bucket and hover picks the wrong one.
 * The widest layout this archive can produce — all 55,821 nodes with Spread at
 * its maximum — reaches 139,566. This leaves an order of magnitude of headroom,
 * and it cannot be reached by hand: at `MIN_SCALE` it is 3,000 px of dragging.
 *
 * It exists mostly to survive a `localStorage` entry that says `1e300`.
 */
export const MAX_DRAG_COORD = 50_000;

/**
 * Where the reader is allowed to put a node.
 *
 * `±Infinity` clamps to the bound it is heading for — it has a direction, and
 * honouring it is what the reader meant. Only `NaN` and non-numbers have nowhere
 * to go, and they survive the clamp untouched, which is how they are caught.
 */
export function clampTarget(x: number, y: number): { x: number; y: number } {
  const c = (v: number) => {
    const n = Math.max(-MAX_DRAG_COORD, Math.min(MAX_DRAG_COORD, Number(v)));
    return Number.isFinite(n) ? n : 0;
  };
  return { x: c(x), y: c(y) };
}

export interface TreeNode {
  node: CorpusNode;
  children: TreeNode[];
}

export interface Placed {
  node: CorpusNode;
  /** Where the dot is drawn: the layout's choice plus the reader's displacement. */
  x: number;
  y: number;
  /** Drawn radius of the dot, in layout units. */
  r: number;
  /** Angle in radians — `relaxAngles` mutates this, then x/y are recomputed. */
  angle: number;
  /** Distance from origin. Set once from depth. Never mutated afterwards. */
  dist: number;
  /** The arc this node's parent handed it. */
  a0: number;
  a1: number;
  /** The reader's displacement, added at render time. Zero until something is dragged. */
  ox: number;
  oy: number;
  /**
   * Path of the nearest displaced ancestor-or-self; `""` if nothing above it moved.
   *
   * Two nodes share a frame exactly when they share a displacement, and a rigid
   * translation is an isometry — so every chord *inside* a frame is what the
   * layout laid down. That is what lets `relaxAngles` keep doing chord geometry
   * on a subtree the reader has flung across the disc.
   */
  frame: string;
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
 * find their session — and when a fetch stops above the tracks, nothing should
 * be orphaned into the root.
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

/**
 * A subtree's angular weight: how much arc it is entitled to.
 *
 * This used to be `leafCount`, which gave a childless `TreeNode` a weight of 1.
 * Once nodes could be *collapsed*, that was wrong in a way that would have been
 * blamed on the physics: a collapsed `Dagshai 2002` holding 1,748 sittings is
 * childless, so it would claim the same sliver of the disc as an empty folder,
 * and opening it would re-angle every other collection. The map would shuffle on
 * every click.
 *
 * A collapsed folder therefore claims arc for what it *holds*. Because the
 * backend rolls each track into every ancestor (`build_nodes`), a container's
 * `n_files` equals the sum of its children's — so `weight(collapsed n)` equals
 * `weight(expanded n)`, and expanding a node moves nothing outside its own
 * subtree. Call it wedge conservation. It is what makes the map feel solid.
 */
function weight(t: TreeNode, memo: Map<TreeNode, number>): number {
  const hit = memo.get(t);
  if (hit !== undefined) return hit;
  let n = 0;
  if (t.children.length === 0) n = Math.max(1, t.node.n_files);
  else for (const c of t.children) n += weight(c, memo);
  memo.set(t, n);
  return n;
}

export interface LayoutOptions {
  /** Smallest allowed distance between consecutive rings. Floored at 2·MAX_NODE_RADIUS. */
  minRingGap?: number;
  /** Radius of the innermost ring. */
  innerRadius?: number;
  /** Arc padding between adjacent dots, in layout units. Floored at 0. */
  padding?: number;
}

/**
 * The radius ring `d` needs to seat its own contents, or `lower`, whichever is
 * larger.
 *
 * A fixed gap per depth is the obvious choice and it is wrong here. Depth 3 can
 * hold 1,748 sittings; on a fixed-gap ring of radius 332 each gets 1.2 units of
 * arc while its dot needs about 7, so the ring is a solid smear and no amount of
 * angular relaxation can fix it — the space simply is not there.
 *
 * The seating condition is stated in *chords*, not arcs. Two dots θ apart on a
 * ring of radius R are `2R·sin(θ/2)` apart, which is strictly less than `Rθ`;
 * the arc form `2πR ≥ Σ(2ρ + pad)` therefore over-promises, and on a tight inner
 * ring it over-promises by more than a dot's width. Each node claims a half-width
 * `h = asin((ρ + pad/2) / R)`, and `Σ2h ≤ 2π` is what we solve for. Since `asin`
 * is convex, `2·asin((ρa' + ρb')/2R) ≤ asin(ρa'/R) + asin(ρb'/R)`, so satisfying
 * the per-node budget guarantees every *pair* fits too — which is exactly the
 * separation `relaxAngles` tries to reach.
 *
 * Radius still encodes depth exactly. It just is not a *linear* encoding of it.
 */
function seatRadius(ring: TreeNode[], padding: number, lower: number): number {
  let arcDemand = 0;
  for (const t of ring) arcDemand += 2 * nodeRadius(t.node) + padding;
  let r = Math.max(lower, arcDemand / TAU, 1);

  // Monotone: `need` falls as r grows, so scaling by need/2π never overshoots
  // past the fixed point and a handful of passes lands within a float of it.
  for (let i = 0; i < 40; i++) {
    let need = 0;
    for (const t of ring) need += 2 * Math.asin(Math.min(1, (nodeRadius(t.node) + padding / 2) / r));
    if (need <= TAU) break;
    r *= Math.max(1.02, need / TAU);
  }
  return r;
}

function ringRadii(
  byDepth: Map<number, TreeNode[]>, innerRadius: number, minRingGap: number, padding: number,
): Map<number, number> {
  const radii = new Map<number, number>([[0, 0]]);
  const maxDepth = Math.max(0, ...byDepth.keys());
  let prev = 0;
  for (let d = 1; d <= maxDepth; d++) {
    const lower = Math.max(prev + minRingGap, d === 1 ? innerRadius : 0);
    const r = seatRadius(byDepth.get(d) ?? [], padding, lower);
    radii.set(d, r);
    prev = r;
  }
  return radii;
}

/**
 * Place every node. Each subtree receives an angular wedge proportional to what
 * it holds, and its children divide that wedge the same way, recursively.
 */
export function radialTree(tree: TreeNode, opts: LayoutOptions = {}): Placed[] {
  // Floors live here, not at the call site, so the invariant checker can prove
  // them against every caller — including a slider someone drags to zero.
  const minRingGap = Math.max(MIN_RING_GAP, opts.minRingGap ?? 130);
  const innerRadius = Math.max(0, opts.innerRadius ?? 150);
  const padding = Math.max(0, opts.padding ?? 2.5);

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
      angle, dist, a0, a1, ox: 0, oy: 0, frame: "", parent,
    };
    out.push(p);

    const total = weight(t, memo);
    let cursor = a0;
    for (const c of t.children) {
      const share = (weight(c, memo) / total) * (a1 - a0);
      place(c, depth + 1, cursor, cursor + share, p);
      cursor += share;
    }
  };

  // Start at -90° so the first collection sits at the top, where the eye lands.
  place(tree, 0, -Math.PI / 2, Math.PI * 1.5, null);
  return out;
}

/**
 * Recompute x/y after `relaxAngles` has moved angles.
 *
 * `dist` and `angle` are the layout; `ox`/`oy` are the reader. The sum is the
 * only place the two ever meet, and it is here.
 */
export function applyAngles(placed: Placed[]): void {
  for (const p of placed) {
    p.x = Math.cos(p.angle) * p.dist + p.ox;
    p.y = Math.sin(p.angle) * p.dist + p.oy;
  }
}

// ---------------------------------------------------------------------------
// Arranging: the reader picks a node up and puts it anywhere.
//
// A drag is not a second layout engine. It is a rigid translation of a subtree,
// applied *after* the layout has run, and recorded in `ox`/`oy`. The layout's
// own `angle` and `dist` are never touched — which is why relaxation, the
// spacing sliders and wedge conservation all keep working on a branch the reader
// has dragged across the disc.
// ---------------------------------------------------------------------------

function childIndex(placed: Placed[]): Map<Placed, Placed[]> {
  const kids = new Map<Placed, Placed[]>();
  for (const p of placed) {
    if (!p.parent) continue;
    const at = kids.get(p.parent);
    if (at) at.push(p);
    else kids.set(p.parent, [p]);
  }
  return kids;
}

function collect(root: Placed, kids: Map<Placed, Placed[]>): Placed[] {
  const out: Placed[] = [];
  const stack: Placed[] = [root];
  while (stack.length > 0) {
    const p = stack.pop()!;
    out.push(p);
    const cs = kids.get(p);
    if (cs) for (const c of cs) stack.push(c);
  }
  return out;
}

/** Every `Placed` at or under `root`, `root` first. Build once per drag, not per move. */
export function subtreeOf(root: Placed, placed: Placed[]): Placed[] {
  return collect(root, childIndex(placed));
}

/**
 * Move a precomputed subtree by one vector, in place.
 *
 * A container carries its contents: moving a collection without its groups would
 * fan its links across the disc and say something false about where its
 * recordings live.
 *
 * `angle` and `dist` are never touched. A rigid translation preserves every chord
 * *inside* the subtree, which is exactly why `relaxAngles` can still separate its
 * rings once the subtree has landed somewhere else.
 */
export function translateSubtree(subtree: Placed[], dx: number, dy: number): void {
  if (dx === 0 && dy === 0) return;
  for (const p of subtree) {
    p.ox += dx;
    p.oy += dy;
    p.x += dx;
    p.y += dy;
  }
}

/**
 * Re-apply the reader's arrangement to a freshly-laid-out tree.
 *
 * Targets are stored **absolute**, not as deltas: a delta drifts as the layout
 * changes underneath it, while an absolute point is what the reader chose. So a
 * moved node holds its pixel while the Spread slider grows the disc around it.
 *
 * Shallowest first. A moved collection carries its groups before a group's own
 * target is resolved against the position it has just been given — and a deeper
 * target then overwrites its own subtree's `frame`, leaving every node tagged
 * with its *nearest* displaced ancestor-or-self.
 *
 * The root is skipped (`dist > 0`): it has no position to remember, and moving
 * everything is a pan, which the background drag already does.
 */
export function applyDisplacements(
  placed: Placed[], moves: ReadonlyMap<string, { x: number; y: number }>,
): void {
  // Wiped first, and `applyAngles` runs unconditionally at the end. Forgetting a
  // move is how the reader puts a node back, so an empty ledger must actively
  // re-render the layout's own position, not skip out and leave the old one.
  for (const p of placed) { p.ox = 0; p.oy = 0; p.frame = ""; }

  const byPath = new Map<string, Placed>();
  for (const p of placed) byPath.set(p.node.path, p);

  const targets: Placed[] = [];
  for (const path of moves.keys()) {
    const p = byPath.get(path);
    if (p && p.dist > 0) targets.push(p);
  }
  targets.sort((a, b) => a.node.depth - b.node.depth);

  if (targets.length > 0) {
    const kids = childIndex(placed);
    for (const p of targets) {
      const m = moves.get(p.node.path)!;
      const want = clampTarget(m.x, m.y);
      // `p.ox` already carries whatever a displaced ancestor did to it.
      const dx = want.x - (Math.cos(p.angle) * p.dist + p.ox);
      const dy = want.y - (Math.sin(p.angle) * p.dist + p.oy);
      for (const q of collect(p, kids)) {
        q.ox += dx;
        q.oy += dy;
        q.frame = p.node.path;
      }
    }
  }
  applyAngles(placed);
}
