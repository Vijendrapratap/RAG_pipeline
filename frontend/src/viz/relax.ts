/**
 * Angular relaxation. Siblings on the same ring push each other apart until
 * their dots stop overlapping.
 *
 * The one invariant, enforced by construction and asserted in the tests:
 * **`dist` never changes.** A node's distance from the centre is what the layout
 * said its depth was worth, so a relaxation that moved nodes radially would make
 * the picture lie — a track would drift into the ring where camps live. Only
 * `angle` moves.
 *
 * Since the reader can drag a subtree anywhere, `dist` no longer *implies* a
 * node's distance from the centre on screen. It still records what the layout
 * chose, and this file still refuses to touch it: the system never moves a node
 * radially, only the reader does.
 *
 * This is what buys the organic, breathing look without a force simulation:
 * enough give that the ring is not a mechanical clock face, no give at all in
 * the dimension that carries meaning.
 */
import type { Placed } from "./radialTree";
import { applyAngles } from "./radialTree";

const TAU = Math.PI * 2;

/** Wrap into (-π, π]. Two nodes at 179° and -179° are 2° apart, not 358°. */
function wrap(a: number): number {
  let x = a % TAU;
  if (x > Math.PI) x -= TAU;
  if (x <= -Math.PI) x += TAU;
  return x;
}

export interface RelaxOptions {
  iterations?: number;
  /** Extra breathing room between dot edges, in layout units. Floored at 0. */
  padding?: number;
  /** 0..1 — how much of each overlap is corrected per pass. */
  strength?: number;
}

/**
 * Push overlapping same-ring siblings apart, in place.
 *
 * Only nodes on the same ring can collide (adjacent rings are at least
 * `2 * MAX_NODE_RADIUS` apart, so their dots cannot reach each other), and each
 * ring relaxes independently. Within a ring the nodes are sorted by angle and
 * only *adjacent* pairs are compared — an O(n log n) pass rather than the O(n²)
 * every-pair check, because a non-adjacent pair cannot overlap without its
 * neighbours overlapping first.
 *
 * The separation two dots need is a **chord**, not an arc. `s = 2R·sin(θ/2)`, so
 * `θ = 2·asin(s / 2R)`. The obvious `θ = s / R` is the small-angle approximation
 * of that, and it under-separates: at R = 60 with 22-unit dots it leaves a real
 * overlap of ~1 unit. At the shipped R = 150 the error is 0.16, which is why the
 * invariant check used to carry a `- 0.5` tolerance. The `asin` is exact, so the
 * tolerance is now `1e-6` and the spacing sliders are safe at every setting.
 */
export function relaxAngles(placed: Placed[], opts: RelaxOptions = {}): void {
  const iterations = opts.iterations ?? 24;
  const padding = Math.max(0, opts.padding ?? 2.5);
  const strength = Math.min(1, Math.max(0, opts.strength ?? 0.5));

  // Bucketed by `(depth, frame)`, not by the float `dist`. Same rings, said in
  // the language the layout actually uses — and no float-equality dependency to
  // break when a spacing slider lands two rings on radii that differ in the last
  // bit.
  //
  // `frame` splits each ring by displacement. Chord geometry is only meaningful
  // between two nodes the reader has moved by the *same* vector, because a rigid
  // translation is the only transform that leaves the chords inside it intact.
  // A dragged node is therefore alone in its bucket — its own frame, at its own
  // depth — and the `< 2` guard below is what makes relaxation leave it exactly
  // where it was put. That is why `Placed` no longer needs a `fixed` flag.
  const rings = new Map<string, Placed[]>();
  for (const p of placed) {
    if (p.dist === 0) continue; // the root has no siblings to shove
    const key = `${p.node.depth} ${p.frame}`;
    const ring = rings.get(key);
    if (ring) ring.push(p);
    else rings.set(key, [p]);
  }

  for (const ring of rings.values()) {
    if (ring.length < 2) continue;
    const dist = ring[0].dist;

    for (let pass = 0; pass < iterations; pass++) {
      // Sorted on the *wrapped* angle, which is the circular order. `place()`
      // hands out angles in [-π/2, 3π/2], so the raw sort happened to be circular
      // — until these very passes shoved an angle across the -π/2 seam. Then the
      // raw sort compares nodes that are not neighbours, and two dots overlap
      // with nothing to push them apart.
      ring.sort((a, b) => wrap(a.angle) - wrap(b.angle));
      let moved = false;

      for (let i = 0; i < ring.length; i++) {
        const a = ring[i];
        const b = ring[(i + 1) % ring.length];
        if (a === b) continue;

        const want = 2 * Math.asin(Math.min(1, (a.r + b.r + padding) / (2 * dist)));
        const gap = wrap(b.angle - a.angle);
        // A negative gap here means the wrap-around pair (last vs first) is
        // already more than half the circle apart — they cannot collide.
        if (gap < 0 || gap >= want) continue;

        const push = (want - gap) * strength;
        a.angle -= push * 0.5;
        b.angle += push * 0.5;
        moved = true;
      }
      // Converged: every adjacent pair on this ring already clears its chord, so
      // the ring is provably non-overlapping and the remaining passes are no-ops.
      if (!moved) break;
    }
  }

  applyAngles(placed);
}

// There was a `drift()` here: a slow per-ring rotation, running every frame.
// It was decoration, and it cost more than it looked like it did — it forced a
// full canvas redraw and a spatial-grid rebuild on all 1,947 nodes 60 times a
// second, forever, so the `dirtyRef` check in BrainView could never short-
// circuit and the map never idled.
//
// It also contradicted the one thing this view is supposed to say: **stillness
// is information**. A glance should tell you whether anything is happening. A
// map that is always moving cannot answer that, and a permanently-spinning disc
// invites the reading that the archive is doing something. It isn't.
//
// The only motion left is the camera easing toward a cluster you clicked, and a
// node following your finger. Both are responses to an action, and both stop.
