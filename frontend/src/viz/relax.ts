/**
 * Angular relaxation. Siblings on the same ring push each other apart until
 * their dots stop overlapping.
 *
 * The one invariant, enforced by construction and asserted in the tests:
 * **`dist` never changes.** A node's distance from the centre encodes its depth
 * in the archive, so a relaxation that moved nodes radially would make the
 * picture lie — a track would drift into the ring where camps live. Only
 * `angle` moves, and only within the wedge its parent gave it.
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
  /** Extra angular breathing room between dot edges, in layout units. */
  padding?: number;
  /** 0..1 — how much of each overlap is corrected per pass. */
  strength?: number;
}

/**
 * Push overlapping same-ring siblings apart, in place.
 *
 * Only nodes on the same ring can collide (different rings are `ringGap` apart
 * radially), so each ring relaxes independently. Within a ring the nodes are
 * sorted by angle and only *adjacent* pairs are compared — an O(n log n) pass
 * rather than the O(n²) every-pair check, because a non-adjacent pair cannot
 * overlap without its neighbours overlapping first.
 */
export function relaxAngles(placed: Placed[], opts: RelaxOptions = {}): void {
  const iterations = opts.iterations ?? 24;
  const padding = opts.padding ?? 2.5;
  const strength = opts.strength ?? 0.5;

  const rings = new Map<number, Placed[]>();
  for (const p of placed) {
    if (p.dist === 0) continue; // the root has no siblings to shove
    const ring = rings.get(p.dist);
    if (ring) ring.push(p);
    else rings.set(p.dist, [p]);
  }

  for (const [dist, ring] of rings) {
    if (ring.length < 2) continue;
    // An arc length converts to an angle by dividing by the radius. On the
    // inner rings that makes every gap enormous; on the outer rings, tiny.
    const toAngle = 1 / dist;

    for (let pass = 0; pass < iterations; pass++) {
      ring.sort((a, b) => a.angle - b.angle);
      let moved = false;

      for (let i = 0; i < ring.length; i++) {
        const a = ring[i];
        const b = ring[(i + 1) % ring.length];
        if (a === b) continue;

        const wantArc = a.r + b.r + padding;
        const gap = wrap(b.angle - a.angle);
        // A negative gap here means the wrap-around pair (last vs first) is
        // already more than half the circle apart — they cannot collide.
        if (gap < 0 || gap >= wantArc * toAngle) continue;

        const push = (wantArc * toAngle - gap) * strength * 0.5;
        a.angle -= push;
        b.angle += push;
        moved = true;
      }
      if (!moved) break; // converged; the remaining passes would be no-ops
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
// The only motion left is the camera easing toward a cluster you clicked, which
// is a response to an action, and which stops.
