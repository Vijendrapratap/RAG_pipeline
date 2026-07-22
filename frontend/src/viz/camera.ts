/**
 * The camera: layout coordinates ↔ CSS pixels.
 *
 * Split out of `draw.ts` because `draw.ts` touches `document` (it rasterises
 * glow sprites into an offscreen canvas) and therefore cannot be loaded by the
 * invariant checker, which runs in plain Node. These three functions are pure
 * arithmetic and carry a proof obligation — see `anchorPan` — so they belong
 * where the proof can be checked.
 */

export interface Viewport {
  /** Pan, in layout units. */
  x: number;
  y: number;
  scale: number;
  /** CSS pixels. */
  width: number;
  height: number;
}

/** Layout coords → CSS pixels. */
export function toScreen(p: { x: number; y: number }, v: Viewport): { x: number; y: number } {
  return {
    x: (p.x - v.x) * v.scale + v.width / 2,
    y: (p.y - v.y) * v.scale + v.height / 2,
  };
}

/** CSS pixels → layout coords. The inverse; used to pick under the cursor. */
export function toLayout(sx: number, sy: number, v: Viewport): { x: number; y: number } {
  return {
    x: (sx - v.width / 2) / v.scale + v.x,
    y: (sy - v.height / 2) / v.scale + v.y,
  };
}

/**
 * The pan that keeps one node under the pixel it already occupied, after a
 * relayout moved it from `before` to `after`.
 *
 * Spacing sliders and expanding a cluster both change ring radii, which moves
 * every node in layout space. Without this the picture lurches, and the reader
 * loses the node they were looking at — which is the only thing on screen they
 * were holding on to.
 *
 *     toScreen(after, anchorPan(v, before, after))
 *       = (after − v − after + before)·s + C
 *       = (before − v)·s + C
 *       = toScreen(before, v)                                              ∎
 *
 * Note what this deliberately does *not* do: it does not touch `scale`. A pure
 * `v ↦ k·v, scale ↦ scale/k` compensation reproduces the whole screen image
 * exactly, which would make the Spread slider a no-op with extra steps. The
 * reader asked for the dots to move apart; they should see them move apart,
 * around a fixed point.
 */
export function anchorPan(
  v: Viewport, before: { x: number; y: number }, after: { x: number; y: number },
): { x: number; y: number } {
  return { x: v.x + after.x - before.x, y: v.y + after.y - before.y };
}
