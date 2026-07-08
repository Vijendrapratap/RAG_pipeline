/**
 * Uniform spatial hash for hover picking.
 *
 * The alternative is scanning all 2,000 nodes on every `mousemove`, which is
 * fine until it isn't — mousemove fires far more often than a frame, and at
 * 27,806 nodes it is 27,806 distance checks per event. A grid turns that into
 * "look in the 9 cells around the cursor", which is O(1) whatever the corpus
 * grows to.
 *
 * A quadtree would also work and is what a library would give you. At this
 * density (nodes are roughly evenly spread over the rings) a flat grid is both
 * faster and forty lines shorter.
 */
import type { Placed } from "./radialTree";

export class SpatialGrid {
  private readonly cell: number;
  private readonly buckets = new Map<number, Placed[]>();

  /** `cell` should be ~2× the largest node radius, so a hit is always within
   *  the 3×3 neighbourhood of the cursor's own cell. */
  constructor(cell = 48) {
    this.cell = cell;
  }

  /** Cantor-ish pairing into a single integer key. Cheaper than a string. */
  private key(cx: number, cy: number): number {
    // 2^16 rows; layout coords stay well inside ±32k / cell.
    return (cx + 32768) * 65536 + (cy + 32768);
  }

  rebuild(placed: Placed[]): void {
    this.buckets.clear();
    for (const p of placed) {
      const k = this.key(Math.floor(p.x / this.cell), Math.floor(p.y / this.cell));
      const b = this.buckets.get(k);
      if (b) b.push(p);
      else this.buckets.set(k, [p]);
    }
  }

  /**
   * The node under (x, y) in layout coordinates, or null.
   *
   * `slop` widens every node's hit area — a 2.2-unit dot is impossible to hit
   * with a mouse, and the outermost ring is all 2.2-unit dots. When two nodes
   * both contain the point, the *smaller* wins: a big collection would
   * otherwise swallow the tracks drawn on top of it.
   */
  pick(x: number, y: number, slop = 4): Placed | null {
    const cx = Math.floor(x / this.cell);
    const cy = Math.floor(y / this.cell);
    let best: Placed | null = null;
    let bestR = Infinity;

    // The 3×3 neighbourhood is only sufficient while `slop` stays inside one
    // cell. Zoomed all the way out, slop is ~80 layout units — a hit two cells
    // away is real, and searching 3×3 would silently miss it.
    const span = 1 + Math.ceil(slop / this.cell);

    for (let ix = cx - span; ix <= cx + span; ix++) {
      for (let iy = cy - span; iy <= cy + span; iy++) {
        const bucket = this.buckets.get(this.key(ix, iy));
        if (!bucket) continue;
        for (const p of bucket) {
          const dx = p.x - x;
          const dy = p.y - y;
          const reach = p.r + slop;
          if (dx * dx + dy * dy <= reach * reach && p.r < bestR) {
            best = p;
            bestR = p.r;
          }
        }
      }
    }
    return best;
  }
}
