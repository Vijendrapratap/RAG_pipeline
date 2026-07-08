/**
 * Per-frame CPU budget for the archive map, measured — not eyeballed.
 *
 * The canvas paint needs a browser, so it is not covered here. What *is* covered
 * is everything the render loop does before it paints: `drift()` mutates every
 * angle and `SpatialGrid.rebuild()` re-buckets every node, once per frame. Those
 * are pure and measurable in Node, and they are where a careless implementation
 * eats the 16 ms frame before a single pixel is drawn.
 *
 *     npm run bench:viz
 */
const { buildTree, radialTree, rootNode } = require("./.viz-build/viz/radialTree.js");
const { relaxAngles, drift } = require("./.viz-build/viz/relax.js");
const { SpatialGrid } = require("./.viz-build/viz/grid.js");

function node(path, depth, isLeaf) {
  return {
    path, name: path.split("/").pop(), depth, is_leaf: isLeaf,
    n_files: 4, n_chunks: 12, duration_sec: 600,
    remembered: 3, written: 1, failed: 0, session_date: null,
  };
}

/** Approximates the real shape: 14 collections, 177 groups, then sittings. */
function corpus(nSittings) {
  const out = [];
  const collections = 14, groups = 177;
  for (let c = 0; c < collections; c++) out.push(node(`C${c}`, 1, false));
  for (let g = 0; g < groups; g++) out.push(node(`C${g % collections}/G${g}`, 2, false));
  for (let s = 0; s < nSittings; s++) {
    const g = s % groups;
    out.push(node(`C${g % collections}/G${g}/S${s}`, 3, false));
  }
  return out;
}

function bench(label, n, fn, iters) {
  fn(); // warm
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn();
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / iters;
  const verdict = ms < 16 ? "ok " : "SLOW";
  console.log(`  ${verdict} ${label.padEnd(34)} ${n.toString().padStart(6)} nodes  ${ms.toFixed(3)} ms`);
  return ms;
}

for (const nSittings of [1748, 9335, 27806]) {
  const nodes = corpus(nSittings);
  const placed = radialTree(buildTree(rootNode(0, 0), nodes));
  const grid = new SpatialGrid();
  const n = placed.length;

  console.log(`\n${nSittings.toLocaleString()} sittings → ${n.toLocaleString()} placed nodes`);
  bench("layout (radialTree)", n, () => radialTree(buildTree(rootNode(0, 0), nodes)), 5);
  bench("relaxAngles (24 passes)", n, () => relaxAngles(placed), 5);

  // The per-frame pair. This is the number that must stay under 16 ms.
  let t = 0;
  const perFrame = bench("PER FRAME: drift + grid rebuild", n, () => {
    drift(placed, (t += 0.016));
    grid.rebuild(placed);
  }, 60);

  grid.rebuild(placed);
  bench("hover pick (×1000)", n, () => {
    for (let i = 0; i < 1000; i++) grid.pick(placed[i % n].x, placed[i % n].y, 5);
  }, 5);

  if (perFrame >= 16) {
    console.log(`\n  Per-frame work exceeds the 16 ms budget at ${n} nodes.`);
    process.exitCode = 1;
  }
}
console.log();
