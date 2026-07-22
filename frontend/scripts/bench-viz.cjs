/**
 * CPU budget for the archive map, measured — not eyeballed.
 *
 * The canvas paint needs a browser, so it is not covered here. What *is* covered
 * is the pure work the map does around it, and the budgets differ by an order of
 * magnitude because the paths do:
 *
 *   - **Per click** — visibleNodes → buildTree → radialTree → applyDisplacements →
 *     relaxAngles → grid.rebuild. This runs once, when you open a cluster, so it
 *     may spend 100 ms. It used to run on every `selectedPath` change too, which
 *     is 24 relaxation passes and a full grid rebuild to update one `.find()`.
 *   - **Per pointermove, while dragging a node** — translateSubtree + grid.rebuild.
 *     This is the one 60 Hz path in the system. 16 ms, and it should not come
 *     close. It got cheaper when the drag stopped being angular: a translation is
 *     two adds per node, where a rotation was two trig calls.
 *   - **Per idle frame** — nothing. `drift()` used to rotate every ring every
 *     frame; it was deleted, and this file used to call it and throw.
 *
 *     npm run bench:viz
 */
const {
  applyDisplacements, buildTree, radialTree, rootNode, subtreeOf, translateSubtree,
} = require("./.viz-build/viz/radialTree.js");
const { relaxAngles } = require("./.viz-build/viz/relax.js");
const { SpatialGrid } = require("./.viz-build/viz/grid.js");
const { spineOf, visibleNodes } = require("./.viz-build/corpus.js");

/** Roll tracks up into their ancestors, exactly as `rag_api.corpus.build_nodes` does. */
function corpusFrom(trackPaths) {
  const acc = new Map();
  for (const track of trackPaths) {
    const parts = track.split("/");
    for (let i = 1; i <= parts.length; i++) {
      const path = parts.slice(0, i).join("/");
      let e = acc.get(path);
      if (!e) {
        e = {
          path, name: parts[i - 1], depth: i, is_leaf: i === parts.length,
          n_files: 0, n_chunks: 0, duration_sec: 0,
          remembered: 0, written: 0, failed: 0, session_date: null,
        };
        acc.set(path, e);
      }
      e.n_files += 1;
      e.n_chunks += 3;
      e.duration_sec += 600;
      e.remembered += 1;
    }
  }
  return [...acc.values()];
}

/**
 * Approximates the real shape: 17 collections, 191 groups, then sittings — with
 * one *fat* group holding a tenth of the archive, the way
 * `Dagshai 2002/03 MAR - 2002` holds 1,748 sittings. That fat group is the worst
 * case the collapsed map can ever be asked to draw, and it is the only reason
 * the per-click budget is interesting at all.
 */
function corpus(nSittings) {
  const collections = 17, groups = 191;
  const fat = Math.round(nSittings * 0.1);
  const tracks = [];
  for (let s = 0; s < fat; s++) tracks.push(`C0/G0/S${s}/01 PRAVACHAN.json`);
  for (let s = fat; s < nSittings; s++) {
    const g = s % groups;
    tracks.push(`C${g % collections}/G${g}/S${s}/01 PRAVACHAN.json`);
  }
  return corpusFrom(tracks);
}

function bench(label, n, budget, fn, iters) {
  fn(); // warm
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn();
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / iters;
  const ok = ms < budget;
  console.log(
    `  ${ok ? "ok  " : "SLOW"} ${label.padEnd(40)} ${n.toString().padStart(6)} nodes  ` +
    `${ms.toFixed(3)} ms  (budget ${budget})`,
  );
  if (!ok) process.exitCode = 1;
  return ms;
}

for (const nSittings of [1748, 9335, 27806]) {
  const nodes = corpus(nSittings);
  const root = rootNode(0, 0);
  const grid = new SpatialGrid();

  // What the reader can actually put on screen at once: the fat group, opened.
  const open = spineOf("C0/G0");
  const visible = visibleNodes(nodes, open);
  const n = visible.length + 1; // + the synthetic root
  const moves = new Map(visible.slice(0, 40).map((v, i) => [v.path, { x: i * 40, y: -i * 25 }]));

  console.log(
    `\n${nSittings.toLocaleString()} sittings → ${nodes.length.toLocaleString()} nodes, ` +
    `${n.toLocaleString()} drawn with the largest group open`,
  );

  bench("layout (radialTree, every node)", nodes.length + 1, 200,
    () => radialTree(buildTree(root, nodes)), 5);

  bench("PER CLICK: visible→tree→layout→relax→grid", n, 100, () => {
    const v = visibleNodes(nodes, open);
    const placed = radialTree(buildTree(root, v));
    applyDisplacements(placed, moves);
    relaxAngles(placed);
    grid.rebuild(placed);
  }, 5);

  // The one 60 Hz path. A drag translates a precomputed subtree and re-buckets the
  // grid; it must never re-run `radialTree` or 24 relaxation passes per move.
  const placed = radialTree(buildTree(root, visible));
  relaxAngles(placed);
  const dragged = placed.find((p) => p.node.path === "C0/G0");
  const sub = subtreeOf(dragged, placed);
  let t = 0;
  bench("PER POINTERMOVE: translateSubtree + grid", n, 16, () => {
    translateSubtree(sub, (t = -t) + 1, 0);
    grid.rebuild(placed);
  }, 60);

  grid.rebuild(placed);
  bench("hover pick (×1000)", n, 16, () => {
    for (let i = 0; i < 1000; i++) grid.pick(placed[i % n].x, placed[i % n].y, 5);
  }, 5);
}

console.log("\n  PER IDLE FRAME: zero work — nothing mutates the layout when nothing happens.\n");
