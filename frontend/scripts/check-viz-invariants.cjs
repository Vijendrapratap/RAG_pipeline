/**
 * Invariant checks for the layout, run against the compiled viz modules.
 *
 * There is no test runner in this project and adding one would mean adding a
 * dependency tree to a frontend whose whole point is that it has two. `tsc` is
 * already here, so: compile the pure modules (everything except draw.ts, which
 * touches `document`) into a temp dir and assert against them from plain Node.
 *
 *     npm run check:viz
 *
 * CommonJS, not ESM: `tsc` emits extensionless relative imports, which Node's
 * ESM resolver refuses. CJS `require` resolves them without a build step.
 *
 * The invariant that matters: **relaxation never moves a node radially.** A
 * node's distance from the centre encodes its depth in the archive. If a
 * relaxation pass could nudge it outward, a recording would drift into the ring
 * where camps live and the picture would quietly start lying — which is the same
 * class of bug as the position-based path parser that cost this project 52% of
 * its session dates.
 */
const assert = require("node:assert/strict");
const { buildTree, radialTree, rootNode, applyAngles } = require("./.viz-build/viz/radialTree.js");
const relax = require("./.viz-build/viz/relax.js");
const { relaxAngles } = relax;
const { SpatialGrid } = require("./.viz-build/viz/grid.js");
const { parentPath, ancestors } = require("./.viz-build/corpus.js");

let checks = 0;
function check(name, fn) {
  fn();
  checks++;
  console.log(`  ok  ${name}`);
}

function node(path, depth, isLeaf, files = 1, chunks = 3, remembered = 1) {
  return {
    path, name: path.split("/").pop(), depth, is_leaf: isLeaf,
    n_files: files, n_chunks: chunks, duration_sec: 60,
    remembered, written: files - remembered, failed: 0, session_date: null,
  };
}

/** A tree shaped like the real archive: Dagshai 4 folders deep, Live Masters 3. */
function corpus(nSessions = 400) {
  const out = [node("Dagshai 2002", 1, false, nSessions, nSessions * 3, nSessions)];
  out.push(node("Dagshai 2002/03 MAR - 2002", 2, false, nSessions, nSessions * 3, nSessions));
  for (let i = 0; i < nSessions; i++) {
    out.push(node(`Dagshai 2002/03 MAR - 2002/${i} MAR - 6 PM`, 3, false));
  }
  out.push(node("Live Masters 2010", 1, false, 5, 15, 5));
  out.push(node("Live Masters 2010/01 NOIDA", 2, false, 5, 15, 5));
  out.push(node("Live Masters 2010/01 NOIDA/7 JAN - 6 PM", 3, false, 5, 15, 5));
  out.push(node("Live Masters 2010/01 NOIDA/7 JAN - 6 PM/04 PRAVACHAN.json", 4, true));
  out.push(node("03 AA HIMMAT.json", 1, true)); // the degenerate key
  return out;
}

function layout(nodes) {
  const placed = radialTree(buildTree(rootNode(0, 0), nodes));
  return placed;
}

// ---------------------------------------------------------------------------

check("every node at a given depth shares one radius", () => {
  const placed = layout(corpus());
  const byDepth = new Map();
  for (const p of placed) {
    const seen = byDepth.get(p.node.depth);
    if (seen === undefined) byDepth.set(p.node.depth, p.dist);
    else assert.equal(p.dist, seen, `depth ${p.node.depth} has two radii`);
  }
  assert.ok(byDepth.size >= 4, "expected at least 4 distinct depths");
});

check("radius is strictly increasing with depth", () => {
  const placed = layout(corpus());
  const radii = [...new Map(placed.map((p) => [p.node.depth, p.dist])).entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, r]) => r);
  for (let i = 1; i < radii.length; i++) {
    assert.ok(radii[i] > radii[i - 1], `ring ${i} is not outside ring ${i - 1}`);
  }
});

check("relaxAngles never moves a node radially", () => {
  const placed = layout(corpus());
  const before = placed.map((p) => p.dist);
  relaxAngles(placed, { iterations: 40 });
  placed.forEach((p, i) => assert.equal(p.dist, before[i], `node ${i} moved radially`));
  // and x/y are still consistent with (dist, angle)
  for (const p of placed) {
    assert.ok(Math.abs(Math.hypot(p.x, p.y) - p.dist) < 1e-6, "x/y desynced from dist");
  }
});

check("nothing animates the layout — the map idles when nothing is happening", () => {
  // `drift()` used to rotate every ring every frame, which forced a full redraw
  // and a spatial-grid rebuild 60×/s forever, and told the operator the archive
  // was busy when it was asleep. Stillness is information; this keeps it.
  assert.equal(relax.drift, undefined, "relax.ts exports drift() again");
  assert.deepEqual(Object.keys(relax).sort(), ["relaxAngles"]);
});

check("a path's ancestors are its real prefixes, split only on the separator", () => {
  // `Live Masters 2010` is a prefix of `Live Masters 2010 B` as a *string*.
  // Focus dimming and subtree pruning both key on paths, so a naive
  // startsWith(focus) lights up a sibling collection and nobody notices.
  assert.equal(parentPath("a/b/c"), "a/b");
  assert.equal(parentPath("a"), "");
  assert.deepEqual(ancestors("Dagshai 2002/03 MAR/CAMP/1 PM/x.json"),
    ["Dagshai 2002", "Dagshai 2002/03 MAR", "Dagshai 2002/03 MAR/CAMP",
     "Dagshai 2002/03 MAR/CAMP/1 PM"]);
  assert.deepEqual(ancestors("x.json"), []);
  assert.ok(!"Live Masters 2010 B".startsWith("Live Masters 2010" + "/"));
});

check("a ring is wide enough to seat its own dots", () => {
  const placed = layout(corpus(400));
  const rings = new Map();
  for (const p of placed) {
    if (p.dist === 0) continue;
    (rings.get(p.dist) ?? rings.set(p.dist, []).get(p.dist)).push(p);
  }
  for (const [dist, ring] of rings) {
    const demand = ring.reduce((s, p) => s + 2 * p.r, 0);
    assert.ok(2 * Math.PI * dist >= demand,
      `ring at ${dist.toFixed(0)} must seat ${demand.toFixed(0)} of arc`);
  }
});

check("relaxation leaves same-ring dots non-overlapping", () => {
  const placed = layout(corpus(120));
  relaxAngles(placed, { iterations: 60 });
  const rings = new Map();
  for (const p of placed) {
    if (p.dist === 0) continue;
    (rings.get(p.dist) ?? rings.set(p.dist, []).get(p.dist)).push(p);
  }
  for (const ring of rings.values()) {
    ring.sort((a, b) => a.angle - b.angle);
    for (let i = 1; i < ring.length; i++) {
      const a = ring[i - 1], b = ring[i];
      const gap = Math.hypot(b.x - a.x, b.y - a.y);
      assert.ok(gap >= a.r + b.r - 0.5, `overlap of ${(a.r + b.r - gap).toFixed(2)}`);
    }
  }
});

check("buildTree attaches a child to its deepest present ancestor", () => {
  // The session's own parent (the month bucket) is absent from this fetch.
  const nodes = [
    node("Dagshai 2002", 1, false),
    node("Dagshai 2002/03 MAR - 2002/26 MAR - 7 PM", 3, false),
  ];
  const tree = buildTree(rootNode(0, 0), nodes);
  assert.equal(tree.children.length, 1, "session should not be orphaned to the root");
  assert.equal(tree.children[0].children.length, 1);
});

check("a folder name containing '$' survives the tree build", () => {
  const nodes = [node("Dagshai 2002", 1, false), node("Dagshai 2002/26 MAR - 1$ - 7 PM", 2, false)];
  const tree = buildTree(rootNode(0, 0), nodes);
  assert.equal(tree.children[0].children[0].node.name, "26 MAR - 1$ - 7 PM");
});

check("the spatial grid finds a node the naive scan finds", () => {
  const placed = layout(corpus(200));
  relaxAngles(placed);
  const grid = new SpatialGrid();
  grid.rebuild(placed);
  let agree = 0;
  for (const target of placed.slice(0, 60)) {
    const hit = grid.pick(target.x, target.y, 0);
    assert.ok(hit, `grid missed a node sitting exactly on a dot centre`);
    // Ties go to the smaller dot, so assert containment, not identity.
    assert.ok(Math.hypot(hit.x - target.x, hit.y - target.y) <= hit.r + 1e-6);
    agree++;
  }
  assert.equal(agree, 60);
});

check("the grid finds nothing in empty space", () => {
  const placed = layout(corpus(50));
  const grid = new SpatialGrid();
  grid.rebuild(placed);
  assert.equal(grid.pick(99999, 99999, 4), null);
});

check("applyAngles keeps x/y on the ring", () => {
  const placed = layout(corpus(30));
  for (const p of placed) p.angle += 1.234;
  applyAngles(placed);
  for (const p of placed) {
    assert.ok(Math.abs(Math.hypot(p.x, p.y) - p.dist) < 1e-6);
  }
});

console.log(`\n${checks} invariant checks passed.`);
