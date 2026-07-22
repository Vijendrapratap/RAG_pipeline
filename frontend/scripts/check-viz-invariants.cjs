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
 * Three invariants carry the picture, and every check below serves one of them:
 *
 *   1. **The system never moves a node radially. Only the reader does.** Neither
 *      relaxation, nor a spacing slider, nor a drag may change `dist` or `angle`
 *      — they are what the layout said the node's depth was worth. A reader may
 *      pick a node up and put it anywhere; that displacement lives in `ox`/`oy`
 *      and is added at render time. Nothing else may move a dot, or a recording
 *      drifts into the ring where camps live and the picture quietly starts
 *      lying. That is the same class of bug as the position-based path parser
 *      that cost this project 52% of its session dates.
 *   2. **Wedge conservation.** Expanding a node changes no angle outside its own
 *      subtree, because a collapsed folder claims arc for what it holds.
 *      Without it the map reshuffles on every click.
 *   3. **The camera compensation is exact.** A relayout may move a node in
 *      layout space; it must not move it on screen.
 */
const assert = require("node:assert/strict");
const {
  MAX_DRAG_COORD, applyAngles, applyDisplacements, buildTree, clampTarget, radialTree,
  rootNode, subtreeOf, translateSubtree,
} = require("./.viz-build/viz/radialTree.js");
const relax = require("./.viz-build/viz/relax.js");
const { relaxAngles } = relax;
const { SpatialGrid } = require("./.viz-build/viz/grid.js");
const { anchorPan, toScreen } = require("./.viz-build/viz/camera.js");
const {
  MAX_NODE_RADIUS, PREFETCH_DEPTH, REVEAL_DEPTH, ancestors, childrenUnfetched,
  parentPath, spineOf, visibleNodes,
} = require("./.viz-build/corpus.js");

const TAU = Math.PI * 2;

let checks = 0;
function check(name, fn) {
  fn();
  checks++;
  console.log(`  ok  ${name}`);
}

/**
 * Build the node list the backend would serve for a set of track paths, rolling
 * every track up into every ancestor exactly the way `rag_api.corpus.build_nodes`
 * does. Hand-writing the counts is how the old fixture ended up with a folder
 * whose `n_files` was 5 and whose only child's was 1 — a shape the backend cannot
 * produce, and one that would fail wedge conservation for a reason nobody would
 * ever find.
 */
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
      e.duration_sec += 60;
      e.remembered += 1;
    }
  }
  return [...acc.values()];
}

/** A tree shaped like the real archive: Dagshai 4 folders deep, Live Masters 3. */
function corpus(nSessions = 400) {
  const tracks = [];
  for (let i = 0; i < nSessions; i++) {
    tracks.push(`Dagshai 2002/03 MAR - 2002/${i} MAR - 6 PM/01 PRAVACHAN.json`);
  }
  for (let i = 0; i < 5; i++) {
    tracks.push(`Live Masters 2010/01 NOIDA/7 JAN - 6 PM/0${i} PRAVACHAN.json`);
  }
  // `Live Masters 2010` is a string prefix of this. Nothing may treat it as one.
  tracks.push("Live Masters 2010 B/01 DELHI/8 JAN - 6 PM/01 PRAVACHAN.json");
  tracks.push("03 AA HIMMAT.json"); // the degenerate key
  return corpusFrom(tracks);
}

/** Everything visible: what the map drew before progressive disclosure landed. */
function layout(nodes, opts) {
  return radialTree(buildTree(rootNode(0, 0), nodes), opts);
}

function ringsOf(placed) {
  const rings = new Map();
  for (const p of placed) {
    if (p.dist === 0) continue;
    const at = rings.get(p.node.depth);
    if (at) at.push(p);
    else rings.set(p.node.depth, [p]);
  }
  return rings;
}

/** The buckets `relaxAngles` actually works on: one depth, one displacement. */
function framesOf(placed) {
  const buckets = new Map();
  for (const p of placed) {
    if (p.dist === 0) continue;
    const key = `${p.node.depth} ${p.frame}`;
    const at = buckets.get(key);
    if (at) at.push(p);
    else buckets.set(key, [p]);
  }
  return buckets;
}

function angleByPath(placed) {
  return new Map(placed.map((p) => [p.node.path, p.angle]));
}

const SWEEP = [];
for (const minRingGap of [60, 130, 420]) {
  for (const padding of [0, 2.5, 24]) SWEEP.push({ minRingGap, padding });
}

// ---------------------------------------------------------------------------
// Radius encodes depth. Nothing may break this — not relaxation, not a slider,
// not a drag.
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

  // The same, with a subtree the reader has flung across the disc. `dist` is now
  // the distance from the node's *frame origin*, not from the screen centre — and
  // relaxation is still forbidden from touching it.
  applyDisplacements(placed, new Map([["Dagshai 2002", { x: -4000, y: 900 }]]));
  const displaced = placed.map((p) => p.dist);
  relaxAngles(placed, { iterations: 40 });
  placed.forEach((p, i) => assert.equal(p.dist, displaced[i], `node ${i} moved radially`));
  let moved = 0;
  for (const p of placed) {
    assert.ok(Math.abs(Math.hypot(p.x - p.ox, p.y - p.oy) - p.dist) < 1e-6,
      `${p.node.path}: x/y desynced from (dist, angle, offset)`);
    if (p.ox !== 0 || p.oy !== 0) moved++;
  }
  assert.ok(moved > 1, "the displacement carried nothing");
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
  // Focus dimming, subtree pruning and visibility all key on paths, so a naive
  // startsWith(focus) lights up a sibling collection and nobody notices.
  assert.equal(parentPath("a/b/c"), "a/b");
  assert.equal(parentPath("a"), "");
  assert.deepEqual(ancestors("Dagshai 2002/03 MAR/CAMP/1 PM/x.json"),
    ["Dagshai 2002", "Dagshai 2002/03 MAR", "Dagshai 2002/03 MAR/CAMP",
     "Dagshai 2002/03 MAR/CAMP/1 PM"]);
  assert.deepEqual(ancestors("x.json"), []);
  assert.ok(!"Live Masters 2010 B".startsWith("Live Masters 2010" + "/"));
});

check("buildTree attaches a child to its deepest present ancestor", () => {
  // The session's own parent (the month bucket) is absent from this fetch.
  const all = corpus(4);
  const nodes = [
    all.find((n) => n.path === "Dagshai 2002"),
    all.find((n) => n.path === "Dagshai 2002/03 MAR - 2002/1 MAR - 6 PM"),
  ];
  const tree = buildTree(rootNode(0, 0), nodes);
  assert.equal(tree.children.length, 1, "session should not be orphaned to the root");
  assert.equal(tree.children[0].children.length, 1);
});

check("a folder name containing '$' survives the tree build", () => {
  const nodes = corpusFrom(["Dagshai 2002/26 MAR - 1$ - 7 PM/01 X.json"]);
  const tree = buildTree(rootNode(0, 0), nodes);
  assert.equal(tree.children[0].children[0].node.name, "26 MAR - 1$ - 7 PM");
});

// ---------------------------------------------------------------------------
// Visibility. The map draws the open branch, not everything it has ever fetched.
// ---------------------------------------------------------------------------

check("nothing open → exactly the collections, and nothing under them", () => {
  const nodes = corpus(30);
  const seen = visibleNodes(nodes, new Set());
  assert.ok(seen.length > 0);
  for (const n of seen) assert.ok(n.depth <= REVEAL_DEPTH, `${n.path} should be hidden`);
  const shallow = nodes.filter((n) => n.depth <= REVEAL_DEPTH);
  assert.equal(seen.length, shallow.length, "every shallow node must be drawn");
  // One click, one level. At REVEAL_DEPTH 2 a collection's children were already
  // on screen, so clicking one changed nothing the reader could see.
  assert.equal(REVEAL_DEPTH, 1);
});

check("opening a collection reveals its groups and no sitting", () => {
  const nodes = corpus(30);
  const collection = "Dagshai 2002";
  const seen = new Set(visibleNodes(nodes, spineOf(collection)).map((n) => n.path));
  assert.ok(seen.has(`${collection}/03 MAR - 2002`), "the groups must appear");
  assert.ok(!seen.has(`${collection}/03 MAR - 2002/7 MAR - 6 PM`), "the sittings must not");
  assert.ok(seen.has("Live Masters 2010"), "the other collections stay on the map");
  assert.ok(!seen.has("Live Masters 2010/01 NOIDA"), "…but stay closed");
});

check("opening a group reveals its children and no grandchild", () => {
  const nodes = corpus(30);
  const group = "Dagshai 2002/03 MAR - 2002";
  const seen = new Set(visibleNodes(nodes, spineOf(group)).map((n) => n.path));
  assert.ok(seen.has(`${group}/7 MAR - 6 PM`), "the sittings must appear");
  assert.ok(!seen.has(`${group}/7 MAR - 6 PM/01 PRAVACHAN.json`), "the tracks must not");
});

check("opening `Live Masters 2010` reveals nothing under `Live Masters 2010 B`", () => {
  const nodes = corpus(4);
  const seen = new Set(visibleNodes(nodes, spineOf("Live Masters 2010/01 NOIDA")).map((n) => n.path));
  assert.ok(seen.has("Live Masters 2010/01 NOIDA/7 JAN - 6 PM"));
  assert.ok(!seen.has("Live Masters 2010 B/01 DELHI/8 JAN - 6 PM"),
    "a string prefix is not an ancestor");
});

check("visibleNodes agrees with the brute-force rule on every path", () => {
  const nodes = corpus(40);
  // Reference: a node is drawn iff every ancestor of it that could have been
  // collapsed (depth >= REVEAL_DEPTH) is open.
  const reference = (n, open) =>
    ancestors(n.path).filter((a) => a.split("/").length >= REVEAL_DEPTH).every((a) => open.has(a));

  for (const focus of [null, ...nodes.map((n) => n.path)].slice(0, 200)) {
    const open = spineOf(focus);
    const seen = new Set(visibleNodes(nodes, open).map((n) => n.path));
    for (const n of nodes) {
      assert.equal(seen.has(n.path), reference(n, open),
        `${n.path} disagrees under focus=${focus}`);
    }
  }
});

check("spineOf is closed under ancestors, and collapse is the inverse of expand", () => {
  const deep = "Dagshai 2002/03 MAR - 2002/7 MAR - 6 PM/01 PRAVACHAN.json";
  const open = spineOf(deep);
  for (const p of open) for (const a of ancestors(p)) assert.ok(open.has(a), `${a} missing`);
  assert.ok(open.has(deep));
  assert.equal(spineOf(null).size, 0);

  // Esc walks focusPath up one level; `open` is derived, so collapse cannot
  // strand a node the way a stored visibility set could.
  const nodes = corpus(20);
  const shut = visibleNodes(nodes, spineOf(null)).map((n) => n.path);
  const roundTrip = visibleNodes(nodes, spineOf(parentPath(parentPath("Dagshai 2002")) || null))
    .map((n) => n.path);
  assert.deepEqual(roundTrip, shut);
});

check("childrenUnfetched is about the network, not about depth alone", () => {
  const nodes = corpus(4);
  const collection = nodes.find((n) => n.path === "Dagshai 2002");
  const session = nodes.find((n) => n.path === "Dagshai 2002/03 MAR - 2002/1 MAR - 6 PM");
  const track = nodes.find((n) => n.is_leaf && n.depth === 4);
  assert.equal(collection.depth, 1);
  assert.ok(collection.depth < PREFETCH_DEPTH);
  assert.equal(childrenUnfetched(collection, new Set()), false, "the skeleton brought these");
  assert.equal(childrenUnfetched(session, new Set()), true);
  assert.equal(childrenUnfetched(session, new Set([session.path])), false, "already fetched");
  assert.equal(childrenUnfetched(track, new Set()), false, "a recording has no children");
});

check("expanding a node changes no angle outside its subtree", () => {
  // Wedge conservation. This is the check that forces `weight()` over
  // `leafCount()`: without it a collapsed 1,748-sitting camp claims the arc of an
  // empty folder, and opening it re-angles the entire disc.
  const nodes = corpus(60);
  const branch = "Dagshai 2002";
  const shut = angleByPath(layout(visibleNodes(nodes, spineOf(null))));
  const open = angleByPath(layout(visibleNodes(nodes, spineOf(`${branch}/03 MAR - 2002`))));

  let compared = 0;
  for (const [path, angle] of shut) {
    if (path === branch || path.startsWith(branch + "/")) continue;
    assert.ok(open.has(path), `${path} vanished when a sibling branch opened`);
    assert.ok(Math.abs(open.get(path) - angle) < 1e-9,
      `${path} moved by ${Math.abs(open.get(path) - angle)}`);
    compared++;
  }
  // Closed, the map is its collections: Dagshai, Live Masters 2010, its
  // near-namesake, and the degenerate bare filename. Three of them are outside
  // the branch being opened.
  assert.ok(compared >= 3, "expected several untouched nodes");
  assert.ok(Math.abs(open.get(branch) - shut.get(branch)) < 1e-9, "the branch itself moved");
});

check("a folder's wedge is exactly the sum of its children's", () => {
  const placed = layout(corpus(25));
  const kids = new Map();
  for (const p of placed) {
    if (!p.parent) continue;
    const at = kids.get(p.parent);
    if (at) at.push(p);
    else kids.set(p.parent, [p]);
  }
  let checked = 0;
  for (const [parent, cs] of kids) {
    const sum = cs.reduce((s, c) => s + (c.a1 - c.a0), 0);
    assert.ok(Math.abs(sum - (parent.a1 - parent.a0)) < 1e-9,
      `${parent.node.path || "root"}: children claim ${sum}, parent has ${parent.a1 - parent.a0}`);
    checked++;
  }
  assert.ok(checked >= 3);
});

// ---------------------------------------------------------------------------
// Spacing. Both sliders sweep their full range; nothing may break anywhere in it.
// ---------------------------------------------------------------------------

check("across the whole slider sweep, radius is still a function of depth", () => {
  for (const opts of SWEEP) {
    const placed = layout(corpus(120), opts);
    const byDepth = new Map();
    for (const p of placed) {
      const seen = byDepth.get(p.node.depth);
      if (seen === undefined) byDepth.set(p.node.depth, p.dist);
      else assert.equal(p.dist, seen, `two radii at depth ${p.node.depth}, opts ${JSON.stringify(opts)}`);
    }
  }
});

check("across the whole sweep, a ring is wide enough to seat its own dots", () => {
  for (const opts of SWEEP) {
    const placed = layout(corpus(400), opts);
    for (const ring of ringsOf(placed).values()) {
      const dist = ring[0].dist;
      // The chord form, which is what the dots actually need. The arc form
      // (2πR >= Σ2r) is its small-angle approximation and over-promises.
      let need = 0;
      for (const p of ring) need += 2 * Math.asin(Math.min(1, (p.r + opts.padding / 2) / dist));
      assert.ok(need <= TAU + 1e-9,
        `ring at ${dist.toFixed(0)} needs ${need.toFixed(3)} rad of ${TAU.toFixed(3)}`);
    }
  }
});

check("across the whole sweep, adjacent rings never touch", () => {
  // Nobody checked this before. It was silently true because 130 > 2 * 22, and a
  // slider that could go below 44 would have put a track's dot on top of a camp's.
  for (const opts of SWEEP) {
    const placed = layout(corpus(120), opts);
    const radii = [...new Set(placed.map((p) => p.dist))].sort((a, b) => a - b);
    for (let i = 1; i < radii.length; i++) {
      assert.ok(radii[i] - radii[i - 1] >= 2 * MAX_NODE_RADIUS - 1e-9,
        `rings ${radii[i - 1].toFixed(1)} and ${radii[i].toFixed(1)} can collide`);
    }
  }
});

check("the floors clamp inside radialTree and relaxAngles, not at the call site", () => {
  const placed = layout(corpus(40), { minRingGap: -1000, padding: -50, innerRadius: -1 });
  const radii = [...new Set(placed.map((p) => p.dist))].sort((a, b) => a - b);
  for (let i = 1; i < radii.length; i++) {
    assert.ok(radii[i] - radii[i - 1] >= 2 * MAX_NODE_RADIUS - 1e-9, "minRingGap floor bypassed");
  }
  const before = placed.map((p) => p.dist);
  relaxAngles(placed, { padding: -50, strength: 99, iterations: 10 });
  placed.forEach((p, i) => assert.equal(p.dist, before[i]));
});

check("the layout stays inside the spatial grid's key space at 27,806 nodes", () => {
  // SpatialGrid.key() packs a cell index as (cx + 32768) * 65536, with cell = 48.
  // Layout coords must therefore stay inside ±32768 * 48 ≈ 1.57M, or two distant
  // nodes silently share a bucket and hover picks the wrong one.
  const tracks = [];
  for (let s = 0; s < 27806; s++) tracks.push(`C${s % 17}/G${s % 191}/S${s}.json`);
  const nodes = corpusFrom(tracks);
  const placed = layout(nodes, { minRingGap: 420, padding: 24 });
  relaxAngles(placed);
  const reach = Math.max(...placed.map((p) => Math.max(Math.abs(p.x), Math.abs(p.y)) + p.r));
  assert.ok(reach < 32768 * 48, `layout reaches ${reach.toFixed(0)}, key space ends at 1572864`);

  const grid = new SpatialGrid();
  grid.rebuild(placed);
  const target = placed[placed.length - 1];
  assert.ok(grid.pick(target.x, target.y, 0), "the outermost node is still pickable");

  // And with the reader's arrangement pushed to its limit in every direction.
  const corners = new Map();
  for (const [i, sign] of [[1, 1], [-1, 1], [1, -1], [-1, -1]].entries()) {
    corners.set(`C${i}`, { x: sign[0] * MAX_DRAG_COORD, y: sign[1] * MAX_DRAG_COORD });
  }
  applyDisplacements(placed, corners);
  relaxAngles(placed);
  const far = Math.max(...placed.map((p) => Math.max(Math.abs(p.x), Math.abs(p.y)) + p.r));
  assert.ok(far < 32768 * 48, `a dragged branch reaches ${far.toFixed(0)}, key space ends at 1572864`);
  grid.rebuild(placed);
  const flung = placed.find((p) => p.node.path === "C0");
  assert.ok(grid.pick(flung.x, flung.y, 0), "a node dragged to the corner is still pickable");
});

check("relaxation leaves same-frame dots non-overlapping, in chord distance", () => {
  // Tolerance 1e-6, not 0.5. The old tolerance was absorbing the small-angle
  // error of `toAngle = 1 / dist`: at minRingGap 60 it hid a 0.98-unit overlap.
  // `relaxAngles` breaks out of its loop the moment a full pass moves nothing, so
  // 400 iterations means "run to convergence", not "run 400 times".
  //
  // Bucketed by `(depth, frame)`, which is what relaxation works on. Two nodes in
  // different frames are separated by the reader's hand, not by chord geometry —
  // and if the reader parks one branch on top of another, that is a choice.
  for (const opts of SWEEP) {
    const placed = layout(corpus(120), opts);
    applyDisplacements(placed, new Map([["Dagshai 2002/03 MAR - 2002", { x: 700, y: -300 }]]));
    relaxAngles(placed, { iterations: 400, padding: opts.padding });
    let checked = 0;
    for (const ring of framesOf(placed).values()) {
      ring.sort((a, b) => a.angle - b.angle);
      for (let i = 1; i < ring.length; i++) {
        const a = ring[i - 1], b = ring[i];
        const gap = Math.hypot(b.x - a.x, b.y - a.y);
        assert.ok(gap >= a.r + b.r - 1e-6,
          `overlap of ${(a.r + b.r - gap).toFixed(3)} at ${JSON.stringify(opts)}`);
        checked++;
      }
    }
    assert.ok(checked > 100, "the sweep compared almost nothing");
  }
});

check("the asin guard never clamps anywhere in the allowed slider range", () => {
  // If (ra + rb + padding) / (2 * dist) ever reached 1, `want` would saturate at
  // π and two dots would be pushed to opposite sides of the ring. The minRingGap
  // floor of 2 * MAX_NODE_RADIUS is what rules it out; this proves it.
  for (const opts of SWEEP) {
    const placed = layout(corpus(120), opts);
    for (const ring of ringsOf(placed).values()) {
      const dist = ring[0].dist;
      for (const a of ring) for (const b of ring) {
        assert.ok((a.r + b.r + opts.padding) / (2 * dist) < 1,
          `asin saturates at depth ${a.node.depth}, opts ${JSON.stringify(opts)}`);
      }
    }
  }
});

check("the camera compensation is exact", () => {
  const v = { x: 137, y: -42, scale: 0.31, width: 1280, height: 800 };
  const before = { x: 300, y: -900 };
  const after = { x: 812.5, y: -1233.25 };
  const moved = { ...v, ...anchorPan(v, before, after) };
  const s0 = toScreen(before, v);
  const s1 = toScreen(after, moved);
  assert.ok(Math.abs(s0.x - s1.x) < 1e-9 && Math.abs(s0.y - s1.y) < 1e-9,
    `anchor moved on screen by ${Math.hypot(s0.x - s1.x, s0.y - s1.y)}`);
  assert.equal(moved.scale, v.scale, "compensation must not touch zoom");
  // A no-op relayout must be a no-op pan.
  assert.deepEqual(anchorPan(v, before, before), { x: v.x, y: v.y });
});

// ---------------------------------------------------------------------------
// Arranging. A drag translates a subtree, changes no angle, and survives both
// relaxation and a relayout.
// ---------------------------------------------------------------------------

check("a drag translates a whole subtree by one vector, and nothing outside it moves", () => {
  const nodes = corpus(20);
  const branch = "Dagshai 2002/03 MAR - 2002";
  const visible = visibleNodes(nodes, spineOf(branch));
  const base = layout(visible);
  const target = base.find((p) => p.node.path === branch);
  const want = { x: target.x - 900, y: target.y + 400 };

  const placed = layout(visible);
  applyDisplacements(placed, new Map([[branch, want]]));
  const moved = placed.find((p) => p.node.path === branch);
  assert.ok(Math.hypot(moved.x - want.x, moved.y - want.y) < 1e-9,
    `asked for (${want.x}, ${want.y}), landed at (${moved.x}, ${moved.y})`);

  const dx = moved.x - target.x, dy = moved.y - target.y;
  let carried = 0, still = 0;
  for (const p of placed) {
    const was = base.find((q) => q.node.path === p.node.path);
    const inside = p.node.path === branch || p.node.path.startsWith(branch + "/");
    assert.ok(Math.abs(p.x - (was.x + (inside ? dx : 0))) < 1e-9, `${p.node.path} x`);
    assert.ok(Math.abs(p.y - (was.y + (inside ? dy : 0))) < 1e-9, `${p.node.path} y`);
    assert.equal(p.frame, inside ? branch : "", `${p.node.path} has the wrong frame`);
    if (inside) carried++; else still++;
  }
  assert.ok(carried > 1 && still > 1, "the fixture proves nothing");
});

check("a drag never changes `dist` or `angle`", () => {
  // The law, stated exactly. The layout owns (angle, dist); the reader owns
  // (ox, oy); x/y is their sum and the only thing the renderer reads.
  const nodes = corpus(20);
  const placed = layout(visibleNodes(nodes, spineOf("Dagshai 2002")));
  const before = placed.map((p) => ({ dist: p.dist, angle: p.angle }));
  applyDisplacements(placed, new Map([
    ["Dagshai 2002", { x: -2200, y: 60 }],
    ["Live Masters 2010", { x: 15, y: -3333 }],
  ]));
  placed.forEach((p, i) => {
    assert.equal(p.dist, before[i].dist, `${p.node.path} moved radially`);
    assert.equal(p.angle, before[i].angle, `${p.node.path} was rotated`);
  });
  for (const p of placed) {
    assert.ok(Math.abs(Math.hypot(p.x - p.ox, p.y - p.oy) - p.dist) < 1e-6, "x/y desynced");
  }
});

check("a rigid translation preserves every chord inside the subtree", () => {
  // The reason frame-bucketed relaxation is still exact. `relaxAngles` separates
  // dots by `2R·sin(θ/2)`, a chord — which survives a translation and nothing else.
  const placed = layout(corpus(60));
  const sub = subtreeOf(placed.find((p) => p.node.path === "Dagshai 2002"), placed);
  const chords = [];
  for (let i = 0; i < sub.length; i += 3) {
    for (let j = i + 1; j < sub.length; j += 7) {
      chords.push([i, j, Math.hypot(sub[i].x - sub[j].x, sub[i].y - sub[j].y)]);
    }
  }
  assert.ok(chords.length > 50, "sampled too few pairs to prove anything");
  translateSubtree(sub, -1234.5, 6789.25);
  for (const [i, j, was] of chords) {
    const now = Math.hypot(sub[i].x - sub[j].x, sub[i].y - sub[j].y);
    assert.ok(Math.abs(now - was) < 1e-9, `chord ${i}-${j} changed by ${now - was}`);
  }
});

check("a node may be dragged to any point, including across the centre", () => {
  // The old design welded radius to depth, so a drag could only slide a dot around
  // its ring. Nothing is welded now: a camp may sit inside the disc, and a single
  // recording may sit outside it. Only the reader can put it there.
  const placed = layout(corpus(20));
  const branch = "Live Masters 2010";
  for (const want of [{ x: 0, y: 0 }, { x: -5, y: -5 }, { x: 4000, y: -9000 }, { x: 1, y: 30000 }]) {
    applyDisplacements(placed, new Map([[branch, want]]));
    const p = placed.find((q) => q.node.path === branch);
    assert.ok(Math.hypot(p.x - want.x, p.y - want.y) < 1e-9,
      `asked for (${want.x}, ${want.y}), landed at (${p.x}, ${p.y})`);
  }
  // …and putting it back is a matter of forgetting, not of undoing.
  const home = layout(corpus(20)).find((q) => q.node.path === branch);
  applyDisplacements(placed, new Map());
  const p = placed.find((q) => q.node.path === branch);
  assert.ok(Math.hypot(p.x - home.x, p.y - home.y) < 1e-9, "clearing the ledger did not snap it home");
});

check("`clampTarget` bounds a hostile localStorage value", () => {
  // `SpatialGrid.key()` overflows its bucket packing past ±1,572,864, and the
  // arrangement ledger is a JSON blob on the reader's disk that can say anything.
  for (const [x, y] of [[1e300, -1e300], [Infinity, -Infinity], [NaN, 5], [-1e9, 1e9]]) {
    const c = clampTarget(x, y);
    assert.ok(Math.abs(c.x) <= MAX_DRAG_COORD && Math.abs(c.y) <= MAX_DRAG_COORD,
      `(${x}, ${y}) escaped to (${c.x}, ${c.y})`);
    assert.ok(Number.isFinite(c.x) && Number.isFinite(c.y), "a NaN survived the clamp");
  }
  assert.deepEqual(clampTarget(NaN, NaN), { x: 0, y: 0 }, "NaN should mean the origin");
  assert.deepEqual(clampTarget(12, -34), { x: 12, y: -34 }, "an honest point must pass through");
  // An infinity has a direction, and honouring it is what the reader meant.
  assert.deepEqual(clampTarget(Infinity, -Infinity), { x: MAX_DRAG_COORD, y: -MAX_DRAG_COORD });

  // And the clamp is applied where it matters: on the way in from disk.
  const placed = layout(corpus(12));
  applyDisplacements(placed, new Map([["Dagshai 2002", { x: 1e300, y: -Infinity }]]));
  const p = placed.find((q) => q.node.path === "Dagshai 2002");
  assert.equal(p.x, MAX_DRAG_COORD);
  assert.equal(p.y, -MAX_DRAG_COORD);
});

check("relaxation separates a ring whose subtree was pushed off the seam", () => {
  // `place()` hands out angles in [-π/2, 3π/2], so sorting on the *raw* angle was
  // the circular order by accident. Relaxation's own passes shove angles across the
  // seam; sorted raw, two true neighbours land at opposite ends of the array and
  // nothing ever pushes them apart. `relaxAngles` sorts the wrapped angle for this.
  const placed = layout(corpus(120));
  const sub = subtreeOf(placed.find((p) => p.node.path === "Dagshai 2002"), placed);
  for (const p of sub) p.angle -= 5.7; // across the -π/2 seam
  applyAngles(placed);
  relaxAngles(placed, { iterations: 400 });

  const circular = (p) => Math.atan2(Math.sin(p.angle), Math.cos(p.angle));
  for (const ring of ringsOf(placed).values()) {
    const sorted = [...ring].sort((a, b) => circular(a) - circular(b));
    for (let i = 1; i < sorted.length; i++) {
      const a = sorted[i - 1], b = sorted[i];
      const gap = Math.hypot(b.x - a.x, b.y - a.y);
      assert.ok(gap >= a.r + b.r - 1e-6,
        `overlap of ${(a.r + b.r - gap).toFixed(3)} at depth ${a.node.depth} after a rotation`);
    }
  }
});

check("a node the reader placed is alone in its (depth, frame) bucket", () => {
  // This is what replaced `Placed.fixed`, and it is a theorem, not a flag. For a
  // dragged node D, frame(D) = D.path; any other node with that frame is a strict
  // descendant, hence deeper. So D's bucket is {D}, `relaxAngles` skips buckets of
  // size < 2, and nothing can shove a node the reader put somewhere.
  const nodes = corpus(120);
  const placed = layout(visibleNodes(nodes, spineOf("Dagshai 2002/03 MAR - 2002")));
  // A collection, a group *inside it*, and an unrelated collection. Nested drags
  // are where a naive `frame` would be wrong.
  const pins = new Map([
    ["Dagshai 2002", { x: -800, y: 250 }],
    ["Dagshai 2002/03 MAR - 2002", { x: -1500, y: 900 }],
    ["Live Masters 2010", { x: 640, y: -70 }],
  ]);
  applyDisplacements(placed, pins);

  const buckets = framesOf(placed);
  for (const path of pins.keys()) {
    const p = placed.find((q) => q.node.path === path);
    const bucket = buckets.get(`${p.node.depth} ${p.frame}`);
    assert.equal(p.frame, path, `${path} is not its own frame`);
    assert.equal(bucket.length, 1, `${path} shares its bucket with ${bucket.length - 1} others`);
  }
  // The payoff: a branch the reader dragged out still relaxes *inside* its frame.
  const inner = [...buckets.entries()].filter(([k, v]) => k.endsWith("03 MAR - 2002") && v.length > 1);
  assert.ok(inner.length > 0, "a dragged branch's own rings stopped relaxing");

  const before = [...pins.keys()].map((path) => {
    const p = placed.find((q) => q.node.path === path);
    return { x: p.x, y: p.y };
  });
  relaxAngles(placed, { iterations: 200 });
  [...pins.keys()].forEach((path, i) => {
    const p = placed.find((q) => q.node.path === path);
    assert.equal(p.x, before[i].x, `${path} was shoved`);
    assert.equal(p.y, before[i].y, `${path} was shoved`);
  });
});

check("a moved node survives a relayout: same point, whatever the spacing", () => {
  // Targets are stored absolute, not as deltas, precisely so this holds. A delta
  // drifts as the rings grow under it; the point the reader chose does not.
  const nodes = corpus(120);
  const visible = visibleNodes(nodes, spineOf("Dagshai 2002"));
  const branch = "Dagshai 2002/03 MAR - 2002";
  const want = { x: -1750.5, y: 620.25 };
  for (const opts of SWEEP) {
    const placed = layout(visible, opts);
    applyDisplacements(placed, new Map([[branch, want]]));
    relaxAngles(placed, { iterations: 400, padding: opts.padding });
    const p = placed.find((q) => q.node.path === branch);
    assert.ok(Math.hypot(p.x - want.x, p.y - want.y) < 1e-9,
      `at ${JSON.stringify(opts)} the node drifted to (${p.x}, ${p.y})`);
  }
});

check("translateSubtree and subtreeOf agree with the parent chain", () => {
  const placed = layout(corpus(12));
  const root = placed.find((p) => p.node.path === "Dagshai 2002");
  const sub = subtreeOf(root, placed);
  const expected = placed.filter((p) => {
    for (let q = p; q; q = q.parent) if (q === root) return true;
    return false;
  });
  assert.equal(sub.length, expected.length);
  const before = new Map(placed.map((p) => [p, { x: p.x, y: p.y }]));
  translateSubtree(sub, 0.75, -0.25);
  for (const p of placed) {
    const moved = Math.abs(p.x - before.get(p).x - 0.75) < 1e-12
      && Math.abs(p.y - before.get(p).y + 0.25) < 1e-12;
    assert.equal(moved, sub.includes(p), `${p.node.path} moved when it should not have`);
  }
});

// ---------------------------------------------------------------------------
// Picking.
// ---------------------------------------------------------------------------

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

check("applyAngles keeps x/y on the ring, offset by exactly what the reader asked", () => {
  const placed = layout(corpus(30));
  for (const p of placed) p.angle += 1.234;
  applyAngles(placed);
  for (const p of placed) {
    assert.equal(p.ox, 0);
    assert.equal(p.oy, 0);
    assert.ok(Math.abs(Math.hypot(p.x, p.y) - p.dist) < 1e-6);
  }

  // With a displacement, x/y is the ring position plus the offset — nothing else.
  applyDisplacements(placed, new Map([["Dagshai 2002", { x: 3000, y: -3000 }]]));
  applyAngles(placed);
  for (const p of placed) {
    assert.ok(Math.abs(p.x - (Math.cos(p.angle) * p.dist + p.ox)) < 1e-9);
    assert.ok(Math.abs(p.y - (Math.sin(p.angle) * p.dist + p.oy)) < 1e-9);
  }
});

console.log(`\n${checks} invariant checks passed.`);
