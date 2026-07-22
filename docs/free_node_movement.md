# Free node movement in the archive map

*Phase 16, slice 1c — amended. 2026-07-08.*

How a feature request that looked like "add a second layout engine" turned out to be
eleven lines of vector arithmetic, a deleted boolean, and two pre-existing bugs.

---

## 1. The request

Slice 1c shipped an **angular** drag: you could slide a dot *around* its ring, but its
distance from the centre was welded to its depth in the tree. The verdict, after using it:

> "i am able to move the node from left to right but i can't move it near or far from the
> center. i want that every node is free to move whether it is primary, secondary or any
> node … and then i can open its inner nodes"

Three requirements, and the third is the hard one:

1. Any node moves to any point.
2. A container carries what's inside it.
3. **You can still open a node where you parked it** — and its children arrive tidy,
   relaxed, non-overlapping, around its new position.

## 2. Why the code said no

`radialTree.ts` carried a comment arguing against exactly this:

> Free `(x, y)` dragging would be a second layout engine, and the first thing it would do
> is put a track in the ring where camps live.

That comment was defending a real invariant — **radius encodes depth** — enforced by 34
checks in `check-viz-invariants.cjs`. Requirement 3 is what makes it look expensive: the
angular relaxation that separates overlapping dots does chord geometry on a ring,
`θ = 2·asin(s / 2R)`, and a ring is only a ring if every node on it shares a radius. Move
one node off, and the geometry stops being true.

So the obvious reading was: free movement means the layout is no longer radial, so
relaxation must be replaced, so this is a rewrite.

That reading was wrong.

## 3. The approach: don't guess, measure

Before writing a plan, two claims had to be settled. Both were checked by running
read-only Node scripts against the already-compiled `scripts/.viz-build/` — no edits, no
guessing.

### Claim 1 — is a Cartesian drag actually more expensive?

The drag is the only 60 Hz path in the system; `bench:viz` holds it to 16 ms.

```
27,806 sittings -> 2,942 drawn, dragged subtree = 2,913 nodes
  ok  PER POINTERMOVE (old): rotateSubtree + grid     0.1322 ms
  ok  PER POINTERMOVE (new): translateSubtree + grid  0.0555 ms
  ok  subtreeOf (once per pointerdown)                0.0575 ms
```

**It is 2.4× cheaper.** A rotation is two trig calls per node. A translation is two adds.
The thing we were reluctant to build was faster than the thing we had.

### Claim 2 — can relaxation survive?

This is the one that decided the design. Stated as a question: *after the reader flings a
subtree across the disc, is there still something relaxation can correctly separate?*

## 4. The structural insight

`Placed.x/y` is not the layout. It is the **render** of the layout. Today the two are
welded:

```ts
p.x = Math.cos(p.angle) * p.dist;
p.y = Math.sin(p.angle) * p.dist;
```

Split them. Add the reader's displacement as a separate term:

```ts
p.x = Math.cos(p.angle) * p.dist + p.ox;   // layout's choice + reader's
p.y = Math.sin(p.angle) * p.dist + p.oy;
```

A drag writes `ox`/`oy` and never touches `angle` or `dist`. So the invariant does not
break — it **sharpens**:

> The law is not *"radius is depth."*
> The law is ***the system never moves a node radially. Only the reader does.***

Layout, relaxation and both spacing sliders remain exactly as pure as they were. Free drag
is not a second layout engine; it is a display-time translation applied after the first one
has run.

## 5. Why relaxation still works — and why `fixed` is a theorem

A rigid translation is an **isometry**: it preserves distance. So every chord *inside* a
translated subtree is exactly what the layout laid down. Relaxation's `2·asin(s / 2R)` is
therefore:

- **still exact** between two nodes that share a displacement, and
- **meaningless** between two nodes that do not.

Which is a bucketing rule, not a rewrite. Give every node a `frame` — the path of its
nearest displaced ancestor-or-self, `""` if nothing above it moved — and have `relaxAngles`
bucket rings by `(depth, frame)` instead of `depth`.

Then something falls out for free.

> **Theorem.** A node the reader placed is alone in its bucket, so relaxation cannot move
> it.
>
> *Proof.* For a dragged node `D`, `frame(D) = D.path`. Any other node `q` with
> `frame(q) = D.path` is a strict descendant of `D`, so `depth(q) > depth(D)`. Therefore
> bucket `(depth(D), D.path) = {D}`. `relaxAngles` skips buckets of size < 2. ∎

The old code carried a `Placed.fixed` boolean, plus two asymmetric-push branches in
`relax.ts`, to stop relaxation from shoving a pinned node. **All of it is redundant.**
Deleted; net −8 lines.

Measured, with three simultaneous drags including a nested one (`C0`, then `C0/G0` *inside*
it, plus an unrelated `C1`):

```
6 relaxation buckets (was 3 rings)
  dragged C0      depth 1 frame "C0"    -> bucket size 1 (alone: relax cannot move it)
  dragged C0/G0   depth 2 frame "C0/G0" -> bucket size 1 (alone: relax cannot move it)
  dragged C1      depth 1 frame "C1"    -> bucket size 1 (alone: relax cannot move it)
  buckets: 1|C0=1  2|C0/G0=1  3|C0/G0=978  2|C0=11  1|C1=1  1|=15
```

`3|C0/G0=978` **is requirement 3, satisfied.** The fat group's 978 sittings still relax
against each other, *inside* the translated frame — so a branch you drag out and then open
is as tidy as one you left alone. `2|C0=11` is C0's other groups relaxing without G0, which
has left their frame. `1|=15` is the untouched collections, undisturbed.

And a bonus nobody asked for: dragging a node **out** of a ring no longer disturbs its old
ring-mates. Relaxation only pushes apart, never pulls, so removing `D` from their bucket
leaves them exactly where they were. That is strictly better than the old `fixed`, under
which `D` still shoved its neighbours while refusing to be shoved.

## 6. Three design decisions worth the ink

**Absolute targets, not deltas.** A stored delta drifts as the layout changes underneath
it. An absolute point is what the reader chose. So a moved node holds its pixel while the
Spread slider grows the disc around it.

**Shallowest-first application.** `applyDisplacements` sorts targets by depth. That is what
makes `q.frame = p.node.path` correct: a moved collection carries its groups first, then a
group's own target resolves against the position it has just been given, and overwrites its
own subtree's `frame` — leaving every node tagged with its *nearest* displaced ancestor.

**Re-derive `frame` on pointerup, not patch it.** `commitNodeDrag` calls
`applyDisplacements(placed, moveRef)` from scratch before relaxing. It costs one pointerup.
It makes the state after a drag byte-identical to the state after a reload, which closes a
whole class of bug that would otherwise only appear after refreshing.

## 7. `MAX_DRAG_COORD` is derived, not chosen

```
SpatialGrid.key() packs (c + 32768) * 65536, cell = 48
  => |coord| must stay under 32768 · 48 = 1,572,864
     or two distant nodes silently share a bucket and hover picks the wrong dot.

widest layout this archive can produce (all 55,821 nodes, Spread 420) = 139,566   [measured]
MAX_DRAG_COORD = 50,000   -> an order of magnitude of headroom
                          -> 3,000 px of dragging at MIN_SCALE: unreachable by hand
```

It exists mostly to survive a `localStorage` entry that says `1e300`. An infinity clamps to
the bound it is heading for — it has a direction, and honouring it is what the reader
meant. Only `NaN` means the origin.

## 8. Two bugs the change exposed

Neither was introduced by this work. Both were found *because* of it.

**1. `frameFor` clipped at `MIN_SCALE = 0.06`.** The "Whole archive" breadcrumb clamps its
computed scale to `MIN_SCALE`, so it *already* silently cropped anything past an extent of
≈ 6,600 — and opening the largest camp at Spread 420 reaches 14,575 (measured). Free
dragging just made it reachable by hand, which is how it surfaced.

`MIN_SCALE` is now `0.006`, which frames an extent of 92,000. That in turn forced a second
fix: `pick`'s slop is measured in **layout** units, so at scale 0.006 an uncapped
`5 / scale` is 833 units → a 19×19 sweep → **1,521 grid cells per pointermove**. Capped at
`4 · MAX_NODE_RADIUS = 88`, it sweeps 49. No dot is wider than the cap, so nothing is
missed.

**2. `applyDisplacements` skipped `applyAngles` on an empty ledger.** *Reset arrangement*
would have left every node rendered at its last dragged position until something else
forced a relayout. Forgetting a move is how the reader puts a node back, so an empty ledger
must *actively* re-render the layout's own position rather than skip out.

Found by the invariant checker, not by hand:

```
AssertionError: clearing the ledger did not snap it home
```

That is the checker earning its keep. It also caught a wrong assertion I had written about
`clampTarget(-Infinity)`, and in fixing it we decided the *code* was wrong too — an
infinity should clamp to its bound, not collapse to the origin.

## 9. What changed

| File | Change |
|---|---|
| `viz/radialTree.ts` | **−** `clampToWedge`, `rotateSubtree`, `applyOverrides`, `Placed.fixed`, `wrap()`, the comment arguing against free drag. **+** `translateSubtree`, `applyDisplacements`, `clampTarget`, `MAX_DRAG_COORD`, `ox`/`oy`/`frame` on `Placed`. |
| `viz/relax.ts` | Buckets by `(depth, frame)`. Both `fixed` branches deleted; the push is always 50/50 again. **No new exports** — `check:viz` asserts `Object.keys(relax) === ["relaxAngles"]`. |
| `components/BrainView.tsx` | `angleRef` → `moveRef` (`Map<string, {x,y}>`). `brain.arrange.v1` → **`.v2`** (v1 held bare angles; reading one as a point would scatter the map — the old key is removed on sight). `Drag` carries a grab offset `gx`/`gy` instead of a grab angle. `MIN_SCALE 0.06 → 0.006`. Capped `pick` slop. |
| `viz/draw.ts`, `styles.css` | **Untouched.** They read `x`/`y` and were already correct. |
| `scripts/check-viz-invariants.cjs` | 34 → **36** checks. |
| `scripts/bench-viz.cjs` | Two renamed imports; budgets unchanged. |

Backend: **zero changes.**

### The invariant checks

Deleted 1 (*"a node clamps into its parent's wedge"* — the constraint no longer exists).
Rewrote 4. Added 3:

- *a rigid translation preserves every chord inside the subtree* — the reason
  frame-bucketed relaxation is exact, asserted rather than argued;
- *a moved node survives a relayout: same point, whatever the spacing* (swept across
  `minRingGap ∈ {60,130,420} × padding ∈ {0,2.5,24}`);
- *`clampTarget` bounds a hostile `localStorage` value* (`1e300`, `±Infinity`, `NaN`).

Plus: *a node the reader placed is alone in its `(depth, frame)` bucket*, and 200
relaxation passes cannot move it — the theorem of §5, executed.

## 10. Verification

```
npm run check:viz   36 invariant checks passed
npm run bench:viz   PER POINTERMOVE: translateSubtree + grid  2942 nodes  0.056 ms  (budget 16)
                    PER CLICK: visible→tree→layout→relax→grid  2942 nodes 10.992 ms  (budget 100)
                    PER IDLE FRAME: zero work
npx tsc --noEmit    clean
npm run build       clean — 217.23 kB / 70.06 kB gzip
dependencies        react, react-dom   (unchanged; the map is canvas 2D and hand-written)
```

Backend untouched, so `pytest` / `ruff` were not re-run.

**Not claimed without a browser, and therefore not claimed:** how the drag *feels*, whether
`DRAG_SLOP_PX = 3` is the right slop, and touch pinch-zoom (`touch-action: none` still
suppresses native gestures and nothing replaces them — separate work).

## 11. The lesson

The comment in `radialTree.ts` was not wrong to defend the invariant. It was wrong about
what the invariant *was*. "Radius is depth" is a statement about pixels. The thing actually
worth protecting is a statement about **authority**:

> The system never moves a node radially. Only the reader does.

Once that was said out loud, the implementation shrank to a vector add, a string key, and
the deletion of a boolean — and got faster. The measurements came before the plan, and the
plan came before the code; both bugs in §8 were found by machinery that already existed,
not by looking at the screen.
