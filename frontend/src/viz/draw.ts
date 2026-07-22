/**
 * Canvas 2D renderer for the archive map.
 *
 * Two decisions carry the frame budget:
 *
 * 1. **Glow is a pre-rendered sprite, not `ctx.shadowBlur`.** shadowBlur runs a
 *    Gaussian blur per draw call; at 2,000 nodes it alone costs more than the
 *    16 ms frame. Each colour's halo is rasterised once into an offscreen
 *    canvas and `drawImage`d thereafter — a blit, not a blur.
 * 2. **Everything is culled against the viewport before it is drawn.** Panning
 *    into a corner of a 27,806-node archive must not cost what drawing all of
 *    it costs.
 *
 * SVG was the alternative. Past ~2,000 nodes the per-element DOM cost dominates
 * and there is no way to opt out of it. Obsidian reaches for WebGL for the same
 * reason; canvas is the middle rung and it is enough here.
 */
import type { CState, CorpusNode } from "../corpus";
import { STATE_COLOR, litRatio, nodeState } from "../corpus";
import type { Viewport } from "./camera";
import { toScreen } from "./camera";
import type { Placed } from "./radialTree";

export interface DrawState {
  placed: Placed[];
  view: Viewport;
  hover: Placed | null;
  selected: Placed | null;
  /** The open branch — `spineOf(focus)`. Its members show their children. */
  open: ReadonlySet<string>;
  /** Containers shallower than this always show their children, open or not. */
  revealDepth: number;
  /** The opened cluster. Its subtree and its ancestry stay lit; the rest fades
   *  back, so "where am I" is answerable without reading a single label. */
  focus: string | null;
  dark: boolean;
}

const SPRITE_RADIUS = 32;
const spriteCache = new Map<string, HTMLCanvasElement>();

/** A radial-gradient halo, rasterised once per colour. */
function glowSprite(color: string): HTMLCanvasElement {
  const hit = spriteCache.get(color);
  if (hit) return hit;

  const size = SPRITE_RADIUS * 2;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(
    SPRITE_RADIUS, SPRITE_RADIUS, 0, SPRITE_RADIUS, SPRITE_RADIUS, SPRITE_RADIUS,
  );
  grad.addColorStop(0, hexToRgba(color, 0.55));
  grad.addColorStop(0.35, hexToRgba(color, 0.18));
  grad.addColorStop(1, hexToRgba(color, 0));
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);

  spriteCache.set(color, c);
  return c;
}

export function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.replace(/./g, (ch) => ch + ch) : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

/** The set to keep bright when hovering: the node, its ancestors, its children. */
function neighbourhood(hover: Placed | null, placed: Placed[]): Set<Placed> | null {
  if (!hover) return null;
  const keep = new Set<Placed>();
  for (let p: Placed | null = hover; p; p = p.parent) keep.add(p);
  for (const p of placed) if (p.parent && keep.has(p.parent)) keep.add(p);
  return keep;
}

/**
 * The set to keep bright while a cluster is open: everything inside it, plus
 * the spine of ancestors back to the centre.
 *
 * Matched on the path string with a trailing separator, never `startsWith(focus)`
 * alone — the archive really does contain `Dagshai 2001` and `Dagshai 2001` is a
 * prefix of nothing else here, but `Live Masters 2010` and `Live Masters 2010 B`
 * are one rename apart, and the bug would be invisible.
 */
function subtree(focus: string | null, placed: Placed[]): Set<Placed> | null {
  if (!focus) return null;
  const prefix = focus + "/";
  const keep = new Set<Placed>();
  for (const p of placed) {
    const path = p.node.path;
    if (p.node.depth === 0 || path === focus || path.startsWith(prefix) ||
        focus.startsWith(path + "/")) {
      keep.add(p);
    }
  }
  return keep;
}

function isVisible(p: Placed, v: Viewport, margin: number): boolean {
  const s = toScreen(p, v);
  return s.x > -margin && s.x < v.width + margin && s.y > -margin && s.y < v.height + margin;
}

/** A container is drawn open when its children are on screen — which is exactly
 *  when `visibleNodes` reveals them. Keep the two rules spelled the same way. */
function isOpen(p: Placed, open: ReadonlySet<string>, revealDepth: number): boolean {
  return !p.node.is_leaf && (p.node.depth < revealDepth || open.has(p.node.path));
}

export function draw(ctx: CanvasRenderingContext2D, st: DrawState, dpr: number): void {
  const { placed, view, hover, selected, open, revealDepth, focus, dark } = st;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, view.width, view.height);
  ctx.fillStyle = dark ? "#12100c" : "#f5eee2";
  ctx.fillRect(0, 0, view.width, view.height);

  // Hover wins over focus: while the pointer is on a dot, the question being
  // asked is "what is this and what is it attached to", not "where am I".
  const keep = neighbourhood(hover, placed) ?? subtree(focus, placed);
  const dimmed = (p: Placed): boolean => !!keep && !keep.has(p);
  const visible = placed.filter((p) => isVisible(p, view, 64));

  // ---- links, beneath everything ------------------------------------------
  ctx.lineWidth = Math.max(0.4, 0.9 * view.scale);
  for (const p of visible) {
    if (!p.parent) continue;
    const a = toScreen(p.parent, view);
    const b = toScreen(p, view);
    const faded = dimmed(p) && dimmed(p.parent);
    ctx.strokeStyle = dark
      ? `rgba(196, 170, 120, ${faded ? 0.04 : 0.16})`
      : `rgba(90, 70, 40, ${faded ? 0.05 : 0.18})`;

    // Bow the edge toward the centre of the disc. Straight radial spokes read
    // as a wheel; a slight curve reads as growth.
    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2;
    const centre = toScreen({ x: 0, y: 0 }, view);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.quadraticCurveTo(mx + (centre.x - mx) * 0.16, my + (centre.y - my) * 0.16, b.x, b.y);
    ctx.stroke();
  }

  // ---- halos ---------------------------------------------------------------
  // Culled on the *drawn halo size*, not on the dot's radius. Zoomed out to the
  // whole archive a dot is sub-pixel but its halo is still several pixels wide —
  // and those overlapping halos are the glow that makes the disc read as one
  // living thing rather than a scatter of dust. Culling on dot radius would
  // delete exactly the picture we came for.
  for (const p of visible) {
    if (dimmed(p)) continue;
    // `emphasised`, not `focus`: `focus` is now the opened cluster's path, and a
    // shadowed name here would be correct only by accident.
    const emphasised = p === hover || p === selected;
    const rr = Math.max(p.r * view.scale, 0.6);
    const size = rr * (emphasised ? 7 : 4.6);
    if (size < 2.5 && !emphasised) continue;

    const state = p.node.depth === 0 ? "remembered" : nodeState(p.node);
    const sprite = glowSprite(STATE_COLOR[state as CState]);
    const s = toScreen(p, view);
    ctx.globalAlpha = emphasised ? 1 : 0.25 + litRatio(p.node) * 0.55;
    ctx.drawImage(sprite, s.x - size / 2, s.y - size / 2, size, size);
  }
  ctx.globalAlpha = 1;

  // ---- dots ----------------------------------------------------------------
  for (const p of visible) {
    const s = toScreen(p, view);
    const rr = Math.max(1.1, p.r * view.scale);
    const state = p.node.depth === 0 ? "remembered" : nodeState(p.node);
    const colour = STATE_COLOR[state as CState];

    // Containers fade with the share of their contents that is dated, so a
    // half-remembered camp looks half-remembered rather than snapping to a
    // single colour. Leaves are one thing or the other.
    const lit = p.node.is_leaf ? 1 : 0.42 + litRatio(p.node) * 0.58;
    ctx.globalAlpha = dimmed(p) ? 0.1 : lit;

    ctx.beginPath();
    ctx.arc(s.x, s.y, rr, 0, Math.PI * 2);
    ctx.fillStyle = colour;
    ctx.fill();

    if (p === selected || p === hover) {
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2;
      ctx.strokeStyle = dark ? "#fff7ea" : "#221d14";
      ctx.stroke();
    } else if (!p.node.is_leaf && !dimmed(p)) {
      if (isOpen(p, open, revealDepth)) {
        // Open: the dot itself is ringed. You are looking at its contents.
        ctx.globalAlpha = 0.7;
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = dark ? "rgba(255,247,234,0.55)" : "rgba(34,29,20,0.5)";
        ctx.stroke();
      } else if (rr > 3) {
        // Closed: a ring held off the dot — "there is more inside". Without it a
        // collapsed collection is indistinguishable from a childless one, and the
        // reader has no way to know the map has anything left to show them.
        ctx.globalAlpha = 0.32;
        ctx.lineWidth = 1;
        ctx.strokeStyle = dark ? "rgba(255,247,234,0.5)" : "rgba(34,29,20,0.45)";
        ctx.beginPath();
        ctx.arc(s.x, s.y, rr + 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }
  ctx.globalAlpha = 1;

  // ---- labels, last, and only where they can be read -----------------------
  drawLabels(ctx, visible, view, hover, selected, keep, dark);
}

function drawLabels(
  ctx: CanvasRenderingContext2D,
  visible: Placed[],
  view: Viewport,
  hover: Placed | null,
  selected: Placed | null,
  keep: Set<Placed> | null,
  dark: boolean,
): void {
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for (const p of visible) {
    const rr = p.r * view.scale;
    const isFocus = p === hover || p === selected;
    // Collections are always named. There are a few dozen of them, they are the
    // whole of the first view, and an unlabelled disc of dots is a picture of an
    // archive rather than a way into one. Below that a label is worth pixels when
    // the dot is big, when the ring is sparse and zoomed in, or when the pointer
    // is on it. Otherwise it is noise.
    const worth = isFocus || p.node.depth <= 1 || rr > 9 ||
      (p.node.depth === 2 && view.scale > 0.42);
    if (!worth) continue;
    if (keep && !keep.has(p) && !isFocus) continue;

    const s = toScreen(p, view);
    const size = isFocus ? 13 : p.node.depth <= 1 ? 13 : 11;
    ctx.font = `${p.node.depth <= 1 || isFocus ? 600 : 400} ${size}px Inter, system-ui, sans-serif`;

    const text = truncate(labelFor(p.node), p.node.depth <= 1 ? 28 : 22);
    const y = s.y - rr - 7;

    ctx.lineWidth = 3;
    ctx.strokeStyle = dark ? "rgba(18,16,12,0.85)" : "rgba(245,238,226,0.9)";
    ctx.strokeText(text, s.x, y);
    ctx.fillStyle = isFocus
      ? (dark ? "#fff7ea" : "#100b04")
      : (dark ? "rgba(238,226,204,0.82)" : "rgba(34,29,20,0.8)");
    ctx.fillText(text, s.x, y);
  }
}

function labelFor(n: CorpusNode): string {
  return n.is_leaf ? n.name.replace(/\.json$/, "") : n.name;
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : s.slice(0, max - 1) + "…";
}
