import { useState } from "react";
import type { ReactNode } from "react";

import type { RetrievalResult } from "../types";

interface Props {
  result: RetrievalResult;
  /** 1-based citation number shown on the card. */
  index?: number;
  /** Scroll-anchor namespace. When set (AnswerPane passes its useId), the card
   *  gets DOM id `{prefix}-cite-{index}` so [N] clicks resolve within their own
   *  turn — turns all number 1..N, so an unprefixed id collides across turns. */
  anchorPrefix?: string;
  /** Highlight pulse, e.g. when a [N] citation marker was clicked. */
  highlight?: boolean;
  /** The user's query. Its content terms are highlighted inside the passage so
   *  the reader can see *why* this source matched. */
  query?: string;
}

// Metadata worth surfacing as a clean, labelled line — the same fields the
// synthesis prompt treats as ground truth (rag_api/synthesis.build_context_block),
// plus the curated-catalog columns. Everything else stays behind "+N more" so
// the footer reads as a caption instead of a raw key dump.
const KNOWN_META: ReadonlyArray<readonly [string, string]> = [
  ["event_id", "Event"],
  ["session_date", "Date"],
  ["session_time", "Time"],
  ["track_title", "Track"],
  ["track_type", "Type"],
  ["location", "Location"],
  ["camp_place", "Place"],
  ["season", "Season"],
  ["year", "Year"],
  ["performers", "Performers"],
];
const KNOWN_KEYS = new Set(KNOWN_META.map(([k]) => k));

// Query words that carry no topical signal — highlighting them would just paint
// the whole passage. Covers the common English + Hindi (Devanagari) function
// words and question words that show up in nearly every query.
const STOP_WORDS = new Set<string>([
  // English
  "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to",
  "of", "in", "on", "at", "for", "and", "or", "but", "if", "so", "as", "by",
  "it", "its", "this", "that", "these", "those", "with", "from", "about",
  "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
  "did", "does", "do", "say", "said", "tell", "told", "me", "i", "you", "he",
  "she", "they", "we", "his", "her", "their", "our", "ji",
  // Hindi
  "क्या", "है", "हैं", "था", "थे", "थी", "की", "के", "का", "को", "में", "से",
  "और", "या", "पर", "यह", "वह", "कौन", "कब", "कहाँ", "कहां", "क्यों", "कैसे",
  "ने", "हो", "कर", "किया", "बारे", "बताओ", "बताइए", "मुझे", "एक",
]);

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Distinct content terms from the query, stop-words and 1-char tokens dropped. */
function queryTerms(query: string): string[] {
  const tokens = query.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  return Array.from(
    new Set(tokens.filter((t) => t.length >= 2 && !STOP_WORDS.has(t))),
  );
}

/** One alternation regex (longest term first) or null when nothing to match. */
function buildHighlightRe(terms: string[]): RegExp | null {
  if (terms.length === 0) return null;
  const alts = terms
    .slice()
    .sort((a, b) => b.length - a.length)
    .map(escapeRegExp);
  return new RegExp(`(${alts.join("|")})`, "giu");
}

/** Split Whisper's run-on transcript into readable paragraphs — group a few
 *  sentences at a time on Devanagari (।/॥) and Latin (.?!) terminators. Purely
 *  cosmetic: the stored text is untouched. */
function toParagraphs(text: string): string[] {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return [];
  const sentences = clean.match(/[^।॥.!?]+[।॥.!?]+|[^।॥.!?]+$/g) ?? [clean];
  const perPara = 3;
  const out: string[] = [];
  for (let i = 0; i < sentences.length; i += perPara) {
    const p = sentences.slice(i, i + perPara).join(" ").trim();
    if (p) out.push(p);
  }
  return out.length ? out : [clean];
}

/** Wrap query-term matches in <mark>; a split on the capture group keeps the
 *  original casing and interleaves plain / matched fragments. */
function highlight(text: string, re: RegExp | null, keyBase: string): ReactNode[] {
  if (!re) return [text];
  return text.split(re).map((part, i) =>
    i % 2 === 1 ? (
      <mark key={`${keyBase}-${i}`} className="hl">{part}</mark>
    ) : (
      <span key={`${keyBase}-${i}`}>{part}</span>
    ),
  );
}

function fmtMetaValue(v: unknown): string {
  return Array.isArray(v) ? v.join(", ") : String(v);
}

function hasValue(v: unknown): boolean {
  if (v == null) return false;
  if (Array.isArray(v)) return v.length > 0;
  return String(v).trim().length > 0;
}

function fmtTime(sec: number | null): string | null {
  if (sec == null) return null;
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

// Turn the fused/rerank score into a human relevance label. Buckets follow the
// reranker's ~[0,1] range (a genuine top hit reranks ~0.8-0.95; the citation
// floor in synthesis.filter_min_score is 0.75).
function relevanceOf(score: number): { label: string; cls: string } {
  if (score >= 0.8) return { label: "Strong match", cls: "rel--strong" };
  if (score >= 0.6) return { label: "Good match", cls: "rel--good" };
  if (score >= 0.4) return { label: "Partial match", cls: "rel--partial" };
  return { label: "Weak match", cls: "rel--weak" };
}

/** Render one chunk- or summary-level result. Reused for /api/search hits
 *  and for /api/query citations. */
export function ResultCard({ result, index, anchorPrefix, highlight: flash, query }: Props) {
  const [showAllMeta, setShowAllMeta] = useState(false);

  const start = fmtTime(result.start_sec);
  const end = fmtTime(result.end_sec);
  const tsRange = start && end ? `${start}–${end}` : start ?? "";

  const isCatalog = result.source === "catalog" || result.result_type === "catalog";
  const meta = result.metadata as Record<string, unknown>;
  // For catalog rows the raw source_file is "catalog:KEY" — show the canonical
  // title / place instead so the sheet entry reads cleanly.
  const sourceLabel = isCatalog
    ? String(meta.track_title ?? meta.camp_place ?? meta.location ?? "Curated catalog")
    : (result.source_file ?? "(unknown)");
  const badgeClass = isCatalog
    ? "badge--catalog"
    : result.result_type === "summary" ? "badge--sum" : "badge--chunk";
  const badgeText = isCatalog ? "Catalog · sheet" : result.result_type;

  const rel = relevanceOf(result.score);
  const re = buildHighlightRe(queryTerms(query ?? ""));
  const paras = toParagraphs(result.text);

  const showAlt =
    !!result.summary_hindi && result.summary_hindi !== result.text;
  const altParas = showAlt ? toParagraphs(result.summary_hindi as string) : [];

  const curated = KNOWN_META.filter(([k]) => hasValue(meta[k]));
  const extraKeys = Object.keys(meta).filter(
    (k) => !KNOWN_KEYS.has(k) && k !== "speakers" && hasValue(meta[k]),
  );

  return (
    <article
      className={"card" + (isCatalog ? " card--catalog" : "") + (flash ? " card--flash" : "")}
      id={index !== undefined && anchorPrefix ? `${anchorPrefix}-cite-${index}` : undefined}
    >
      <header className="card-head">
        {index !== undefined && <span className="cite-num">[{index}]</span>}
        <span className="card-source">{sourceLabel}</span>
        <span className={"badge " + badgeClass}>{badgeText}</span>
        {tsRange && <span className="card-time">{tsRange}</span>}
        <span
          className={"rel " + rel.cls}
          title={`fused score ${result.score.toFixed(3)}`}
        >
          {rel.label}
        </span>
      </header>

      <div className="card-text">
        {paras.map((p, i) => (
          <p key={i} className="card-para">{highlight(p, re, `t${i}`)}</p>
        ))}
      </div>

      {altParas.length > 0 && (
        <div className="card-text card-text--alt">
          {altParas.map((p, i) => (
            <p key={i} className="card-para">{highlight(p, re, `a${i}`)}</p>
          ))}
        </div>
      )}

      <footer className="card-foot">
        {result.speakers.length > 0 && (
          <span className="card-speakers">
            {result.speakers.join(", ")}
          </span>
        )}
        {curated.map(([k, label]) => (
          <span key={k} className="meta-pill">
            <em>{label}</em> {fmtMetaValue(meta[k])}
          </span>
        ))}
        {showAllMeta &&
          extraKeys.map((k) => (
            <span key={k} className="meta-pill meta-pill--extra">
              <em>{k}</em> {fmtMetaValue(meta[k])}
            </span>
          ))}
        {extraKeys.length > 0 && (
          <button
            className="meta-more"
            onClick={() => setShowAllMeta((v) => !v)}
          >
            {showAllMeta ? "Show less" : `+${extraKeys.length} more`}
          </button>
        )}
      </footer>
    </article>
  );
}
