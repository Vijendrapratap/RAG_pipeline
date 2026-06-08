/**
 * Filter helpers shared by the filter panel and the detected-filter chips.
 *
 * The dashboard offers one single-select dropdown per file_meta facet. List-
 * typed filter fields (track_type, topics, …) still send an array — a
 * one-element array from a single pick — because that is what the backend
 * FilterModel expects.
 */
import type { Detection, FilterOptions, Filters } from "./types";

export interface Facet {
  /** Key in the /api/filters options payload. */
  optionKey: keyof FilterOptions;
  /** Field name in the request FilterModel. */
  field: keyof Filters;
  label: string;
  /** True when the backend field is a list (value must be wrapped). */
  isList: boolean;
}

/** Facets that have a dropdown, in display order. */
export const FACETS: Facet[] = [
  { optionKey: "seasons", field: "season", label: "Season", isList: false },
  { optionKey: "locations", field: "location", label: "Location", isList: false },
  { optionKey: "event_types", field: "event_type", label: "Event type", isList: false },
  { optionKey: "track_types", field: "track_type", label: "Track type", isList: true },
  { optionKey: "primary_languages", field: "primary_language", label: "Language", isList: false },
  { optionKey: "event_ids", field: "event_id", label: "Event", isList: false },
  { optionKey: "topics", field: "topics", label: "Topic", isList: true },
  { optionKey: "scriptures_referenced", field: "scriptures_referenced", label: "Scripture", isList: true },
  { optionKey: "performers", field: "performers", label: "Performer", isList: true },
];

/** Filter fields whose value is a list in the backend FilterModel. */
export const LIST_FIELDS = new Set<keyof Filters>([
  "track_type",
  "topics",
  "scriptures_referenced",
  "people_named",
  "performers",
]);

/** Drop null / empty-string / empty-array entries — what is actually sent. */
export function cleanFilters(f: Filters): Filters {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(f)) {
    if (v == null) continue;
    if (Array.isArray(v) && v.length === 0) continue;
    if (typeof v === "string" && v.trim() === "") continue;
    out[k] = v;
  }
  return out as Filters;
}

export function countActive(f: Filters): number {
  return Object.keys(cleanFilters(f)).length;
}

/** Merge a detected signal into the active filter set (used by soft chips). */
export function applyDetection(f: Filters, d: Detection): Filters {
  const next: Filters = { ...f };
  const field = d.field as keyof Filters;
  if (field === "date_range") {
    next.date_range = d.value as [string, string];
  } else if (LIST_FIELDS.has(field)) {
    const current = (next[field] as string[] | undefined) ?? [];
    const value = String(d.value);
    if (!current.includes(value)) {
      (next[field] as string[]) = [...current, value];
    }
  } else {
    (next[field] as string) = String(d.value);
  }
  return next;
}

/** Is this detection's value already present in the active filters? */
export function detectionActive(f: Filters, d: Detection): boolean {
  const field = d.field as keyof Filters;
  const current = f[field];
  if (current == null) return false;
  if (field === "date_range") {
    return JSON.stringify(current) === JSON.stringify(d.value);
  }
  if (Array.isArray(current)) return current.includes(String(d.value));
  return current === d.value;
}

/** Human-readable rendering of a filter value for chips / summaries. */
export function formatValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}
