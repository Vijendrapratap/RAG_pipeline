import { useEffect } from "react";

import { FACETS, countActive } from "../filters";
import type { Filters, FilterOptions } from "../types";

interface Props {
  filters: Filters;
  options: Partial<FilterOptions> | null;
  dbOk: boolean;
  onChange: (next: Filters) => void;
  onClose: () => void;
}

/**
 * Slide-in filter drawer. One dropdown per file_meta facet, plus a date-range
 * picker. List-typed fields still send a one-element array — that is what the
 * backend expects. (No speaker filter: the corpus is a single voice, Guruji.)
 */
export function FilterPanel({ filters, options, dbOk, onChange, onClose }: Props) {
  const active = countActive(filters);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  function setScalar(field: keyof Filters, value: string) {
    onChange({ ...filters, [field]: value || null });
  }
  function setList(field: keyof Filters, value: string) {
    onChange({ ...filters, [field]: value ? [value] : null });
  }
  // Year range (coarse) — drives the same date_range as the precise picker
  // below, so the two never conflict. From/To are forgiving of order.
  const years = (options?.years ?? []) as string[];
  const dr = filters.date_range;
  const yearFrom = dr?.[0]?.slice(0, 4) ?? "";
  const yearTo = dr?.[1]?.slice(0, 4) ?? "";
  function setYear(which: "from" | "to", y: string) {
    const f = which === "from" ? y : yearFrom;
    const t = which === "to" ? y : yearTo;
    if (!f && !t) {
      onChange({ ...filters, date_range: null });
      return;
    }
    const lo = f || t;
    const hi = t || f;
    const [a, b] = Number(lo) <= Number(hi) ? [lo, hi] : [hi, lo];
    onChange({ ...filters, date_range: [`${a}-01-01`, `${b}-12-31`] });
  }

  function setDate(idx: 0 | 1, value: string) {
    const current = filters.date_range ?? ["", ""];
    const next: [string, string] = [current[0], current[1]];
    next[idx] = value;
    onChange({
      ...filters,
      date_range: next[0] && next[1] ? next : null,
    });
  }

  return (
    <>
      <div className="filter-drawer-backdrop" onClick={onClose} />
      <aside className="filter-drawer" role="dialog" aria-label="Filters">
        <header className="filter-drawer-head">
          <h3>Filters</h3>
          <button className="btn btn--ghost" onClick={onClose} aria-label="Close filters">
            ✕
          </button>
        </header>

        <div className="filter-drawer-body">
          {!dbOk && (
            <div className="filters-warn">
              Filter options unavailable — Postgres unreachable. You can still
              type a query.
            </div>
          )}

          {FACETS.map((f) => {
            const opts = (options?.[f.optionKey] ?? []) as string[];
            if (opts.length === 0) return null;
            const raw = filters[f.field];
            const current = Array.isArray(raw) ? raw[0] ?? "" : (raw ?? "") as string;
            return (
              <label className="field" key={f.field}>
                <span>{f.label}</span>
                <select
                  value={current}
                  onChange={(e) =>
                    f.isList
                      ? setList(f.field, e.target.value)
                      : setScalar(f.field, e.target.value)
                  }
                >
                  <option value="">(any)</option>
                  {opts.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
            );
          })}

          {years.length > 0 && (
            <div className="field field--range">
              <span>Year range</span>
              <div className="range-inputs">
                <select value={yearFrom} onChange={(e) => setYear("from", e.target.value)}>
                  <option value="">(any)</option>
                  {years.map((y) => (
                    <option key={y} value={y}>{y}</option>
                  ))}
                </select>
                <select value={yearTo} onChange={(e) => setYear("to", e.target.value)}>
                  <option value="">(any)</option>
                  {years.map((y) => (
                    <option key={y} value={y}>{y}</option>
                  ))}
                </select>
              </div>
            </div>
          )}

          <div className="field field--range">
            <span>Date range (precise)</span>
            <div className="range-inputs">
              <input
                type="date"
                value={filters.date_range?.[0] ?? ""}
                onChange={(e) => setDate(0, e.target.value)}
              />
              <input
                type="date"
                value={filters.date_range?.[1] ?? ""}
                onChange={(e) => setDate(1, e.target.value)}
              />
            </div>
          </div>
        </div>

        <footer className="filter-drawer-foot">
          <button
            className="btn"
            onClick={() => onChange({})}
            disabled={active === 0}
          >
            Clear {active > 0 ? `(${active})` : ""}
          </button>
          <button className="btn btn--primary" onClick={onClose}>
            Done
          </button>
        </footer>
      </aside>
    </>
  );
}
