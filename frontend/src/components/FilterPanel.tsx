import { FACETS, countActive } from "../filters";
import type { Filters, FilterOptions } from "../types";

interface Props {
  filters: Filters;
  options: Partial<FilterOptions> | null;
  dbOk: boolean;
  onChange: (next: Filters) => void;
}

/**
 * Sidebar with one dropdown per file_meta facet, plus a date-range picker
 * and a free-text speaker field. List-typed fields (track_type, topics, …)
 * still send a one-element array — that is what the backend expects.
 */
export function FilterPanel({ filters, options, dbOk, onChange }: Props) {
  const active = countActive(filters);

  function setScalar(field: keyof Filters, value: string) {
    onChange({ ...filters, [field]: value || null });
  }
  function setList(field: keyof Filters, value: string) {
    onChange({ ...filters, [field]: value ? [value] : null });
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
    <aside className="filters">
      <div className="filters-head">
        <span>Filters</span>
        {active > 0 && (
          <button className="link-btn" onClick={() => onChange({})}>
            Clear ({active})
          </button>
        )}
      </div>

      {!dbOk && (
        <div className="filters-warn">
          Filter options unavailable — Postgres unreachable. You can still
          type a query.
        </div>
      )}

      <label className="field">
        <span>Speaker</span>
        <input
          type="text"
          value={filters.speaker ?? ""}
          placeholder="e.g. Swami Ji"
          onChange={(e) =>
            onChange({ ...filters, speaker: e.target.value || null })
          }
        />
      </label>

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

      <div className="field field--range">
        <span>Date range</span>
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
    </aside>
  );
}
