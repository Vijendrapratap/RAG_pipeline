import { useEffect, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";

import { ApiError, deleteHistoryItem, listHistory } from "../api";
import type { ConversationSummary, Health } from "../types";
import { BrandMark } from "./BrandMark";

type Tab = "search" | "analytics";

interface Props {
  health: Health | null;
  tab: Tab;
  onTab: (t: Tab) => void;
  canLogout: boolean;
  onLogout: () => void;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  activeConversationId: string | null;
  onSelectConversation: (id: string | null) => void;
  /** Starts a fresh chat (clears active id + remounts SearchView). */
  onNewChat: () => void;
  /** Bumped by App after a turn saves, so the sidebar refetches. */
  historyVersion: number;
  onAuthFail: () => void;
}

/**
 * One-line health summary. Green dot + "All systems online" when everything's
 * up; amber + count when something is degraded. Detailed per-service dots are
 * shown on hover via the tooltip — keeps the footer clean for non-tech users.
 */
function HealthSummary({ health }: { health: Health }) {
  const svc = health.services;
  const states = [
    { name: "Answer model", up: svc.ollama },
    { name: "Search index", up: svc.qdrant },
    { name: "Reranker", up: svc.reranker },
  ];
  const down = states.filter((s) => !s.up);
  const bm25Off = !health.bm25_enabled;
  const allOk = down.length === 0 && !bm25Off;
  const tooltip = states.map((s) => `${s.name}: ${s.up ? "online" : "offline"}`).join("\n")
    + (bm25Off ? "\nKeyword index: not loaded" : "");
  return (
    <span className="health" title={tooltip}>
      <span className={`health-dot ${allOk ? "health-dot--ok" : "health-dot--warn"}`} />
      {allOk
        ? "All systems online"
        : down.length > 0
          ? `${down.length} service${down.length > 1 ? "s" : ""} offline`
          : "Keyword search unavailable"}
    </span>
  );
}

function ChatIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function ChartIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="18" y1="20" x2="18" y2="10" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6"  y1="20" x2="6"  y2="14" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5"  y1="12" x2="19" y2="12" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}

function CollapseIcon({ collapsed }: { collapsed: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
         style={{ transform: collapsed ? "rotate(180deg)" : undefined, transition: "transform 150ms ease" }}>
      <polyline points="15 18 9 12 15 6" />
    </svg>
  );
}

/**
 * Left sidebar: brand, "New chat" + tab navigation, conversation history
 * grouped by recency, upstream-service status, sign-out.
 *
 * History is refetched whenever `historyVersion` bumps — App increments it
 * after a turn saves or a deletion happens, instead of doing optimistic
 * client-side updates.
 */
export function Sidebar({
  health, tab, onTab, canLogout, onLogout, collapsed, onToggleCollapsed,
  activeConversationId, onSelectConversation, onNewChat,
  historyVersion, onAuthFail,
}: Props) {
  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await listHistory();
        if (!cancelled) {
          setItems(r.items);
          setHistoryError(null);
        }
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 401) {
          onAuthFail();
          return;
        }
        setHistoryError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { cancelled = true; };
  }, [historyVersion, onAuthFail]);

  async function onDelete(id: string, e: ReactMouseEvent) {
    e.stopPropagation();
    if (!window.confirm("Delete this conversation?")) return;
    try {
      await deleteHistoryItem(id);
      setItems((prev) => prev.filter((c) => c.id !== id));
      if (activeConversationId === id) onNewChat();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) onAuthFail();
      else setHistoryError(err instanceof Error ? err.message : String(err));
    }
  }

  const groups = groupByRecency(items);

  return (
    <aside className={"sidebar" + (collapsed ? " sidebar--collapsed" : "")}>
      <div className="sidebar-head">
        <span className="brand">
          <BrandMark className="brand-mark brand-mark--logo" decorative />
          <span className="brand-text">
            <span className="brand-name">Vishvas Foundation</span>
            <span className="brand-sub">Discourse Archive</span>
          </span>
        </span>
        <button
          className="sidebar-toggle"
          onClick={onToggleCollapsed}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <CollapseIcon collapsed={collapsed} />
        </button>
      </div>

      <div className="sidebar-newchat">
        <button
          className="newchat-btn"
          onClick={onNewChat}
          title="New chat"
        >
          <PlusIcon />
          <span>New chat</span>
        </button>
      </div>

      <nav className="sidebar-nav" aria-label="Primary">
        <button
          className={"nav-item" + (tab === "search" && !activeConversationId ? " nav-item--active" : "")}
          onClick={() => { onTab("search"); }}
          title="Ask"
        >
          <span className="nav-item-icon"><ChatIcon /></span>
          <span>Ask</span>
        </button>
        <button
          className={"nav-item" + (tab === "analytics" ? " nav-item--active" : "")}
          onClick={() => onTab("analytics")}
          title="Analytics"
        >
          <span className="nav-item-icon"><ChartIcon /></span>
          <span>Analytics</span>
        </button>
      </nav>

      <div className="sidebar-history">
        <div className="sidebar-section-title">Recent</div>
        {historyError && (
          <div className="sidebar-history-error" title={historyError}>
            History unavailable
          </div>
        )}
        {!historyError && items.length === 0 && (
          <div className="sidebar-history-empty">No chats yet</div>
        )}
        {groups.map((g) => (
          <div className="history-group" key={g.label}>
            <div className="history-group-label">{g.label}</div>
            {g.items.map((c) => (
              <button
                key={c.id}
                className={"history-item" + (c.id === activeConversationId ? " history-item--active" : "")}
                onClick={() => {
                  onTab("search");
                  onSelectConversation(c.id);
                }}
                title={c.title}
              >
                <span className="history-item-title">{c.title}</span>
                <span
                  className="history-item-del"
                  onClick={(e) => onDelete(c.id, e)}
                  role="button"
                  aria-label="Delete conversation"
                  title="Delete"
                >
                  <TrashIcon />
                </span>
              </button>
            ))}
          </div>
        ))}
      </div>

      <div className="sidebar-foot">
        {health && <HealthSummary health={health} />}
        {canLogout && (
          <button className="sidebar-signout" onClick={onLogout}>
            Sign out
          </button>
        )}
      </div>
    </aside>
  );
}

interface Group {
  label: string;
  items: ConversationSummary[];
}

/** Bucket conversations by created_at into Claude-style date groups. */
function groupByRecency(items: ConversationSummary[]): Group[] {
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const dayMs = 24 * 60 * 60 * 1000;
  const startOfYesterday = startOfToday - dayMs;
  const startOfWeek = startOfToday - 6 * dayMs;
  const startOfMonth = startOfToday - 29 * dayMs;

  const buckets: Group[] = [
    { label: "Today", items: [] },
    { label: "Yesterday", items: [] },
    { label: "Previous 7 days", items: [] },
    { label: "Previous 30 days", items: [] },
    { label: "Older", items: [] },
  ];

  for (const c of items) {
    const t = Date.parse(c.created_at);
    if (Number.isNaN(t)) { buckets[4].items.push(c); continue; }
    if (t >= startOfToday) buckets[0].items.push(c);
    else if (t >= startOfYesterday) buckets[1].items.push(c);
    else if (t >= startOfWeek) buckets[2].items.push(c);
    else if (t >= startOfMonth) buckets[3].items.push(c);
    else buckets[4].items.push(c);
  }

  return buckets.filter((g) => g.items.length > 0);
}
