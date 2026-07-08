import { useCallback, useEffect, useState } from "react";

import { ApiError, clearPassword, getFilters, getHealth, getPassword } from "./api";
import type { Health } from "./types";
import { Sidebar } from "./components/Sidebar";
import { Login } from "./components/Login";
import { SearchView } from "./components/SearchView";
import { AnalyticsView } from "./components/AnalyticsView";
import { BrainView } from "./components/BrainView";

type Tab = "search" | "analytics" | "archive";

const COLLAPSE_KEY = "vishvas.sidebar.collapsed";

/**
 * Root component. Owns the health probe, auth gate, sidebar collapse state,
 * the currently-open conversation id, and a monotonic `historyVersion` that
 * the sidebar watches to refetch the Recent list after saves / deletes.
 */
export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [authed, setAuthed] = useState(false);
  const [checkingAuth, setCheckingAuth] = useState(true);
  const [tab, setTab] = useState<Tab>("search");
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(() => {
    try { return localStorage.getItem(COLLAPSE_KEY) === "1"; } catch { return false; }
  });
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [historyVersion, setHistoryVersion] = useState(0);
  // Bumped whenever the user explicitly starts a new chat (sidebar "New chat"
  // button or deleting the conversation they were viewing). SearchView keys
  // off it to wipe its state cleanly without race-prone effects.
  const [chatSessionKey, setChatSessionKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const h = await getHealth();
        if (cancelled) return;
        setHealth(h);
        if (!h.auth_required) {
          setAuthed(true);
          setCheckingAuth(false);
          return;
        }
        if (getPassword()) {
          try {
            await getFilters();
            if (!cancelled) setAuthed(true);
          } catch (e) {
            if (e instanceof ApiError && e.status === 401) clearPassword();
          }
        }
      } catch (e) {
        if (!cancelled) {
          setHealthError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (!cancelled) setCheckingAuth(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleAuthFail = useCallback(() => {
    clearPassword();
    setAuthed(false);
  }, []);

  const handleLogout = useCallback(() => {
    clearPassword();
    setAuthed(false);
  }, []);

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((c) => {
      const next = !c;
      try { localStorage.setItem(COLLAPSE_KEY, next ? "1" : "0"); } catch { /* ignore */ }
      return next;
    });
  }, []);

  const bumpHistory = useCallback(() => {
    setHistoryVersion((v) => v + 1);
  }, []);

  const onNewChat = useCallback(() => {
    setActiveConversationId(null);
    setChatSessionKey((k) => k + 1);
    setTab("search");
  }, []);

  if (checkingAuth) {
    return <div className="app-status">Connecting…</div>;
  }

  if (healthError) {
    return (
      <div className="app-status app-status--error">
        <p>Cannot reach the backend.</p>
        <code>{healthError}</code>
        <p className="muted">
          Check that the API is running and <code>VITE_API_BASE</code> points
          at it.
        </p>
      </div>
    );
  }

  if (health?.auth_required && !authed) {
    return <Login onAuthed={() => setAuthed(true)} />;
  }

  return (
    <div className="app">
      <Sidebar
        health={health}
        tab={tab}
        onTab={setTab}
        canLogout={!!health?.auth_required}
        onLogout={handleLogout}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={toggleSidebar}
        activeConversationId={activeConversationId}
        onSelectConversation={setActiveConversationId}
        onNewChat={onNewChat}
        historyVersion={historyVersion}
        onAuthFail={handleAuthFail}
      />
      <main className="app-main">
        {tab === "search" && (
          <SearchView
            key={chatSessionKey}
            onAuthFail={handleAuthFail}
            activeConversationId={activeConversationId}
            onActiveConversationChange={setActiveConversationId}
            onConversationSaved={bumpHistory}
          />
        )}
        {tab === "analytics" && <AnalyticsView onAuthFail={handleAuthFail} />}
        {tab === "archive" && <BrainView onAuthFail={handleAuthFail} />}
      </main>
    </div>
  );
}
