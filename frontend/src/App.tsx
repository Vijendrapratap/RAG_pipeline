import { useCallback, useEffect, useState } from "react";

import { ApiError, clearPassword, getFilters, getHealth, getPassword } from "./api";
import type { Health } from "./types";
import { Header } from "./components/Header";
import { Login } from "./components/Login";
import { SearchView } from "./components/SearchView";
import { AnalyticsView } from "./components/AnalyticsView";

type Tab = "search" | "analytics";

/**
 * Root component. Owns the health probe and the auth gate, then hands off to
 * the Search / Analytics views.
 *
 * Auth: the backend reports `auth_required` from /api/health. When it is on,
 * a stored password is validated against an authed endpoint before the app
 * renders; a bad or missing one drops to the Login screen.
 */
export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [authed, setAuthed] = useState(false);
  const [checkingAuth, setCheckingAuth] = useState(true);
  const [tab, setTab] = useState<Tab>("search");

  // Probe health, then resolve the auth state.
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
        // Auth required: validate any stored password silently.
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

  // A 401 from any view means the session is no longer valid — drop to Login.
  const handleAuthFail = useCallback(() => {
    clearPassword();
    setAuthed(false);
  }, []);

  const handleLogout = useCallback(() => {
    clearPassword();
    setAuthed(false);
  }, []);

  if (checkingAuth) {
    return <div className="app-status">Connecting to rag-api…</div>;
  }

  if (healthError) {
    return (
      <div className="app-status app-status--error">
        <p>Cannot reach the rag-api backend.</p>
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
      <Header
        health={health}
        tab={tab}
        onTab={setTab}
        canLogout={!!health?.auth_required}
        onLogout={handleLogout}
      />
      <main className="app-main">
        {tab === "search" ? (
          <SearchView onAuthFail={handleAuthFail} />
        ) : (
          <AnalyticsView onAuthFail={handleAuthFail} />
        )}
      </main>
    </div>
  );
}
