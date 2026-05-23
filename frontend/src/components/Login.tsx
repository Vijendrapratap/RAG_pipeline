import { useState } from "react";
import type { FormEvent } from "react";

import { ApiError, getFilters, setPassword } from "../api";

/**
 * Shared-password gate. Shown only when /api/health reports
 * `auth_required: true`. The entered password is stored, then validated by
 * calling an authed endpoint — a 401 means it is wrong.
 */
export function Login({ onAuthed }: { onAuthed: () => void }) {
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!value || busy) return;
    setBusy(true);
    setError(null);
    setPassword(value);
    try {
      await getFilters(); // any authed endpoint — validates the password
      onAuthed();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setError("Incorrect password.");
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login">
      <form className="login-card" onSubmit={submit}>
        <h1>Transcript RAG</h1>
        <p className="muted">Enter the dashboard password to continue.</p>
        <input
          type="password"
          value={value}
          autoFocus
          placeholder="Password"
          onChange={(e) => setValue(e.target.value)}
        />
        {error && <div className="login-error">{error}</div>}
        <button type="submit" disabled={busy || !value}>
          {busy ? "Checking…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
