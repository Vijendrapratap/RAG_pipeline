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
        <div className="login-brand">
          <img className="brand-mark brand-mark--lg brand-mark--logo" src="/logo.png" alt="Vishvas Foundation" />
          <h1>Vishvas Foundation</h1>
          <span className="login-sub">Discourse Archive</span>
        </div>
        <p>Enter the archive password to continue.</p>
        <label className="login-field">
          <span>Password</span>
          <input
            type="password"
            value={value}
            autoFocus
            placeholder="••••••••"
            onChange={(e) => setValue(e.target.value)}
          />
        </label>
        {error && <div className="login-error">{error}</div>}
        <button type="submit" disabled={busy || !value}>
          {busy ? "Checking…" : "Enter archive"}
        </button>
      </form>
    </div>
  );
}
