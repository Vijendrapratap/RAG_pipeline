# Vishvas Foundation — Desktop shell

A thin Electron wrapper that lets a non-technical client launch the whole
local RAG stack and use the archive without ever touching Docker or a terminal.

It does **not** bundle the frontend — the `rag-api` container already serves the
built React UI at `/`. This app just brings the stack up and opens a window onto
`http://localhost:<port>`.

## Lifecycle: "warm after first open"

- **First open of a session** (e.g. just after a reboot): a branded splash shows
  while it starts Docker Desktop (if the daemon is down), runs
  `docker compose up -d`, waits for `/api/health`, and pre-warms the answer model.
- **Every open after that**: a fast health probe passes, so the window opens
  immediately — no splash, no wait.
- **Closing the window** hides it to the system tray; the stack stays warm.
- The tray menu offers **Open**, live **status**, **Restart services**,
  **Quit (keep services warm)**, and **Quit & stop services** (the only path
  that tears the stack down / frees VRAM).

Because the containers use `restart: unless-stopped`, once they've been started
they survive Docker restarts on their own — this app only needs to do the work
on a genuine cold start.

## Run in dev

```powershell
cd desktop
npm install
npm start
```

With no `desktop.config.json` present, it targets the repo root (one level up)
for `docker compose`, and probes ports 8080 then 8081 (the dev override) for the
UI — so it works against your existing stack as-is.

## Configure

Copy `desktop.config.example.json` → `desktop.config.json` (gitignored) and edit,
or set env vars. Precedence: `VISHVAS_*` env > `desktop.config.json` > defaults.

| Key | Env | Default | Meaning |
|---|---|---|---|
| `projectDir` | `VISHVAS_PROJECT_DIR` | repo root | folder with `docker-compose.yml` + `.env` |
| `appPort` | `VISHVAS_APP_PORT` | `8080` | canonical rag-api port |
| `probePorts` | — | `[8080, 8081]` | ports probed for a reachable UI |
| `dockerDesktop` | `VISHVAS_DOCKER_DESKTOP` | `C:\Program Files\Docker\Docker\Docker Desktop.exe` | started if daemon is down |
| `chatModel` | `VISHVAS_CHAT_MODEL` | `qwen3.5:9b` | model to pre-warm |

## Package an installer

1. Add `assets/icon.ico` (see `assets/README.md`).
2. Build:
   ```powershell
   cd desktop
   npm install
   npm run dist
   ```
   Output: `desktop/dist/Vishvas Foundation Setup <version>.exe` (NSIS installer
   with Start-menu + desktop shortcuts).

> The packaged app needs the compose files + `.env` + `data/` on the client
> machine and `projectDir` in `desktop.config.json` pointing at them. The
> installer should place those (or the client copies the project folder), since
> the stack itself is what does the real work — this app only orchestrates it.

## First-run note

The very first `docker compose up -d` pulls images and the ollama model, so the
first launch on a fresh machine is slow (minutes). Every launch after that is
warm. The splash communicates this while it works.
