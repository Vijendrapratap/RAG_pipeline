"use strict";

/**
 * Vishvas Foundation desktop shell.
 *
 * Lifecycle = "warm after first open":
 *   - On launch, probe for an already-running UI. If reachable, open the
 *     window immediately (the common case once the stack is warm).
 *   - If nothing answers, show a branded splash and bring the stack up:
 *     start Docker Desktop if the daemon is down, `docker compose up -d`,
 *     wait for /api/health, pre-warm the model, then open the window.
 *   - Closing the window hides it to the tray; the stack stays warm until
 *     reboot. Only "Quit & stop services" tears the stack down.
 */

const { app, BrowserWindow, Tray, Menu, ipcMain, shell, nativeImage } = require("electron");
const { execFile, spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");

const { cfg } = require("./config");

let splashWin = null;
let mainWin = null;
let tray = null;
let resolvedBase = null; // e.g. "http://localhost:8080"
let lastHealth = null;
app.isQuitting = false;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, { windowsHide: true, ...opts }, (err, stdout, stderr) => {
      if (err) {
        err.stdout = stdout;
        err.stderr = stderr;
        reject(err);
      } else {
        resolve({ stdout, stderr });
      }
    });
  });
}

async function waitFor(fn, timeoutMs, everyMs = 2000) {
  const end = Date.now() + timeoutMs;
  let lastErr = null;
  while (Date.now() < end) {
    try {
      const v = await fn();
      if (v) return v;
    } catch (e) {
      lastErr = e;
    }
    await sleep(everyMs);
  }
  throw lastErr || new Error("timed out after " + timeoutMs + "ms");
}

async function fetchWithTimeout(url, opts = {}, timeoutMs = 4000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, { ...opts, signal: ctrl.signal });
  } finally {
    clearTimeout(t);
  }
}

// ── Docker ────────────────────────────────────────────────────────────────

/**
 * Run a shell command INSIDE the WSL distro that owns the data.
 *
 * This is the crux of the empty-mount fix. The override's bind mounts use
 * absolute paths (/home/pc/transcript-rag-data/...) that resolve to the real
 * data ONLY when `docker compose` runs from inside Ubuntu. Invoking `docker`
 * from the Windows side — or letting Docker Desktop auto-start the stack —
 * resolves those same paths inside Docker's own VM, where they are empty.
 */
function runWsl(shellCmd, opts = {}) {
  return run("wsl", ["-d", cfg.wslDistro, "-e", "bash", "-lc", shellCmd], opts);
}

/** `docker compose <subcmd>` from inside the WSL project dir. */
function composeWsl(subcmd) {
  return runWsl(`cd '${cfg.wslProjectDir}' && docker compose ${subcmd}`);
}

/** Daemon AND WSL integration ready — checked from inside Ubuntu, the same
 *  context compose will actually run in (so a half-up daemon doesn't pass). */
async function dockerDaemonReady() {
  try {
    await runWsl("docker info --format '{{.ServerVersion}}'");
    return true;
  } catch {
    return false;
  }
}

function startDockerDesktop() {
  if (!fs.existsSync(cfg.dockerDesktop)) {
    throw new Error("Docker Desktop not found at " + cfg.dockerDesktop);
  }
  const child = spawn(cfg.dockerDesktop, [], { detached: true, stdio: "ignore" });
  child.unref();
}

async function composeUp({ forceRecreate = false } = {}) {
  await composeWsl(`up -d${forceRecreate ? " --force-recreate" : ""}`);
}

async function composeStop() {
  await composeWsl("stop");
}

async function composeRestart() {
  await composeWsl("restart");
}

// ── Health probing ──────────────────────────────────────────────────────────

/** Returns the parsed /api/health body if `base` is reachable, else null. */
async function probeHealth(base, timeoutMs = 3000) {
  try {
    const r = await fetchWithTimeout(base + "/api/health", {}, timeoutMs);
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

/** Empty-mount detector. A correctly-mounted ollama reports its pulled
 *  models; an empty bind mount (the Docker-Desktop auto-start failure mode)
 *  reports zero. This is what separates "warm and real" from "came up blank". */
async function ollamaHasModels() {
  try {
    const r = await fetchWithTimeout(`http://localhost:${cfg.ollamaPort}/api/tags`, {}, 4000);
    if (!r.ok) return false;
    const body = await r.json();
    return Array.isArray(body.models) && body.models.length > 0;
  } catch {
    return false;
  }
}

/** Find the first candidate port whose UI answers; remember it as the base. */
async function resolveReachableBase(timeoutMs = 1500) {
  for (const port of cfg.probePorts) {
    const base = `http://localhost:${port}`;
    const body = await probeHealth(base, timeoutMs);
    if (body) {
      resolvedBase = base;
      lastHealth = body;
      return base;
    }
  }
  return null;
}

// ── Pre-warm ─────────────────────────────────────────────────────────────────

/** Best-effort: load the chat model into VRAM so the first query is fast.
 *  Never throws — a failed warm-up must not block opening the window. */
async function prewarmModel() {
  try {
    await fetchWithTimeout(
      `http://localhost:${cfg.ollamaPort}/api/generate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: cfg.chatModel,
          prompt: "ok",
          stream: false,
          keep_alive: -1,
        }),
      },
      120000,
    );
  } catch {
    // ignore — model loads lazily on the first real query if this missed.
  }
}

// ── Windows ───────────────────────────────────────────────────────────────────

function setSplashStatus(msg) {
  if (splashWin && !splashWin.isDestroyed()) {
    splashWin.webContents.send("status", msg);
  }
}

function createSplash() {
  splashWin = new BrowserWindow({
    width: 460,
    height: 320,
    frame: false,
    resizable: false,
    center: true,
    show: false,
    backgroundColor: "#f9f2e3",
    webPreferences: { preload: path.join(__dirname, "preload.js") },
  });
  splashWin.loadFile(path.join(__dirname, "splash.html"));
  splashWin.once("ready-to-show", () => splashWin.show());
}

function createMainWindow(base) {
  mainWin = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    show: false,
    backgroundColor: "#f9f2e3",
    title: "Vishvas Foundation — Discourse Archive",
    icon: iconPath(),
    autoHideMenuBar: true,
  });

  mainWin.loadURL(base);
  mainWin.once("ready-to-show", () => {
    mainWin.show();
    if (splashWin && !splashWin.isDestroyed()) splashWin.close();
    splashWin = null;
  });

  // Open external links in the system browser, keep app links in-window.
  mainWin.webContents.setWindowOpenHandler(({ url }) => {
    if (!url.startsWith(base)) {
      shell.openExternal(url);
      return { action: "deny" };
    }
    return { action: "allow" };
  });

  // Closing hides to tray instead of quitting — keeps the stack warm.
  mainWin.on("close", (e) => {
    if (!app.isQuitting) {
      e.preventDefault();
      mainWin.hide();
    }
  });
}

function showError(message) {
  if (splashWin && !splashWin.isDestroyed()) {
    splashWin.webContents.send("error", message);
  }
}

// ── Tray ──────────────────────────────────────────────────────────────────────

function iconPath() {
  const p = path.join(__dirname, "assets", "icon.ico");
  return fs.existsSync(p) ? p : undefined;
}

function trayImage() {
  const p = iconPath();
  if (p) return nativeImage.createFromPath(p);
  // 1x1 transparent fallback so the tray still works before an icon is added.
  return nativeImage.createEmpty();
}

function healthLabel() {
  if (!lastHealth) return "Status: starting…";
  const s = lastHealth.services || {};
  const down = ["ollama", "qdrant", "reranker"].filter((k) => s[k] === false);
  if (down.length === 0) return "● All systems online";
  return `● ${down.length} service${down.length > 1 ? "s" : ""} offline`;
}

function buildTrayMenu() {
  return Menu.buildFromTemplate([
    {
      label: "Open Vishvas Archive",
      click: () => {
        if (mainWin) {
          mainWin.show();
          mainWin.focus();
        }
      },
    },
    { label: healthLabel(), enabled: false },
    { type: "separator" },
    {
      label: "Restart services",
      click: async () => {
        try {
          await composeRestart();
        } catch (e) {
          // surfaced via tray tooltip; non-fatal
          if (tray) tray.setToolTip("Restart failed: " + (e.message || e));
        }
      },
    },
    { type: "separator" },
    {
      label: "Quit (keep services warm)",
      click: () => {
        app.isQuitting = true;
        app.quit();
      },
    },
    {
      label: "Quit & stop services",
      click: async () => {
        app.isQuitting = true;
        try {
          await composeStop();
        } catch {
          // best effort — quit regardless
        }
        app.quit();
      },
    },
  ]);
}

function createTray() {
  tray = new Tray(trayImage());
  tray.setToolTip("Vishvas Foundation");
  tray.setContextMenu(buildTrayMenu());
  tray.on("double-click", () => {
    if (mainWin) {
      mainWin.show();
      mainWin.focus();
    }
  });
}

/** Periodically refresh tray health so the menu reflects reality. */
function startHealthPolling() {
  setInterval(async () => {
    if (!resolvedBase) return;
    const body = await probeHealth(resolvedBase, 3000);
    if (body) lastHealth = body;
    if (tray) {
      tray.setContextMenu(buildTrayMenu());
      tray.setToolTip("Vishvas Foundation — " + healthLabel());
    }
  }, 15000);
}

// ── Startup orchestration ─────────────────────────────────────────────────────

async function bringStackUp({ forceRecreate = false } = {}) {
  setSplashStatus("Starting Docker…");
  if (!(await dockerDaemonReady())) {
    startDockerDesktop();
    await waitFor(dockerDaemonReady, cfg.dockerStartTimeout, 2500);
  }

  setSplashStatus(forceRecreate ? "Repairing the archive…" : "Starting services…");
  await composeUp({ forceRecreate });

  setSplashStatus("Waking the archive…");
  const base = await waitFor(async () => {
    return await resolveReachableBase(2000);
  }, cfg.healthTimeout, 2500);

  setSplashStatus("Warming the answer model…");
  await prewarmModel();

  return base;
}

async function boot() {
  // Fast path: stack already warm AND mounted to the real data → open now.
  const existing = await resolveReachableBase(1200);
  if (existing && (await ollamaHasModels())) {
    createMainWindow(existing);
    return;
  }

  // Otherwise bring everything up from inside WSL. If something answered but
  // ollama has no models, the stack came up on the empty Docker-Desktop
  // mounts — force a recreate so the WSL-native bind mounts (the real data)
  // take effect instead of silently showing a blank archive.
  createSplash();
  try {
    const base = await bringStackUp({ forceRecreate: Boolean(existing) });
    createMainWindow(base);
  } catch (e) {
    const msg =
      "Could not start the archive.\n\n" +
      (e && e.message ? e.message : String(e)) +
      "\n\nMake sure Docker Desktop is installed and the project files are present.";
    showError(msg);
  }
}

// ── App wiring ────────────────────────────────────────────────────────────────

// Single instance: re-clicking the icon focuses the existing window.
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWin) {
      mainWin.show();
      mainWin.focus();
    }
  });

  ipcMain.handle("retry", async () => {
    await boot();
  });

  app.whenReady().then(async () => {
    createTray();
    startHealthPolling();
    await boot();
  });

  // Don't quit when the window is hidden to tray.
  app.on("window-all-closed", (e) => {
    // no-op: we keep running in the tray on Windows
  });

  app.on("activate", () => {
    if (mainWin) mainWin.show();
  });
}
