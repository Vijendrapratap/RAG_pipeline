"use strict";

/**
 * Resolved runtime configuration for the desktop shell.
 *
 * Precedence (highest first):
 *   1. Environment variables (VISHVAS_*).
 *   2. A `desktop.config.json` placed next to the executable / project root.
 *   3. The defaults below.
 *
 * On a dev box the project root is the repo (one level above this folder),
 * so `docker compose` runs against the same compose + `.env` you use by hand.
 * On a client install, point `projectDir` at wherever the compose files and
 * `.env` were placed by the installer.
 */

const path = require("node:path");
const fs = require("node:fs");

const REPO_ROOT = path.resolve(__dirname, "..");

function readJsonConfig() {
  // Look beside the app first (packaged), then at the repo root (dev).
  const candidates = [
    process.env.VISHVAS_CONFIG,
    path.join(__dirname, "desktop.config.json"),
    path.join(process.cwd(), "desktop.config.json"),
    path.join(REPO_ROOT, "desktop.config.json"),
  ].filter(Boolean);
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) {
        return JSON.parse(fs.readFileSync(p, "utf8"));
      }
    } catch {
      // Malformed config should not crash the launcher — fall back to defaults.
    }
  }
  return {};
}

const file = readJsonConfig();

const DEFAULTS = {
  // Directory containing docker-compose.yml + .env (+ optional override).
  // Used only as a label / Windows-side cwd; compose itself runs in WSL.
  projectDir: REPO_ROOT,
  // WSL distro that owns the real data and runs the Docker engine. Compose
  // MUST run inside this distro: the override's bind mounts use absolute
  // paths (/home/pc/transcript-rag-data/...) that resolve to the real data
  // ONLY from inside Ubuntu. Run from the Windows / Docker-Desktop context
  // and the same paths resolve to EMPTY dirs in Docker's own VM — that is the
  // "everything came up blank after a reboot" bug.
  wslDistro: "Ubuntu-24.04",
  // The project dir as seen from inside the WSL distro (where compose +
  // override live). DrvFs path is fine here — the heavy data is bind-mounted
  // from native ext4 by the override, not from this path.
  wslProjectDir: "/mnt/d/Vishvas-rag-pipeline",
  // Canonical published port for rag-api. The launcher also probes the dev
  // override (8081) automatically, so this only needs to be right for prod.
  appPort: 8080,
  // Extra ports to probe for a reachable UI, in order.
  probePorts: [8080, 8081],
  // Path to Docker Desktop on Windows; started if the daemon is down.
  dockerDesktop: "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe",
  // ollama host port + model to pre-warm so the first query is not slow.
  ollamaPort: 11434,
  chatModel: "qwen3.5:9b",
  // Timeouts (ms).
  dockerStartTimeout: 120000,
  healthTimeout: 240000,
};

const cfg = {
  projectDir: process.env.VISHVAS_PROJECT_DIR || file.projectDir || DEFAULTS.projectDir,
  wslDistro: process.env.VISHVAS_WSL_DISTRO || file.wslDistro || DEFAULTS.wslDistro,
  wslProjectDir:
    process.env.VISHVAS_WSL_PROJECT_DIR || file.wslProjectDir || DEFAULTS.wslProjectDir,
  appPort: Number(process.env.VISHVAS_APP_PORT || file.appPort || DEFAULTS.appPort),
  probePorts: file.probePorts || DEFAULTS.probePorts,
  dockerDesktop: process.env.VISHVAS_DOCKER_DESKTOP || file.dockerDesktop || DEFAULTS.dockerDesktop,
  ollamaPort: Number(file.ollamaPort || DEFAULTS.ollamaPort),
  chatModel: process.env.VISHVAS_CHAT_MODEL || file.chatModel || DEFAULTS.chatModel,
  dockerStartTimeout: file.dockerStartTimeout || DEFAULTS.dockerStartTimeout,
  healthTimeout: file.healthTimeout || DEFAULTS.healthTimeout,
};

// Make sure the configured appPort is probed first.
cfg.probePorts = [cfg.appPort, ...cfg.probePorts.filter((p) => p !== cfg.appPort)];

module.exports = { cfg, REPO_ROOT };
