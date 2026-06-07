"use strict";

const { contextBridge, ipcRenderer } = require("electron");

// Minimal, safe bridge for the splash screen: receive status/error updates
// from the main process and offer a single "retry" action.
contextBridge.exposeInMainWorld("vishvas", {
  onStatus: (cb) => ipcRenderer.on("status", (_e, msg) => cb(msg)),
  onError: (cb) => ipcRenderer.on("error", (_e, msg) => cb(msg)),
  retry: () => ipcRenderer.invoke("retry"),
});
