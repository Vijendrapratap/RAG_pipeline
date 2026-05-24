/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Override the API origin (same-origin if unset). */
  readonly VITE_API_BASE?: string;
  /** Vite dev-server proxy target for `/api`. */
  readonly VITE_API_TARGET?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
