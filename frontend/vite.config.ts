import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Dev: Vite on :5173 proxies `/api` to the rag-api backend, so the app
// can use same-origin relative URLs (`/api/...`) in both dev and prod.
// Prod: the built `dist/` is served by the rag-api container at `/`,
// so `/api/...` is already same-origin — no env var needed.
// Override the proxy target with `VITE_API_TARGET` if your backend is
// not on :8080.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_API_TARGET || "http://localhost:8080";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": { target, changeOrigin: true },
      },
    },
  };
});
