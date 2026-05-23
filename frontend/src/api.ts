/**
 * Typed client for the rag-api backend.
 *
 * Auth is a shared password sent in the `X-Dashboard-Password` header. It is
 * held in sessionStorage so a tab reload keeps the user signed in but closing
 * the tab clears it. Because the password rides in a header, the `/api/query`
 * SSE stream is consumed via `fetch` + ReadableStream, not `EventSource`
 * (EventSource cannot set request headers).
 */
import type {
  FilterOptionsResponse,
  Health,
  MentionsResponse,
  QueryBody,
  QueryMeta,
  SearchBody,
  SearchResponse,
  SpeakersResponse,
  TranscriptsResponse,
} from "./types";

// Same-origin by default: dev → Vite proxies `/api` to the backend;
// prod → the rag-api container serves both the UI and the API.
// Set VITE_API_BASE only when hosting the UI on a different origin.
const API_BASE = (import.meta.env.VITE_API_BASE ?? "")
  .replace(/\/+$/, "");

const PW_KEY = "rag_dashboard_password";

export function getPassword(): string {
  return sessionStorage.getItem(PW_KEY) ?? "";
}
export function setPassword(pw: string): void {
  sessionStorage.setItem(PW_KEY, pw);
}
export function clearPassword(): void {
  sessionStorage.removeItem(PW_KEY);
}

/** An HTTP-level failure. `status` lets callers special-case 401 (re-auth). */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function authHeaders(): Record<string, string> {
  const pw = getPassword();
  return pw ? { "X-Dashboard-Password": pw } : {};
}

async function errorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.detail === "string") return body.detail;
    return JSON.stringify(body);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

async function getJson<T>(path: string, auth: boolean): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: auth ? authHeaders() : {},
  });
  if (!res.ok) throw new ApiError(res.status, await errorDetail(res));
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await errorDetail(res));
  return res.json() as Promise<T>;
}

// ---- endpoints -----------------------------------------------------------

export function getHealth(): Promise<Health> {
  return getJson<Health>("/api/health", false);
}

export function getFilters(): Promise<FilterOptionsResponse> {
  return getJson<FilterOptionsResponse>("/api/filters", true);
}

export function search(body: SearchBody): Promise<SearchResponse> {
  return postJson<SearchResponse>("/api/search", body);
}

export function analyticsMentions(
  term: string,
  speaker?: string,
): Promise<MentionsResponse> {
  const qs = new URLSearchParams({ term });
  if (speaker) qs.set("speaker", speaker);
  return getJson<MentionsResponse>(`/api/analytics/mentions?${qs}`, true);
}

export function analyticsSpeakers(
  term: string,
  limit = 10,
): Promise<SpeakersResponse> {
  const qs = new URLSearchParams({ term, limit: String(limit) });
  return getJson<SpeakersResponse>(`/api/analytics/speakers?${qs}`, true);
}

export function analyticsTranscripts(
  term: string,
  limit = 20,
): Promise<TranscriptsResponse> {
  const qs = new URLSearchParams({ term, limit: String(limit) });
  return getJson<TranscriptsResponse>(`/api/analytics/transcripts?${qs}`, true);
}

// ---- /api/query streaming ------------------------------------------------

export interface StreamHandlers {
  onMeta: (meta: QueryMeta) => void;
  onToken: (text: string) => void;
  onDone: () => void;
  onError: (detail: string) => void;
}

/** Parse one raw SSE block ("event: x\ndata: {...}") into its parts. */
function parseSseBlock(block: string): { event: string; data: string } | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

/**
 * Open a streaming `/api/query` turn. Resolves when the stream ends.
 * Throws ApiError on an HTTP-level failure (e.g. 401) so the caller can
 * react before any handler fires; per-stream `error` events go to onError.
 */
export async function streamQuery(
  body: Omit<QueryBody, "stream">,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  if (!res.ok) throw new ApiError(res.status, await errorDetail(res));
  if (!res.body) throw new ApiError(0, "streaming not supported by response");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const evt = parseSseBlock(block);
      if (!evt) continue;
      try {
        if (evt.event === "meta") handlers.onMeta(JSON.parse(evt.data));
        else if (evt.event === "token")
          handlers.onToken(JSON.parse(evt.data).text ?? "");
        else if (evt.event === "done") handlers.onDone();
        else if (evt.event === "error")
          handlers.onError(JSON.parse(evt.data).detail ?? "stream error");
      } catch (e) {
        handlers.onError(`malformed stream event: ${String(e)}`);
      }
    }
  }
}
