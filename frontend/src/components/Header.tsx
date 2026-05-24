import type { Health } from "../types";

type Tab = "search" | "analytics";

interface Props {
  health: Health | null;
  tab: Tab;
  onTab: (t: Tab) => void;
  canLogout: boolean;
  onLogout: () => void;
}

/** A coloured dot reporting one upstream service's reachability. */
function ServiceDot({ name, up }: { name: string; up: boolean }) {
  return (
    <span className="svc" title={`${name}: ${up ? "up" : "down"}`}>
      <span className={`svc-dot ${up ? "svc-dot--up" : "svc-dot--down"}`} />
      {name}
    </span>
  );
}

/** Top bar: title, tab switch, upstream-service status, model, logout. */
export function Header({ health, tab, onTab, canLogout, onLogout }: Props) {
  const svc = health?.services;
  return (
    <header className="header">
      <div className="header-left">
        <span className="header-title">Transcript RAG</span>
        <nav className="tabs">
          <button
            className={tab === "search" ? "tab tab--active" : "tab"}
            onClick={() => onTab("search")}
          >
            Search
          </button>
          <button
            className={tab === "analytics" ? "tab tab--active" : "tab"}
            onClick={() => onTab("analytics")}
          >
            Analytics
          </button>
        </nav>
      </div>
      <div className="header-right">
        {svc && (
          <div className="svc-row">
            <ServiceDot name="ollama" up={svc.ollama} />
            <ServiceDot name="qdrant" up={svc.qdrant} />
            <ServiceDot name="reranker" up={svc.reranker} />
            {!health?.bm25_enabled && (
              <span className="svc svc--warn" title="BM25 index not loaded">
                bm25 off
              </span>
            )}
          </div>
        )}
        {health && <span className="header-model">{health.chat_model}</span>}
        {canLogout && (
          <button className="link-btn" onClick={onLogout}>
            Sign out
          </button>
        )}
      </div>
    </header>
  );
}
