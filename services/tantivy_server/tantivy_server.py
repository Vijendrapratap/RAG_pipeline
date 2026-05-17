import os
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
from contextlib import asynccontextmanager
import tantivy

TANTIVY_DIR = Path(os.environ.get("TANTIVY_DIR", "./data/tantivy"))

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not TANTIVY_DIR.exists():
        raise RuntimeError(f"Tantivy index dir not found: {TANTIVY_DIR}")
    state["index"] = tantivy.Index.open(str(TANTIVY_DIR))
    state["index"].reload()
    state["searcher"] = state["index"].searcher()
    yield
    state.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"ok": True, "docs": state["searcher"].num_docs}

@app.get("/search")
def search(q: str = Query(...), k: int = Query(40, ge=1, le=200)):
    try:
        idx = state["index"]
        parser = idx.parse_query(q, ["text"])
        hits = state["searcher"].search(parser, limit=k).hits
        out = []
        for score, doc_addr in hits:
            doc = state["searcher"].doc(doc_addr)
            out.append({
                "chunk_id": doc["chunk_id"][0],
                "text": doc["text"][0],
                "source_file": doc["source_file"][0],
                "score": float(score),
            })
        return {"hits": out, "count": len(out)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
def reload_idx():
    state["index"].reload()
    state["searcher"] = state["index"].searcher()
    return {"ok": True, "docs": state["searcher"].num_docs}
