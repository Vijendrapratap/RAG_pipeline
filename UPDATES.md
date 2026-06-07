# Updates

A plain-language log of changes to the Vishvas Foundation discourse archive.
Newest first. For the detailed technical build log see [`doc.md`](doc.md).

---

## 2026-06-02

### Faster answers
- The answer model (`qwen3.5:9b`) is a "thinking" model: it used to write a
  long hidden reasoning chain before showing a single word, which is why you
  waited 10–30 s staring at the spinner. Added a `CHAT_THINK` setting and set
  it to `off`, so the model now answers directly. Much snappier.
  - To change it: edit `CHAT_THINK` in `.env` (`off` = fast, `auto` = let the
    model think). If you switch `CHAT_MODEL` to a non-thinking model such as
    `qwen2.5:7b-instruct`, set `CHAT_THINK=auto`.
- Models are now kept loaded in the GPU permanently (`OLLAMA_KEEP_ALIVE=-1`),
  so you no longer pay a 1–2 minute "cold start" reload after the system has
  been idle. The first load after a full restart is still a one-time wait.

### Choose the answer language (Hindi / English)
- Added a visible **Auto · English · हिंदी** toggle right in the ask box, so
  you can pick the answer language in one click. It used to be hidden inside
  the "Advanced" panel.
  - **Auto** matches the language of your question (ask in Hindi → Hindi
    answer). **English** / **हिंदी** force that language regardless of how you
    typed the question.

### Infrastructure / housekeeping (behind the scenes)
- **GPU:** confirmed the new RTX 5090 (32 GB) now runs the large models fully
  on the GPU. The earlier slowness was the old 16 GB card forcing the model
  onto the CPU.
- **Search 404 fixed:** the stack was starting with empty data when launched
  the wrong way; it now loads the real Qdrant / Postgres / index data. Always
  start it from the Ubuntu terminal with `docker compose up -d`.
- **Disk cleanup:** removed ~23 GB of retired images (old Open WebUI), unused
  containers, build cache, and stray volumes.

---

## How to apply updates after editing

Frontend or backend code changes are baked into the `rag-api` container, so
rebuild and restart it **from the Ubuntu terminal**:

```bash
cd /mnt/d/Vishvas-rag-pipeline
docker compose build rag-api
docker compose up -d rag-api
```

Then hard-refresh the browser (Ctrl+Shift+R) at http://localhost:8081.
