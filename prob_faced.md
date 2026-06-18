# Problems Faced — Engineering Incident Log

A look-back record of the real problems hit while building this local RAG system,
written for a technical reader. Each entry is **Symptom → Root cause → Solution →
Prevention**. The goal is production hindsight: where this system has failed, *why*,
how we fixed it, and how to stop it recurring at scale.

The chronological narrative (with commands and verification output) lives in
[`doc.md`](doc.md); this file is the distilled, categorized version.

**Legend**
- ⭐ = recurring / high-impact (the ones that bit us more than once)
- 🟢 fixed & verified · 🟡 mitigated, residual risk · 🔵 open / known-limit

---

## Table of contents
1. [Infrastructure: Docker, WSL & mounts](#1-infrastructure-docker-wsl--mounts)
2. [GPU, VRAM & model loading](#2-gpu-vram--model-loading)
3. [Retrieval quality](#3-retrieval-quality)
4. [Answer synthesis (the LLM step)](#4-answer-synthesis-the-llm-step)
5. [Ingestion pipeline](#5-ingestion-pipeline)
6. [Dev-environment gotchas](#6-dev-environment-gotchas)
7. [Cross-cutting lessons / production checklist](#7-cross-cutting-lessons--production-checklist)
8. [Open risks](#8-open-risks-still-live)

---

## 1. Infrastructure: Docker, WSL & mounts

### 1.1 ⭐🟢 The "everything blank after reboot" empty-mount bug

**The single most recurring failure** (hit 2026-06-02, 06-08, and 06-10 before a
durable fix).

**Symptom.** After a reboot or a Docker Desktop (DD) auto-start, the app would
come up "successfully" but **empty**: `/api/embed` 404, "History unavailable",
Qdrant `404 /collections/transcripts`, `ollama list` blank, fresh empty Postgres.
Everything looked healthy; there was just no data.

**Root cause.** The heavy data (72 GB Ollama models, Qdrant index, reranker cache,
Postgres) is bind-mounted from **WSL2-native ext4** at `/home/pc/transcript-rag-data/...`
via *absolute* paths in `docker-compose.override.yml`. An absolute Linux path like
`/home/pc/...` only resolves to the real Ubuntu data **when `docker compose` is run
from inside the Ubuntu-24.04 distro.** When Docker Desktop auto-starts the
containers at boot (its `restart: unless-stopped` policy), or when `docker` is
invoked from the Windows side, that same path is resolved inside Docker's *own*
`docker-desktop` utility VM — where `/home/pc/...` does not exist. Docker silently
**creates an empty directory and mounts that.** Result: a structurally-healthy
stack pointed at empty folders. The data was never lost; the containers were just
looking at the wrong place.

**Solution (evolved over three rounds):**
- *First aid (06-02, 06-08):* recreate the stack **from inside WSL** —
  `docker compose down && docker compose up -d` — so the binds resolve to the real
  Ubuntu fs. `--force-recreate` is required because compose compares the mount
  *spec string* (unchanged) and won't otherwise recreate a wrong-context container.
- *Durable fix (06-10):* (1) set `restart: "no"` on all five services in the
  override so DD never auto-resurrects them; applied live with
  `docker update --restart no …` (non-destructive). (2) The desktop launcher
  (`desktop/main.js`) now runs compose **inside WSL** (`runWsl`/`composeWsl`), and
  checks daemon readiness via WSL too. (3) Self-heal: `ollamaHasModels()` detects
  the empty-mount signature (a service answers but Ollama reports 0 models) and
  `boot()` force-recreates from WSL ("Repairing the archive…").

**Prevention / production note.** Absolute host paths in bind mounts are
context-sensitive across the Windows/WSL/DD boundary — they are a footgun. The
launcher is now the **single authoritative starter**; nothing auto-starts at boot.
For a Linux-native production host this class of bug disappears entirely (no DD VM
indirection); the more robust long-term option is **named volumes** instead of
absolute bind paths, so resolution can't drift. Never use DD's per-image **Run ▶**
button (spawns stray one-off containers outside compose).

### 1.2 🟢 Port 8080 hijacked by a backup-agent service

**Symptom.** After a rebuild, `rag-api` failed to recreate:
`bind: Only one usage of each socket address … 0.0.0.0:8080`.

**Root cause.** `Get-NetTCPConnection -LocalPort 8080` → PID →
`MiniTool ShadowMaker\AgentService.exe` (Windows service `MTAgentService`,
StartMode Auto). It boots before Docker Desktop and grabs 8080.

**Solution.** `docker-compose.override.yml` remaps the dashboard to host port
**8081** (`ports: !override - "8081:8080"`). The container still listens on 8080
internally, so in-network URLs (`OLLAMA_URL`, `QDRANT_URL`) are unaffected. The
launcher probes `[8080, 8081]` so it finds whichever is live.

**Prevention.** Canonical port stays 8080 for clean installs; the override is
local-only. Permanent fix on this box would be `Set-Service MTAgentService
-StartupType Manual`. Lesson: on Windows hosts, **audit auto-start services for
port squatters** before assuming a port is free.

### 1.3 🟢 DrvFs vs native ext4 — broken locks, slow I/O

**Symptom.** Postgres instability and slow Qdrant HNSW when data lived on the
Windows drive (`/mnt/d/...`, DrvFs); a separate "BM25 disabled,
`FileDoesNotExist("meta.json")`" when a Tantivy mount landed empty.

**Root cause.** DrvFs (the Windows-drive projection into WSL) has **unreliable
`fcntl` locks** (corrupts Postgres), poor `fsync` semantics, and slow random I/O
(hurts Qdrant). Separately, Tantivy 0.26's reader allocates a lockfile even when
opening read-only, so a `:ro` bind failed with `Failed to acquire Lockfile:
ReadOnlyFilesystem`.

**Solution.** The override moves all stateful data to native ext4
(`/home/pc/transcript-rag-data/...`) and mounts Tantivy **rw** (the API never
writes, but the reader needs its lockfile). Tantivy/pageindex stay as *relative*
`./data/...` paths (read access on DrvFs is fine).

**Prevention.** Keep databases and index files on a **native Linux filesystem**,
never on a 9p/DrvFs projection. (This is also why 1.1's mounts use `/home/pc`.)

### 1.4 🟢 Electron launches as Node under the dev harness

**Symptom.** `npm start` in `desktop/` crashed: `TypeError: Cannot set properties
of undefined (setting 'isQuitting')` at `main.js`.

**Root cause.** The VSCode/Claude Code harness sets `ELECTRON_RUN_AS_NODE=1`,
which forces `electron.exe` into plain-Node mode → `require("electron")` returns a
path string, so `app` is `undefined`. Not a code bug; a real user's environment
won't have this var.

**Solution.** `Remove-Item Env:ELECTRON_RUN_AS_NODE` before `npm start` for dev
launches from this environment.

**Prevention.** Be aware that agent/IDE shells inject env vars that change tool
behavior — when a "works for users, breaks in dev" gap appears, diff the env.

---

## 2. GPU, VRAM & model loading

### 2.1 🟢 "Gemma is slow / CPU-bound" — it was a VRAM ceiling, not a GPU fault

**Symptom.** A ~26–31B Gemma ran very slowly with high CPU and seemingly no GPU
use, despite an RTX 5090 32 GB on paper.

**Root cause.** GPU passthrough and CUDA were fine (Blackwell sm_120 supported).
The Ollama logs revealed the GPU at the time of the slow runs was actually a
**5070 Ti (15.9 GiB)** — a 26B/31B model can't fit 16 GB, so layers spilled to
CPU. That spill was the entire slowdown.

**Solution.** Hardware upgrade to the 5090; verified live `gemma4:31b` at **100%
GPU, 27 GB @ 32k ctx, 63 tok/s**. No code change.

**Prevention.** Diagnose "slow GPU model" by checking **actual resident VRAM**
(`ollama ps` → `PROCESSOR` must read `100% GPU`; `nvidia-smi`), not the spec sheet.
CPU spill, not GPU capability, is the usual culprit.

### 2.2 ⭐🟢 VRAM contention — a resident model starves the next one

**Symptom.** First model-comparison run thrashed `gemma4:26b` (112 s, then a 5-min
`/api/chat` timeout → 502).

**Root cause.** `qwen3.5:9b` (~9 GB) was still resident when `gemma4:26b` (~18 GB)
loaded; only ~18 GiB was free, so Gemma spilled to CPU. This box has **32 GB VRAM
but only ~31 GB system RAM (~4.5 free)**, so any CPU spill collapses to ~2 tok/s.

**Solution.** The eval harness now **unloads other models (`keep_alive:0`) before
each model** so each gets a clean VRAM budget. Runtime keeps `MAX_LOADED_MODELS=2`
(bge-m3 + one chat model) and `CHAT_NUM_CTX=8192`.

**Prevention.** On a single-GPU box, **keep exactly one large chat model resident**
and never let two big models co-reside. System RAM being smaller than VRAM means
spill is catastrophic, not graceful — treat "fits in VRAM" as a hard constraint.

### 2.3 🟢 35B MoE doesn't fit; 12B won't pull

**Symptom.** (a) `qwen3.6:35b-a3b` (MoE) gave ~60 s answers; (b) `gemma4:12b`
refused to pull: `412: requires a newer version`.

**Root cause.** (a) At 8192-ctx the KV cache grew and forced a **7% CPU spill**
(`on_gpu` 100%→93%) → ~4.7 tok/s; structural, because the reranker (~2–3 GB) +
bge-m3 (~1.3 GB) permanently occupy ~4 GB. (b) `gemma4:12b`'s manifest needs a
newer Ollama than the pinned 0.24.0; 26b/31b use the older format and pull fine.

**Solution.** Dropped the 35B (`ollama rm`); the practical big-model pick on this
box is `gemma4:26b` (~18 GB, fits with headroom, 50–60 tok/s). Skipped 12b rather
than upgrade the engine mid-project.

**Prevention.** Budget VRAM for the **co-resident** reranker + embedder, not just
the chat model. Pin the inference engine version and validate model-manifest
compatibility before relying on a tag.

### 2.4 🟢 Degenerate repetition loop

**Symptom.** `qwen3.5:9b` occasionally degenerated into a repetition loop (~64 s of
one sentence repeated hundreds of times) — a reliability hazard for a user-facing
dashboard.

**Root cause.** Small-model answer-generation degeneration (not the thinking chain;
`CHAT_THINK=off` was already set).

**Solution.** Added `repeat_penalty` (`CHAT_REPEAT_PENALTY`, default **1.1**) in
`synthesis._ollama_options()`, wired through config/.env/compose. Stops the loop
without truncating good answers (quality-neutral).

**Prevention.** Ship a small anti-repetition penalty by default for small models;
1.1 is safe. Add a max-token safety cap only if you can guarantee it won't truncate
legitimate long answers.

---

## 3. Retrieval quality

### 3.1 ⭐🟢 Catalog/sheet rows rank near-zero ("right data, score 0.03")

**Symptom.** For a query like "Top discourses on dharma from 2015", genuinely
relevant catalog sittings displayed at ~0.019–0.03 with **flat ordering** — the
right data, but scored as if irrelevant.

**Root cause.** The displayed score is the `bge-reranker-v2-m3` `relevance_score`
(a query↔text cross-encoder). Catalog `track_title` rows carried only metadata into
the reranker — `[Catalog | LOC | DATE | Performers: …]\nTitle: …` — with **no prose
for the cross-encoder to judge**, so everything clustered near zero. A live reranker
probe confirmed a composed natural-language sentence scores ~**8×** higher than the
bare metadata line.

**Solution.** In `rag_api/retrieval.py`, three pure/tested helpers: `rerank_text()`
composes a natural-language sentence from a title row's facets (and appends the
sitting's prose body when available); `collect_sitting_bodies()` harvests sitting
prose already in the candidate pool; `Retriever._fetch_sitting_bodies()` does one
batched Qdrant scroll for the rest. The **display text is untouched** — only what
the reranker *reads* changes. Result: content-bearing sittings now score 0.12–0.18
and rank first; bare-metadata rows sit below at ~0.05.

**Prevention.** A cross-encoder needs **prose, not key-value metadata**. Any
metadata-only document type must be given a textual surface (composed sentence
and/or a joined body) before reranking, or it will always lose.

### 3.2 ⭐🟢 Quote search returns noise citations → the answer model refuses

**Symptom.** Pasting an exact Hindi passage retrieved the correct source as
citation **[1]** (~0.80), yet the model answered "not found" / empty / timed out.

**Root cause.** A controlled experiment (same passage, varying citation count)
isolated it: `find_quote` returns 6 chunks but only [1] is real; [2]–[6] are noise
(~0.04). **1–2 citations → correct; 6 citations → refusal**, regardless of
thinking on/off or model size. A small model shown mostly-irrelevant passages
concludes "absent." `gemma4:26b` tolerates the noise; the qwen family does not. It
was never the model or the retrieval — it was the **noise floor in the prompt.**

**Solution.** `synthesis.trim_by_relevance()` gated by `SYNTH_MIN_SCORE_RATIO`
(default **0.2**): before synthesis, drop passages scoring below 20% of the top
hit. The kept set is a score-sorted prefix so citation numbers still line up with
the UI list; display citations are unchanged. Applied in `generate()` and
`stream()`. After the fix, the fast default `qwen3.5:9b` answers quote queries
correctly — gemma-quality, no model switch. Semantic/catalog clusters (where no
single hit dominates) keep all passages → no regression.

**Prevention.** Don't feed an LLM low-relevance citations "just in case." Trim to
the passages that actually clear a relative score bar; one strong hit surrounded by
noise is worse than the strong hit alone.

### 3.3 🔵 Cross-lingual ceiling (English query vs Hindi corpus)

**Symptom.** English queries against the Devanagari corpus underperform native
Hindi queries.

**Root cause.** Even with bge-m3 (multilingual) and the cross-encoder, an
English-query / Hindi-document pairing is inherently weaker than same-language
retrieval.

**Status / next lever (not yet built).** Query translation before rerank
(translate the English query to Hindi, retrieve, rerank) is the obvious optional
lever. Left open deliberately.

### 3.4 🟡 Reranker 422 on empty/short text

**Symptom.** A short Hindi query ("प्रवचन क्या है") occasionally made the Infinity
reranker throw a transient **422**.

**Root cause.** Likely an empty-text candidate reaching the reranker.

**Solution / status.** The pipeline degrades gracefully — it falls back to RRF
ordering and never 500s — so this is mitigated, not eliminated. An empty-text
**guard before the reranker call** is the proper fix and remains open.

---

## 4. Answer synthesis (the LLM step)

### 4.1 🟢 Thinking-model latency (10–30 s spinner)

**Symptom.** Every query sat on a spinner for 10–30 s before the first visible
token, even though warm generation was fast.

**Root cause.** `qwen3.5:9b` is a **thinking model** — it emits a hidden reasoning
chain before the visible answer, producing many tokens first.

**Solution.** Added `CHAT_THINK` (auto/off/on) in config/synthesis; `.env` sets
`CHAT_THINK=off`, which disables the `<think>` chain → ~3.7–4.2 s answers. Also kept
models resident (`OLLAMA_KEEP_ALIVE=-1`) so warm latency dominates.

**Prevention.** Know whether your model "thinks." For latency-sensitive RAG with a
thinking model, disabling thinking is the single biggest lever; weigh it against
reasoning quality on complex questions.

### 4.2 🟢 Thinking model swallows JSON (`format=json` → empty `response`)

**Symptom.** The PageIndex backend always returned "no relevant passage"; the tree
builder logged "section summary parse failed … char 0" for every node.

**Root cause.** With `format=json` **and** a thinking model, Ollama routes the
model's output into the `thinking` field and returns an **empty `response`**. So
every JSON parse saw an empty string.

**Solution.** Set `"think": False` in the single shared helper
`rag_api/pageindex.ollama_generate_json` (verified safe on non-thinking models too).

**Prevention.** When you require structured/JSON output from an Ollama model,
**explicitly disable thinking** for that call — thinking and `format=json` conflict.

### 4.3 🟢 Language + citation discipline with thinking off

**Symptom.** With thinking off, `qwen3.5:9b` initially ignored the Hindi
instruction (answered in English) and dumped raw citation blocks into the answer.

**Root cause.** The language/citation rules were not prominent enough in the prompt
once the reasoning chain (which had been "figuring it out") was removed.

**Solution.** Front-loaded an emphatic LANGUAGE rule in the system prompt + a
closing language/inline-cite reminder in the user prompt; `build_user_prompt` takes
the resolved answer language.

**Prevention.** Removing a model's reasoning chain can expose weak instruction
following — make critical output rules (language, citation format) explicit and
front-loaded.

### 4.4 🟢 Cold-start latency

**Symptom.** First query after a stack (re)create took 1–2.5 min.

**Root cause.** Cold load of bge-m3 + the chat model off the WSL disk (~100 MB/s).
One-time per model load.

**Solution.** `OLLAMA_KEEP_ALIVE=-1` keeps models resident for the session; the
desktop launcher **pre-warms** the chat model right after the stack comes up.

**Prevention.** Pre-warm on launch and pin models resident; surface a "warming"
state in the UI so the first-query cost isn't mistaken for a hang.

---

## 5. Ingestion pipeline

### 5.1 ⭐🟢 `bulk_ingest` doesn't read `.env` → 401 on upsert

**Symptom.** Ingestion's startup health check passed, then **upserts 401'd** —
confusing, because the run "started fine."

**Root cause.** `bulk_ingest_hardened` reads `QDRANT_API_KEY` from `os.environ` and
does **not** load `.env`. The health check hits an *unauthenticated* endpoint (so it
passes), but the authenticated upsert fails. A classic "passes the check, fails the
work" trap.

**Solution.** Always `set -a; source .env; set +a` before ingest. The new
`scripts/add_transcripts.sh` wrapper does this automatically.

**Prevention.** Make secrets-loading part of the entry point, not a manual
pre-step. Health checks should exercise the **authenticated** path so they fail
loudly when creds are missing.

### 5.2 🟢 Stem collisions silently dropped chunks

**Symptom.** A recursive corpus run over the nested layout dropped chunks: e.g. four
distinct `06 SAMBODHAN` tracks across sessions collapsed to one (8 skipped, ~40
chunks silently lost).

**Root cause.** `bulk_ingest` globs the output dir **flat** and `source_file` is the
global identity key; duplicate bare stems across sessions produced the same flat
output filename → first-write-wins.

**Solution.** `chunker_cleaned.qualified_source()` builds an event/session-qualified
key (e.g. `Live Masters 2010/01 NOIDA …/7 JAN - 1$ - 6 PM/06 SAMBODHAN.json`) as
both `source_file` and a flat-but-unique `output_basename` (`/`→`__`, long names get
a sha1 suffix). Re-running the NOIDA event went from 72 chunks/8 skipped → **112
chunks/0 skipped** — the +40 were content the collision had been dropping.

**Prevention.** Never key document identity on a bare filename in a corpus with
repeated track names across folders. Qualify by path; assert uniqueness; treat
"skipped" counts as a red flag, not a benign stat.

### 5.3 🟡 `--base-dir` drift causes duplicates

**Symptom.** Re-running the chunker with a different `--base-dir` made every track
look "new" and threatened duplicate ingestion.

**Root cause.** The incremental skip is keyed on `output_basename()`, which is
derived from `--base-dir`. Change the base and the qualified key shifts, so prior
outputs no longer match.

**Solution.** Keep `--base-dir` constant. `scripts/add_transcripts.sh` **pins** it to
the source root so it can't be set wrong by hand; the chunker also warns when
`--base-dir` is omitted.

**Prevention.** When an incremental key depends on a CLI arg, **fix that arg in a
wrapper** rather than trusting operators to pass it identically every time.

### 5.4 ⭐🟢 Stale progress entries skip re-added tracks (Tantivy is append-only)

**Symptom (2026-06-10).** After re-chunking the corpus, `verify_ingestion` sat at
**97.9%** — exactly 3 files / 12 chunks missing, even though ingest reported
`failed=0`.

**Root cause (a connected one).** On 2026-06-09 those 3 top-level tracks
(`03 AA HIMMAT KA EK KADAM`, `03 PRAVACHAN IN MEDITATION`, `05 MAUSAM KOI HO` =
5+6+1 chunks) had been **surgically deleted** from Qdrant/Tantivy/Postgres to scope
the index to "Live Masters only" — but their rows in `ingest_progress.sqlite` were
**not** cleared. When they were re-chunked and re-ingested today, `bulk_ingest` saw
their flat names still marked `ok` and **skipped them wholesale**. Their chunk IDs
are content-dependent (`chunk_uuid(source_file, idx, text)`), so the new chunks were
genuinely absent from the index.

**Solution (had to be surgical).** A blind full re-ingest was unsafe because
`TantivyWriter.add()` is **append-only (no delete-by-term)** — re-ingesting already
present files would **double** the BM25 docs. So: a probe computed each file's cids,
checked Qdrant presence, and reset **only** the files with missing chunks via
`ProgressDB.reset_to_pending` (keyed on `fpath.name`); confirmed exactly the 3 files;
re-ingested (processed=3, 12 chunks, 153 skipped → Tantivy not doubled); restarted
rag-api. Re-verify: **Qdrant 100% / Tantivy 100%**, points 551→563.

**Prevention.** A delete must clear **all four** stores: Qdrant, Tantivy, Postgres,
**and the progress DB**. The same trap recurs for a *re-transcribed* existing track
(progress `ok` by flat name → skipped) — reset its progress row first. Long-term,
`bulk_ingest` should detect content change (chunk-set hash) rather than trust a
file-name `ok` flag.

### 5.5 🟢 PageIndex returned no results — missing mount + thinking-JSON

**Symptom.** `backend=pageindex` always returned the "no relevant passage"
fallback; `/api/health` showed `pageindex_trees:0`.

**Root cause (two stacked).** (1) `docker-compose.yml` mounted only
`./data/tantivy`; `./data/pageindex` was **unmounted**, so the container saw an
empty dir (the same empty-dir trap as the mounts in §1.1) — and because the override
uses `volumes: !override`, the mount had to be added in **both** files. (2) The
thinking-model-eats-JSON bug (§4.2) made every tree-node title/summary and the
retriever's node-selection parse to empty.

**Solution.** Added `./data/pageindex:/app/data/pageindex:ro` to **both** compose
files; set `think:False` in the shared JSON helper; rebuilt rag-api; built sample
trees. `pageindex_trees:3`, queries return reranked section hits.

**Prevention.** With `volumes: !override`, every mount must be repeated in the
override or it's lost. Verify new mounts actually exist *inside the container*, not
just on the host.

---

## 6. Dev-environment gotchas

Smaller, environment-specific traps that cost time. All worked around, none are
product bugs.

| Problem | Root cause | Workaround |
|---|---|---|
| 🟢 `npm install` fails `UNABLE_TO_VERIFY_LEAF_SIGNATURE` | Corporate cert / SSL inspection on this machine intercepts the npm registry | Build the frontend inside Docker (the multi-stage Dockerfile) or on the WSL side; don't rely on host npm |
| 🟢 PowerShell mangles UTF-8 / Devanagari in API tests | PS test client re-encodes request/response bytes | Send/decode **explicit UTF-8 bytes**; browsers handle UTF-8 JSON natively, so the UI is fine |
| 🟢 `psql` DELETE fails inside `wsl -lc '…'` | SQL single-quotes collide with the WSL command quoting | Write a temp `.sql` file and pipe it on **stdin** instead of inline `-c` |
| 🟢 Reranker JSON shell-escaping mangles Devanagari | Inline `curl` in Git Bash mangles multibyte | Run via WSL `python3` / write a script file rather than inline curl |
| 🟢 `python3` not on Git Bash PATH | The Bash tool here is Git Bash (MINGW), not WSL | Run Python via WSL (`/mnt/d/...`/`.venv/bin/python`) |
| 🟢 `κ`/Unicode `UnicodeEncodeError` in cp1252 terminal | Windows console default encoding | `PYTHONIOENCODING=utf-8` for CLI runs |

**Prevention.** On a Windows+WSL box, do anything touching **non-ASCII text,
secrets, or SQL** from **inside WSL with script files**, not inline through nested
PowerShell/Git-Bash quoting.

---

## 7. Cross-cutting lessons / production checklist

The themes that repeat across the incidents above:

1. **Always (re)create the stack from inside WSL** — never let Docker Desktop
   auto-start it. (§1.1) Now enforced by `restart:"no"` + the launcher.
2. **Always `source .env` before any ingest/enrich run.** (§5.1) Use
   `scripts/add_transcripts.sh`, which does it for you.
3. **A delete is not done until all four stores agree** — Qdrant, Tantivy,
   Postgres, *and* `ingest_progress.sqlite`. Verify with `verify_ingestion`
   (expect 100%). (§5.4)
4. **One big model resident at a time**; budget VRAM for the co-resident reranker +
   embedder; spill is catastrophic on this box (RAM < VRAM). (§2.1, §2.2)
5. **Cross-encoders need prose**; metadata-only rows must be given a textual
   surface before reranking. (§3.1)
6. **Don't feed the LLM low-relevance citations** — trim to what clears a relative
   score bar. (§3.2)
7. **Thinking models**: disable thinking for latency and for any `format=json`
   call. (§4.1, §4.2)
8. **Keep data on native ext4**, never DrvFs. (§1.3)
9. **"Healthy but empty" is a real failure mode** — health checks must exercise the
   authenticated/data path, and the UI should detect an empty index, not show a
   blank archive as success. (§1.1, §5.1)

---

## 8. Open risks (still live)

Things not yet fully solved — the honest backlog for a production hardening pass:

- 🔵 **Cross-lingual retrieval** (English query → Hindi corpus): query translation
  before rerank is the unbuilt lever. (§3.3)
- 🟡 **Reranker empty-text guard**: the 422 is handled by graceful fallback but the
  root-cause guard isn't in place. (§3.4)
- 🟡 **Re-transcription / re-add skip**: re-adding or re-transcribing a track whose
  name is still `ok` in the progress DB is silently skipped; needs a content-hash
  change-detector in `bulk_ingest` rather than a name flag. (§5.4)
- 🟡 **Bind-mount fragility**: the durable fix relies on the launcher always
  starting from WSL. A Linux-native host or named volumes would remove the class of
  bug entirely. (§1.1)
- 🔵 **Scale**: the largest verified index here is ~563 chunks (a deliberately
  scoped corpus). The PRD targets hundreds of millions of records over a multi-day
  ingest — the resumability, dead-letter, and progress-DB machinery exist for it,
  but the failure modes at that scale (Qdrant 503 on low disk, FTS index rebuild
  time, Tantivy lock recovery) are documented in [`docs/runbook.md`](docs/runbook.md)
  but not yet exercised at volume.
