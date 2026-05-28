# Model Configuration in Open WebUI

> ⚠️ **Phase F — retired.** Open WebUI is no longer in the stack and the
> per-model system-prompt / tool-calling settings described here do not
> apply to the new dashboard. The dashboard's answer model is set
> entirely by `CHAT_MODEL` in `.env` (see
> [dashboard.md](dashboard.md)); there is no per-model UI to click
> through. This document is retained only as a record of the old setup.

This is the human runbook for configuring the two chat models so they
actually invoke the Phase 7 function tools. Open WebUI does not configure
its models automatically — every setting below has to be clicked through
the admin UI once per fresh install.

Phase 8 deliverable per PRD §6 Phase 8.

## Prerequisites

Before touching the UI, confirm:

```bash
# 1. Both models are pulled into Ollama (Phase 2 already did this).
docker exec ollama ollama list | grep -E 'qwen2.5:7b-instruct-q4_K_M|deepseek-r1:7b-qwen-distill-q4_K_M'

# 2. Both function tools installed in Open WebUI (Phase 7 + docs/install_functions.md).
#    Visit http://localhost:8080/admin/functions and confirm both rows are toggled on.

# 3. Tantivy sidecar reachable from inside the open-webui container.
docker exec open-webui curl -fsS http://host.docker.internal:8765/health
# Expect: {"ok":true,"docs":<some integer>}
```

If any of these fail, fix them before continuing. The UI steps below assume
the models exist in Ollama and the tools exist in Open WebUI.

## Per-model configuration

Repeat the full flow below for **each** of the two models. The settings are
identical except for the base model name.

1. Sign in to <http://localhost:8080> as the admin user.
2. **Workspace → Models** (top-left burger menu → Workspace, then Models tab).
3. Click **+ (New Model)**. If the model already exists as a clone, click
   the pencil ("Edit") icon next to it instead.
4. Fill in the fields exactly as below.

### Settings table

| Setting | Qwen 2.5 7B | DeepSeek-R1 7B |
|---|---|---|
| Name | `Qwen 2.5 7B (transcripts)` | `DeepSeek-R1 7B (synthesis)` |
| Base model | `qwen2.5:7b-instruct-q4_K_M` | `deepseek-r1:7b-qwen-distill-q4_K_M` |
| Context Length | `16384` | `16384` |
| Function Calling | `native` | `native` |
| Builtin Tools | (none — uncheck Web Search, Code Interpreter, etc.) | (none) |
| Attached Tools | `search_transcripts`, `analytics` | `search_transcripts`, `analytics` |

> **Function Calling = native** is the critical knob. The "default" mode
> uses a regex-based pseudo-function-calling shim that the LLM cannot
> reliably trigger on small (7B) models. `native` uses Ollama's tool-call
> JSON support, which Qwen 2.5 and DeepSeek-R1 both implement.

> **Context Length 16384** matches Ollama's default `num_ctx` for these
> Q4_K_M quants. Going higher works but eats VRAM (~10 GB at 32k); going
> lower (4096 default in some Open WebUI installs) truncates retrieved
> chunks mid-sentence and breaks citations.

### System prompt (paste verbatim into both models)

```
You are a transcript research assistant. You have these tools available:

1. search_transcripts — for finding content, topics, and quotes by meaning
2. find_quote — when the user is hunting for a specific phrase or exact words
3. count_mentions — for "how many times" / "how often" queries
4. top_speakers_for_topic — for "who talks most about X" queries
5. list_transcripts_mentioning — for "which transcripts mention X" queries

Rules:
- ALWAYS call a tool before answering questions about transcript content.
- For quote-finding queries ("who said X", "find when X was said"), use find_quote.
- For "how often" / "how many" / "which speaker most" questions, use the analytics tools.
- For everything else, use search_transcripts.
- Cite source file and timestamp for every claim. If timestamps are unavailable
  (plain-text source), say so explicitly.
- Never invent quotes. If the retrieved chunks don't contain the answer, say so plainly.
- Do not speculate beyond what the retrieved chunks support.
```

5. **Save** at the bottom of the model edit page. Do this for both models.

## Acceptance smoke tests

After saving both models, run these exact tests from the chat UI. They map
1:1 to PRD §6 Phase 8 acceptance criteria.

### 1. Both models appear in the model dropdown

Open a fresh chat. The model picker (top of chat) should now list both
`Qwen 2.5 7B (transcripts)` and `DeepSeek-R1 7B (synthesis)` alongside any
default `qwen2.5:7b-instruct-q4_K_M` / `deepseek-r1:...` entries Ollama
exposes by default. The named clones are the ones with the system prompt
and tools attached — pick those, not the bare base model.

### 2. Find-quote routing

Select the **Qwen 2.5 7B (transcripts)** clone and send exactly:

```
find the moment someone said "we have to focus"
```

**Expect:** Open WebUI's tool-call indicator (small badge near the assistant
message) shows `find_quote` was invoked. The assistant's reply either lists
matching chunks with `Source:` + timestamps + speakers, or — if no fixture
contains that phrase — says so plainly without inventing one.

If the model answers the question without calling a tool, it lost the
system-prompt routing. Re-open the model edit page and confirm
`Function Calling = native` and both tools are still attached.

### 3. Analytics routing

In a fresh chat (still on the Qwen clone), send:

```
how many times is "platform team" mentioned?
```

**Expect:** Tool-call badge shows `count_mentions` was invoked. The assistant
reports a number ≥ 1 (the sample_whisperx fixture contains "platform team"
once).

### 4. Source + timestamp citations

Send:

```
search for content about the new analytics dashboard
```

**Expect:** Tool-call badge shows `search_transcripts`. The assistant's
prose answer includes citations of the form
`(source: sample_whisperx.json, 46.9s → 52.2s, SPEAKER_02)` or similar —
the exact citation rendering is governed by Open WebUI's `citation` flag,
which both Phase 7 tools set to `True`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Model not in dropdown after Save | Open WebUI cache | Browser hard refresh (Ctrl+Shift+R); then admin → users → log out / log back in if needed |
| Tool badge never appears | Function Calling = default (regex shim) | Re-edit model, set Function Calling = native, save |
| `find_quote` answers without tool call | System prompt got truncated or model is the base (no clone) | Verify you're chatting with the *named clone* (e.g. `Qwen 2.5 7B (transcripts)`) and not the bare `qwen2.5:7b-instruct-q4_K_M` |
| Tool invoked but returns `No results.` | Empty Qdrant collection | Confirm ingestion has run: `curl -s -H "api-key: $QDRANT_API_KEY" http://localhost:6333/collections/transcripts \| jq '.result.points_count'` — must be > 0 |
| Tool returns a `ConnectionError` to `host.docker.internal:8765` | Tantivy sidecar not running on host | Start it: `bash scripts/run_tantivy_server.sh` (or install via the systemd unit in [runbook.md](runbook.md)) |
| Analytics returns 0 for everything | Postgres `chunk_meta` empty | Run Phase 6 init: `bash scripts/03_init_postgres.sh`, then re-ingest |
| `find_quote` slow (10+ s before any output) | First call on a freshly-pulled model — Ollama loading into VRAM | Wait ~30 s for the first inference; subsequent calls warm |

## Why two models?

Both models are configured identically because:

- **Qwen 2.5 7B Instruct** is the workhorse — fast, good at structured
  tool calls, reliable citation formatting. Default for daily research use.
- **DeepSeek-R1 7B (Qwen-distill)** is the synthesis option — slower (it
  emits a `<think>` reasoning trace before the user-visible answer) but
  better at multi-step questions like "summarize the financial trajectory
  across these three transcripts" where it needs to call `search_transcripts`
  multiple times and reason over the union.

The system prompt is identical because both should follow the same
tool-routing discipline; only their reasoning depth differs at inference
time.

Phase 9's eval harness will measure both against the golden query set so
the right default is data-driven, not guessed.
