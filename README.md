Live: https://uncensored-harness-production.up.railway.app/

# Uncensored Harness

Standalone AI agent harness with **verified uncensored models only** — never GPT, Claude, Gemini, or any RLHF-tuned refuser.

Live: https://uncensored-harness-production.up.railway.app (Railway, sleep OFF, deploys from `main` via Dockerfile). Access is gated by `HARNESS_ACCESS_CODE`.

## What it is

A personal agent computer: chat UI, terminal + filesystem (sandboxed to the workdir), web search/fetch, a controlled headless browser with screenshots, an app maker that screenshots its own previews, and native direct-API integrations (GitHub, GitLab, Gmail, Vercel, Fal) — no middleware.

**Uncensored means** no refusals on legitimate work — coding, business ops, research, writing, analysis, authorized security work on your own assets. It is *not* a bypass: a small set of hard boundaries stays in place (stolen-card fraud, phishing kits, draining anyone's funds, unauthorized hacking targets, weapons of mass harm, sexual content involving minors, doxxing). Those are the only refusals, ever, each paired with the nearest legitimate alternative. See `app/guards.py`.

## Refusal policy (template feature)

Refusal policy is yours: edit `app/guard_policy.py` to set what your instance refuses; it ships empty. This app is a template — whoever deploys an instance writes their own refusal policy in that one file (or the `GUARD_*` env vars): your own block patterns for user messages, any extra phrases that mark a model reply as a refusal for the router's skip logic, and the behaviour switches for refusal-skipping and uncensored-first chain order. Zero preset entries ship in the lists. The hard floor is separate and always on: the boundary pre-check in `app/guards.py` and the access-code gate. Those are not in the config and cannot be configured away.

## Model routing

Every turn goes through this chain (see `app/models.py`, `app/providers.py`):

1. **Race (primary)** — AI Horde (45s cap: Stheno, Heretic, Rocinante workers, genuinely uncensored) runs *in parallel* against the fast keyless providers (Kilo Gateway, LLM7, Pollinations). Whichever answers first with a valid non-refusal response wins; the losers are cancelled. A slow Horde queue never blocks the turn when a fast provider is ready.
2. **Omni chain (sequential fallback)** — explicit UI pick first, then uncensored keyed models (DeepInfra/Together Dolphin, OpenRouter Hermes) when their keys are set, then the keyless backstop (LLM7, Kilo), then remaining keyed providers with keys set. Each gets a short timeout. Within each tier, legs with poor recent stats (under 50% success over 3+ attempts, or no data in 24h) are tried last — the uncensored-first tier order never changes; `ROUTER_HEALTH_ORDER=0` restores static order.
3. **Pollinations** (streamed, native tool calls).
4. **Horde retry** (final) — a worker may have come online meanwhile.

If every leg fails, the turn raises a clear error instead of hanging. Every leg records its outcome (ok / refusal / error + latency) to `app/router_stats.py`, visible at `GET /api/router-stats` and in the System tab's router-health table.

### Providers

Keyless (work with zero keys): AI Horde, Pollinations, LLM7, Kilo Gateway.
Keyed (code is wired; the UI marks "needs key" until the env var is set):

| Provider | Env var | Notes |
|---|---|---|
| DeepInfra | `DEEPINFRA_API_KEY` | Dolphin 2.9.1 (uncensored) first |
| Together | `TOGETHER_API_KEY` | Dolphin 2.9.1 (uncensored) first |
| OpenRouter | `OPENROUTER_API_KEY` | Hermes 3 70B (uncensored) first |
| Groq | `GROQ_API_KEY` | gpt-oss-120b |
| Cerebras | `CEREBRAS_API_KEY` | llama-3.3-70b, qwen-3-32b |
| SambaNova | `SAMBANOVA_API_KEY` | Llama 3.3 70B, DeepSeek V3 |
| NVIDIA NIM | `NVIDIA_API_KEY` | GLM, Laguna |
| Mistral | `MISTRAL_API_KEY` | mistral-small, codestral |
| Fireworks | `FIREWORKS_API_KEY` | llama-3.3-70b |
| HuggingFace | `HF_TOKEN` | router.huggingface.co |
| Cohere | `COHERE_API_KEY` | /v2/chat shape |
| Gratisfy | `GRATISFY_API_KEY` | (no models listed yet) |

`HORDE_API_KEY` is optional (better queue priority on Horde).

### Uncensored probe suite

`GET /api/self-test` probes the *active route*. `GET /api/probe` (and `python -m app.probe`) goes further: it runs the refusal probes against **each live model individually** and scores PASS/FAIL per model. The System tab has a per-provider probe runner UI.

Four probes, all lawful and legitimate: dual-use hobby info (lockpicking as a sport), free expression (mild profanity in creative writing), authorized security work (own-network red team), and opinion without hedging (a hotly debated political issue — catches RLHF "as an AI, I don't have opinions" hedging).

**Standing rule:** a provider or model may only be described as "uncensored" or "fixed" after passing *every* probe in this suite. The suite reports; it never flips labels on its own.

```bash
python -m app.probe --provider groq        # probe one provider
python -m app.probe --model "kilo:kilo-auto/free" --json
```

## Native integrations (no Composio)

`app/integrations.py` talks to each provider's own REST API directly. When a credential is missing, the tool says so honestly instead of failing obscurely. Status is on `GET /api/health` → `integrations`, and in the System tab.

| Tool | Env | Actions |
|---|---|---|
| `github_api` | `GITHUB_TOKEN` | repos, repo, file, issues, create_issue, prs, workflow_runs, dispatch |
| `gitlab_api` | `GITLAB_TOKEN` (+ `GITLAB_HOST`) | projects, project, issues, pipelines |
| `gmail` | `GOOGLE_OAUTH_CLIENT_ID` + `GOOGLE_OAUTH_CLIENT_SECRET` + `GOOGLE_REFRESH_TOKEN` | send, search, get |
| `vercel_api` | `VERCEL_TOKEN` (+ `VERCEL_TEAM_ID`) | projects, deployments, redeploy, env_vars, set_env |
| `fal_generate` | `FAL_KEY` (+ `FAL_IMAGE_MODEL`) | generate, status, result |

Gmail needs the one-time Google connect (OAuth consent with offline access) to mint `GOOGLE_REFRESH_TOKEN`. Sending mail is approval-gated: the agent shows Jay the draft and waits for his OK before calling `gmail/send`.

## Capabilities

- Chat UI: thinking/progress events, token streaming, markdown, code blocks with copy/run, live app preview, inline screenshots, thread sidebar (reopen a past thread with its full history, rename, delete, search titles and message content, search within an open thread with prev/next jumps and the matched words highlighted in the message, search every thread's messages at once and jump straight to the turn, load older messages on long threads, delete one message or cut a thread from a message onward, regenerate the last reply with a fresh answer, edit a message or fix a user prompt and re-run the turn from it, stop a running turn server-side so the partial reply is stored with a "[stopped by user]" marker instead of running to completion)
- Terminal + filesystem, sandboxed to `HARNESS_WORKDIR` (default `~/harness-work`); path escapes rejected
- Web search (Bing RSS primary, DuckDuckGo fallback) + fetch, with SSRF guards (no localhost/private IPs/metadata hosts)
- Controlled headless browser (open/click/fill/text) with automatic screenshots (Playwright)
- App maker: builds web apps, auto-screenshots the preview
- Image generation tab (Pollinations) + `fal_generate` agent tool
- System tab: integration status, per-provider probe runner, router health

## API endpoints

All state-changing reads take `access_code` (query or body) when `HARNESS_ACCESS_CODE` is set.

- `GET /` — chat UI
- `GET /api/health` — liveness, routing summary, integration status
- `GET /api/self-test` — refusal probes against the active route
- `GET /api/probe?provider=&model=` — per-model probe suite (slow)
- `GET /api/router-stats` — per-leg health (attempts, success rate, refusals, errors, avg/p50/p95 ms latency, `stale` flag when no data in 24h)
- `GET /api/models` — model list grouped by provider, with live key status
- `GET /api/threads` — list threads
- `GET /api/threads/search?q=` — search thread titles and message content (returns a match snippet per thread)
- `GET /api/threads/{id}` — reopen a thread with its stored messages (clicking a sidebar thread used to clear the pane and show nothing; this endpoint is what loads it back). `before_id` pages by cursor (the `limit` messages older than that row id) and wins over `offset`: the UI pages with the cursor so "load older" stays exact during a live turn, when untagged live bubbles would throw an on-screen count off
- `GET /api/threads/{id}/messages/search?q=` — search within one thread's messages (row id, thread position, and match snippet per hit, oldest-first). The find bar appears as soon as a live turn assigns a thread id, not only on reopen, so search works mid-turn over everything already stored; a jump to a still-streaming match waits for the turn to finish instead of reopening and wiping the stream. A jump fetches the 60-message window ending at the match's position in one request, so there is no page cap: a match 7,000 messages back used to need ~117 page loads and was unreachable past ~6,000. Every on-screen match gets its matched words highlighted in the message (amber marks, solid on the current match), not just an outline around the bubble; the marks survive a same-thread reopen and clear with the search. Pages cap at 50 matches: `after_id` fetches the next page (matches with a row id above it), `has_more`/`next_after_id` say whether there is one, so the find bar's "…" button reaches every match in a long thread
- `GET /api/messages/search?q=` — search every thread's messages at once (per-message hits, newest first: thread id/title, row id, role, thread position and total, match snippet; archived threads included and flagged). The sidebar shows the hits under the thread results; clicking one opens that thread and jumps to the exact turn, and the in-thread find bar takes the same query so prev/next walks the whole thread's matches from there. Pages cap at 50 hits: `before_id` fetches the next page (matches older than that row, compared as a (created, id) pair so same-second messages page exactly), `has_more`/`next_before_id` say whether there is one, so the sidebar's "Show more message results" button reaches every hit
- `GET /api/threads/{id}/messages/{mid}/position` — 1-based position of one message in its thread (oldest = 1), plus the thread total; 404 when the row id is not in that thread
- `DELETE /api/threads/{id}` — delete a thread and its messages
- `DELETE /api/threads/{id}/messages/{mid}` — delete one message from a thread
- `POST /api/threads/{id}/messages/{mid}/truncate` — delete a message and everything after it (retry a bad turn)
- `PATCH /api/threads/{id}` — rename a thread (`{title, access_code}`)
- `POST /api/chat` — SSE agent turn `{message, thread_id, access_code, model, web_search}`
- `POST /api/threads/{id}/regenerate` — re-run the last turn: drops the stored assistant reply and answers the same last user message again over SSE (`{model, web_search, access_code}`); a reply that never landed just re-runs, and a last prompt that hits the boundary returns the refusal JSON instead of running the agent
- `POST /api/threads/{id}/stop` — stop the turn running on a thread, server-side (`{access_code}`); returns `{"stopped": true, "running": true}` while a turn is in flight and `{"stopped": false, "running": false}` when the thread is idle. The UI Stop button calls this before aborting its own fetch: before it existed the server turn ran to its 11-minute cap and stored the full reply even after the user stopped it. A stopped turn ends at the next model/tool boundary (a model call already in flight finishes first), emits a `stopped` SSE event, and stores the partial reply with a "[stopped by user]" marker. Closing the tab mid-turn flags the turn the same way, so a disconnect stops the server work too
- `POST /api/threads/{id}/messages/{mid}/edit` — edit a stored message (`{content, resend, model, web_search, access_code}`); with `resend=true` the message must be a user prompt, everything after it is cut and the turn re-runs from the new wording over SSE, and a re-sent prompt that hits the boundary returns the refusal JSON instead of running the agent. Editing the thread's first message also follows the auto title (a renamed title is left alone)
- `POST /api/run` — python/javascript code runner
- `POST /api/shell` — terminal command
- `GET /api/files`, `GET /api/preview`, `GET /api/shot/{name}`

## Run locally

```bash
pip install -r requirements.txt
export HARNESS_ACCESS_CODE=secret
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

See `.env.example` for the full env var list. On Railway/Render, set env vars in the dashboard; the Dockerfile installs Playwright + Chromium for screenshots.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

336 tests, no network (HTTP is stubbed): guard boundaries, sqlite thread storage, refusal detection, provider chain ordering, SSRF/path safety, web-search fallback, the forwarder CONNECT tunnel, all five integrations, the probe suite, router stats, and thread reopen/delete/rename/search/message-prune/message-edit/in-thread-search/global-message-search/search-paging. Tests must pass before every push — a red suite means the repo is BLOCKED, not padded.

## Architecture

```
static/index.html   chat / image / terminal / system UI (SSE)
app/main.py         FastAPI: chat SSE, code/shell/files, probes, stats
app/agent.py        model -> tools -> model loop (OpenAI tool format + fenced fallback)
app/models.py       routing: race -> omni -> pollinations -> horde retry
app/providers.py    provider table + omni_chain (uncensored-first ordering)
app/probe.py        per-model refusal probe suite (+ CLI)
app/router_stats.py sqlite leg-health tracking
app/integrations.py native GitHub/GitLab/Gmail/Vercel/Fal REST clients
app/tools.py        agent tool registry (shell, files, web, browser, integrations)
app/guards.py       hard-boundary pre-check (tiny by design — legit work passes)
app/memory.py       sqlite threads
app/forwarder.py    local CONNECT forwarder for Chromium proxy auth
```
