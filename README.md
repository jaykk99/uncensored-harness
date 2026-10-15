# Uncensored Harness

Standalone AI agent harness with **verified uncensored models only** — never GPT, Claude, Gemini, or any RLHF-tuned refuser.

## Model routing (verified 2026-10-15)

1. **Pollinations** (primary) — keyless, OpenAI-compatible, `gpt-oss-20b` open weights.
   Verified: alive, answered a dual-use hobby probe directly with zero refusal framing.
2. **AI Horde** (fallback) — keyless, genuinely uncensored workers, verified online:
   Stheno (`koboldcpp/L3-8B-Stheno-v3.2`, 4 workers), Heretic variants,
   Rocinante (`koboldcpp/Rocinante-X-12B-v1`, 2 workers).

## Capabilities

- Chat with Claude-like UI: thinking indicator, markdown, code blocks with copy/run,
  live app preview, inline screenshots
- Terminal + filesystem (sandboxed to workdir)
- Web search + fetch, Composio integrations
- Controlled headless browser (open/click/fill) with automatic screenshots
- App maker: builds web apps, screenshots the preview automatically
- `GET /api/self-test` — refusal-probe verification endpoint

## Run

```bash
cd ~/workspace/uncensored-harness
./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Env: `HARNESS_ACCESS_CODE` (optional gate), `HARNESS_WORKDIR` (default ~/harness-work),
`COMPOSIO_API_KEY` (optional), `HORDE_API_KEY` (optional, better queue priority).

## Endpoints

- `GET /` — chat UI
- `GET /api/health` — liveness + routing
- `GET /api/self-test` — refusal probes, verdict
- `POST /api/chat` — SSE agent turn
- `POST /api/run` — python/javascript code runner
- `POST /api/shell` — terminal
- `GET /api/files`, `GET /api/preview`, `GET /api/shot/{name}`
