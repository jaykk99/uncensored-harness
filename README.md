# Uncensored Harness

Standalone AI agent harness with **verified uncensored models only** — never GPT, Claude, Gemini, or any RLHF-tuned refuser.

## Model routing (horde primary since 2026-10-17)

1. **AI Horde** (primary) — keyless, genuinely uncensored workers:
   Stheno (`koboldcpp/L3-8B-Stheno-v3.2`), Heretic variants, Rocinante
   (`koboldcpp/Rocinante-X-12B-v1`). Verified genuinely uncensored.
2. **Pollinations** (fallback) — keyless, OpenAI-compatible, `gpt-oss-20b`
   open weights. Kept for its native tool-call support; answers are
   refusal-probed and a refused/failing leg retries Horde once more.
   A final Horde retry closes out the legs if Pollinations fails.

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
