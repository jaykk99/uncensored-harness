"""uncensored-harness — FastAPI server.

Endpoints:
  GET  /                  — chat UI
  GET  /api/health        — liveness + model routing + self-test summary
  GET  /api/self-test     — refusal-probe verification (access_code)
  GET  /api/threads       — list threads (access_code)
  POST /api/chat          — agent turn, SSE stream {message, thread_id, access_code}
  POST /api/run           — code runner {language: python|javascript, code, access_code}
  POST /api/shell         — terminal {command, access_code}
  GET  /api/files         — list workdir files (access_code)
  GET  /api/preview       — serve a workdir HTML file for live preview (path)
  GET  /api/shot/{name}   — serve a screenshot PNG (access_code, gated)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import agent as agent_mod
from . import integrations, memory, models
from . import providers as providers_mod
from . import tools as T
from .guards import check_hard_boundary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="uncensored-harness")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


def _access_code() -> str:
    return os.environ.get("HARNESS_ACCESS_CODE", "").strip()


def _check_access(provided: str) -> None:
    required = _access_code()
    if required and provided != required:
        raise HTTPException(status_code=401, detail="wrong access code")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = ""
    access_code: str = ""
    model: str = ""        # preferred Horde model id from the UI picker ("" = auto)
    web_search: bool = False


class RunRequest(BaseModel):
    language: str = "python"
    code: str = ""
    access_code: str = ""


class ShellRequest(BaseModel):
    command: str = ""
    access_code: str = ""


class RouterResetRequest(BaseModel):
    provider: str = ""
    model: str = ""
    access_code: str = ""


class ThreadRenameRequest(BaseModel):
    title: str = ""
    access_code: str = ""


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "uncensored-harness",
        "primary": "ai-horde (stheno/heretic/rocinante, genuinely uncensored)",
        "fallback": "pollinations (gpt-oss-20b, native tool calls)",
        "never": ["gpt", "claude", "gemini"],
        "integrations": integrations.integration_status(),
        "composio": "removed — native direct APIs only (see integrations)",
        "access_gate": bool(_access_code()),
    }


@app.get("/api/self-test")
def self_test(access_code: str = ""):
    _check_access(access_code)
    return models.self_test()


@app.get("/api/probe")
def probe(provider: str = "", model: str = "", access_code: str = ""):
    """Per-provider uncensored probe suite (slow — hits live providers).

    Probes each live model individually with the refusal probes and
    reports PASS/FAIL per model. A model may only be called 'uncensored'
    after passing every probe here.
    """
    _check_access(access_code)
    from . import probe as probe_mod
    # Guardrails: bound the work so one HTTP call can't run for an hour.
    known_providers = {p.name for p in providers_mod.ALL_PROVIDERS}
    known_providers |= {"horde", "pollinations"}
    if provider and provider not in known_providers:
        raise HTTPException(status_code=400, detail="unknown provider")
    if model:
        allowed = {m["id"] for m in providers_mod.list_models()}
        if model not in allowed:
            raise HTTPException(status_code=400, detail="unknown model id")
    return probe_mod.run_suite(provider=provider, model_id=model)


@app.get("/api/router-stats")
def router_stats(access_code: str = ""):
    """Per-leg routing health: attempts/successes/refusals/errors + latency."""
    _check_access(access_code)
    from . import router_stats as rs_mod
    return {"health": rs_mod.health(), "legs": rs_mod.snapshot()}


@app.post("/api/router-stats/reset")
def router_stats_reset(req: RouterResetRequest):
    """Clear router leg stats (all, or one provider/model).

    Stale stats drag the health table down long after a provider outage
    is fixed; this gives the System tab a clean slate. The stats rebuild
    naturally as new turns run.
    """
    _check_access(req.access_code)
    from . import router_stats as rs_mod
    n = rs_mod.reset(provider=req.provider, model=req.model)
    return {"cleared": n}


@app.get("/api/threads")
def threads(access_code: str = ""):
    _check_access(access_code)
    return {"threads": memory.list_threads()}


@app.get("/api/threads/search")
def thread_search(q: str = "", access_code: str = ""):
    """Search thread titles and message content.

    Registered before /api/threads/{thread_id} so "search" is never
    treated as a thread id. Auth is checked before the lookup, same as
    the other thread endpoints.
    """
    _check_access(access_code)
    return {"threads": memory.search_threads(q)}


@app.get("/api/threads/{thread_id}")
def thread_detail(thread_id: str, access_code: str = "", limit: int = 60,
                  offset: int = 0):
    """One thread with its messages, for reopening it in the UI.

    Until now the sidebar only listed threads: clicking one cleared the
    pane and showed nothing, because no endpoint ever served the stored
    history back. Auth is checked before the lookup so a wrong code
    cannot probe whether a thread id exists.
    `offset` pages backwards (skip that many of the newest messages) so
    a thread longer than `limit` can be read in full; `total` tells the
    UI whether older messages remain.
    """
    _check_access(access_code)
    thread = memory.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="no such thread")
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    return {"thread": thread,
            "messages": memory.get_history(thread_id, limit=limit,
                                           offset=offset),
            "total": memory.count_messages(thread_id)}


@app.delete("/api/threads/{thread_id}")
def thread_delete(thread_id: str, access_code: str = ""):
    _check_access(access_code)
    if not memory.delete_thread(thread_id):
        raise HTTPException(status_code=404, detail="no such thread")
    return {"deleted": True, "thread_id": thread_id}


@app.patch("/api/threads/{thread_id}")
def thread_rename(thread_id: str, req: ThreadRenameRequest):
    _check_access(req.access_code)
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="empty title")
    if not memory.rename_thread(thread_id, title):
        raise HTTPException(status_code=404, detail="no such thread")
    return {"renamed": True, "thread": memory.get_thread(thread_id)}


@app.get("/api/models")
def api_models(access_code: str = ""):
    _check_access(access_code)
    from . import providers as providers_mod
    models = providers_mod.list_models()
    groups: dict[str, dict] = {}
    order: list[str] = []
    for m in models:
        g = m["provider"]
        if g not in groups:
            groups[g] = {"provider": g, "company": m["company"],
                         "keyless": m["keyless"], "items": []}
            order.append(g)
        groups[g]["items"].append(m)
    return {"groups": [groups[g] for g in order],
            "count": len(models)}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    _check_access(req.access_code)
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="empty message")

    hit = check_hard_boundary(message)
    if hit:
        thread_id = req.thread_id or memory.create_thread(message[:40])
        memory.add_message(thread_id, "user", message)
        reply = f"{hit.reason} {hit.alternative}"
        memory.add_message(thread_id, "assistant", reply)
        return {"thread_id": thread_id, "reply": reply, "blocked": True}

    thread_id = (req.thread_id if req.thread_id and memory.thread_exists(req.thread_id)
                 else memory.create_thread(message[:40]))
    memory.add_message(thread_id, "user", message)
    history = memory.get_history(thread_id)

    async def stream():
        yield f"data: {json.dumps({'type': 'thread', 'thread_id': thread_id})}\n\n"

        queue: asyncio.Queue = asyncio.Queue()

        async def run_agent():
            try:
                # agent uses a sync-style emit; adapt with a simple collector
                events: list[dict] = []

                async def collect(ev: dict):
                    events.append(ev)
                    await queue.put(f"data: {json.dumps(ev)}\n\n")

                # Hard cap: a turn must never "think forever". If the model
                # legs stall, end the SSE stream with a clear error instead.
                try:
                    text = await asyncio.wait_for(
                        agent_mod.run_turn(history, collect,
                                           model=req.model or "",
                                           web_search=req.web_search),
                        timeout=660)
                except asyncio.TimeoutError:
                    err = ("Timed out after 11 minutes with no answer. "
                           "The model workers are likely overloaded — "
                           "try again in a bit.")
                    logger.warning("agent turn timed out for thread %s",
                                   thread_id)
                    await queue.put(
                        f"data: {json.dumps({'type': 'error', 'error': err})}\n\n")
                    memory.add_message(thread_id, "assistant", f"[timeout] {err}")
                    return
                memory.add_message(thread_id, "assistant", text or "(no reply)")
            except Exception as e:
                logger.exception("agent turn failed")
                await queue.put(f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n")
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_agent())

        # Keepalive: some proxies (Railway included) kill SSE streams that go
        # quiet. Send a comment ping every 15s so the connection stays alive
        # through long model waits. Clients ignore SSE comments.
        stop_ping = asyncio.Event()

        async def pinger():
            try:
                while not stop_ping.is_set():
                    await asyncio.sleep(15)
                    if stop_ping.is_set():
                        break
                    await queue.put(": ping\n\n")
            except asyncio.CancelledError:
                pass

        ping_task = asyncio.create_task(pinger())
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            stop_ping.set()
            ping_task.cancel()
        await task

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/run")
async def run_code(req: RunRequest):
    """Built-in code runner: python or javascript. Returns stdout/stderr."""
    _check_access(req.access_code)
    lang = (req.language or "python").lower()
    code = req.code or ""
    if not code.strip():
        raise HTTPException(status_code=400, detail="empty code")
    if lang.startswith("py"):
        cmd: list[str] = ["python3", "-c", code]
    elif lang.startswith("js") or lang.startswith("javascript") or lang.startswith("node"):
        cmd = ["node", "-e", code]
    else:
        raise HTTPException(status_code=400, detail="language must be python or javascript")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=T.WORKDIR,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
            return {"ok": False, "output": "[timed out after 60s]"}
        text = out.decode("utf-8", "replace")
        return {"ok": proc.returncode == 0, "exit": proc.returncode,
                "output": text[:20000]}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"runtime not found: {cmd[0]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/shell")
async def shell(req: ShellRequest):
    """Terminal: run a command in the workdir."""
    _check_access(req.access_code)
    if not (req.command or "").strip():
        raise HTTPException(status_code=400, detail="empty command")
    out = await T.shell_exec(req.command)
    return {"output": out}


@app.get("/api/files")
async def files(access_code: str = "", path: str = "."):
    _check_access(access_code)
    return {"listing": await T.file_list(path)}


@app.get("/api/preview")
def preview(path: str = "", access_code: str = ""):
    """Serve a workdir HTML file for live in-chat preview."""
    _check_access(access_code)
    try:
        p = T._safe_path(path)
    except ValueError:
        raise HTTPException(status_code=400, detail="bad path")
    if not os.path.isfile(p) or not p.lower().endswith((".html", ".htm")):
        raise HTTPException(status_code=404, detail="no such html file")
    return FileResponse(p, media_type="text/html")


@app.get("/api/shot/{name}")
def shot(name: str, access_code: str = ""):
    # Gated like every other endpoint: screenshots can contain anything the
    # agent did (browsers, files, chats), so they must not be publicly
    # fetchable. The UI appends ?access_code= to every shot <img> src.
    _check_access(access_code)
    p = os.path.join(T.SHOTS_DIR, os.path.basename(name))
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="no such screenshot")
    return FileResponse(p, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
