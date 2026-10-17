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
  GET  /api/shot/{name}   — serve a screenshot PNG
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
from . import memory, models
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


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "uncensored-harness",
        "primary": "pollinations (gpt-oss-20b, verified uncensored)",
        "fallback": "ai-horde (stheno/heretic/rocinante, verified online)",
        "never": ["gpt", "claude", "gemini"],
        "composio_configured": bool(os.environ.get("COMPOSIO_API_KEY", "").strip()),
        "access_gate": bool(_access_code()),
        "workdir": T.WORKDIR,
    }


@app.get("/api/self-test")
def self_test(access_code: str = ""):
    _check_access(access_code)
    return models.self_test()


@app.get("/api/threads")
def threads(access_code: str = ""):
    _check_access(access_code)
    return {"threads": memory.list_threads()}


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

        async def emit(ev: dict):
            yield f"data: {json.dumps(ev)}\n\n"

        # Wrap emit for the agent (agent calls emit as async fn, not async-gen)
        async def emit_cb(ev: dict):
            nonlocal_gen = emit(ev)
            async for chunk in nonlocal_gen:
                # push chunk into the outer generator via queue
                await queue.put(chunk)

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
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
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
def shot(name: str):
    p = os.path.join(T.SHOTS_DIR, os.path.basename(name))
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="no such screenshot")
    return FileResponse(p, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
