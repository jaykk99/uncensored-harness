"""uncensored-harness — FastAPI server.

Endpoints:
  GET  /                  — chat UI
  GET  /api/health        — liveness + model routing + self-test summary
  GET  /api/self-test     — refusal-probe verification (access_code)
  GET  /api/threads       — list threads (access_code, archived=true for the archive view)
  POST /api/threads/import — restore a thread from an exported JSON file
  GET  /api/threads/{id}/export — download a whole thread, md or json
  POST /api/threads/{id}/duplicate — fork a thread with all its messages
  POST /api/threads/{id}/pin — pin/unpin a thread (pinned sort first)
  POST /api/threads/{id}/archive — archive/unarchive a thread (hidden from the sidebar list)
  DELETE /api/threads/{id}/messages/{mid} — delete one message
  POST /api/threads/{id}/messages/{mid}/truncate — delete a message and everything after it
  POST /api/threads/{id}/messages/{mid}/edit — edit a message; resend=true re-runs the turn from it (SSE)
  GET  /api/threads/{id}/messages/search — search within one thread's messages
  GET  /api/messages/search — search every thread's messages, jump to the turn
  POST /api/chat          — agent turn, SSE stream {message, thread_id, access_code}
  POST /api/threads/{id}/regenerate — redo the last turn, same prompt, new reply (SSE)
  POST /api/threads/{id}/stop — stop the turn running on a thread, server-side
  POST /api/run           — code runner {language: python|javascript, code, access_code}
  POST /api/shell         — terminal {command, access_code}
  GET  /api/files         — list workdir files (access_code)
  GET  /api/preview       — serve a workdir HTML file for live preview (path)
  GET  /api/shot/{name}   — serve a screenshot PNG (access_code, gated)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import agent as agent_mod
from . import integrations, memory, models
from . import providers as providers_mod
from . import tools as T
from .guards import check_hard_boundary
from .guard_policy import check_owner_boundary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="uncensored-harness")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

# One stop flag per thread with a turn in flight. _stream_turn registers
# the flag when the turn starts and drops it when the turn ends; the stop
# endpoint (and a client disconnect) sets it, and the agent loop checks it
# before every model and tool call. In-memory only: a server restart drops
# in-flight flags, and with them any turn that was running — nothing is
# persisted, so a stale flag can never stop a later turn.
_stop_events: dict[str, asyncio.Event] = {}


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


class RegenerateRequest(BaseModel):
    model: str = ""        # preferred Horde model id ("" = auto)
    web_search: bool = False
    access_code: str = ""


class StopRequest(BaseModel):
    access_code: str = ""


class MessageEditRequest(BaseModel):
    content: str = ""
    resend: bool = False   # re-run the turn from this message (user only)
    model: str = ""        # preferred Horde model id ("" = auto)
    web_search: bool = False
    access_code: str = ""


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


class ThreadImportRequest(BaseModel):
    # Mirrors the JSON shape thread_export emits: {"thread": {...},
    # "messages": [...]}. Extra keys are ignored; the exported id is
    # never reused (see memory.import_thread). thread/messages are
    # deliberately untyped here so the endpoint itself can 400 on every
    # malformed shape, instead of pydantic 422ing some and the handler
    # 400ing others.
    thread: object = {}
    messages: object = []
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
def threads(access_code: str = "", archived: bool = False):
    _check_access(access_code)
    if archived:
        return {"threads": memory.list_archived_threads()}
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


@app.post("/api/threads/import")
def thread_import(req: ThreadImportRequest):
    """Restore a thread from an exported JSON file.

    Registered before /api/threads/{thread_id} so "import" is never
    treated as a thread id. Auth is checked before any validation, same
    as the other thread endpoints, so a bad payload cannot probe
    anything. Malformed payloads 400 with a plain reason instead of
    importing a half-parsed thread: messages must be a list of objects
    with string role and content. Title falls back to "untitled" when
    the export carries none; the 60-char cap comes from create_thread.
    """
    _check_access(req.access_code)
    if not isinstance(req.messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")
    for m in req.messages:
        if (not isinstance(m, dict)
                or not isinstance(m.get("role"), str)
                or not isinstance(m.get("content"), str)):
            raise HTTPException(
                status_code=400,
                detail="each message needs a string role and content")
    title = req.thread.get("title") if isinstance(req.thread, dict) else None
    title = title.strip() if isinstance(title, str) else ""
    new_id = memory.import_thread(title or "untitled", req.messages)
    return {"imported": True, "thread": memory.get_thread(new_id),
            "messages": len(req.messages)}


@app.get("/api/threads/{thread_id}")
def thread_detail(thread_id: str, access_code: str = "", limit: int = 60,
                  offset: int = 0, before_id: int | None = None):
    """One thread with its messages, for reopening it in the UI.

    Until now the sidebar only listed threads: clicking one cleared the
    pane and showed nothing, because no endpoint ever served the stored
    history back. Auth is checked before the lookup so a wrong code
    cannot probe whether a thread id exists.
    `offset` pages backwards (skip that many of the newest messages) so
    a thread longer than `limit` can be read in full; `total` tells the
    UI whether older messages remain. `before_id` pages by cursor
    instead (the `limit` messages older than that row id) and wins over
    `offset` when given: offset paging drifts mid-turn, when live
    bubbles on screen are not stored/tagged yet, while a cursor stays
    exact. The UI pages with the cursor so in-thread search jumps stay
    correct during a live turn.
    """
    _check_access(access_code)
    thread = memory.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="no such thread")
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    return {"thread": thread,
            "messages": memory.get_history_with_ids(thread_id, limit=limit,
                                                    offset=offset,
                                                    before_id=before_id),
            "total": memory.count_messages(thread_id)}


@app.get("/api/messages/search")
def global_message_search(q: str = "", access_code: str = "",
                          limit: int = 50, before_id: int | None = None):
    """Search every thread's messages at once, newest match first.

    The sidebar thread search returns one snippet per matching thread
    and opens it at the newest 60; the in-thread search only works
    inside the thread already open. This returns per-message hits
    across all threads, each carrying its thread id/title, row id,
    role, 1-based position in its thread and the thread total — the
    address the UI's jump machinery needs to land on the exact turn.
    `limit` clamps like the other paged endpoints; `total` still
    reports every match. `before_id` pages backwards: only matches
    older than that row are returned, so the sidebar's "show more"
    button can fetch the next page with the last returned id;
    `has_more` and `next_before_id` say whether there is one and
    what cursor to pass.
    """
    _check_access(access_code)
    limit = max(1, min(limit, 50))
    result = memory.search_all_messages(q, limit=limit,
                                        before_id=before_id)
    return {"query": (q or "").strip(), **result}


@app.get("/api/threads/{thread_id}/messages/search")
def thread_message_search(thread_id: str, q: str = "",
                          access_code: str = "", limit: int = 50,
                          after_id: int | None = None):
    """Search within one thread's messages, oldest-first.

    The sidebar /api/threads/search finds threads; this finds a turn
    inside the thread already open, which is what a long reopened chat
    actually needs. Auth, then thread existence, are checked in that
    order, same as the other thread endpoints, so a wrong code or an
    unknown id cannot probe anything. `limit` clamps like the other
    paged endpoints; `total` still reports every match. `after_id`
    pages forward: only matches with a row id above it are returned,
    so the find bar's "more matches" button can fetch the next page
    with the last returned id; `has_more` and `next_after_id` say
    whether there is one and what cursor to pass.
    """
    _check_access(access_code)
    if memory.get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail="no such thread")
    limit = max(1, min(limit, 50))
    result = memory.search_thread_messages(thread_id, q, limit=limit,
                                           after_id=after_id)
    return {"thread_id": thread_id, "query": (q or "").strip(), **result}


@app.get("/api/threads/{thread_id}/messages/{message_id}/position")
def thread_message_position(thread_id: str, message_id: int,
                            access_code: str = ""):
    """Where one message sits in its thread (oldest = 1), plus total.

    The search endpoint already reports a match's position, but a
    caller that only has a row id — a jump after a delete, a
    deep-link to one turn — had no way to place it without paging.
    Auth, then thread existence, then the row's membership in this
    thread, in that order, same as the other thread endpoints, so a
    wrong code or a foreign id cannot probe anything.
    """
    _check_access(access_code)
    if memory.get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail="no such thread")
    pos = memory.message_position(thread_id, message_id)
    if pos is None:
        raise HTTPException(status_code=404, detail="no such message")
    return {"thread_id": thread_id, "message_id": message_id, **pos}


def _thread_markdown(thread: dict, messages: list[dict]) -> str:
    """A thread as a readable Markdown transcript, oldest-first.

    The title is flattened to one line: it comes from a user message
    (or a rename), so it may contain a newline that would otherwise
    break the heading. Message bodies go in verbatim — this is the
    user's own transcript, and escaping it would corrupt code blocks.
    """
    title = (thread.get("title") or "untitled").replace("\n", " ")
    lines = [f"# {title}", ""]
    created = thread.get("created")
    if created:
        started = datetime.datetime.fromtimestamp(
            created, datetime.timezone.utc).isoformat()
        lines.append(f"Thread {thread['id']}, started {started}.")
        lines.append("")
    labels = {"user": "You", "assistant": "Assistant"}
    for m in messages:
        lines.append(f"## {labels.get(m['role'], m['role'].capitalize())}")
        lines.append("")
        lines.append(m["content"])
        lines.append("")
    return "\n".join(lines)


@app.get("/api/threads/{thread_id}/export")
def thread_export(thread_id: str, format: str = "md",
                  access_code: str = ""):
    """Download a whole thread: Markdown transcript (default) or JSON.

    Reopening a thread in the UI pages 60 messages at a time, so a long
    chat could never leave the harness in one piece. An export always
    carries the full thread, for saving or pasting elsewhere. Auth is
    checked before the lookup so a wrong code cannot probe whether a
    thread id exists.
    """
    _check_access(access_code)
    thread = memory.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="no such thread")
    fmt = (format or "md").lower()
    if fmt not in ("md", "json"):
        raise HTTPException(status_code=400, detail="format must be md or json")
    messages = memory.get_all_messages(thread_id)
    if fmt == "json":
        body = json.dumps({"thread": thread, "messages": messages,
                           "total": len(messages)}, indent=2)
        return Response(content=body, media_type="application/json",
                        headers={"Content-Disposition":
                                 f'attachment; filename="thread-{thread_id}.json"'})
    return Response(content=_thread_markdown(thread, messages),
                    media_type="text/markdown",
                    headers={"Content-Disposition":
                             f'attachment; filename="thread-{thread_id}.md"'})


class ThreadAccessRequest(BaseModel):
    access_code: str = ""


class ThreadPinRequest(BaseModel):
    pinned: bool | None = None  # None = toggle the current state
    access_code: str = ""


class ThreadArchiveRequest(BaseModel):
    archived: bool | None = None  # None = toggle the current state
    access_code: str = ""


@app.post("/api/threads/{thread_id}/duplicate")
def thread_duplicate(thread_id: str, req: ThreadAccessRequest):
    """Fork a thread: copy it and all its messages under a new id.

    Auth is checked before the lookup so a wrong code cannot probe
    whether a thread id exists, same as the other thread endpoints.
    """
    _check_access(req.access_code)
    new_id = memory.duplicate_thread(thread_id)
    if new_id is None:
        raise HTTPException(status_code=404, detail="no such thread")
    return {"duplicated": True, "thread": memory.get_thread(new_id),
            "messages": memory.count_messages(new_id)}


@app.post("/api/threads/{thread_id}/pin")
def thread_pin(thread_id: str, req: ThreadPinRequest):
    """Pin a thread to the top of the sidebar, or unpin it.

    With no `pinned` value the call toggles, which is what the sidebar
    star button needs; an explicit value sets the state outright.
    Auth is checked before the lookup so a wrong code cannot probe
    whether a thread id exists, same as the other thread endpoints.
    """
    _check_access(req.access_code)
    thread = memory.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="no such thread")
    pinned = (not thread["pinned"]) if req.pinned is None else req.pinned
    memory.set_thread_pinned(thread_id, pinned)
    return {"pinned": pinned, "thread": memory.get_thread(thread_id)}


@app.post("/api/threads/{thread_id}/archive")
def thread_archive(thread_id: str, req: ThreadArchiveRequest):
    """Archive a thread out of the sidebar list, or unarchive it back.

    With no `archived` value the call toggles, which is what the
    sidebar archive button needs; an explicit value sets the state
    outright. Auth is checked before the lookup so a wrong code cannot
    probe whether a thread id exists, same as the other thread endpoints.
    Archiving never deletes anything: the thread stays reopenable,
    searchable, and exportable, and unarchiving restores it to the list.
    """
    _check_access(req.access_code)
    thread = memory.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="no such thread")
    archived = (not thread["archived"]) if req.archived is None else req.archived
    memory.set_thread_archived(thread_id, archived)
    return {"archived": archived, "thread": memory.get_thread(thread_id)}


@app.delete("/api/threads/{thread_id}/messages/{message_id}")
def thread_message_delete(thread_id: str, message_id: int,
                          access_code: str = ""):
    """Delete one message from a thread, leaving the rest intact.

    Auth is checked before the lookup so a wrong code cannot probe
    whether a thread or message id exists, same as the other thread
    endpoints. Unknown thread and unknown-in-this-thread message both
    404; a message id from another thread never deletes here (see
    memory.delete_message's thread_id guard).
    """
    _check_access(access_code)
    if memory.get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail="no such thread")
    if not memory.delete_message(thread_id, message_id):
        raise HTTPException(status_code=404, detail="no such message")
    return {"deleted": True, "message_id": message_id,
            "remaining": memory.count_messages(thread_id)}


@app.post("/api/threads/{thread_id}/messages/{message_id}/truncate")
def thread_message_truncate(thread_id: str, message_id: int,
                            req: ThreadAccessRequest):
    """Cut a thread off at a message: delete it and everything after.

    The retry case for a bad turn — deleting one message alone leaves
    orphan replies that still reference it. Auth, then thread, then
    message existence are checked in that order, so a wrong code or a
    foreign message id can never cut a thread it does not own.
    """
    _check_access(req.access_code)
    if memory.get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail="no such thread")
    deleted = memory.truncate_thread_from(thread_id, message_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="no such message")
    return {"truncated": True, "deleted": deleted,
            "remaining": memory.count_messages(thread_id)}


@app.post("/api/threads/{thread_id}/messages/{message_id}/edit")
async def thread_message_edit(thread_id: str, message_id: int,
                              req: MessageEditRequest):
    """Edit a stored message's text, optionally re-running from it.

    The fix-a-typo case: truncate + retype loses the turn, and
    regenerate only re-runs the same words. With resend=false the edit
    is in place (either role — a stored assistant reply can be
    corrected too) and the thread is untouched otherwise. With
    resend=true the message must be a user prompt: the new text is
    stored, everything after it is cut (replies that answered the old
    wording are stale), and the turn re-runs over SSE exactly like
    /api/chat and regenerate. A re-sent prompt that hits the hard
    boundary is refused the same way, with the refusal stored as the
    reply instead of running the agent. Auth, then thread, then the
    row's membership in this thread, then the role check, in that
    order — a wrong code or a foreign message id can never rewrite a
    thread it does not own, and a re-send never runs on an assistant
    message (there is no prompt to answer).
    """
    _check_access(req.access_code)
    if not memory.thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="no such thread")
    content = (req.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="empty message")
    existing = memory.get_message(thread_id, message_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="no such message")
    if req.resend and existing["role"] != "user":
        raise HTTPException(status_code=400,
                            detail="only a user message can be re-sent")
    edited = memory.edit_message(thread_id, message_id, content)
    if edited is None:
        raise HTTPException(status_code=404, detail="no such message")
    if not req.resend:
        return {"edited": True,
                "message": {"id": edited["id"], "role": edited["role"],
                            "content": edited["content"]}}
    memory.delete_after(thread_id, message_id)
    hit = (check_hard_boundary(content)
           or check_owner_boundary(content))
    if hit:
        reply = f"{hit.reason} {hit.alternative}"
        memory.add_message(thread_id, "assistant", reply)
        return {"thread_id": thread_id, "reply": reply, "blocked": True}
    history = memory.get_history(thread_id)
    return _stream_turn(thread_id, history, model=req.model or "",
                        web_search=req.web_search)


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


def _stream_turn(thread_id: str, history: list[dict], model: str = "",
                 web_search: bool = False) -> StreamingResponse:
    """Stream one agent turn over SSE, storing the reply when it lands.

    Shared by /api/chat (fresh user message) and regenerate (same
    prompt, new answer): the 11-minute cap, the 15s keepalive pings,
    the timeout message stored as the assistant reply, and the error
    event are identical in both cases, so they live here once instead
    of drifting apart.
    """
    async def stream():
        yield f"data: {json.dumps({'type': 'thread', 'thread_id': thread_id})}\n\n"

        queue: asyncio.Queue = asyncio.Queue()
        stop_event = asyncio.Event()
        _stop_events[thread_id] = stop_event

        async def run_agent():
            try:
                # agent uses a sync-style emit; adapt with a simple collector
                async def collect(ev: dict):
                    await queue.put(f"data: {json.dumps(ev)}\n\n")

                # Hard cap: a turn must never "think forever". If the model
                # legs stall, end the SSE stream with a clear error instead.
                try:
                    text = await asyncio.wait_for(
                        agent_mod.run_turn(history, collect,
                                           model=model,
                                           web_search=web_search,
                                           should_stop=stop_event.is_set),
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
            # A disconnect lands here too (client closed the tab, hit
            # Stop, network dropped). Flag the turn so the agent stops
            # at its next boundary instead of running the full turn and
            # storing a reply nobody is reading. Setting it after a
            # normal finish is a no-op: the flag is dropped below.
            stop_event.set()
            stop_ping.set()
            ping_task.cancel()
        await task
        # Only drop our own flag: two tabs can run turns on one thread,
        # and the second turn's flag must survive the first turn ending.
        if _stop_events.get(thread_id) is stop_event:
            _stop_events.pop(thread_id, None)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    _check_access(req.access_code)
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="empty message")

    # Per-deployment refusal policy (app/guard_policy.py): this instance's
    # own owner-written patterns. Ships empty — nothing extra refused until
    # the owner adds an entry. The hard floor above always applies first.
    hit = check_hard_boundary(message) or check_owner_boundary(message)
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
    return _stream_turn(thread_id, history, model=req.model or "",
                        web_search=req.web_search)


@app.post("/api/threads/{thread_id}/regenerate")
async def thread_regenerate(thread_id: str, req: RegenerateRequest):
    """Re-run the last turn: drop the stored assistant reply (if one
    landed) and answer the same last user message again, streaming
    the new reply over SSE exactly like /api/chat.

    The old reply is dropped before the turn runs so the model never
    sees the stale answer in its history. When the last message is
    the user's own (the reply never got stored — the turn errored
    before saving), there is nothing to drop and the same prompt
    just runs again. A last user message that hits the hard boundary
    now is refused the same way /api/chat refuses it, and the stored
    reply is replaced with that refusal instead of re-running the
    agent on a blocked prompt.
    """
    _check_access(req.access_code)
    if not memory.thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="no such thread")
    messages = memory.get_all_messages(thread_id)
    if not messages:
        raise HTTPException(status_code=400, detail="empty thread")
    last_user = next((m for m in reversed(messages)
                      if m["role"] == "user"), None)
    if last_user is None:
        raise HTTPException(status_code=400,
                            detail="no user message to regenerate from")
    hit = (check_hard_boundary(last_user["content"])
           or check_owner_boundary(last_user["content"]))
    if hit:
        memory.drop_last_assistant(thread_id)
        reply = f"{hit.reason} {hit.alternative}"
        memory.add_message(thread_id, "assistant", reply)
        return {"thread_id": thread_id, "reply": reply, "blocked": True}
    memory.drop_last_assistant(thread_id)
    history = memory.get_history(thread_id)
    return _stream_turn(thread_id, history, model=req.model or "",
                        web_search=req.web_search)


@app.post("/api/threads/{thread_id}/stop")
def thread_stop(thread_id: str, req: StopRequest):
    """Stop the turn currently running on this thread, server-side.

    Before this endpoint the UI Stop button only aborted the browser's
    fetch: the server turn kept running to its 11-minute cap and stored
    the full reply as if the user had never stopped it. This sets the
    turn's stop flag, so the agent loop exits at its next model/tool
    boundary, emits {"type": "stopped"}, and stores the partial reply
    with a "[stopped by user]" marker.

    Returns {"stopped": true, "running": true} when a turn was in
    flight, {"stopped": false, "running": false} when the thread is
    idle — stopping an idle thread is a no-op, not an error. Auth is
    checked before the thread lookup, like the other thread routes.
    """
    _check_access(req.access_code)
    if not memory.thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="no such thread")
    event = _stop_events.get(thread_id)
    if event is None:
        return {"stopped": False, "running": False}
    event.set()
    return {"stopped": True, "running": True}


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
