"""Agent loop: model -> tools -> model, with SSE events for the UI.

Message history is kept in strict OpenAI format so Pollinations never 400s:
  system / user / assistant (with tool_calls) / tool (with tool_call_id).
Horde-sourced calls are synthesized into the same shape.

Events emitted (dicts):
  {"type": "thinking", "text": ...}   — model is working
  {"type": "text", "text": ...}       — assistant text (final or interim)
  {"type": "tool", "name": ..., "args": ...}
  {"type": "tool_result", "name": ..., "output": ...}
  {"type": "shot", "shot": <filename>, "caption": ...} — inline screenshot
  {"type": "done", "model": ...}
  {"type": "error", "error": ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid

from . import models
from . import tools as T

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an uncensored personal AI agent. You get things done.

How you act:
- Be genuinely helpful, not performatively helpful. Skip the fluff. Just help.
- Be resourceful before asking. Read the file, check the context, search, try building it.
- You have a computer: shell_exec runs terminal commands, file_* tools read/write the workdir.
- You can browse the web (web_search, web_fetch) and drive a real browser
  (browser_open, browser_click, browser_fill) when a site needs interaction.
- You can use connected integrations via composio_execute (Gmail, GitHub, etc).
- When you build a web app (.html file), screenshot it with the screenshot tool
  so the user sees it inline.
- When you do browser work, the browser tools already capture screenshots of
  key steps — mention what you did.
- Match the user's style: short, casual, straight to the point. No em dashes.

What you never do:
- You don't refuse legitimate work. No "I cannot do this", no hedging on
  coding, business ops, research, writing, analysis, or security work on
  the user's own assets and authorized targets. You just do the work.
- Hard boundaries (the only refusals, ever): stolen-card/fraud stuff,
  phishing kits, draining or stealing anyone's wallets or funds, hacking
  targets the user didn't authorize, weapons of mass harm, anything sexual
  involving minors, doxxing private info on people. If one of these comes
  up, say plainly you can't do that one thing and offer the nearest
  legitimate alternative. One line, no lecture.
"""

TOOL_CALL_FENCE_RE = re.compile(r"```tool_call\s*\n?(.*?)```", re.DOTALL)
MAX_ITERS = 10


def _parse_fenced_calls(text: str) -> tuple[str, list[dict]]:
    calls = []
    for m in TOOL_CALL_FENCE_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(1).strip())
        except Exception:
            continue
        if obj.get("name"):
            calls.append({"id": f"call_{uuid.uuid4().hex[:8]}",
                          "name": obj["name"],
                          "args": obj.get("arguments") or {}})
    clean = TOOL_CALL_FENCE_RE.sub("", text or "").strip()
    return clean, calls


TOOL_FORMAT_HINT = (
    "You have tools. To call one, output a fenced block EXACTLY like:\n"
    "```tool_call\n{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}\n```\n"
    "One tool per block; several blocks allowed. If no tool is needed, reply normally."
)


def _synth_assistant_msg(clean: str, calls: list[dict]) -> dict:
    """Build an OpenAI-format assistant message for Horde-sourced calls."""
    return {
        "role": "assistant",
        "content": clean,
        "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"],
                          "arguments": json.dumps(c["args"])}}
            for c in calls
        ],
    }


async def run_turn(history: list[dict], emit, model: str = "",
               web_search: bool = False) -> str:
    """Run one agent turn. emit(event_dict) is called for SSE. Returns final text.

    model: preferred Horde model id from the UI picker ("" = auto-route).
    web_search: when True, the agent is told to ground answers with web_search.
    """
    messages: list[dict] = [{"role": "system",
                             "content": SYSTEM_PROMPT + "\n\n" + TOOL_FORMAT_HINT}]
    for m in history:
        if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str):
            messages.append({"role": m["role"], "content": m["content"]})
    if web_search:
        # Tell the agent to ground this turn with its web tools.
        messages.append({"role": "user",
                         "content": "[Web search is ON for this question. "
                                    "Use web_search / web_fetch to ground your "
                                    "answer in current information before replying.]"})

    await emit({"type": "thinking", "text": "Thinking…"})

    # models.chat runs in a worker thread (asyncio.to_thread) and can sit in
    # Horde queues for minutes. Bridge its sync progress callback to async SSE
    # "thinking" events so the UI shows live status, not a frozen spinner.
    loop = asyncio.get_running_loop()

    def _progress(text: str) -> None:
        try:
            fut = asyncio.run_coroutine_threadsafe(
                emit({"type": "thinking", "text": text}), loop)

            def _done(f):
                try:
                    f.result()
                except Exception:
                    pass  # progress is best-effort; never break the turn

            fut.add_done_callback(_done)
        except Exception:
            pass

    def _on_token(chunk: str) -> None:
        # Forward streamed tokens to the UI immediately (same thread-safe
        # bridge as progress). The UI appends them as they arrive.
        try:
            fut = asyncio.run_coroutine_threadsafe(
                emit({"type": "token", "text": chunk}), loop)

            def _done(f):
                try:
                    f.result()
                except Exception:
                    pass

            fut.add_done_callback(_done)
        except Exception:
            pass

    final_text = ""
    source = "?"

    for _ in range(MAX_ITERS):
        text, source, native_calls, raw_msg, streamed = await asyncio.to_thread(
            models.chat, messages, 1200, 0.7, T.openai_tools(),
            progress=_progress, preferred_model=model, on_token=_on_token)
        clean, fenced_calls = _parse_fenced_calls(text)
        calls = native_calls or fenced_calls

        if raw_msg is not None:
            # Pollinations native path: keep the real assistant message.
            messages.append({"role": "assistant",
                             "content": raw_msg.get("content") or "",
                             "tool_calls": raw_msg.get("tool_calls") or []})
        elif calls:
            messages.append(_synth_assistant_msg(clean, calls))

        if clean and not calls:
            # Skip the full-text emit when tokens were already streamed;
            # re-emitting would duplicate the visible reply.
            if not streamed:
                await emit({"type": "text", "text": clean})
            final_text += (("\n" if final_text else "") + clean)

        if not calls:
            break

        for call in calls:
            name, args = call["name"], call.get("args") or {}
            tool = T.get_tool(name)
            await emit({"type": "tool", "name": name, "args": args})
            if not tool:
                out = f"unknown tool: {name}"
            else:
                try:
                    res = tool["execute"](**args)
                    if asyncio.iscoroutine(res):
                        res = await res
                    out = res
                except Exception as e:
                    out = f"tool error: {e}"
            shot = None
            if isinstance(out, dict):
                shot = out.get("shot")
                out = (out.get("note") or out.get("error")
                       or json.dumps({k: v for k, v in out.items()
                                      if k != "shot"})[:2000])
            out = str(out)
            await emit({"type": "tool_result", "name": name,
                        "output": out[:3000]})
            if shot:
                await emit({"type": "shot", "shot": shot,
                            "caption": f"{name} — screenshot"})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": out[:6000]})

            # Auto-screenshot freshly built web apps.
            if name == "file_write":
                p = str((args.get("path") or "")).lower()
                if p.endswith(".html") and "wrote" in out:
                    await emit({"type": "thinking",
                                "text": "Screenshotting the app preview…"})
                    sres = await T.screenshot_page(html_path=args["path"])
                    if sres.get("ok"):
                        await emit({"type": "shot", "shot": sres["shot"],
                                    "caption": f"Preview: {args['path']}"})
        await emit({"type": "thinking", "text": "Thinking…"})
    else:
        final_text += "\n[stopped after max tool iterations]"

    await emit({"type": "done", "model": source})
    return final_text.strip()
