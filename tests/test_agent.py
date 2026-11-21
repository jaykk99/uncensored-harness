"""Tests for app/agent.py — the model -> tools -> model loop.

models.chat is stubbed with canned turns (no network); tool execution
runs against the real registry with stubbed tool functions where the
test needs them. Async is driven with tests.conftest.run.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from app import agent, models
from app.agent import _parse_fenced_calls, _synth_assistant_msg, run_turn
from tests.conftest import run


# ---------------------------------------------------------------------------
# _parse_fenced_calls
# ---------------------------------------------------------------------------

def test_parse_single_fenced_call():
    text = ('Let me check.\n```tool_call\n'
            '{"name": "web_search", "arguments": {"query": "x"}}\n```\nDone.')
    clean, calls = _parse_fenced_calls(text)
    assert clean == "Let me check.\n\nDone."
    assert len(calls) == 1
    assert calls[0]["name"] == "web_search"
    assert calls[0]["args"] == {"query": "x"}
    assert calls[0]["id"].startswith("call_")


def test_parse_multiple_fenced_calls():
    text = ('```tool_call\n{"name": "a", "arguments": {}}\n```\n'
            '```tool_call\n{"name": "b", "arguments": {"k": 1}}\n```')
    clean, calls = _parse_fenced_calls(text)
    assert clean == ""
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[1]["args"] == {"k": 1}


def test_parse_malformed_fence_skipped():
    text = '```tool_call\nnot json {\n```\nplain text'
    clean, calls = _parse_fenced_calls(text)
    assert calls == []
    assert clean == "plain text"


def test_parse_fence_without_name_skipped():
    text = '```tool_call\n{"arguments": {"k": 1}}\n```'
    assert _parse_fenced_calls(text)[1] == []


def test_parse_no_fence():
    assert _parse_fenced_calls("just an answer") == ("just an answer", [])
    assert _parse_fenced_calls("") == ("", [])
    assert _parse_fenced_calls(None) == ("", [])


def test_parse_missing_arguments_defaults_empty():
    _, calls = _parse_fenced_calls('```tool_call\n{"name": "x"}\n```')
    assert calls[0]["args"] == {}


# ---------------------------------------------------------------------------
# _synth_assistant_msg
# ---------------------------------------------------------------------------

def test_synth_assistant_msg_shape():
    msg = _synth_assistant_msg("working on it", [
        {"id": "call_1", "name": "shell_exec", "args": {"command": "ls"}}])
    assert msg["role"] == "assistant"
    assert msg["content"] == "working on it"
    (tc,) = msg["tool_calls"]
    assert tc["id"] == "call_1" and tc["type"] == "function"
    assert tc["function"]["name"] == "shell_exec"
    assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}


def test_synth_assistant_msg_no_calls():
    msg = _synth_assistant_msg("done", [])
    assert msg["tool_calls"] == []


# ---------------------------------------------------------------------------
# run_turn — stubbed models.chat
# ---------------------------------------------------------------------------

class _ChatStub:
    """Canned chat turns. Each call pops the next (text, source, calls,
    raw_msg, streamed) tuple; progress/on_token callbacks are exercised
    when the turn supplies them."""

    def __init__(self, turns, progress_texts=None, tokens=None):
        self.turns = list(turns)
        self.calls = []
        self.progress_texts = progress_texts or []
        self.tokens = tokens or []

    def __call__(self, messages, *a, **k):
        self.calls.append(messages)
        for t in self.progress_texts:
            k.get("progress") and k["progress"](t)
        for tok in self.tokens:
            k.get("on_token") and k["on_token"](tok)
        return self.turns.pop(0)


def _events():
    seen = []

    async def emit(ev):
        seen.append(ev)
    return seen, emit


def test_run_turn_plain_answer(monkeypatch):
    stub = _ChatStub([("Hello there", "kilo/m", [], None, False)])
    monkeypatch.setattr(models, "chat", stub)
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "hi"}], emit))
    assert out == "Hello there"
    assert {"type": "thinking", "text": "Thinking…"} in seen
    assert {"type": "text", "text": "Hello there"} in seen
    assert {"type": "done", "model": "kilo/m"} in seen
    # History reaches the model: system prompt + user message.
    assert stub.calls[0][0]["role"] == "system"
    assert stub.calls[0][1] == {"role": "user", "content": "hi"}


def test_run_turn_streamed_answer_not_reemitted(monkeypatch):
    stub = _ChatStub([("streamed text", "pollinations", [], None, True)])
    monkeypatch.setattr(models, "chat", stub)
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "hi"}], emit))
    assert out == "streamed text"
    assert not [e for e in seen if e["type"] == "text"]
    assert {"type": "done", "model": "pollinations"} in seen


def test_run_turn_native_tool_call_then_answer(monkeypatch):
    calls = [{"id": "c1", "name": "shell_exec", "args": {"command": "echo hi"}}]
    stub = _ChatStub([
        ("", "pollinations", calls, {"content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "shell_exec",
                          "arguments": '{"command": "echo hi"}'}}]}, False),
        ("The output was hi", "pollinations", [], None, False),
    ])
    monkeypatch.setattr(models, "chat", stub)

    async def fake_shell(command, timeout=60):
        return "$ echo hi\n(exit 0)\nhi"
    from app import tools as T
    tool = T.get_tool("shell_exec")
    monkeypatch.setitem(tool, "execute", lambda command, timeout=60: fake_shell(command, timeout))

    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "run it"}], emit))
    assert out == "The output was hi"
    tool_evs = [e for e in seen if e["type"] == "tool"]
    assert tool_evs and tool_evs[0]["name"] == "shell_exec"
    res_evs = [e for e in seen if e["type"] == "tool_result"]
    assert res_evs and "hi" in res_evs[0]["output"]
    # Second model call saw the tool result in history.
    assert stub.calls[1][-1]["role"] == "tool"
    assert "hi" in stub.calls[1][-1]["content"]


def test_run_turn_fenced_tool_call(monkeypatch):
    text = ('Checking.\n```tool_call\n'
            '{"name": "file_list", "arguments": {}}\n```')
    stub = _ChatStub([
        (text, "horde", [], None, False),
        ("Files listed.", "horde", [], None, False),
    ])
    monkeypatch.setattr(models, "chat", stub)
    from app import tools as T
    tool = T.get_tool("file_list")
    monkeypatch.setitem(tool, "execute", lambda path=".": "a.txt\nb.txt")
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "list files"}], emit))
    # The "Checking." preamble is not part of the final reply: clean text
    # is only accumulated when the turn makes no tool calls.
    assert out == "Files listed."
    assert [e for e in seen if e["type"] == "tool" and e["name"] == "file_list"]
    # Horde-sourced calls get a synthesized assistant message in history
    # (after the system prompt + user message).
    assert stub.calls[1][2]["role"] == "assistant"
    assert stub.calls[1][2]["tool_calls"][0]["function"]["name"] == "file_list"


def test_run_turn_unknown_tool(monkeypatch):
    calls = [{"id": "c1", "name": "nope_tool", "args": {}}]
    stub = _ChatStub([
        ("", "horde", calls, None, False),
        ("Sorry, that tool does not exist.", "horde", [], None, False),
    ])
    monkeypatch.setattr(models, "chat", stub)
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "hi"}], emit))
    assert out == "Sorry, that tool does not exist."
    res = [e for e in seen if e["type"] == "tool_result"][0]
    assert res["output"] == "unknown tool: nope_tool"


def test_run_turn_tool_exception(monkeypatch):
    calls = [{"id": "c1", "name": "shell_exec", "args": {"command": "x"}}]
    stub = _ChatStub([
        ("", "horde", calls, None, False),
        ("It errored.", "horde", [], None, False),
    ])
    monkeypatch.setattr(models, "chat", stub)
    from app import tools as T
    tool = T.get_tool("shell_exec")
    monkeypatch.setitem(tool, "execute",
                        lambda command, timeout=60: (_ for _ in ()).throw(RuntimeError("boom")))
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "hi"}], emit))
    assert out == "It errored."
    res = [e for e in seen if e["type"] == "tool_result"][0]
    assert res["output"] == "tool error: boom"


def test_run_turn_screenshot_result_emits_shot(monkeypatch):
    calls = [{"id": "c1", "name": "screenshot",
              "args": {"url": "https://example.com/"}}]
    stub = _ChatStub([
        ("", "horde", calls, None, False),
        ("Screenshotted.", "horde", [], None, False),
    ])
    monkeypatch.setattr(models, "chat", stub)
    from app import tools as T
    tool = T.get_tool("screenshot")
    monkeypatch.setitem(
        tool, "execute",
        lambda url="", html_path="": {"ok": True, "shot": "page-1.png",
                                      "note": "screenshotted x"})
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "shot it"}], emit))
    assert out == "Screenshotted."
    shots = [e for e in seen if e["type"] == "shot"]
    assert shots and shots[0]["shot"] == "page-1.png"
    res = [e for e in seen if e["type"] == "tool_result"][0]
    assert res["output"] == "screenshotted x"


def test_run_turn_file_write_html_auto_screenshots(monkeypatch):
    calls = [{"id": "c1", "name": "file_write",
              "args": {"path": "app.html", "content": "<h1>hi</h1>"}}]
    stub = _ChatStub([
        ("", "horde", calls, None, False),
        ("Built it.", "horde", [], None, False),
    ])
    monkeypatch.setattr(models, "chat", stub)
    from app import tools as T
    tool = T.get_tool("file_write")
    monkeypatch.setitem(tool, "execute",
                        lambda path, content: "wrote 14 chars to app.html")

    async def fake_shot(html_path=None, **k):
        return {"ok": True, "shot": "page-2.png"}
    monkeypatch.setattr(T, "screenshot_page", fake_shot)
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "build an app"}], emit))
    assert out == "Built it."
    shots = [e for e in seen if e["type"] == "shot"]
    assert shots and shots[0]["shot"] == "page-2.png"
    assert shots[0]["caption"] == "Preview: app.html"


def test_run_turn_file_write_non_html_no_screenshot(monkeypatch):
    calls = [{"id": "c1", "name": "file_write",
              "args": {"path": "notes.txt", "content": "x"}}]
    stub = _ChatStub([
        ("", "horde", calls, None, False),
        ("Wrote it.", "horde", [], None, False),
    ])
    monkeypatch.setattr(models, "chat", stub)
    from app import tools as T
    tool = T.get_tool("file_write")
    monkeypatch.setitem(tool, "execute",
                        lambda path, content: "wrote 1 chars to notes.txt")

    async def boom(**k):
        raise AssertionError("no screenshot expected")
    monkeypatch.setattr(T, "screenshot_page", boom)
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "write notes"}], emit))
    assert out == "Wrote it."
    assert not [e for e in seen if e["type"] == "shot"]


def test_run_turn_stop_before_first_call(monkeypatch):
    stub = _ChatStub([("never used", "horde", [], None, False)])
    monkeypatch.setattr(models, "chat", stub)
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "hi"}], emit,
                       should_stop=lambda: True))
    assert out == "[stopped by user]"
    assert {"type": "stopped"} in seen
    assert stub.calls == []  # model never called


def test_run_turn_stop_between_tool_calls(monkeypatch):
    calls = [
        {"id": "c1", "name": "shell_exec", "args": {"command": "echo a"}},
        {"id": "c2", "name": "shell_exec", "args": {"command": "echo b"}},
    ]
    stub = _ChatStub([("", "horde", calls, None, False)])
    monkeypatch.setattr(models, "chat", stub)
    from app import tools as T
    tool = T.get_tool("shell_exec")
    monkeypatch.setitem(tool, "execute",
                        lambda command, timeout=60: f"ran {command}")
    state = {"n": 0}

    def stop_after_first():
        return state["n"] >= 1
    seen, emit = _events()

    orig_emit = emit

    async def counting_emit(ev):
        if ev["type"] == "tool_result":
            state["n"] += 1
        await orig_emit(ev)

    out = run(run_turn([{"role": "user", "content": "hi"}], counting_emit,
                       should_stop=stop_after_first))
    assert "[stopped by user]" in out
    assert state["n"] == 1  # second tool call never ran


def test_run_turn_max_iterations(monkeypatch):
    calls = [{"id": "c1", "name": "shell_exec", "args": {"command": "echo x"}}]
    stub = _ChatStub([("", "horde", calls, None, False)] * (agent.MAX_ITERS + 2))
    monkeypatch.setattr(models, "chat", stub)
    from app import tools as T
    tool = T.get_tool("shell_exec")
    monkeypatch.setitem(tool, "execute", lambda command, timeout=60: "ok")
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "hi"}], emit))
    assert "[stopped after max tool iterations]" in out
    assert len(stub.calls) == agent.MAX_ITERS


def test_run_turn_extra_instructions_and_web_search(monkeypatch):
    stub = _ChatStub([("done", "horde", [], None, False)])
    monkeypatch.setattr(models, "chat", stub)
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "hi"}], emit,
                       web_search=True, extra_instructions="Answer in French."))
    assert out == "done"
    msgs = stub.calls[0]
    assert msgs[0]["role"] == "system"
    assert "Answer in French." in msgs[1]["content"]
    assert msgs[1]["role"] == "system"
    assert msgs[2] == {"role": "user", "content": "hi"}
    assert "Web search is ON" in msgs[3]["content"]


def test_run_turn_history_filters_non_text_roles(monkeypatch):
    stub = _ChatStub([("ok", "horde", [], None, False)])
    monkeypatch.setattr(models, "chat", stub)
    seen, emit = _events()
    history = [
        {"role": "user", "content": "first"},
        {"role": "tool", "content": "should be dropped"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": [{"type": "text", "text": "list"}]},
    ]
    run(run_turn(history, emit))
    msgs = stub.calls[0]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant"]


def test_run_turn_progress_and_tokens_bridged(monkeypatch):
    stub = _ChatStub([("done", "horde", [], None, False)],
                     progress_texts=["working…"], tokens=["to", "ken"])
    monkeypatch.setattr(models, "chat", stub)
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "hi"}], emit))
    assert out == "done"
    assert {"type": "thinking", "text": "working…"} in seen
    assert {"type": "token", "text": "to"} in seen
    assert {"type": "token", "text": "ken"} in seen


def test_run_turn_model_error_propagates(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("all legs failed")
    monkeypatch.setattr(models, "chat", boom)
    seen, emit = _events()
    with pytest.raises(RuntimeError, match="all legs failed"):
        run(run_turn([{"role": "user", "content": "hi"}], emit))


def test_run_turn_tool_result_truncated(monkeypatch):
    calls = [{"id": "c1", "name": "shell_exec", "args": {"command": "x"}}]
    stub = _ChatStub([
        ("", "horde", calls, None, False),
        ("done", "horde", [], None, False),
    ])
    monkeypatch.setattr(models, "chat", stub)
    from app import tools as T
    tool = T.get_tool("shell_exec")
    monkeypatch.setitem(tool, "execute",
                        lambda command, timeout=60: "z" * 5000)
    seen, emit = _events()
    run(run_turn([{"role": "user", "content": "hi"}], emit))
    res = [e for e in seen if e["type"] == "tool_result"][0]
    assert len(res["output"]) == 3000
    # History copy is capped at 6000.
    assert len(stub.calls[1][-1]["content"]) == 5000


def test_run_turn_sync_tool_result(monkeypatch):
    calls = [{"id": "c1", "name": "file_list", "args": {}}]
    stub = _ChatStub([
        ("", "horde", calls, None, False),
        ("listed", "horde", [], None, False),
    ])
    monkeypatch.setattr(models, "chat", stub)
    from app import tools as T
    tool = T.get_tool("file_list")
    # A sync (non-coroutine) execute result is used as-is.
    monkeypatch.setitem(tool, "execute", lambda path=".": "sync result")
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "hi"}], emit))
    assert out == "listed"
    res = [e for e in seen if e["type"] == "tool_result"][0]
    assert res["output"] == "sync result"


# ---------------------------------------------------------------------------
# run_turn — progress/token bridge failure modes (best-effort by design)
# ---------------------------------------------------------------------------

class _SlowStub(_ChatStub):
    """Like _ChatStub, but lingers in the worker thread after firing the
    progress/token callbacks, so the event loop gets to run the bridged
    emits (and their done-callbacks) before the turn moves on."""

    def __call__(self, messages, *a, **k):
        out = super().__call__(messages, *a, **k)
        time.sleep(0.3)
        return out


def test_run_turn_progress_callback_error_swallowed(monkeypatch):
    # A bridged emit that raises fails the run_coroutine_threadsafe
    # future; _done's f.result() then raises and is swallowed, and the
    # turn completes normally — progress is best-effort, never fatal.
    stub = _SlowStub([("done", "horde", [], None, False)],
                     progress_texts=["working…"], tokens=["tok"])
    monkeypatch.setattr(models, "chat", stub)
    seen, emit = _events()

    async def flaky_emit(ev):
        if ev.get("text") in ("working…", "tok"):
            raise RuntimeError("sse gone")
        seen.append(ev)

    out = run(run_turn([{"role": "user", "content": "hi"}], flaky_emit))
    assert out == "done"
    assert {"type": "done", "model": "horde"} in seen
    assert not [e for e in seen if e.get("text") in ("working…", "tok")]


def test_run_turn_progress_bridge_failure_swallowed(monkeypatch):
    # If run_coroutine_threadsafe itself raises, both _progress and
    # _on_token swallow it and the turn still completes.
    def boom(coro, loop):
        coro.close()  # never started: close it so no warning is raised
        raise RuntimeError("loop gone")
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", boom)
    stub = _ChatStub([("done", "horde", [], None, False)],
                     progress_texts=["working…"], tokens=["tok"])
    monkeypatch.setattr(models, "chat", stub)
    seen, emit = _events()
    out = run(run_turn([{"role": "user", "content": "hi"}], emit))
    assert out == "done"
    assert {"type": "done", "model": "horde"} in seen
