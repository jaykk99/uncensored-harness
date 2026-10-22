"""Tests for app/tools.py safety rails and web helpers.

Importing app.tools creates WORKDIR dirs as a side effect; tests redirect
WORKDIR-independent pieces only (_safe_path/_blocked_url are pure).
Network paths are stubbed via monkeypatched _aclient factories.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from app import tools as T
from tests.conftest import run


# ---------------------------------------------------------------------------
# _safe_path — workdir sandboxing
# ---------------------------------------------------------------------------

def test_safe_path_normal_relative():
    p = T._safe_path("sub/dir/file.txt")
    assert p == os.path.join(os.path.realpath(T.WORKDIR), "sub/dir/file.txt")


def test_safe_path_dot_resolves_to_workdir():
    assert T._safe_path(".") == os.path.realpath(T.WORKDIR)


def test_safe_path_rejects_parent_escape():
    with pytest.raises(ValueError, match="escapes workdir"):
        T._safe_path("../outside.txt")


def test_safe_path_rejects_nested_escape():
    with pytest.raises(ValueError, match="escapes workdir"):
        T._safe_path("sub/../../etc/passwd")


def test_safe_path_rejects_absolute_outside():
    with pytest.raises(ValueError, match="escapes workdir"):
        T._safe_path("/etc/passwd")


def test_safe_path_allows_dotdot_that_stays_inside():
    p = T._safe_path("sub/../file.txt")
    assert p == os.path.join(os.path.realpath(T.WORKDIR), "file.txt")


# ---------------------------------------------------------------------------
# _blocked_url — SSRF guard
# ---------------------------------------------------------------------------

BLOCKED = [
    "http://localhost:8000/x",
    "http://127.0.0.1/",
    "http://[::1]/",
    "http://10.0.0.5/admin",
    "http://192.168.1.1/",
    "http://172.16.9.9/",
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata
    "http://metadata.google.internal/",
    "http://x.local/",
    "ftp://example.com/file",
    "not a url",
    "http:///nohost",
]

ALLOWED = [
    "https://example.com/",
    "http://example.com:8080/path?q=1",
    "https://sub.domain.co.uk/deep/page.html",
]


def _public_dns(monkeypatch):
    """Stub DNS so _blocked_url's rebinding check is offline-deterministic."""
    monkeypatch.setattr(
        T.socket, "getaddrinfo",
        lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])


@pytest.mark.parametrize("url", BLOCKED)
def test_blocked_urls_rejected(url, monkeypatch):
    _public_dns(monkeypatch)
    assert T._blocked_url(url) is not None, f"should block: {url}"


@pytest.mark.parametrize("url", ALLOWED)
def test_public_urls_allowed(url, monkeypatch):
    _public_dns(monkeypatch)
    assert T._blocked_url(url) is None, f"should allow: {url}"


def test_blocked_url_dns_rebinding(monkeypatch):
    # A hostname that resolves to a private IP must be blocked even though
    # the name itself looks public.
    monkeypatch.setattr(
        T.socket, "getaddrinfo",
        lambda host, port, *a, **k: [(2, 1, 6, "", ("127.0.0.1", 0))])
    reason = T._blocked_url("https://totally-public.example/")
    assert reason is not None and "non-public IP" in reason


def test_blocked_url_dns_failure_fails_open(monkeypatch):
    # Unresolvable names fail open — the connection errors out on its own.
    def boom(host, port, *a, **k):
        raise OSError("no dns")
    monkeypatch.setattr(T.socket, "getaddrinfo", boom)
    assert T._blocked_url("https://example.com/") is None


# ---------------------------------------------------------------------------
# _html_to_text
# ---------------------------------------------------------------------------

def test_html_to_text_strips_tags_and_scripts():
    html = ("<html><head><title>T</title><style>.a{}</style></head>"
            "<body><script>alert(1)</script><h1>Hi</h1><p>World&nbsp;&amp;&nbsp;Co</p></body></html>")
    text = T._html_to_text(html)
    assert "alert" not in text
    assert ".a{}" not in text
    assert "Hi" in text and "World & Co" in text
    assert "<" not in text


def test_html_to_text_truncates():
    assert len(T._html_to_text("<p>" + "x" * 9000 + "</p>", max_chars=100)) <= 100


# ---------------------------------------------------------------------------
# web_search fallback: bing first, ddg on failure (stubbed HTTP)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, text="", status_code=200):
        self.text = text
        self.status_code = status_code
        self.headers = {"content-type": "text/html"}


class _FakeClient:
    """Minimal async context manager mimicking the httpx surface tools use."""
    instances: list = []

    def __init__(self, handler):
        self._handler = handler
        _FakeClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kw):
        return self._handler(url, kw)


BING_RSS = """<?xml version="1.0"?>
<rss><channel>
<item><title>Result One</title><link>https://one.example/</link>
<description>First <b>snippet</b>.</description></item>
<item><title>Result Two</title><link>https://two.example/</link>
<description>Second snippet.</description></item>
</channel></rss>"""


def _client_returning(text=None, exc=None):
    def factory(timeout, **kw):
        def handler(url, kw2):
            if exc is not None:
                raise exc
            return _FakeResp(text or "")
        return _FakeClient(handler)
    return factory


def test_web_search_uses_bing_first(monkeypatch):
    seen = []
    monkeypatch.setattr(T, "_aclient", _client_returning(BING_RSS))
    out = run(T.web_search("test query", max_results=5))
    assert "Result One" in out and "https://one.example/" in out
    assert "First snippet." in out  # tags stripped


def test_web_search_falls_back_to_ddg(monkeypatch):
    ddg_html = ('<a class="result__a" href="https://ddg.example/">DDG Title</a>'
                ' junk <a class="result__snippet" href="#">DDG Snip</a>')
    calls = {"n": 0}

    def factory(timeout, **kw):
        def handler(url, kw2):
            calls["n"] += 1
            if "bing.com" in url:
                raise RuntimeError("bing down")
            return _FakeResp(ddg_html)
        return _FakeClient(handler)

    monkeypatch.setattr(T, "_aclient", factory)
    out = run(T.web_search("q"))
    assert "DDG Title" in out and "https://ddg.example/" in out


def test_web_search_empty_when_both_fail(monkeypatch):
    monkeypatch.setattr(T, "_aclient",
                        _client_returning(exc=RuntimeError("down")))
    assert run(T.web_search("q")) == "No results found."


def test_web_fetch_blocks_ssrf_urls(monkeypatch):
    # Must not even construct an HTTP client for blocked hosts.
    def boom(*a, **k):
        raise AssertionError("client must not be created for blocked URL")
    monkeypatch.setattr(T, "_aclient", boom)
    out = run(T.web_fetch("http://127.0.0.1/admin"))
    assert out.startswith("Blocked:")


def test_web_fetch_ok_path(monkeypatch):
    _public_dns(monkeypatch)
    html = "<html><head><title>My Page</title></head><body><p>Hello</p></body></html>"
    resp = _FakeResp(html)
    resp.headers = {"content-type": "text/html"}
    monkeypatch.setattr(T, "_aclient", _client_returning(html))
    out = run(T.web_fetch("https://example.com/"))
    assert "My Page" in out and "Hello" in out


def _redir_resp(location: str) -> _FakeResp:
    r = _FakeResp("", status_code=302)
    r.headers = {"location": location}
    return r


def test_web_fetch_blocks_redirect_to_private(monkeypatch):
    # A public URL that 302s to 169.254.169.254 must be refused WITHOUT
    # ever requesting the private hop.
    _public_dns(monkeypatch)
    seen: list = []

    def factory(timeout, **kw):
        def handler(url, kw2):
            seen.append(url)
            return _redir_resp("http://169.254.169.254/latest/meta-data/")
        return _FakeClient(handler)

    monkeypatch.setattr(T, "_aclient", factory)
    out = run(T.web_fetch("https://example.com/start"))
    assert out.startswith("Blocked redirect:")
    assert seen == ["https://example.com/start"]


def test_web_fetch_follows_legit_redirect(monkeypatch):
    _public_dns(monkeypatch)
    final = _FakeResp(
        "<html><head><title>End</title></head><body><p>Arrived</p></body></html>")

    def factory(timeout, **kw):
        def handler(url, kw2):
            return final if url.endswith("/final") else _redir_resp("/final")
        return _FakeClient(handler)

    monkeypatch.setattr(T, "_aclient", factory)
    out = run(T.web_fetch("https://example.com/start"))
    assert "Arrived" in out
    assert "URL: https://example.com/final" in out


def test_web_fetch_too_many_redirects(monkeypatch):
    _public_dns(monkeypatch)

    def factory(timeout, **kw):
        def handler(url, kw2):
            return _redir_resp("/loop")
        return _FakeClient(handler)

    monkeypatch.setattr(T, "_aclient", factory)
    out = run(T.web_fetch("https://example.com/loop"))
    assert out.startswith("Too many redirects")


def test_browser_fill_error_returns_dict(monkeypatch):
    async def boom():
        raise RuntimeError("no browser here")
    monkeypatch.setattr(T, "_get_page", boom)
    res = run(T.browser_fill("#missing", "x"))
    assert isinstance(res, dict)
    assert res["ok"] is False
    assert "browser_fill failed" in res["error"]


# ---------------------------------------------------------------------------
# Tool registry shape
# ---------------------------------------------------------------------------

def test_openai_tools_shape():
    tools = T.openai_tools()
    assert len(tools) >= 10
    for t in tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert T.get_tool(fn["name"]) is not None


def test_get_tool_unknown_returns_none():
    assert T.get_tool("does_not_exist") is None
