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


# ---------------------------------------------------------------------------
# _proxy / _pw_proxy / _dns_blocked / _strip_tags
# ---------------------------------------------------------------------------

def test_proxy_helpers(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    assert T._proxy() is None
    assert T._pw_proxy() is None
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    assert T._proxy() is None  # loopback ignored
    monkeypatch.setenv("HTTPS_PROXY", "http://user:pass@proxy.example:8080")
    assert T._proxy() == "http://user:pass@proxy.example:8080"
    assert T._pw_proxy() == {"server": "http://proxy.example:8080",
                             "username": "user", "password": "pass"}
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    assert T._pw_proxy() == {"server": "http://proxy.example:8080"}


def test_dns_blocked_direct(monkeypatch):
    monkeypatch.setattr(T.socket, "getaddrinfo",
                        lambda h, p, *a, **k: [(2, 1, 6, "", ("8.8.8.8", 0))])
    assert T._dns_blocked("example.com") is None
    monkeypatch.setattr(T.socket, "getaddrinfo",
                        lambda h, p, *a, **k: [(2, 1, 6, "", ("10.1.2.3", 0))])
    assert "non-public IP" in (T._dns_blocked("example.com") or "")
    monkeypatch.setattr(T.socket, "getaddrinfo",
                        lambda h, p, *a, **k: (_ for _ in ()).throw(OSError("nx")))
    assert T._dns_blocked("example.com") is None


def test_strip_tags():
    assert T._strip_tags("<b>Hi</b> &amp; bye &quot;q&quot;") == 'Hi & bye "q"'
    assert T._strip_tags("  a   b  ") == "a b"


def test_verify_bundle(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    assert T._verify() == "/etc/ssl/certs/ca-certificates.crt"
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    assert T._verify() is True


# ---------------------------------------------------------------------------
# shell_exec / file_* (tmp workdir)
# ---------------------------------------------------------------------------

@pytest.fixture()
def _tmp_workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "WORKDIR", str(tmp_path))
    monkeypatch.setattr(T, "SHOTS_DIR", str(tmp_path / ".shots"))
    os.makedirs(tmp_path / ".shots", exist_ok=True)
    return tmp_path


def test_shell_exec_success(_tmp_workdir):
    out = run(T.shell_exec("echo hello"))
    assert "$ echo hello" in out
    assert "(exit 0)" in out
    assert "hello" in out


def test_shell_exec_nonzero_exit(_tmp_workdir):
    out = run(T.shell_exec("exit 3"))
    assert "(exit 3)" in out


def test_shell_exec_timeout(_tmp_workdir):
    out = run(T.shell_exec("sleep 5", timeout=1))
    assert out == "[timed out after 1s]"


def test_shell_exec_truncates(_tmp_workdir):
    out = run(T.shell_exec("python3 -c \"print('x'*13000)\""))
    assert "…[truncated]…" in out
    assert len(out) < 14000


def test_file_write_read_list_delete(_tmp_workdir):
    assert run(T.file_write("sub/note.txt", "hello world")) == "wrote 11 chars to sub/note.txt"
    assert run(T.file_read("sub/note.txt")) == "hello world"
    listing = run(T.file_list("sub"))
    assert listing == "note.txt"
    assert run(T.file_list(".")) == ".shots/\nsub/"  # fixture makes .shots/
    assert run(T.file_delete("sub/note.txt")) == "deleted sub/note.txt"
    assert "read error" in run(T.file_read("sub/note.txt"))


def test_file_read_truncates(_tmp_workdir):
    run(T.file_write("big.txt", "y" * 300))
    out = run(T.file_read("big.txt", max_chars=100))
    assert out.endswith("…[truncated]…")
    assert len(out) == 100 + len("\n…[truncated]…")


def test_file_list_empty_and_missing(_tmp_workdir):
    os.makedirs(_tmp_workdir / "empty", exist_ok=True)
    assert run(T.file_list("empty")) == "(empty)"
    assert "list error" in run(T.file_list("nope"))


def test_file_delete_directory(_tmp_workdir):
    run(T.file_write("d/f.txt", "x"))
    assert run(T.file_delete("d")) == "deleted d"
    assert "delete error" in run(T.file_delete("d"))


def test_file_ops_reject_escape(_tmp_workdir):
    assert "read error" in run(T.file_read("../outside.txt"))
    assert "write error" in run(T.file_write("../outside.txt", "x"))


# ---------------------------------------------------------------------------
# web_fetch edge cases
# ---------------------------------------------------------------------------

def test_web_fetch_http_error(monkeypatch):
    _public_dns(monkeypatch)
    monkeypatch.setattr(T, "_aclient",
                        _client_returning("", exc=None))
    # _client_returning with text="" returns a 200 empty body; use a 404 resp.
    def factory(timeout, **kw):
        return _FakeClient(lambda url, kw2: _FakeResp("", status_code=404))
    monkeypatch.setattr(T, "_aclient", factory)
    out = run(T.web_fetch("https://example.com/missing"))
    assert out == "HTTP 404 fetching https://example.com/missing"


def test_web_fetch_non_html_returns_raw(monkeypatch):
    _public_dns(monkeypatch)
    resp = _FakeResp('{"a": 1}')
    resp.headers = {"content-type": "application/json"}
    monkeypatch.setattr(T, "_aclient", _client_returning('{"a": 1}'))
    # _client_returning's resp has text/html headers; patch content-type via factory.
    def factory(timeout, **kw):
        return _FakeClient(lambda url, kw2: resp)
    monkeypatch.setattr(T, "_aclient", factory)
    out = run(T.web_fetch("https://example.com/api"))
    assert '{"a": 1}' in out
    assert "Title:" in out


def test_web_fetch_exception(monkeypatch):
    _public_dns(monkeypatch)
    monkeypatch.setattr(T, "_aclient",
                        _client_returning(exc=RuntimeError("conn reset")))
    out = run(T.web_fetch("https://example.com/"))
    assert out.startswith("Fetch failed: conn reset")


def test_web_fetch_redirect_without_location(monkeypatch):
    _public_dns(monkeypatch)
    r = _FakeResp("", status_code=302)
    r.headers = {}
    monkeypatch.setattr(T, "_aclient",
                        _client_returning(""))
    def factory(timeout, **kw):
        return _FakeClient(lambda url, kw2: r)
    monkeypatch.setattr(T, "_aclient", factory)
    out = run(T.web_fetch("https://example.com/start"))
    # A redirect with no Location is treated as the final response.
    assert out.startswith("URL: https://example.com/start")


# ---------------------------------------------------------------------------
# _bing_rss_search / _ddg_search internals
# ---------------------------------------------------------------------------

def test_bing_rss_http_error(monkeypatch):
    def factory(timeout, **kw):
        return _FakeClient(lambda url, kw2: _FakeResp("", status_code=500))
    monkeypatch.setattr(T, "_aclient", factory)
    with pytest.raises(RuntimeError, match="bing HTTP 500"):
        run(T._bing_rss_search("q", 5))


def test_bing_rss_skips_items_without_link(monkeypatch):
    rss = ('<rss><channel>'
           '<item><title>No link here</title><description>d</description></item>'
           '<item><title>Good</title><link>https://good.example/</link>'
           '<description>ok</description></item></channel></rss>')
    monkeypatch.setattr(T, "_aclient", _client_returning(rss))
    out = run(T._bing_rss_search("q", 5))
    assert len(out) == 1 and "Good" in out[0]


def test_bing_rss_respects_max_results(monkeypatch):
    items = "".join(
        f'<item><title>T{i}</title><link>https://x{i}.example/</link>'
        f'<description>d{i}</description></item>' for i in range(5))
    monkeypatch.setattr(T, "_aclient", _client_returning(f"<rss>{items}</rss>"))
    out = run(T._bing_rss_search("q", 2))
    assert len(out) == 2


def test_ddg_http_error(monkeypatch):
    def factory(timeout, **kw):
        return _FakeClient(lambda url, kw2: _FakeResp("", status_code=403))
    monkeypatch.setattr(T, "_aclient", factory)
    with pytest.raises(RuntimeError, match="ddg HTTP 403"):
        run(T._ddg_search("q", 5))


def test_ddg_parses_results(monkeypatch):
    html = ('<a class="result__a" href="https://a.example/">A Title</a>'
            ' x <a class="result__snippet" href="#">A snip</a>'
            '<a class="result__a" href="https://b.example/">B Title</a>'
            ' x <a class="result__snippet" href="#">B snip</a>')
    monkeypatch.setattr(T, "_aclient", _client_returning(html))
    out = run(T._ddg_search("q", 1))
    assert len(out) == 1 and "A Title" in out[0]


def test_web_search_bing_empty_falls_to_ddg(monkeypatch):
    # Bing returns 200 with no items: web_search falls through to DDG.
    ddg_html = ('<a class="result__a" href="https://ddg.example/">DDG</a>'
                ' x <a class="result__snippet" href="#">snip</a>')

    def factory(timeout, **kw):
        def handler(url, kw2):
            if "bing.com" in url:
                return _FakeResp("<rss><channel></channel></rss>")
            return _FakeResp(ddg_html)
        return _FakeClient(handler)
    monkeypatch.setattr(T, "_aclient", factory)
    out = run(T.web_search("q"))
    assert "DDG" in out


# ---------------------------------------------------------------------------
# _integration_call
# ---------------------------------------------------------------------------

def test_integration_call_str_passthrough():
    assert T._integration_call(lambda a, args: "raw text", "x", {}) == "raw text"


def test_integration_call_dict_json():
    out = T._integration_call(lambda a, args: {"k": 1}, "x", {})
    assert out.strip() == '{\n "k": 1\n}'


def test_integration_call_value_error():
    assert T._integration_call(
        lambda a, args: (_ for _ in ()).throw(ValueError("no token")),
        "x", {}).startswith("integration error: no token")


def test_integration_call_generic_error():
    out = T._integration_call(
        lambda a, args: (_ for _ in ()).throw(RuntimeError("boom")),
        "x", {})
    assert out == "integration error: boom"


def test_integration_call_unserializable_falls_back_to_str():
    class Odd:
        def __str__(self):
            return "odd-str"
    # A non-dict, non-str return value that json cannot serialize falls
    # back to str().
    out = T._integration_call(lambda a, args: Odd(), "x", {})
    assert out == "odd-str"


# ---------------------------------------------------------------------------
# screenshot / browser control (stubbed Playwright)
# ---------------------------------------------------------------------------

class _FakePage:
    def __init__(self):
        self.url = "https://example.com/page"
        self._closed = False

    async def goto(self, url, **kw):
        self.url = url

    async def wait_for_timeout(self, ms):
        return None

    async def screenshot(self, path=None, **kw):
        if path:
            open(path, "wb").write(b"PNG")
        return b"PNG"

    async def close(self):
        self._closed = True

    async def title(self):
        return "Fake Title"

    async def click(self, selector, **kw):
        return None

    async def fill(self, selector, text, **kw):
        return None

    async def inner_text(self, selector):
        return "page body text"

    def is_closed(self):
        return self._closed


class _FakeBrowser:
    def __init__(self):
        self.page = _FakePage()

    async def new_page(self, **kw):
        return self.page


def _stub_browser(monkeypatch):
    async def fake_get_browser():
        return _FakeBrowser()
    monkeypatch.setattr(T, "_get_browser", fake_get_browser)


def test_screenshot_page_requires_url_or_html():
    res = run(T.screenshot_page())
    assert res == {"ok": False, "error": "url or html_path required"}


def test_screenshot_page_missing_html_file(_tmp_workdir):
    res = run(T.screenshot_page(html_path="nope.html"))
    assert res["ok"] is False and "no such file" in res["error"]


def test_screenshot_page_html_path_escape(_tmp_workdir):
    res = run(T.screenshot_page(html_path="../outside.html"))
    assert res["ok"] is False and "escapes workdir" in res["error"]


def test_screenshot_page_blocked_url(monkeypatch):
    res = run(T.screenshot_page(url="http://127.0.0.1/x"))
    assert res["ok"] is False and res["error"].startswith("Blocked:")


def test_screenshot_page_url_success(monkeypatch, _tmp_workdir):
    _public_dns(monkeypatch)
    _stub_browser(monkeypatch)
    res = run(T.screenshot_page(url="https://example.com/"))
    assert res["ok"] is True
    assert res["shot"].startswith("page-") and res["shot"].endswith(".png")


def test_screenshot_page_html_success(monkeypatch, _tmp_workdir):
    _public_dns(monkeypatch)
    _stub_browser(monkeypatch)
    run(T.file_write("app.html", "<h1>hi</h1>"))
    res = run(T.screenshot_page(html_path="app.html"))
    assert res["ok"] is True and res["shot"].endswith(".png")


def test_screenshot_page_browser_failure(monkeypatch):
    _public_dns(monkeypatch)
    async def boom():
        raise RuntimeError("chromium down")
    monkeypatch.setattr(T, "_get_browser", boom)
    res = run(T.screenshot_page(url="https://example.com/"))
    assert res["ok"] is False and "screenshot failed" in res["error"]


def test_browser_open_blocked(monkeypatch):
    res = run(T.browser_open("http://169.254.169.254/"))
    assert res["ok"] is False and res["error"].startswith("Blocked:")


def test_browser_open_click_fill_text(monkeypatch, _tmp_workdir):
    _public_dns(monkeypatch)
    _stub_browser(monkeypatch)
    # _get_page uses the same browser stub via _get_browser.
    async def fake_get_page():
        return _FakeBrowser().page
    monkeypatch.setattr(T, "_get_page", fake_get_page)
    res = run(T.browser_open("https://example.com/"))
    assert res["ok"] is True and res["title"] == "Fake Title"
    assert res["shot"] is not None and res["shot"].startswith("browse-")
    res = run(T.browser_click("#btn"))
    assert res["ok"] is True
    res = run(T.browser_fill("#q", "hello"))
    assert res["ok"] is True
    assert run(T.browser_text()) == "page body text"


def test_browser_open_failure(monkeypatch):
    _public_dns(monkeypatch)
    async def boom():
        raise RuntimeError("no page")
    monkeypatch.setattr(T, "_get_page", boom)
    res = run(T.browser_open("https://example.com/"))
    assert res["ok"] is False and "browser_open failed" in res["error"]
    res = run(T.browser_click("#x"))
    assert res["ok"] is False and "browser_click failed" in res["error"]
    assert run(T.browser_text()).startswith("browser_text failed")


def test_page_shot_failure_returns_none(monkeypatch):
    async def boom():
        raise RuntimeError("no page")
    monkeypatch.setattr(T, "_get_page", boom)
    assert run(T._page_shot("x")) is None


# ---------------------------------------------------------------------------
# Tool registry execute lambdas (end to end through get_tool)
# ---------------------------------------------------------------------------

def test_tool_execute_lambdas(_tmp_workdir, monkeypatch):
    _public_dns(monkeypatch)
    monkeypatch.setattr(T, "_aclient", _client_returning(BING_RSS))
    shell = T.get_tool("shell_exec")
    assert "hello" in run(shell["execute"](command="echo hello"))
    wr = T.get_tool("file_write")
    assert "wrote" in run(wr["execute"](path="a.txt", content="hi"))
    rd = T.get_tool("file_read")
    assert run(rd["execute"](path="a.txt")) == "hi"
    ls = T.get_tool("file_list")
    assert "a.txt" in run(ls["execute"]())
    dl = T.get_tool("file_delete")
    assert "deleted" in run(dl["execute"](path="a.txt"))
    ws = T.get_tool("web_search")
    assert "Result One" in run(ws["execute"](query="q"))
    shot = T.get_tool("screenshot")
    res = run(shot["execute"](url="http://127.0.0.1/"))
    assert res["ok"] is False
    bt = T.get_tool("browser_text")
    assert "browser_text failed" in run(bt["execute"]())


def test_shot_path_naming(_tmp_workdir):
    path, name = T._shot_path("page")
    assert name.startswith("page-") and name.endswith(".png")
    assert path == os.path.join(str(_tmp_workdir / ".shots"), name)

# ---------------------------------------------------------------------------
# Coverage batch B (2026-11-21): _proxy/_pw_proxy/_aclient edges, shell
# spawn error, _blocked_url/_dns_blocked edges, _integration_call, and
# Playwright browser/page setup (fake playwright module — not installed).
# ---------------------------------------------------------------------------

import sys
import types

import httpx


def test_proxy_malformed_url_returns_none(monkeypatch):
    # urlparse raises ValueError on a malformed IPv6 proxy URL.
    for var in ("https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://[::1")
    assert T._proxy() is None


def test_pw_proxy_invalid_port_returns_none(monkeypatch):
    for var in ("https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:99999")
    assert T._proxy() == "http://proxy.example:99999"
    assert T._pw_proxy() is None  # .port raises ValueError -> except


def test_aclient_factory_passes_verify_and_proxy(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    seen = {}

    class FakeAsyncClient:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    T._aclient(25)
    assert seen["timeout"] == 25
    assert seen["proxy"] is None
    assert seen["trust_env"] is False
    assert seen["verify"] == T._verify()


def test_shell_exec_spawn_error(monkeypatch, _tmp_workdir):
    def boom(*a, **k):
        raise RuntimeError("no shell here")
    monkeypatch.setattr(asyncio, "create_subprocess_shell", boom)
    assert run(T.shell_exec("echo hi")) == "shell error: no shell here"


def test_blocked_url_unparseable(monkeypatch):
    monkeypatch.setattr(T, "urlparse",
                        lambda raw: (_ for _ in ()).throw(
                            ValueError("bad url")))
    assert T._blocked_url("http://example.com/") == "not a valid URL"


def test_dns_blocked_skips_unparseable_ip(monkeypatch):
    monkeypatch.setattr(T.socket, "getaddrinfo",
                        lambda h, p, *a, **k: [(2, 1, 6, "", ("not-an-ip", 0))])
    assert T._dns_blocked("example.com") is None


def test_integration_call_generic_exception():
    def boom(action, arguments):
        raise KeyError("kaboom")
    out = T._integration_call(boom, "act", {})
    assert out.startswith("integration error: ")
    assert "kaboom" in out


class _BatchBFakePage:
    """Minimal stand-in for a Playwright page (no Chromium launched)."""

    def __init__(self, closed=False):
        self._closed = closed

    def is_closed(self):
        return self._closed


class _BatchBFakeBrowser:
    """Minimal stand-in for a Playwright browser (no Chromium launched)."""

    def __init__(self):
        self.pages = []

    async def new_page(self, viewport=None):
        page = _BatchBFakePage()
        self.pages.append(page)
        return page


def _install_fake_playwright(monkeypatch, browser, seen):
    """Inject fake playwright + playwright.async_api modules (not installed)."""
    pkg = types.ModuleType("playwright")
    pkg.__path__ = []

    class _Chromium:
        async def launch(self, proxy=None, args=None):
            seen["proxy"] = proxy
            seen["args"] = args
            return browser

    class _PW:
        async def start(self):
            return self

        @property
        def chromium(self):
            return _Chromium()

    api = types.ModuleType("playwright.async_api")
    api.async_playwright = lambda: _PW()
    monkeypatch.setitem(sys.modules, "playwright", pkg)
    monkeypatch.setitem(sys.modules, "playwright.async_api", api)


def test_get_browser_launches_once(monkeypatch):
    import app.forwarder as fwd_mod
    browser = _BatchBFakeBrowser()
    seen = {}
    _install_fake_playwright(monkeypatch, browser, seen)

    async def fake_fwd():
        return "http://127.0.0.1:18080"
    monkeypatch.setattr(fwd_mod, "ensure_forwarder", fake_fwd)
    monkeypatch.setattr(T, "_playwright_browser", None)
    got = run(T._get_browser())
    assert got is browser
    assert seen["proxy"] == {"server": "http://127.0.0.1:18080"}
    assert seen["args"] == ["--ignore-certificate-errors"]
    assert browser._pw is not None  # handle kept so it can be stopped later
    assert run(T._get_browser()) is browser  # cached: no second launch


def test_get_page_creates_and_reuses(monkeypatch):
    browser = _BatchBFakeBrowser()

    async def fake_browser():
        return browser
    monkeypatch.setattr(T, "_get_browser", fake_browser)
    monkeypatch.setattr(T, "_browser_page", None)
    first = run(T._get_page())
    assert run(T._get_page()) is first  # cached while open
    assert len(browser.pages) == 1
    monkeypatch.setattr(T, "_browser_page", _BatchBFakePage(closed=True))
    second = run(T._get_page())
    assert second is not first  # closed page is replaced
    assert len(browser.pages) == 2
