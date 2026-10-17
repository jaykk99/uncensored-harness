"""Agent tools: terminal, filesystem, web, Composio, screenshots, browser control.

All filesystem/shell work is sandboxed to WORKDIR (~/harness-work by default).
Screenshots go to WORKDIR/.shots/ and are served to the chat UI inline.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import shlex
import time
from typing import Any
from urllib.parse import urlparse

import httpx

WORKDIR = os.environ.get("HARNESS_WORKDIR", os.path.expanduser("~/harness-work"))
SHOTS_DIR = os.path.join(WORKDIR, ".shots")
os.makedirs(WORKDIR, exist_ok=True)
os.makedirs(SHOTS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

COMPOSIO_BASE = "https://backend.composio.dev/api/v3"


def _proxy() -> str | None:
    p = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
         or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or None)
    if p:
        try:
            from urllib.parse import urlparse
            if (urlparse(p).hostname or "") in ("localhost", "127.0.0.1", "::1"):
                return None
        except Exception:
            return None
    return p


def _verify():
    p = "/etc/ssl/certs/ca-certificates.crt"
    return p if os.path.isfile(p) else True


def _aclient(timeout: int, **kw) -> httpx.AsyncClient:
    # Explicit proxy + trust_env=False: works around httpx choking on the
    # environment's no_proxy entries (*[::1]) during proxy auto-detection.
    # System CA bundle: the egress proxy re-signs TLS, certifi lacks its CA.
    kw.setdefault("verify", _verify())
    return httpx.AsyncClient(timeout=timeout, proxy=_proxy(), trust_env=False, **kw)


def _pw_proxy() -> dict | None:
    """Proxy config for Playwright (it ignores env proxy vars)."""
    raw = _proxy()
    if not raw:
        return None
    try:
        u = urlparse(raw)
        d: dict = {"server": f"{u.scheme}://{u.hostname}:{u.port}"}
        if u.username:
            d["username"] = u.username
        if u.password:
            d["password"] = u.password
        return d
    except Exception:
        return None


def _safe_path(rel: str) -> str:
    """Resolve a relative path inside WORKDIR; reject escapes."""
    p = os.path.normpath(os.path.join(WORKDIR, rel or "."))
    if not p.startswith(os.path.realpath(WORKDIR) + os.sep) and p != os.path.realpath(WORKDIR):
        raise ValueError("path escapes workdir")
    return p


# ---------------------------------------------------------------- terminal

async def shell_exec(command: str, timeout: int = 60) -> str:
    """Run a shell command in WORKDIR. Returns combined output (truncated)."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command, cwd=WORKDIR,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"[timed out after {timeout}s]"
        text = out.decode("utf-8", "replace")
        if len(text) > 12000:
            text = text[:12000] + "\n…[truncated]…"
        return f"$ {command}\n(exit {proc.returncode})\n{text}"
    except Exception as e:
        return f"shell error: {e}"


# ---------------------------------------------------------------- filesystem

async def file_read(path: str, max_chars: int = 20000) -> str:
    try:
        p = _safe_path(path)
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_chars + 1)
        note = "" if len(text) <= max_chars else "\n…[truncated]…"
        return text[:max_chars] + note
    except Exception as e:
        return f"read error: {e}"


async def file_write(path: str, content: str) -> str:
    try:
        p = _safe_path(path)
        os.makedirs(os.path.dirname(p) or WORKDIR, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"write error: {e}"


async def file_list(path: str = ".") -> str:
    try:
        p = _safe_path(path)
        items = []
        for name in sorted(os.listdir(p)):
            full = os.path.join(p, name)
            items.append(name + ("/" if os.path.isdir(full) else ""))
        return "\n".join(items) or "(empty)"
    except Exception as e:
        return f"list error: {e}"


async def file_delete(path: str) -> str:
    try:
        p = _safe_path(path)
        if os.path.isdir(p):
            import shutil
            shutil.rmtree(p)
        else:
            os.remove(p)
        return f"deleted {path}"
    except Exception as e:
        return f"delete error: {e}"


# ---------------------------------------------------------------- web

def _blocked_url(raw: str) -> str | None:
    try:
        u = urlparse(raw)
    except Exception:
        return "not a valid URL"
    if u.scheme not in ("http", "https"):
        return "only http(s) URLs are fetchable"
    host = (u.hostname or "").lower()
    if not host:
        return "no host"
    if (host == "localhost" or host.endswith(".localhost") or host.endswith(".local")
            or host.endswith(".internal") or host in ("metadata.google.internal", "metadata.google.com")):
        return "private/local hosts are not fetchable"
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return "private IP ranges are not fetchable"
    except ValueError:
        pass
    return None


def _html_to_text(html: str, max_chars: int = 8000) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, ch in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                    ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")]:
        text = text.replace(ent, ch)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


async def web_fetch(url: str, max_chars: int = 8000) -> str:
    blocked = _blocked_url(url)
    if blocked:
        return f"Blocked: {blocked}"
    try:
        async with _aclient(25, follow_redirects=True,
                               headers={"User-Agent": "uncensored-harness/1.0"}) as c:
            r = await c.get(url)
            if r.status_code >= 400:
                return f"HTTP {r.status_code} fetching {url}"
            ctype = r.headers.get("content-type", "")
            text = _html_to_text(r.text, max_chars) if "html" in ctype else r.text[:max_chars]
            m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.I | re.S)
            title = m.group(1).strip()[:200] if m else ""
            return f"URL: {url}\nTitle: {title}\n\n{text}"
    except Exception as e:
        return f"Fetch failed: {e}"


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).replace("&amp;", "&").replace("&quot;", '"')\
        .replace("&#39;", "'").replace("&lt;", "<").replace("&gt;", ">").strip()


async def _bing_rss_search(query: str, max_results: int) -> list[str]:
    """Bing RSS: keyless, no anti-bot page, clean titles/links/snippets."""
    async with _aclient(20) as c:
        r = await c.get(
            "https://www.bing.com/search",
            params={"q": query, "format": "rss"},
            headers={"User-Agent": "uncensored-harness/1.0"})
        if r.status_code >= 400:
            raise RuntimeError(f"bing HTTP {r.status_code}")
    out = []
    for m in re.finditer(r"<item>(.*?)</item>", r.text, re.S):
        block = m.group(1)
        def grab(tag):
            mm = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.S)
            return _strip_tags(mm.group(1)) if mm else ""
        title, link, desc = grab("title"), grab("link"), grab("description")
        if title and link:
            out.append(f"- {title}\n  {link}\n  {desc[:250]}")
            if len(out) >= max_results:
                break
    return out


async def _ddg_search(query: str, max_results: int) -> list[str]:
    """DuckDuckGo HTML (fallback). Often serves an anti-bot page; bing first."""
    async with _aclient(20) as c:
        r = await c.get("https://html.duckduckgo.com/html/", params={"q": query},
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                                               "Chrome/128.0.0.0 Safari/537.36"})
        if r.status_code >= 400:
            raise RuntimeError(f"ddg HTTP {r.status_code}")
    out = []
    for m in re.finditer(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            r.text, re.S):
        url, title, snippet = m.group(1), m.group(2), m.group(3)
        out.append(f"- {_strip_tags(title)}\n  {_strip_tags(url)}\n  {_strip_tags(snippet)[:250]}")
        if len(out) >= max_results:
            break
    return out


async def web_search(query: str, max_results: int = 5) -> str:
    # Bing RSS primary (stable, keyless); DDG fallback when bing blips.
    try:
        out = await _bing_rss_search(query, max_results)
        if out:
            return "\n".join(out)
    except Exception as e:
        logger.warning("bing search failed: %s", e)
    try:
        out = await _ddg_search(query, max_results)
        if out:
            return "\n".join(out)
    except Exception as e:
        logger.warning("ddg search failed: %s", e)
    return "No results found."


# ---------------------------------------------------------------- composio

async def composio_execute(tool_slug: str, arguments: dict | None = None,
                           connected_account_id: str = "") -> str:
    key = os.environ.get("COMPOSIO_API_KEY", "").strip()
    if not key:
        return "Composio is not configured — COMPOSIO_API_KEY is missing."
    try:
        async with _aclient(60) as c:
            body: dict[str, Any] = {"tool_slug": tool_slug,
                                   "arguments": arguments or {}}
            if connected_account_id:
                body["connected_account_id"] = connected_account_id
            r = await c.post(f"{COMPOSIO_BASE}/tools/execute", json=body,
                             headers={"x-api-key": key, "Content-Type": "application/json"})
            if r.status_code == 401:
                return "Composio key is invalid (401)."
            data = r.json()
            if r.status_code >= 400:
                return f"Composio error {r.status_code}: {str(data)[:500]}"
            return str(data)[:4000]
    except Exception as e:
        return f"Composio failed: {e}"


# ---------------------------------------------------------------- screenshots (playwright)

_playwright_browser = None


async def _get_browser():
    global _playwright_browser
    if _playwright_browser is None:
        from playwright.async_api import async_playwright
        from .forwarder import ensure_forwarder
        pw = await async_playwright().start()
        # Route Chromium through the local CONNECT forwarder: some Chromium
        # builds fail the egress proxy's auth handshake directly
        # (ERR_TUNNEL_CONNECTION_FAILED); the forwarder chains with proper
        # Proxy-Authorization.
        fwd = await ensure_forwarder()
        _playwright_browser = await pw.chromium.launch(
            proxy={"server": fwd},
            args=["--ignore-certificate-errors"],
        )
        _playwright_browser._pw = pw  # keep handle to stop later
    return _playwright_browser


def _shot_path(prefix: str = "shot") -> str:
    name = f"{prefix}-{int(time.time() * 1000)}.png"
    return os.path.join(SHOTS_DIR, name), name


async def screenshot_page(url: str = "", html_path: str = "",
                          width: int = 1280, height: int = 800,
                          wait_ms: int = 1500) -> dict:
    """Screenshot a URL or a local HTML file in WORKDIR. Returns {ok, shot, note}."""
    if not url and not html_path:
        return {"ok": False, "error": "url or html_path required"}
    if html_path:
        try:
            p = _safe_path(html_path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not os.path.isfile(p):
            return {"ok": False, "error": f"no such file: {html_path}"}
        url = "file://" + p
    else:
        blocked = _blocked_url(url)
        if blocked:
            return {"ok": False, "error": f"Blocked: {blocked}"}
    try:
        browser = await _get_browser()
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(wait_ms)
        fspath, name = _shot_path("page")
        await page.screenshot(path=fspath)
        await page.close()
        return {"ok": True, "shot": name,
                "note": f"screenshotted {url[:120]}"}
    except Exception as e:
        return {"ok": False, "error": f"screenshot failed: {e}"}


# ---------------------------------------------------------------- browser control (playwright)

_browser_page = None


async def _get_page():
    global _browser_page
    if _browser_page is None or _browser_page.is_closed():
        browser = await _get_browser()
        _browser_page = await browser.new_page(
            viewport={"width": 1280, "height": 800})
    return _browser_page


async def _page_shot(prefix: str) -> str | None:
    try:
        page = await _get_page()
        fspath, name = _shot_path(prefix)
        await page.screenshot(path=fspath)
        return name
    except Exception:
        return None


async def browser_open(url: str) -> dict:
    blocked = _blocked_url(url)
    if blocked:
        return {"ok": False, "error": f"Blocked: {blocked}"}
    try:
        page = await _get_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1200)
        shot = await _page_shot("browse")
        return {"ok": True, "title": await page.title(),
                "url": page.url, "shot": shot}
    except Exception as e:
        return {"ok": False, "error": f"browser_open failed: {e}"}


async def browser_click(selector: str) -> dict:
    try:
        page = await _get_page()
        await page.click(selector, timeout=10000)
        await page.wait_for_timeout(1200)
        shot = await _page_shot("click")
        return {"ok": True, "title": await page.title(), "url": page.url,
                "shot": shot}
    except Exception as e:
        return {"ok": False, "error": f"browser_click failed: {e}"}


async def browser_fill(selector: str, text: str) -> dict:
    try:
        page = await _get_page()
        await page.fill(selector, text, timeout=10000)
        await page.wait_for_timeout(600)
        shot = await _page_shot("fill")
        return {"ok": True, "url": page.url, "shot": shot}
    except Exception as e:
        return f"browser_fill failed: {e}"


async def browser_text(max_chars: int = 6000) -> str:
    try:
        page = await _get_page()
        text = await page.inner_text("body")
        return text[:max_chars]
    except Exception as e:
        return f"browser_text failed: {e}"


# ---------------------------------------------------------------- tool registry (OpenAI function format)

def _tool(name: str, desc: str, params: dict, fn) -> dict:
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": params},
            "execute": fn}


TOOLS: list[dict] = [
    _tool("shell_exec", "Run a shell command in the workdir. Returns exit code and output.",
          {"command": {"type": "string", "description": "Shell command to run"},
           "timeout": {"type": "integer", "description": "Timeout seconds (default 60)"}},
          lambda command, timeout=60: shell_exec(command, int(timeout or 60))),
    _tool("file_read", "Read a text file from the workdir.",
          {"path": {"type": "string", "description": "Relative path in workdir"}},
          lambda path: file_read(path)),
    _tool("file_write", "Write a text file into the workdir (creates dirs).",
          {"path": {"type": "string", "description": "Relative path in workdir"},
           "content": {"type": "string", "description": "File content"}},
          lambda path, content: file_write(path, content)),
    _tool("file_list", "List files in a workdir directory.",
          {"path": {"type": "string", "description": "Relative dir (default .)"}},
          lambda path=".": file_list(path)),
    _tool("file_delete", "Delete a file or directory in the workdir.",
          {"path": {"type": "string", "description": "Relative path"}},
          lambda path: file_delete(path)),
    _tool("web_search", "Search the web. Returns titles, URLs, snippets.",
          {"query": {"type": "string"}, "max_results": {"type": "integer"}},
          lambda query, max_results=5: web_search(query, int(max_results or 5))),
    _tool("web_fetch", "Fetch a public URL as text.",
          {"url": {"type": "string"}},
          lambda url: web_fetch(url)),
    _tool("composio_execute", "Run an action on Jay's connected integrations (Gmail, GitHub, etc).",
          {"tool_slug": {"type": "string", "description": "e.g. GMAIL_SEND_EMAIL"},
           "arguments": {"type": "object", "description": "Tool arguments"},
           "connected_account_id": {"type": "string"}},
          lambda tool_slug, arguments=None, connected_account_id="": composio_execute(
              tool_slug, arguments or {}, connected_account_id)),
    _tool("screenshot", "Screenshot a public URL or a workdir HTML file. Returns inline image.",
          {"url": {"type": "string", "description": "Public URL to screenshot"},
           "html_path": {"type": "string", "description": "Workdir HTML file to screenshot"}},
          lambda url="", html_path="": screenshot_page(url=url, html_path=html_path)),
    _tool("browser_open", "Open a URL in the controlled browser. Returns title + screenshot.",
          {"url": {"type": "string"}}, lambda url="": browser_open(url)),
    _tool("browser_click", "Click a CSS selector in the controlled browser. Returns screenshot.",
          {"selector": {"type": "string", "description": "CSS selector"}},
          lambda selector: browser_click(selector)),
    _tool("browser_fill", "Fill a form field (CSS selector) with text. Returns screenshot.",
          {"selector": {"type": "string"}, "text": {"type": "string"}},
          lambda selector, text: browser_fill(selector, text)),
    _tool("browser_text", "Read the current page's text content.",
          {}, lambda: browser_text()),
]


def openai_tools() -> list[dict]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["parameters"]}} for t in TOOLS]


def get_tool(name: str) -> dict | None:
    return next((t for t in TOOLS if t["name"] == name), None)
