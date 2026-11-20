"""Tests for app/forwarder.py — the local CONNECT forwarder.

The forwarder exists because some Chromium builds fail the sandbox egress
proxy's auth handshake directly. Tests verify credential parsing and run a
real loopback CONNECT exchange against a fake upstream proxy.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from app import forwarder
from tests.conftest import run


def test_egress_parses_host_port_and_auth(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://user:pass@proxy.example:3128")
    monkeypatch.delenv("https_proxy", raising=False)
    host, port, auth = forwarder._egress()
    assert (host, port) == ("proxy.example", 3128)
    assert base64.b64decode(auth).decode() == "user:pass"


def test_egress_default_port(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://u:p@proxy.example")
    monkeypatch.delenv("https_proxy", raising=False)
    host, port, _auth = forwarder._egress()
    assert port == 3128


async def _fake_upstream(reader, writer):
    """Accept a CONNECT, demand Proxy-Authorization, then echo one blob."""
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = await reader.read(4096)
        if not chunk:
            writer.close()
            return
        head += chunk
    text = head.decode("latin1")
    if "Proxy-Authorization: Basic" not in text:
        writer.write(b"HTTP/1.1 407 Proxy Auth Required\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await writer.drain()
    # Echo back whatever the client sends through the tunnel.
    data = await reader.read(1024)
    writer.write(b"ECHO:" + data)
    await writer.drain()
    writer.close()


def test_connect_tunnel_roundtrip(monkeypatch):
    async def scenario():
        server = await asyncio.start_server(_fake_upstream, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("HTTPS_PROXY", f"http://u:p@127.0.0.1:{port}")
        monkeypatch.delenv("https_proxy", raising=False)

        # Start the forwarder on an ephemeral port (bypass ensure_forwarder's
        # fixed LISTEN_PORT to avoid clashes in CI).
        fwd = await asyncio.start_server(forwarder._handle, "127.0.0.1", 0)
        fwd_port = fwd.sockets[0].getsockname()[1]
        try:
            r, w = await asyncio.open_connection("127.0.0.1", fwd_port)
            w.write(b"CONNECT example.com:443 HTTP/1.1\r\n"
                    b"Host: example.com:443\r\n\r\n")
            await w.drain()
            resp = await r.readuntil(b"\r\n\r\n")
            assert b"200 Connection Established" in resp
            w.write(b"hello-tunnel")
            await w.drain()
            echo = await r.read(1024)
            assert echo == b"ECHO:hello-tunnel"
            w.close()
        finally:
            fwd.close()
            server.close()

    run(scenario())


def test_ensure_forwarder_returns_local_url(monkeypatch):
    monkeypatch.setattr(forwarder, "_server", None)
    url = run(forwarder.ensure_forwarder())
    assert url == f"http://{forwarder.LISTEN_HOST}:{forwarder.LISTEN_PORT}"
    # Second call reuses the same server instance.
    url2 = run(forwarder.ensure_forwarder())
    assert url2 == url
    # Clean up so the loopback listener doesn't leak between test runs.
    srv, forwarder._server = forwarder._server, None
    run(_close(srv))


async def _close(srv):
    srv.close()
    await srv.wait_closed()


# ---------------------------------------------------------------------------
# _egress without a proxy configured
# ---------------------------------------------------------------------------

def test_egress_no_proxy_env(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    host, port, auth = forwarder._egress()
    assert (host, port) == ("localhost", 3128)
    # No credentials in the URL: urlparse yields None for both, so the
    # auth header is the literal "None:None" pair.
    assert base64.b64decode(auth).decode() == "None:None"


# ---------------------------------------------------------------------------
# _handle — malformed / oversized / refused / plain-HTTP paths
# ---------------------------------------------------------------------------

async def _run_fwd(handler, monkeypatch, payload: bytes, read_all=True):
    """Start a fake upstream + the forwarder, send payload, return what the
    client read back and what the upstream saw (or None if never hit)."""
    seen = []
    upstream = await asyncio.start_server(handler, "127.0.0.1", 0)
    uport = upstream.sockets[0].getsockname()[1]
    monkeypatch.setenv("HTTPS_PROXY", f"http://u:p@127.0.0.1:{uport}")
    monkeypatch.delenv("https_proxy", raising=False)
    fwd = await asyncio.start_server(forwarder._handle, "127.0.0.1", 0)
    fport = fwd.sockets[0].getsockname()[1]
    try:
        r, w = await asyncio.open_connection("127.0.0.1", fport)
        w.write(payload)
        await w.drain()
        out = b""
        try:
            while True:
                chunk = await asyncio.wait_for(r.read(4096), timeout=2)
                if not chunk:
                    break
                out += chunk
                if not read_all:
                    break
        except asyncio.TimeoutError:
            pass
        w.close()
        return out, seen
    finally:
        fwd.close()
        upstream.close()


def test_handle_client_closes_immediately(monkeypatch):
    hit = []

    async def upstream(reader, writer):
        hit.append(True)
        writer.close()

    async def scenario():
        out, _ = await _run_fwd(upstream, monkeypatch, b"")
        assert out == b""
    run(scenario())
    assert hit == []  # no request head, upstream never contacted


def test_handle_malformed_request_line(monkeypatch):
    hit = []

    async def upstream(reader, writer):
        hit.append(True)
        writer.close()

    async def scenario():
        out, _ = await _run_fwd(upstream, monkeypatch, b"GARBAGE\r\n\r\n")
        assert out == b""  # connection closed, nothing proxied
    run(scenario())
    assert hit == []


def test_handle_oversized_head_dropped(monkeypatch):
    hit = []

    async def upstream(reader, writer):
        hit.append(True)
        writer.close()

    async def scenario():
        out, _ = await _run_fwd(upstream, monkeypatch, b"A" * 70000)
        assert out == b""
    run(scenario())
    assert hit == []


def test_handle_connect_non200_becomes_502(monkeypatch):
    async def upstream(reader, writer):
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = await reader.read(4096)
            if not chunk:
                writer.close()
                return
            head += chunk
        writer.write(b"HTTP/1.1 407 Proxy Auth Required\r\n\r\n")
        await writer.drain()
        writer.close()

    async def scenario():
        out, _ = await _run_fwd(upstream, monkeypatch,
                                b"CONNECT example.com:443 HTTP/1.1\r\n"
                                b"Host: example.com:443\r\n\r\n")
        assert out == b"HTTP/1.1 502 Bad Gateway\r\n\r\n"
    run(scenario())


def test_handle_connect_upstream_closes(monkeypatch):
    async def upstream(reader, writer):
        writer.close()  # dies before answering

    async def scenario():
        out, _ = await _run_fwd(upstream, monkeypatch,
                                b"CONNECT example.com:443 HTTP/1.1\r\n"
                                b"Host: example.com:443\r\n\r\n")
        assert out == b""  # forwarder bails quietly, no 502 without a status
    run(scenario())


def test_handle_plain_http_rewrites_request(monkeypatch):
    seen = []

    async def upstream(reader, writer):
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = await reader.read(4096)
            if not chunk:
                writer.close()
                return
            head += chunk
        # Body already in this chunk, if any.
        body = head.split(b"\r\n\r\n", 1)[1]
        try:
            body += await asyncio.wait_for(reader.read(4096), timeout=0.5)
        except asyncio.TimeoutError:
            pass
        seen.append(head.split(b"\r\n\r\n", 1)[0] + b"\r\n\r\n" + body)
        writer.close()

    async def scenario():
        await _run_fwd(upstream, monkeypatch,
                       b"GET http://example.com/some/path?x=1 HTTP/1.1\r\n"
                       b"Host: example.com\r\n"
                       b"Proxy-Authorization: Basic b2xk\r\n"
                       b"Proxy-Connection: keep-alive\r\n"
                       b"\r\nBODYBYTES")
    run(scenario())
    assert len(seen) == 1
    text = seen[0].decode("latin1")
    lines = text.split("\r\n")
    assert lines[0] == "GET /some/path?x=1 HTTP/1.1"
    # Egress auth injected right after the request line; the stale
    # client auth and hop headers are stripped.
    assert lines[1] == "Proxy-Authorization: Basic " + base64.b64encode(
        b"u:p").decode()
    assert "Proxy-Connection:" not in text
    assert "Basic b2xk" not in text
    assert text.endswith("BODYBYTES")


def test_handle_plain_http_bare_path(monkeypatch):
    seen = []

    async def upstream(reader, writer):
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = await reader.read(4096)
            if not chunk:
                writer.close()
                return
            head += chunk
        seen.append(head)
        writer.close()

    async def scenario():
        await _run_fwd(upstream, monkeypatch,
                       b"GET http://example.com HTTP/1.1\r\n"
                       b"Host: example.com\r\n\r\n")
    run(scenario())
    assert seen[0].decode("latin1").split("\r\n")[0] == "GET / HTTP/1.1"


def test_handle_plain_http_bad_target_keeps_path(monkeypatch):
    # A target that is not a URL keeps its literal path (urlparse treats
    # the whole thing as the path); only a missing path becomes "/".
    seen = []

    async def upstream(reader, writer):
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = await reader.read(4096)
            if not chunk:
                writer.close()
                return
            head += chunk
        seen.append(head)
        writer.close()

    async def scenario():
        await _run_fwd(upstream, monkeypatch,
                       b"GET not-a-url HTTP/1.1\r\nHost: x\r\n\r\n")
    run(scenario())
    assert seen[0].decode("latin1").split("\r\n")[0] == "GET not-a-url HTTP/1.1"
