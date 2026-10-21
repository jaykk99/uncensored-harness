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
