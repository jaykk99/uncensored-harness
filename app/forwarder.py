"""Local CONNECT forwarder: chains Chromium -> this proxy -> egress proxy.

Some Chromium builds fail proxy auth handshakes (ERR_TUNNEL_CONNECTION_FAILED)
against the sandbox egress proxy. This forwarder accepts unauthenticated
CONNECT/HTTP requests on 127.0.0.1 and relays them to the egress proxy with
the proper Proxy-Authorization header.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 18923

_server = None


def _egress() -> tuple[str, int, str]:
    u = urlparse(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "")
    creds = f"{u.username}:{u.password}"
    auth = base64.b64encode(creds.encode()).decode()
    return u.hostname or "localhost", u.port or 3128, auth


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _handle(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter):
    try:
        # Read request head
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = await client_r.read(4096)
            if not chunk:
                return
            head += chunk
            if len(head) > 65536:
                return
        header_text = head.decode("latin1")
        request_line = header_text.split("\r\n", 1)[0]
        parts = request_line.split(" ")
        if len(parts) < 2:
            return
        method, target = parts[0], parts[1]

        host, port, auth = _egress()
        up_r, up_w = await asyncio.open_connection(host, port)

        if method.upper() == "CONNECT":
            up_w.write(f"CONNECT {target} HTTP/1.1\r\n"
                       f"Host: {target}\r\n"
                       f"Proxy-Authorization: Basic {auth}\r\n\r\n".encode())
            await up_w.drain()
            # read connect response
            resp = b""
            while b"\r\n\r\n" not in resp:
                chunk = await up_r.read(4096)
                if not chunk:
                    break
                resp += chunk
            status = resp.decode("latin1", "replace").split("\r\n", 1)[0]
            if " 200 " not in status:
                client_w.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await client_w.drain()
                return
            client_w.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await client_w.drain()
        else:
            # Plain HTTP: rewrite request line to origin form, inject auth
            lines = header_text.split("\r\n")
            # target is absolute URI; extract path
            try:
                u = urlparse(target)
                path = u.path or "/"
                if u.query:
                    path += "?" + u.query
            except Exception:
                path = "/"
            lines[0] = f"{method} {path} {parts[2] if len(parts) > 2 else 'HTTP/1.1'}"
            # strip hop headers, add proxy auth
            out_lines = [l for l in lines
                         if l and not l.lower().startswith(
                             ("proxy-authorization:", "proxy-connection:"))]
            out_lines.insert(1, f"Proxy-Authorization: Basic {auth}")
            up_w.write(("\r\n".join(out_lines) + "\r\n\r\n").encode("latin1"))
            # forward any body already read
            body_start = head.find(b"\r\n\r\n") + 4
            if len(head) > body_start:
                up_w.write(head[body_start:])
            await up_w.drain()

        await asyncio.gather(_pipe(client_r, up_w), _pipe(up_r, client_w))
    except Exception as e:
        logger.debug("forwarder: %s", e)
    finally:
        try:
            client_w.close()
        except Exception:
            pass


async def ensure_forwarder() -> str:
    """Start the forwarder if needed. Returns proxy server URL for Playwright."""
    global _server
    if _server is None:
        _server = await asyncio.start_server(_handle, LISTEN_HOST, LISTEN_PORT)
    return f"http://{LISTEN_HOST}:{LISTEN_PORT}"
