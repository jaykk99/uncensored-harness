"""Shared pytest fixtures for the uncensored-harness test suite.

Tests never hit the network: anything that would dial out (httpx, the
agent's tool lambdas, Playwright) is monkeypatched at the test site.
Async functions are driven with asyncio.run() so the suite needs no
pytest-asyncio plugin.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Repo root on sys.path so `import app.*` works from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(coro):
    """Drive an async callable synchronously (no plugin needed)."""
    return asyncio.run(coro)
