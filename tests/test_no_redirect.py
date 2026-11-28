"""Every SDK call that carries a bearer token refuses redirects.

A redirect is answered with an error naming it; nothing, and in particular
not the ``Authorization`` header, is sent to the ``Location``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from genfleet.sdk import config
from genfleet.sdk._http import REDIRECT_REFUSED
from genfleet.sdk.channels import (
    CHANNELS_TOKEN_ENV,
    CHANNELS_URL_ENV,
    ChannelSendError,
    send_message,
)
from genfleet.sdk.config import AGENT_CONFIG_ENV
from genfleet.sdk.handoffs import (
    HANDOFFS_TOKEN_ENV,
    HANDOFFS_URL_ENV,
    HandoffResult,
    handoff,
)
from genfleet.sdk.memory.platform import PlatformMemory, PlatformMemoryError
from genfleet.sdk.platform_state import PlatformStateError, PlatformStateStore

from .conftest import RawServer

REDIRECTS = [301, 302, 303, 307, 308]


@pytest.fixture
def elsewhere():
    server = RawServer()
    server.answer = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
    yield server
    server.close()


def _redirect(raw_server: RawServer, status: int, elsewhere: RawServer) -> None:
    raw_server.answer = (
        f"HTTP/1.1 {status} Moved\r\nLocation: http://127.0.0.1:{elsewhere.port}/steal\r\n"
        "Content-Length: 0\r\nConnection: close\r\n\r\n"
    ).encode()


@pytest.mark.parametrize("status", REDIRECTS)
async def test_hosted_state_refuses_a_redirect(raw_server, elsewhere, status):
    _redirect(raw_server, status, elsewhere)
    store = PlatformStateStore(f"http://127.0.0.1:{raw_server.port}/state", "spawn-token")
    with pytest.raises(PlatformStateError) as caught:
        await store.create("agent", "owner", {"step": 0})
    assert caught.value.status == status and REDIRECT_REFUSED in str(caught.value)
    assert len(raw_server.requests) == 1 and elsewhere.requests == []


@pytest.mark.parametrize("status", REDIRECTS)
async def test_platform_memory_refuses_a_redirect(raw_server, elsewhere, status):
    _redirect(raw_server, status, elsewhere)
    memory = PlatformMemory(f"http://127.0.0.1:{raw_server.port}/memory", "spawn-token")
    with pytest.raises(PlatformMemoryError) as caught:
        await memory.load("session-1")
    assert caught.value.status == status and REDIRECT_REFUSED in str(caught.value)
    assert len(raw_server.requests) == 1 and elsewhere.requests == []


async def test_channels_name_the_redirect(raw_server, elsewhere, monkeypatch):
    _redirect(raw_server, 302, elsewhere)
    monkeypatch.setenv(CHANNELS_URL_ENV, f"http://127.0.0.1:{raw_server.port}/channels")
    monkeypatch.setenv(CHANNELS_TOKEN_ENV, "spawn-token")
    with pytest.raises(ChannelSendError, match="redirect") as caught:
        await send_message("whatsapp:2010", "hi")
    assert caught.value.status == 302 and caught.value.delivery_unknown
    assert elsewhere.requests == []


async def test_a_handoff_redirect_is_not_followed(raw_server, elsewhere, monkeypatch, tmp_path):
    config._load.cache_clear()
    path = tmp_path / "agent-config.json"
    path.write_text(json.dumps({"version": 1, "handoffs": {"lead": {"kind": "lead.v1", "wired": True}}}))
    monkeypatch.setenv(AGENT_CONFIG_ENV, str(path))
    monkeypatch.setenv(HANDOFFS_URL_ENV, f"http://127.0.0.1:{raw_server.port}/handoffs")
    monkeypatch.setenv(HANDOFFS_TOKEN_ENV, "grant")
    _redirect(raw_server, 307, elsewhere)
    try:
        assert await handoff("lead", "{}") == HandoffResult("not_delivered", "unreachable")
    finally:
        config._load.cache_clear()
    assert len(raw_server.requests) == 1 and elsewhere.requests == []


def test_no_sdk_module_opens_a_url_with_the_redirect_following_default():
    """``urllib.request.urlopen`` follows redirects; the package uses the shared opener instead."""
    package = Path(__file__).resolve().parents[1] / "genfleet" / "sdk"
    offenders = []
    for source in package.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names = [node.attr] if isinstance(node, ast.Attribute) else (
                [alias.name for alias in node.names] if isinstance(node, ast.ImportFrom) else []
            )
            if {"urlopen", "build_opener"} & set(names):
                offenders.append(f"{source.relative_to(package)}:{node.lineno}")
    assert offenders and all(o.startswith("_http.py:") for o in offenders), offenders
