"""PlatformMemory talks to the sandbox's memory proxy over plain HTTP.

A tiny stdlib server stands in for the proxy: what matters is the wire shape
(path, bearer token, snake_case body, no tenant or agent in it) and how
refusals surface — not the proxy's behaviour, which lives in the engine.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from genfleet.sdk.memory import PlatformMemory
from genfleet.sdk.memory.platform import PlatformMemoryError
from genfleet.sdk.schemas import Message, ToolCall


class _Proxy(BaseHTTPRequestHandler):
    calls: list[tuple[str, dict, str | None]] = []
    responses: dict[str, tuple[int, object]] = {}

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        _Proxy.calls.append((self.path, body, self.headers.get("authorization")))
        status, payload = _Proxy.responses.get(self.path, (200, {}))
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def proxy():
    _Proxy.calls = []
    _Proxy.responses = {}
    server = HTTPServer(("127.0.0.1", 0), _Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/memory"
    server.shutdown()


@pytest.mark.asyncio
async def test_append_sends_the_turn_with_the_token_and_nothing_about_scope(proxy):
    mem = PlatformMemory(proxy, "spawn-token")
    await mem.append(
        "whatsapp:1",
        [Message(role="assistant", content="", tool_calls=[ToolCall(id="c1", name="t", arguments={"q": 1})])],
        subject="cust-1",
        metadata={"channel": "whatsapp"},
    )
    (path, body, auth), = _Proxy.calls
    assert path == "/memory/episodic/append"
    assert auth == "Bearer spawn-token"
    assert body == {
        "session_id": "whatsapp:1",
        "subject": "cust-1",
        "metadata": {"channel": "whatsapp"},
        "messages": [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "t", "arguments": {"q": 1}}]}],
    }
    assert not {"tenant_id", "agent_id", "tenantId", "agentName"} & body.keys()


@pytest.mark.asyncio
async def test_load_returns_messages_and_skips_a_row_it_cannot_decode(proxy):
    _Proxy.responses["/memory/episodic/load"] = (200, {
        "messages": [{"role": "user", "content": "hi"}, {"role": "martian"}, {"role": "assistant", "content": "hello"}],
    })
    mem = PlatformMemory(proxy, "t")
    history = await mem.load("s", limit=50)
    assert [m.content for m in history] == ["hi", "hello"]
    assert _Proxy.calls[0][1] == {"session_id": "s", "limit": 50}


@pytest.mark.asyncio
async def test_append_with_nothing_to_append_makes_no_call(proxy):
    await PlatformMemory(proxy, "t").append("s", [])
    assert _Proxy.calls == []


@pytest.mark.asyncio
async def test_save_is_clear_then_append(proxy):
    await PlatformMemory(proxy, "t").save("s", [Message(role="user", content="x")])
    assert [c[0] for c in _Proxy.calls] == ["/memory/episodic/clear", "/memory/episodic/append"]


@pytest.mark.asyncio
async def test_search_and_purge(proxy):
    _Proxy.responses["/memory/episodic/search"] = (200, {
        "results": [{"session_id": "s", "seq": 3, "message": {"role": "user", "content": "boiler"}}],
    })
    _Proxy.responses["/memory/episodic/purge"] = (200, {"sessions": 2, "facts": 0})
    mem = PlatformMemory(proxy, "t")
    hits = await mem.search("cust-1", "boiler", limit=5)
    assert hits[0]["seq"] == 3
    assert await mem.purge("cust-1") == {"sessions": 2, "facts": 0}
    assert _Proxy.calls[0][1] == {"subject": "cust-1", "query": "boiler", "limit": 5}


@pytest.mark.asyncio
async def test_a_refusal_surfaces_the_status_and_the_proxy_message(proxy):
    _Proxy.responses["/memory/episodic/load"] = (400, {"error": 'unknown agent "ghost"'})
    with pytest.raises(PlatformMemoryError) as exc:
        await PlatformMemory(proxy, "t").load("s")
    assert exc.value.status == 400
    assert 'unknown agent "ghost"' in str(exc.value)


@pytest.mark.asyncio
async def test_an_unreachable_proxy_is_an_error_not_a_hang():
    with pytest.raises(PlatformMemoryError) as exc:
        await PlatformMemory("http://127.0.0.1:1/memory", "t").load("s")
    assert exc.value.status == 0


def test_from_env_requires_both_variables(monkeypatch):
    monkeypatch.setenv("GENFLEET_MEMORY_URL", "http://x")
    monkeypatch.delenv("GENFLEET_MEMORY_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GENFLEET_MEMORY_TOKEN"):
        PlatformMemory.from_env()


@pytest.mark.asyncio
async def test_a_body_that_is_not_json_is_an_error_not_a_crash(proxy):
    class _Broken(_Proxy):
        def do_POST(self):  # noqa: N802
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", "5")
            self.end_headers()
            self.wfile.write(b"<html")

    server = HTTPServer(("127.0.0.1", 0), _Broken)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(PlatformMemoryError, match="undecodable"):
            await PlatformMemory(f"http://127.0.0.1:{server.server_port}", "t").load("s")
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_a_response_of_the_wrong_shape_is_an_error_not_an_attributeerror(proxy):
    _Proxy.responses["/memory/episodic/load"] = (200, ["not", "an", "object"])
    _Proxy.responses["/memory/episodic/search"] = (200, ["not", "a", "results", "object"])
    mem = PlatformMemory(proxy, "t")
    with pytest.raises(PlatformMemoryError, match="expected an object"):
        await mem.load("s")
    with pytest.raises(PlatformMemoryError, match="results"):
        await mem.search("cust-1", "boiler")


@pytest.mark.asyncio
async def test_append_sends_the_turn_id_as_the_idempotency_key(proxy):
    await PlatformMemory(proxy, "t").append(
        "s", [Message(role="user", content="hi")], turn_id="turn-7"
    )
    (_, body, _), = _Proxy.calls
    assert body["turn_id"] == "turn-7"
