"""call_connection talks to the sandbox's channels proxy.

A stdlib server stands in for the proxy: the wire shape (path, bearer token,
the call's fields and never a header), how the API's answer is decoded, and
how refusals and failures surface.
"""
from __future__ import annotations

import base64
import json
import threading
from typing import ClassVar
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from genfleet.sdk import ConnectionCallError, ConnectionResponse, call_connection
from genfleet.sdk.channels import CHANNELS_TOKEN_ENV, CHANNELS_URL_ENV


def _answer(status: int = 200, body: bytes = b'{"ok": true}', headers: dict | None = None) -> dict:
    return {"status": status, "headers": headers or {"Content-Type": "application/json"}, "bodyBase64": base64.b64encode(body).decode()}


class _Proxy(BaseHTTPRequestHandler):
    calls: ClassVar[list[tuple[str, dict, str | None]]] = []
    response: ClassVar[tuple[int, object]] = (200, _answer())

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        _Proxy.calls.append((self.path, json.loads(self.rfile.read(length)), self.headers.get("authorization")))
        status, payload = _Proxy.response
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def proxy(monkeypatch):
    _Proxy.calls = []
    _Proxy.response = (200, _answer())
    server = HTTPServer(("127.0.0.1", 0), _Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(CHANNELS_URL_ENV, f"http://127.0.0.1:{server.server_port}/channels")
    monkeypatch.setenv(CHANNELS_TOKEN_ENV, "spawn-token")
    yield
    server.shutdown()


@pytest.mark.asyncio
async def test_posts_the_call_fields_with_the_spawn_token(proxy):
    response = await call_connection(
        "erp", "post", "/integrations/leads?x=1", {"phone": "2010"}, idempotency_key="evt-1:lead", timeout=5
    )
    assert response == ConnectionResponse(200, {"content-type": "application/json"}, b'{"ok": true}')
    assert response.ok and response.json() == {"ok": True} and response.text == '{"ok": true}'
    ((path, body, auth),) = _Proxy.calls
    assert (path, auth) == ("/channels/connection-call", "Bearer spawn-token")
    assert body == {
        "connection": "erp", "method": "POST", "path": "/integrations/leads?x=1",
        "body": {"phone": "2010"}, "idempotencyKey": "evt-1:lead", "timeoutSeconds": 5.0,
    }


@pytest.mark.asyncio
async def test_a_get_sends_no_body_key_and_any_api_status_is_a_response(proxy):
    _Proxy.response = (200, _answer(403, b"error code: 1010", {"cf-ray": "abc"}))
    response = await call_connection("erp", "GET", "/x")
    assert (response.status, response.ok, response.headers) == (403, False, {"cf-ray": "abc"})
    assert "body" not in _Proxy.calls[0][1] and "idempotencyKey" not in _Proxy.calls[0][1]
    with pytest.raises(ValueError):
        response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "transport", "too_large"])
async def test_a_failure_raises_with_its_code_and_may_have_reached_the_api(proxy, failure):
    _Proxy.response = (200, {"failure": failure})
    with pytest.raises(ConnectionCallError) as caught:
        await call_connection("erp", "POST", "/x", {"a": 1})
    assert caught.value.code == failure
    assert caught.value.may_have_reached_api


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "code"), [(404, "unknown_connection"), (422, "agent_not_allowed"), (422, "host_refused"), (400, "invalid_request")])
async def test_a_platform_refusal_raises_with_its_code(proxy, status, code):
    _Proxy.response = (status, {"error": f"words for {code}", "code": code})
    with pytest.raises(ConnectionCallError) as caught:
        await call_connection("erp", "GET", "/x")
    assert (caught.value.code, caught.value.status, str(caught.value)) == (code, status, f"words for {code}")
    assert not caught.value.may_have_reached_api


@pytest.mark.asyncio
async def test_a_platform_fault_is_unknown(proxy):
    _Proxy.response = (502, {"error": "channels backend error"})
    with pytest.raises(ConnectionCallError) as caught:
        await call_connection("erp", "POST", "/x", {"a": 1})
    assert caught.value.code is None and caught.value.may_have_reached_api


@pytest.mark.asyncio
async def test_an_unreadable_answer_raises(proxy):
    _Proxy.response = (200, {"status": 200, "bodyBase64": "not base64!"})
    with pytest.raises(ConnectionCallError):
        await call_connection("erp", "GET", "/x")


@pytest.mark.asyncio
@pytest.mark.parametrize(("args", "words"), [(("erp", "TRACE", "/x"), "method"), (("erp", "GET", "x"), "path"), (("", "GET", "/x"), "connection")])
async def test_bad_arguments_raise_before_any_call(proxy, args, words):
    with pytest.raises(ValueError, match=words):
        await call_connection(*args)
    assert _Proxy.calls == []


@pytest.mark.asyncio
async def test_outside_a_sandbox_it_says_so(monkeypatch):
    monkeypatch.delenv(CHANNELS_URL_ENV, raising=False)
    monkeypatch.delenv(CHANNELS_TOKEN_ENV, raising=False)
    with pytest.raises(ConnectionCallError) as caught:
        await call_connection("erp", "GET", "/x")
    assert caught.value.code == "not_in_sandbox"
