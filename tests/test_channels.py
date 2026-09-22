"""send_message talks to the sandbox's channels proxy (ADR-0019 §5).

A stdlib server stands in for the proxy: what matters is the wire shape
(path, bearer token, body with no scope in it) and how refusals surface —
as a typed error to code, as words to the model.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from genfleet.sdk import ChannelSendError, send_message, send_message_tool
from genfleet.sdk.channels import CHANNELS_TOKEN_ENV, CHANNELS_URL_ENV
from genfleet.sdk.schemas import ToolCall


class _Proxy(BaseHTTPRequestHandler):
    calls: list[tuple[str, dict, str | None]] = []
    response: tuple[int, object] = (200, {"sent": True, "messageIds": ["wamid.1"]})

    def do_POST(self):  # noqa: N802
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
    _Proxy.response = (200, {"sent": True, "messageIds": ["wamid.1"]})
    server = HTTPServer(("127.0.0.1", 0), _Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(CHANNELS_URL_ENV, f"http://127.0.0.1:{server.server_port}/channels")
    monkeypatch.setenv(CHANNELS_TOKEN_ENV, "spawn-token")
    yield
    server.shutdown()


@pytest.mark.asyncio
async def test_text_goes_to_the_proxy_with_the_token_and_nothing_about_scope(proxy):
    assert await send_message("whatsapp:2010", "Leak reported by Mona") == {"sent": True, "messageIds": ["wamid.1"]}
    ((path, body, auth),) = _Proxy.calls
    assert path == "/channels/send"
    assert auth == "Bearer spawn-token"
    assert body == {"to": "whatsapp:2010", "text": "Leak reported by Mona"}


@pytest.mark.asyncio
async def test_a_template_carries_its_name_language_and_params(proxy):
    await send_message("whatsapp:2010", template="heatec_escalation", params=["Mona", "leak"], language="ar")
    ((_, body, _),) = _Proxy.calls
    assert body == {
        "to": "whatsapp:2010",
        "template": {"name": "heatec_escalation", "language": "ar", "params": ["Mona", "leak"]},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{}, {"text": "x", "template": "t"}])
async def test_exactly_one_of_text_or_template(proxy, kwargs):
    with pytest.raises(ValueError):
        await send_message("whatsapp:1", **kwargs)
    assert _Proxy.calls == []


@pytest.mark.asyncio
async def test_a_refusal_is_a_typed_error_with_the_platforms_words(proxy):
    _Proxy.response = (409, {"error": "has not written in 24 hours; send a template", "code": "outside_window"})
    with pytest.raises(ChannelSendError) as info:
        await send_message("whatsapp:2010", "hi")
    assert (info.value.status, info.value.code) == (409, "outside_window")
    assert "send a template" in str(info.value)


@pytest.mark.asyncio
async def test_outside_a_sandbox_it_says_so_instead_of_pretending(monkeypatch):
    monkeypatch.delenv(CHANNELS_URL_ENV, raising=False)
    monkeypatch.delenv(CHANNELS_TOKEN_ENV, raising=False)
    with pytest.raises(ChannelSendError, match="platform sandbox") as info:
        await send_message("whatsapp:1", "x")
    assert info.value.code == "not_in_sandbox"


@pytest.mark.asyncio
async def test_the_tool_answers_the_model_rather_than_raising(proxy):
    _Proxy.response = (409, {"error": "has not written in 24 hours; send a template", "code": "outside_window"})
    out = await send_message_tool.call(ToolCall(id="c1", name="send_message", arguments={"to": "whatsapp:2010", "text": "hi"}))
    assert out == "not sent: has not written in 24 hours; send a template"
    _Proxy.response = (200, {"sent": True})
    assert await send_message_tool(to="whatsapp:2010", template="alert") == "sent"


def test_the_tool_schema_types_every_parameter_for_the_model():
    schema = send_message_tool.schema()
    assert schema.name == "send_message"
    assert schema.parameters["required"] == ["to"]
    assert schema.parameters["properties"] == {
        "to": {"type": "string"},
        "text": {"type": "string"},
        "template": {"type": "string"},
        "params": {"type": "array", "items": {"type": "string"}},
        "language": {"type": "string"},
    }


@pytest.mark.asyncio
async def test_the_tool_passes_the_template_language(proxy):
    await send_message_tool(to="whatsapp:2010", template="alert", params=["x"], language="ar")
    ((_, body, _),) = _Proxy.calls
    assert body["template"]["language"] == "ar"


@pytest.mark.asyncio
async def test_string_params_are_refused_not_split_into_characters(proxy):
    with pytest.raises(ValueError, match="list of strings"):
        await send_message("whatsapp:1", template="alert", params="Mona")  # type: ignore[arg-type]
    assert _Proxy.calls == []


@pytest.mark.asyncio
async def test_a_proxy_fault_is_delivery_unknown_not_a_refusal(proxy):
    _Proxy.response = (502, {"error": "channels backend error"})
    with pytest.raises(ChannelSendError) as info:
        await send_message("whatsapp:1", "x")
    assert (info.value.status, info.value.code, info.value.delivery_unknown) == (502, None, True)
    out = await send_message_tool(to="whatsapp:1", text="x")
    assert out.startswith("delivery unknown") and "do not resend" in out


@pytest.mark.asyncio
async def test_an_unreachable_proxy_is_delivery_unknown(monkeypatch):
    monkeypatch.setenv(CHANNELS_URL_ENV, "http://127.0.0.1:9/channels")
    monkeypatch.setenv(CHANNELS_TOKEN_ENV, "t")
    with pytest.raises(ChannelSendError) as info:
        await send_message("whatsapp:1", "x")
    assert info.value.status == 0 and info.value.delivery_unknown


@pytest.mark.asyncio
async def test_a_non_json_error_body_is_kept_as_text(proxy, monkeypatch):
    class _Html(_Proxy):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("content-length", 0)))
            raw = b"<html>bad gateway</html>"
            self.send_response(502)
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(("127.0.0.1", 0), _Html)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(CHANNELS_URL_ENV, f"http://127.0.0.1:{server.server_port}/channels")
    try:
        with pytest.raises(ChannelSendError, match="bad gateway") as info:
            await send_message("whatsapp:1", "x")
        assert info.value.code is None
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_a_2xx_that_is_not_an_object_is_an_error(proxy):
    _Proxy.response = (200, ["unexpected"])
    with pytest.raises(ChannelSendError, match="expected an object"):
        await send_message("whatsapp:1", "x")
