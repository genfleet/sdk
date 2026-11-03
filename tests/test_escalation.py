"""escalate and staff_directory talk to the sandbox's channels proxy.

A stdlib server stands in for the proxy: the wire shape (path, bearer token,
a body naming staff, never an address), the session default from the
running turn, and how outcomes and refusals surface — typed to code, plain
words to the model.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from genfleet.sdk import (
    EscalationError,
    EscalationRecipient,
    EscalationResult,
    StaffMember,
    escalate,
    escalate_tool,
    staff_directory,
)
from genfleet.sdk import turn as _turn
from genfleet.sdk.channels import CHANNELS_TOKEN_ENV, CHANNELS_URL_ENV

SENT = {"outcome": "sent", "recipients": [{"name": "Ahmed", "outcome": "sent"}]}


class _Proxy(BaseHTTPRequestHandler):
    calls: list[tuple[str, str, dict | None, str | None]] = []
    response: tuple[int, object] = (200, SENT)

    def _answer(self, method: str, body: dict | None) -> None:
        _Proxy.calls.append((method, self.path, body, self.headers.get("authorization")))
        status, payload = _Proxy.response
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        self._answer("POST", json.loads(self.rfile.read(length)))

    def do_GET(self):  # noqa: N802
        self._answer("GET", None)

    def log_message(self, *args):
        pass


@pytest.fixture
def proxy(monkeypatch):
    _Proxy.calls = []
    _Proxy.response = (200, SENT)
    server = HTTPServer(("127.0.0.1", 0), _Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(CHANNELS_URL_ENV, f"http://127.0.0.1:{server.server_port}/channels")
    monkeypatch.setenv(CHANNELS_TOKEN_ENV, "spawn-token")
    yield
    server.shutdown()


@pytest.mark.asyncio
async def test_escalate_posts_names_text_and_session_with_the_token(proxy):
    result = await escalate(["Ahmed", "role:sales"], "Customer wants a callback", session="whatsapp:2010")
    assert result == EscalationResult("sent", (EscalationRecipient("Ahmed", "sent"),))
    assert result.sent
    ((method, path, body, auth),) = _Proxy.calls
    assert (method, path, auth) == ("POST", "/channels/escalate", "Bearer spawn-token")
    assert body == {"to": ["Ahmed", "role:sales"], "text": "Customer wants a callback", "sessionKey": "whatsapp:2010"}


@pytest.mark.asyncio
async def test_the_session_defaults_to_the_running_turn_and_one_name_may_be_a_string(proxy):
    await _turn.call_within({"session_id": "telegram:42"}, escalate("default", "help"))
    ((_, _, body, _),) = _Proxy.calls
    assert body == {"to": ["default"], "text": "help", "sessionKey": "telegram:42"}


@pytest.mark.asyncio
async def test_no_session_is_refused_before_any_call(proxy):
    with pytest.raises(EscalationError) as err:
        await escalate("default", "help")
    assert err.value.code == "no_session"
    assert not err.value.delivery_unknown
    assert _Proxy.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("to", [[], ["a", "b", "c", "d", "e", "f"], [""], [3]])
async def test_bad_targets_are_a_value_error(proxy, to):
    with pytest.raises(ValueError):
        await escalate(to, "help", session="s")
    assert _Proxy.calls == []


@pytest.mark.asyncio
async def test_too_long_text_is_a_value_error(proxy):
    with pytest.raises(ValueError):
        await escalate("default", "x" * 4001, session="s")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [(404, "unknown_recipient"), (409, "already_escalated"), (429, "rate_limited"), (422, "invalid_request")],
)
async def test_a_refusal_is_a_typed_error_with_the_platforms_words(proxy, status, code):
    _Proxy.response = (status, {"error": f"words for {code}", "code": code})
    with pytest.raises(EscalationError) as err:
        await escalate("Mona", "help", session="s")
    assert (err.value.code, err.value.status, str(err.value)) == (code, status, f"words for {code}")
    assert not err.value.delivery_unknown


@pytest.mark.asyncio
async def test_a_platform_fault_is_delivery_unknown(proxy):
    _Proxy.response = (502, {"error": "channels backend error"})
    with pytest.raises(EscalationError) as err:
        await escalate("default", "help", session="s")
    assert err.value.code is None
    assert err.value.delivery_unknown


@pytest.mark.asyncio
async def test_outside_a_sandbox_it_says_so(monkeypatch):
    monkeypatch.delenv(CHANNELS_URL_ENV, raising=False)
    monkeypatch.delenv(CHANNELS_TOKEN_ENV, raising=False)
    for call in (escalate("default", "help", session="s"), escalate("default", "help"), staff_directory()):
        with pytest.raises(EscalationError) as err:
            await call
        assert err.value.code == "not_in_sandbox"
        assert err.value.status == 0


@pytest.mark.asyncio
async def test_staff_directory_is_names_and_roles(proxy):
    _Proxy.response = (200, {"staff": [{"name": "Ahmed", "role": "sales", "isDefault": True}, {"bad": 1}]})
    assert await staff_directory() == [StaffMember("Ahmed", "sales", True)]
    ((method, path, body, auth),) = _Proxy.calls
    assert (method, path, body, auth) == ("GET", "/channels/staff-directory", None, "Bearer spawn-token")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "answer"),
    [
        ((200, SENT), "sent"),
        (
            (200, {"outcome": "sent", "recipients": [{"name": "A", "outcome": "already_escalated"}, {"name": "B", "outcome": "sent"}]}),
            "sent (already escalated to A)",
        ),
        ((200, {"outcome": "unknown", "recipients": []}), "delivery unknown, do not resend"),
        ((200, {"outcome": "not_sent", "recipients": []}), "not sent: staff could not be reached; tell the person you could not pass it on right now"),
        ((409, {"error": "already", "code": "already_escalated"}), "already escalated: staff were already alerted about this conversation; do not resend"),
        ((404, {"error": "No staff member matches \"Mona\". Valid: Ahmed, default.", "code": "unknown_recipient"}), 'not sent: No staff member matches "Mona". Valid: Ahmed, default.'),
        ((502, {"error": "channels backend error"}), "delivery unknown, do not resend"),
        ((200, {"nonsense": True}), "delivery unknown, do not resend"),
    ],
)
async def test_the_tool_answers_in_plain_words(proxy, response, answer):
    _Proxy.response = response
    result = await _turn.call_within({"session_id": "whatsapp:2010"}, escalate_tool(to=["Mona"], text="help"))
    assert result == answer


@pytest.mark.asyncio
async def test_the_tool_outside_a_sandbox_is_not_sent(monkeypatch):
    monkeypatch.delenv(CHANNELS_URL_ENV, raising=False)
    result = await escalate_tool(to=["default"], text="help")
    assert result.startswith("not sent: escalate needs a platform sandbox")


def test_the_tool_schema_takes_names_and_text_only():
    schema = escalate_tool.schema()
    assert schema.name == "escalate"
    assert schema.parameters["properties"] == {"to": {"type": "array", "items": {"type": "string"}}, "text": {"type": "string"}}
    # Customer-facing agents need it; the model picks only from the staff list.
    assert escalate_tool.audiences == frozenset({"operator", "customer"})
