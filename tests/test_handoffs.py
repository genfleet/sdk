"""`genfleet.sdk.handoffs`: sending a hand-off to wherever the workspace wired it."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from genfleet.sdk import config, turn
from genfleet.sdk.config import AGENT_CONFIG_ENV
from genfleet.sdk.handoffs import (
    HANDOFFS_TOKEN_ENV,
    HANDOFFS_URL_ENV,
    MAX_PAYLOAD_CHARS,
    HandoffResult,
    handoff,
    is_wired,
    option,
    read_receipt,
)

RECEIPT = '{"schema":"receipt.v1","ok":true,"status":"created"}'


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    config._load.cache_clear()
    for name in (AGENT_CONFIG_ENV, HANDOFFS_URL_ENV, HANDOFFS_TOKEN_ENV):
        monkeypatch.delenv(name, raising=False)
    yield
    config._load.cache_clear()


def _configure(tmp_path, monkeypatch, handoffs) -> None:
    path = tmp_path / "agent-config.json"
    path.write_text(json.dumps({"version": 1, "handoffs": handoffs}))
    monkeypatch.setenv(AGENT_CONFIG_ENV, str(path))


@pytest.fixture
def platform(monkeypatch):
    """A stand-in for the platform's hand-off endpoint: records each request, answers what the test set."""
    seen: list[dict] = []
    answer = {"status": 200, "body": json.dumps({"outcome": "sent", "content": RECEIPT})}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append({"path": self.path, "auth": self.headers.get("authorization"), "body": body})
            raw = answer["body"].encode()
            self.send_response(answer["status"])
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(HANDOFFS_URL_ENV, f"http://127.0.0.1:{server.server_port}/handoffs")
    monkeypatch.setenv(HANDOFFS_TOKEN_ENV, "grant")
    yield seen, answer
    server.shutdown()
    server.server_close()


WIRED = {"lead": {"kind": "lead.v1", "wired": True, "options": {"full_details": True}}, "ticket": {"kind": "ticket.v1", "wired": False}}


def test_wiring_and_options_are_read(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, WIRED)
    assert is_wired("lead") and not is_wired("ticket") and not is_wired("unknown")
    assert option("lead", "full_details") is True
    assert option("lead", "ad_text", False) is False
    assert option("unknown", "x", "d") == "d"


async def test_outside_the_platform_everything_goes_nowhere():
    assert not is_wired("lead")
    assert await handoff("lead", {"a": 1}) == HandoffResult("nowhere")


async def test_an_unwired_handoff_makes_no_call(tmp_path, monkeypatch, platform):
    seen, _ = platform
    _configure(tmp_path, monkeypatch, WIRED)
    assert await handoff("ticket", "{}") == HandoffResult("nowhere")
    assert await handoff("unknown", "{}") == HandoffResult("nowhere")
    assert seen == []


async def test_a_wired_handoff_without_the_platform_endpoint_goes_nowhere(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, WIRED)
    assert await handoff("lead", "{}") == HandoffResult("nowhere")


async def test_a_filed_lead_is_delivered(tmp_path, monkeypatch, platform):
    seen, _ = platform
    _configure(tmp_path, monkeypatch, WIRED)
    result = await turn.call_within({"session_id": "s1", "turn_id": "t1"}, handoff("lead", {"schema": "lead.v1"}))
    assert result == HandoffResult("delivered", "created") and result.delivered
    assert seen == [{
        "path": "/handoffs/send",
        "auth": "Bearer grant",
        "body": {"key": "lead", "payload": '{"schema":"lead.v1"}', "session_id": "s1", "turn_id": "t1"},
    }]


async def test_outside_a_turn_no_turn_is_named(tmp_path, monkeypatch, platform):
    seen, _ = platform
    _configure(tmp_path, monkeypatch, WIRED)
    await handoff("lead", "text payload")
    assert seen[0]["body"] == {"key": "lead", "payload": "text payload"}


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (200, {"outcome": "sent", "content": '{"schema":"receipt.v1","ok":true,"status":"duplicate"}'}, ("delivered", "duplicate")),
        (200, {"outcome": "sent", "content": '{"schema":"receipt.v1","ok":false,"status":"not_filed","reason":"sink_down"}'}, ("not_delivered", "sink_down")),
        (200, {"outcome": "sent", "content": '{"schema":"receipt.v1","ok":false,"status":"not_filed"}'}, ("not_delivered", "not_filed")),
        (200, {"outcome": "sent", "content": "The peer agent couldn't take this request."}, ("not_delivered", "no_receipt")),
        (200, {"outcome": "sent"}, ("not_delivered", "no_receipt")),
        (200, {"outcome": "nowhere"}, ("nowhere", None)),
        (200, {"outcome": "not_delivered", "reason": "unavailable"}, ("not_delivered", "unavailable")),
        (200, {"outcome": "not_delivered", "reason": "Not A Slug"}, ("not_delivered", "refused")),
        (200, {"outcome": "something new"}, ("not_delivered", "refused")),
        (200, ["sent"], ("not_delivered", "unreachable")),
        (409, {"error": "the named turn is not running"}, ("not_delivered", "refused")),
        (429, {}, ("not_delivered", "refused")),
        (502, {}, ("not_delivered", "unreachable")),
    ],
)
async def test_outcomes(tmp_path, monkeypatch, platform, status, body, expected):
    _, answer = platform
    answer.update(status=status, body=json.dumps(body))
    _configure(tmp_path, monkeypatch, WIRED)
    result = await handoff("lead", "{}")
    assert (result.outcome, result.reason) == expected


async def test_an_undecodable_answer_is_not_delivered(tmp_path, monkeypatch, platform):
    _, answer = platform
    answer.update(body="<html>")
    _configure(tmp_path, monkeypatch, WIRED)
    assert await handoff("lead", "{}") == HandoffResult("not_delivered", "unreachable")


async def test_an_unreachable_platform_is_not_delivered(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, WIRED)
    monkeypatch.setenv(HANDOFFS_URL_ENV, "http://127.0.0.1:9/handoffs")
    monkeypatch.setenv(HANDOFFS_TOKEN_ENV, "grant")
    assert await handoff("lead", "{}") == HandoffResult("not_delivered", "unreachable")


async def test_a_kind_without_a_reply_shape_is_delivered_when_sent(tmp_path, monkeypatch, platform):
    _, answer = platform
    answer.update(body=json.dumps({"outcome": "sent", "content": "ok"}))
    _configure(tmp_path, monkeypatch, {"ticket": {"kind": "ticket.v1", "wired": True}})
    assert await handoff("ticket", "{}") == HandoffResult("delivered")


@pytest.mark.parametrize("key, payload", [("Lead", "{}"), ("", "{}"), (3, "{}"), ("lead", ""), ("lead", "  "), ("lead", 3), ("lead", "x" * (MAX_PAYLOAD_CHARS + 1))])
async def test_bad_arguments_raise(key, payload):
    with pytest.raises(ValueError):
        await handoff(key, payload)


@pytest.mark.parametrize(
    "reply, expected",
    [
        (RECEIPT, (True, "created")),
        ('{"schema":"receipt.v1","ok":true,"status":"appended"}', (True, "appended")),
        ('{"schema":"receipt.v1","ok":true,"status":"not_filed"}', (False, "not_filed")),
        ('{"schema":"receipt.v2","ok":true,"status":"created"}', (False, "no_receipt")),
        ('{"schema":"receipt.v1","ok":"true","status":"created"}', (False, "not_filed")),
        ("not json", (False, "no_receipt")),
        ("[" * 100_000, (False, "no_receipt")),
        (None, (False, "no_receipt")),
        (RECEIPT + " " * 5000, (False, "no_receipt")),
    ],
)
def test_read_receipt(reply, expected):
    assert read_receipt(reply) == expected
