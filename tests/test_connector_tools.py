"""Connector tools and reply talk to the sandbox's channels proxy.

A stdlib server stands in for the proxy: the wire shape (path, bearer token,
the turn reference and the listing's tool and hash), what the model reads for
each answer, the refusal codes, and the retry rule (only
``backend_unreachable`` is retried).
"""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from genfleet.sdk import (
    Agent,
    AgentInput,
    AgentOutput,
    ConnectorToolError,
    ReplyResult,
    ToolCall,
    config,
    connector_tools,
    reply,
    reply_tool,
)
from genfleet.sdk import connector_calls as ct
from genfleet.sdk.caller import (
    CALLER_METADATA_KEY,
    EVERYONE,
    OPERATOR_ONLY,
    TOOL_AUDIENCES_METADATA_KEY,
)
from genfleet.sdk.channels import CHANNELS_TOKEN_ENV, CHANNELS_URL_ENV
from genfleet.sdk.confirmations import SENSITIVE_TOOLS_KEY
from genfleet.sdk.turn import call_within

FIXTURES = Path(__file__).parent / "fixtures"
CODES = json.loads((FIXTURES / "connector-tool-codes.json").read_text())
EVERYTHING = json.loads((FIXTURES / "agent-config.json").read_text())["cases"][0]["file"]
TURN = {"session_id": "whatsapp:201001234567", "turn_id": "turn-1"}
API_ANSWER = {
    "tool": "erp__create_lead", "status": 201, "ok": True, "contentType": "application/json",
    "body": '{"id": 7}', "truncated": False, "bytes": 9, "bodyWithheld": False, "note": None,
}


class _Proxy(BaseHTTPRequestHandler):
    calls: ClassVar[list[tuple[str, dict, str | None]]] = []
    #: Answers in order; the last one repeats.
    responses: ClassVar[list[tuple[int, object]]] = []

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        _Proxy.calls.append((self.path, json.loads(self.rfile.read(length)), self.headers.get("authorization")))
        status, payload = _Proxy.responses.pop(0) if len(_Proxy.responses) > 1 else _Proxy.responses[0]
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture(autouse=True)
def listing(tmp_path, monkeypatch):
    """The agent-config file of the shared 'everything set' case."""
    config._load.cache_clear()
    path = tmp_path / "agent-config.json"
    path.write_text(json.dumps(EVERYTHING))
    monkeypatch.setenv(config.AGENT_CONFIG_ENV, str(path))
    monkeypatch.setattr(ct, "_RETRY_DELAYS_S", (0.0, 0.0))
    yield
    config._load.cache_clear()


@pytest.fixture
def proxy(monkeypatch):
    _Proxy.calls = []
    _Proxy.responses = [(200, API_ANSWER)]
    server = HTTPServer(("127.0.0.1", 0), _Proxy)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    monkeypatch.setenv(CHANNELS_URL_ENV, f"http://127.0.0.1:{server.server_port}/channels")
    monkeypatch.setenv(CHANNELS_TOKEN_ENV, "spawn-token")
    yield _Proxy
    server.shutdown()


def _tool(name: str):
    return next(t for t in connector_tools() if t.schema().name == name)


async def _call(name: str, arguments: dict, turn: dict = TURN) -> str:
    return await call_within(turn, _tool(name)(**arguments))


# --- The listing as tools ------------------------------------------------------


def test_tools_carry_the_listing_exactly():
    tools = {t.schema().name: t for t in connector_tools()}
    assert list(tools) == ["erp__create_lead", "erp__list_sales_people", "crm__find_contact"]
    lead = EVERYTHING["connectors"]["erp"]["tools"][0]
    schema = tools["erp__create_lead"].schema()
    assert (schema.description, schema.parameters) == (lead["description"], lead["input_schema"])
    assert all(t.audiences == OPERATOR_ONLY and not t.sensitive for t in tools.values())


def test_a_slot_gives_only_its_connection():
    assert [t.schema().name for t in connector_tools("erp")] == ["erp__create_lead", "erp__list_sales_people"]
    assert connector_tools("crm") == []
    assert [t.schema().name for t in config.connector_tools("erp")] == ["erp__create_lead", "erp__list_sales_people"]


def test_reply_tool_is_the_listed_one_for_everyone():
    tool = reply_tool()
    listed = EVERYTHING["channels"]["whatsapp"]["tools"][1]
    assert tool.schema().model_dump() == {"name": "reply", "description": listed["description"], "parameters": listed["input_schema"]}
    assert tool.audiences == EVERYONE


def test_a_file_without_tools_gives_none(tmp_path, monkeypatch):
    config._load.cache_clear()
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": 1, "connectors": {"erp": {"connection": "erp", "provider": "custom_api"}}}))
    monkeypatch.setenv(config.AGENT_CONFIG_ENV, str(path))
    assert connector_tools() == [] and connector_tools("erp") == [] and reply_tool() is None


def test_no_file_gives_none(monkeypatch):
    config._load.cache_clear()
    monkeypatch.delenv(config.AGENT_CONFIG_ENV)
    assert connector_tools() == [] and reply_tool() is None


# --- The call ------------------------------------------------------------------


async def test_posts_the_turn_and_the_listings_tool_and_hash(proxy):
    text = await _call("erp__create_lead", {"phone": "2010"})
    ((path, body, auth),) = proxy.calls
    assert (path, auth) == ("/channels/connector-tool-call", "Bearer spawn-token")
    assert body == {
        "session_id": "whatsapp:201001234567", "turn_id": "turn-1", "connection": "heatec-erp",
        "tool": "create_lead", "arguments": {"phone": "2010"}, "definitionHash": "1" * 64,
    }
    assert json.loads(text) == API_ANSWER


async def test_a_slot_tool_names_the_slots_connection(proxy):
    tool = connector_tools("erp")[0]
    await call_within(TURN, tool(phone="1"))
    assert proxy.calls[0][1]["connection"] == "heatec-erp"


@pytest.mark.parametrize("turn", [{}, {"session_id": "s"}, {"turn_id": "t"}, {"session_id": "", "turn_id": "t"}, {"session_id": "s", "turn_id": "x" * 257}])
async def test_no_running_turn_sends_nothing(proxy, turn):
    text = await _call("erp__create_lead", {"phone": "1"}, turn)
    assert text.startswith("no_turn: erp__create_lead can only be called while a turn is running")
    assert proxy.calls == []


async def test_outside_the_sandbox_sends_nothing(monkeypatch):
    monkeypatch.delenv(CHANNELS_URL_ENV, raising=False)
    text = await _call("erp__create_lead", {"phone": "1"})
    assert text.startswith("not_in_sandbox: ")


async def test_a_non_2xx_api_answer_is_a_result(proxy):
    proxy.responses = [(200, {**API_ANSWER, "status": 404, "ok": False, "body": "nope", "truncated": True, "note": "cut at 64 KB"})]
    got = json.loads(await _call("erp__create_lead", {"phone": "1"}))
    assert (got["status"], got["ok"], got["truncated"], got["note"]) == (404, False, True, "cut at 64 KB")


@pytest.mark.parametrize("failure", ["timeout", "transport", "too_large"])
async def test_an_api_failure_may_have_completed(proxy, failure):
    proxy.responses = [(200, {"tool": "erp__create_lead", "failure": failure})]
    text = await _call("erp__create_lead", {"phone": "1"})
    assert "erp__create_lead may already have completed" in text
    assert "Do not call erp__create_lead again in this turn" in text
    assert len(proxy.calls) == 1


REFUSALS = sorted(set(CODES["refusal"]) | set(CODES["route_refusal"]))


@pytest.mark.parametrize("code", REFUSALS)
async def test_every_refusal_reaches_the_model_unchanged(proxy, code):
    message = f"the platform's own words for {code}"
    proxy.responses = [(409, {"code": code, "message": message, "tool": "erp__create_lead"})]
    assert await _call("erp__create_lead", {"phone": "1"}) == f"{code}: {message}"
    assert len(proxy.calls) == 1


@pytest.mark.parametrize("status, code", [(504, "backend_timeout"), (500, "backend_error")])
async def test_an_unknown_outcome_is_never_retried(proxy, status, code):
    proxy.responses = [(status, {"code": code, "message": "m", "retryable": False})]
    text = await _call("erp__create_lead", {"phone": "1"})
    assert text == (
        "erp__create_lead may already have completed: the platform did not confirm the result, so the result is unknown. "
        "Do not call erp__create_lead again in this turn. "
        "Tell the person what you tried to do, so they can check whether it happened."
    )
    assert len(proxy.calls) == 1


@pytest.mark.parametrize("status, payload", [(503, {"detail": "not configured"}), (500, b"Internal Server Error"), (200, b"not json"), (200, [1])])
async def test_an_unreadable_answer_is_an_unknown_outcome(proxy, status, payload):
    proxy.responses = [(status, payload)]
    assert "may already have completed" in await _call("erp__create_lead", {"phone": "1"})
    assert len(proxy.calls) == 1


async def test_a_4xx_without_a_code_is_a_refusal(proxy):
    proxy.responses = [(413, {"detail": "too large"})]
    assert await _call("erp__create_lead", {"phone": "1"}) == "erp__create_lead was refused (413): erp__create_lead was refused"


async def test_backend_unreachable_is_retried_then_answered(proxy):
    unreachable = (502, {"code": "backend_unreachable", "message": "the platform could not be reached; the tool did not run", "retryable": True})
    proxy.responses = [unreachable, (200, API_ANSWER)]
    assert json.loads(await _call("erp__create_lead", {"phone": "1"})) == API_ANSWER
    assert len(proxy.calls) == 2

    proxy.calls = []
    proxy.responses = [unreachable]
    assert await _call("erp__create_lead", {"phone": "1"}) == "backend_unreachable: the platform could not be reached; the tool did not run"
    assert len(proxy.calls) == 3


async def test_engine_rate_limit_is_not_retried(proxy):
    proxy.responses = [(429, {"code": "engine_rate_limited", "message": "wait", "tool": None})]
    assert await _call("erp__create_lead", {"phone": "1"}) == "engine_rate_limited: wait"
    assert len(proxy.calls) == 1


async def test_no_answer_at_all_is_an_unknown_outcome(monkeypatch):
    monkeypatch.setenv(CHANNELS_URL_ENV, "http://127.0.0.1:9/channels")
    monkeypatch.setenv(CHANNELS_TOKEN_ENV, "t")
    assert "may already have completed" in await _call("erp__create_lead", {"phone": "1"})


def test_the_http_timeout_is_longer_than_the_platforms_limit():
    assert ct._TIMEOUT_S > 30


def test_codes_match_the_platforms_list():
    assert ct.REFUSAL_CODES == set(CODES["refusal"]) | set(CODES["route_refusal"])
    assert ct.UNKNOWN_OUTCOME_CODES == set(CODES["unknown_outcome"])
    assert ct.RETRYABLE_CODES == set(CODES["retryable"])
    assert not ct.LOCAL_CODES & (ct.REFUSAL_CODES | ct.UNKNOWN_OUTCOME_CODES | ct.RETRYABLE_CODES)


def test_error_properties():
    assert ConnectorToolError("m", status=504, code="backend_timeout").outcome_unknown
    assert ConnectorToolError("m", status=0).delivery_unknown
    assert not ConnectorToolError("m", status=409, code="tool_changed").outcome_unknown
    assert ConnectorToolError("m", status=502, code="backend_unreachable").retryable
    assert not ConnectorToolError("m", status=502).retryable


# --- reply -----------------------------------------------------------------------


async def test_reply_names_neither_connection_nor_slot(proxy):
    proxy.responses = [(200, {"sent": True, "messageIds": ["wamid.1"], "quoted": True})]
    got = await call_within(TURN, reply("One moment, checking.", quote="wamid.0"))
    assert got == ReplyResult(message_ids=("wamid.1",), quoted=True)
    ((_, body, _),) = proxy.calls
    assert body == {
        "session_id": "whatsapp:201001234567", "turn_id": "turn-1", "tool": "reply",
        "arguments": {"text": "One moment, checking.", "quote": "wamid.0"}, "definitionHash": "5" * 64,
    }


async def test_reply_without_a_quote(proxy):
    proxy.responses = [(200, {"sent": True, "messageIds": ["m"]})]
    assert await call_within(TURN, reply("hi")) == ReplyResult(message_ids=("m",), quoted=None)
    assert proxy.calls[0][1]["arguments"] == {"text": "hi"}


async def test_reply_refusal_raises_with_the_platforms_message(proxy):
    proxy.responses = [(422, {"code": "no_conversation", "message": "reply works only in a turn answering a channel conversation", "tool": "reply"})]
    with pytest.raises(ConnectorToolError) as caught:
        await call_within(TURN, reply("hi"))
    assert (caught.value.code, caught.value.status, caught.value.tool) == ("no_conversation", 422, "reply")
    assert str(caught.value) == "reply works only in a turn answering a channel conversation"


async def test_reply_not_listed(proxy, tmp_path, monkeypatch):
    config._load.cache_clear()
    path = tmp_path / "none.json"
    path.write_text(json.dumps({"version": 1}))
    monkeypatch.setenv(config.AGENT_CONFIG_ENV, str(path))
    with pytest.raises(ConnectorToolError) as caught:
        await call_within(TURN, reply("hi"))
    assert caught.value.code == "tool_not_listed"
    monkeypatch.delenv(CHANNELS_URL_ENV)
    with pytest.raises(ConnectorToolError) as caught:
        await call_within(TURN, reply("hi"))
    assert caught.value.code == "not_in_sandbox"
    assert proxy.calls == []


async def test_reply_checks_its_arguments():
    with pytest.raises(ValueError):
        await reply("  ")
    with pytest.raises(ValueError):
        await reply("hi", quote="")


@pytest.mark.parametrize("answer, text", [
    ({"sent": True, "messageIds": ["m"]}, "sent"),
    ({"sent": True, "messageIds": ["m"], "quoted": True}, "sent"),
    ({"sent": True, "messageIds": ["m"], "quoted": False}, "sent, without the quote: this channel cannot quote a message"),
])
async def test_reply_tool_answers(proxy, answer, text):
    proxy.responses = [(200, answer)]
    assert await call_within(TURN, reply_tool()(text="hi", quote="q")) == text


async def test_reply_tool_outside_a_conversation(proxy):
    proxy.responses = [(422, {"code": "no_conversation", "message": "reply works only in a turn answering a channel conversation", "tool": "reply"})]
    assert await call_within(TURN, reply_tool()(text="hi")) == "no_conversation: reply works only in a turn answering a channel conversation"


# --- In an Agent ---------------------------------------------------------------------


class _Model:
    def __init__(self, *rounds: list[AgentOutput]) -> None:
        self.rounds = list(rounds)
        self.offered: list[str] = []
        self.results: list[str] = []
        self.complete = self._complete

    async def _complete(self, messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        self.offered = [t.name for t in tools or []]
        self.results += [m["content"] for m in messages if m.get("role") == "tool"]
        for chunk in self.rounds.pop(0):
            yield chunk


_DONE = [AgentOutput(content="ok", done=True)]


def _agent(model: _Model) -> Agent:
    tools = [*connector_tools(), reply_tool()]
    agent = Agent(role="r", model={"model": "openai/gpt-4o-mini", "api_key": "k"}, tools=tools)
    agent._provider = model
    return agent


def _meta(**extra) -> dict:
    return {**TURN, CALLER_METADATA_KEY: {"role": "customer", "private": True, "channel": "whatsapp"}, **extra}


async def test_the_platforms_audience_setting_decides_by_listed_name(proxy):
    model = _Model(_DONE)
    [o async for o in _agent(model).run(AgentInput(message="hi", metadata=_meta()))]
    assert model.offered == ["reply"]

    model = _Model([AgentOutput(tool_calls=[ToolCall(id="1", name="erp__create_lead", arguments={"phone": "1"})])], _DONE)
    meta = _meta(**{TOOL_AUDIENCES_METADATA_KEY: {"erp__create_lead": ["customer"], "reply": ["operator"]}})
    [o async for o in _agent(model).run(AgentInput(message="hi", metadata=meta))]
    assert model.offered == ["erp__create_lead"]
    assert proxy.calls[0][1]["session_id"] == TURN["session_id"] and proxy.calls[0][1]["turn_id"] == "turn-1"
    assert json.loads(model.results[0]) == API_ANSWER


async def test_a_tool_the_platform_flags_waits_for_approval(proxy):
    model = _Model([AgentOutput(tool_calls=[ToolCall(id="1", name="erp__create_lead", arguments={"phone": "1"})])], _DONE)
    meta = _meta(**{TOOL_AUDIENCES_METADATA_KEY: {"erp__create_lead": ["customer"]}, SENSITIVE_TOOLS_KEY: {"erp__create_lead": True}})
    [o async for o in _agent(model).run(AgentInput(message="hi", metadata=meta))]
    assert proxy.calls == []
