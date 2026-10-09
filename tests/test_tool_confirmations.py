"""Sensitive tools and their confirmations (ADR-0028 §8a, SDK 0.19). Tier 1, no network."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from genfleet.sdk import (
    APPROVED_CALL_KEY,
    CONFIRMATION_REQUESTS_KEY,
    SENSITIVE_TOOLS_KEY,
    Agent,
    AgentInput,
    AgentOutput,
    ToolCall,
    confirmations,
    sign_approved_call,
    tool,
)
from genfleet.sdk import agent as agent_module
from genfleet.sdk.caller import (
    _LEGACY_HOSTED_ENV,
    CALLER_METADATA_KEY,
    HOSTED_ENV,
    TOOL_AUDIENCES_METADATA_KEY,
)
from genfleet.sdk.confirmations import (
    APPROVAL_KEY_ENV,
    PEER_REFUSAL,
    PEER_TURN_KEY,
    PENDING_RESULT,
    ApprovedCall,
    approved_call_of,
    canonical_json,
    is_peer_turn,
    mark_run,
    sensitive_flags_of,
)
from genfleet.sdk.manifest import load_tool_manifest
from genfleet.sdk.tool import wrap_tool

KEY = "spawn-key-1"
NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
LATER = "2026-10-09T12:30:00+00:00"
EARLIER = "2026-10-09T11:30:00+00:00"


def _call(**over) -> dict:
    call = {
        "confirmation_id": "c-1",
        "tool": "refund",
        "arguments": {"order": "A1", "amount": 5},
        "args_hash": "abc123",
        "expires_at": LATER,
    }
    call.update(over)
    return call


def _signed(key: str = KEY, **over) -> dict:
    call = _call(**over)
    call["signature"] = sign_approved_call(call, key)
    return call


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ran.clear()
    confirmations._RUN.clear()
    monkeypatch.delenv(HOSTED_ENV, raising=False)
    monkeypatch.delenv(_LEGACY_HOSTED_ENV, raising=False)
    monkeypatch.delenv(APPROVAL_KEY_ENV, raising=False)
    yield
    confirmations._RUN.clear()


# ---------------------------------------------------------------------------
# canonical_json and the signature's exact form
# ---------------------------------------------------------------------------

def test_canonical_json_sorts_keys_and_is_compact():
    assert canonical_json({"b": 1, "a": {"d": [1, 2], "c": None}}) == b'{"a":{"c":null,"d":[1,2]},"b":1}'


def test_canonical_json_keeps_unicode_raw_as_utf8():
    out = canonical_json({"name": "café ☕"})
    assert out == '{"name":"café ☕"}'.encode()
    assert b"\\u" not in out


def test_signature_golden_vector_pins_the_form_both_sides_use():
    call = {
        "confirmation_id": "c-9",
        "tool": "@acme/crm",
        "arguments": {"z": "é", "a": [1, True, None]},
        "args_hash": "deadbeef",
        "expires_at": "2026-10-09T12:30:00Z",
        "signature": "ignored",
        "extra": "ignored",
    }
    expected_bytes = (
        '{"args_hash":"deadbeef","arguments":{"a":[1,true,null],"z":"é"},"confirmation_id":"c-9",'
        '"expires_at":"2026-10-09T12:30:00Z","tool":"@acme/crm"}'
    ).encode()
    expected = hmac.new(b"k", expected_bytes, hashlib.sha256).hexdigest()
    assert sign_approved_call(call, "k") == expected
    assert sign_approved_call(call, b"k") == expected
    assert len(expected) == 64


def test_missing_signed_fields_are_signed_as_null():
    expected_bytes = b'{"args_hash":null,"arguments":null,"confirmation_id":"c","expires_at":null,"tool":null}'
    assert sign_approved_call({"confirmation_id": "c"}, "k") == hmac.new(b"k", expected_bytes, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# approved_call_of
# ---------------------------------------------------------------------------

def test_a_valid_call_verifies():
    call, reason = approved_call_of({APPROVED_CALL_KEY: _signed()}, now=NOW, key=KEY)
    assert reason is None
    assert call == ApprovedCall("c-1", "refund", {"order": "A1", "amount": 5})


def test_no_approved_call_is_none_none():
    assert approved_call_of(None, now=NOW, key=KEY) == (None, None)
    assert approved_call_of({}, now=NOW, key=KEY) == (None, None)
    assert approved_call_of({"other": 1}, now=NOW, key=KEY) == (None, None)


def test_key_comes_from_the_environment(monkeypatch):
    meta = {APPROVED_CALL_KEY: _signed()}
    assert approved_call_of(meta, now=NOW) == (None, "no approval key in this environment")
    monkeypatch.setenv(APPROVAL_KEY_ENV, "")
    assert approved_call_of(meta, now=NOW) == (None, "no approval key in this environment")
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    call, reason = approved_call_of(meta, now=NOW)
    assert reason is None and call is not None


def test_an_explicit_key_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv(APPROVAL_KEY_ENV, "other")
    call, reason = approved_call_of({APPROVED_CALL_KEY: _signed()}, now=NOW, key=KEY)
    assert reason is None and call is not None


def test_a_signature_from_another_key_is_refused():
    meta = {APPROVED_CALL_KEY: _signed(key="someone-else")}
    assert approved_call_of(meta, now=NOW, key=KEY) == (None, "bad signature")


@pytest.mark.parametrize("signature", [None, "", "x", "0" * 64, "0" * 65, 5, ["a"]])
def test_garbage_or_wrong_length_signatures_are_refused(signature):
    raw = _call()
    if signature is not None:
        raw["signature"] = signature
    assert approved_call_of({APPROVED_CALL_KEY: raw}, now=NOW, key=KEY) == (None, "bad signature")


@pytest.mark.parametrize(
    "field, value",
    [
        ("tool", "other"),
        ("arguments", {"order": "A1", "amount": 500}),
        ("args_hash", "zzz"),
        ("expires_at", "2030-01-01T00:00:00+00:00"),
        ("confirmation_id", "c-2"),
    ],
)
def test_tampering_with_any_signed_field_breaks_the_signature(field, value):
    raw = _signed()
    raw[field] = value
    assert approved_call_of({APPROVED_CALL_KEY: raw}, now=NOW, key=KEY) == (None, "bad signature")


def test_the_signature_field_is_not_part_of_the_signed_bytes():
    raw = _signed()
    assert sign_approved_call(raw, KEY) == sign_approved_call({**raw, "signature": "other"}, KEY)
    # Unsigned extras don't matter either.
    raw["note"] = "free text"
    call, reason = approved_call_of({APPROVED_CALL_KEY: raw}, now=NOW, key=KEY)
    assert reason is None and call is not None


@pytest.mark.parametrize("raw", ["text", ["a"], 5, True])
def test_a_non_dict_approved_call_is_malformed(raw):
    assert approved_call_of({APPROVED_CALL_KEY: raw}, now=NOW, key=KEY) == (None, "malformed")


@pytest.mark.parametrize(
    "over",
    [
        {"confirmation_id": ""},
        {"confirmation_id": 7},
        {"tool": ""},
        {"tool": None},
        {"arguments": "x"},
        {"arguments": None},
        {"expires_at": "not a time"},
        {"expires_at": None},
        {"expires_at": "2026-10-09T12:30:00"},  # naive
    ],
)
def test_validly_signed_but_malformed_calls_are_refused(over):
    assert approved_call_of({APPROVED_CALL_KEY: _signed(**over)}, now=NOW, key=KEY) == (None, "malformed")


def test_expiry_is_exclusive_and_accepts_z():
    z = "2026-10-09T12:30:00Z"
    call, reason = approved_call_of({APPROVED_CALL_KEY: _signed(expires_at=z)}, now=NOW, key=KEY)
    assert reason is None and call is not None
    at = datetime(2026, 10, 9, 12, 30, tzinfo=UTC)
    meta = {APPROVED_CALL_KEY: _signed(expires_at=z)}
    assert approved_call_of(meta, now=at, key=KEY) == (None, "expired")
    assert approved_call_of(meta, now=at + timedelta(seconds=1), key=KEY) == (None, "expired")
    assert approved_call_of(meta, now=at - timedelta(microseconds=1), key=KEY)[1] is None
    assert approved_call_of({APPROVED_CALL_KEY: _signed(expires_at=EARLIER)}, now=NOW, key=KEY) == (None, "expired")


def test_expiry_defaults_to_the_real_clock():
    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    future = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    assert approved_call_of({APPROVED_CALL_KEY: _signed(expires_at=past)}, key=KEY) == (None, "expired")
    assert approved_call_of({APPROVED_CALL_KEY: _signed(expires_at=future)}, key=KEY)[1] is None


def test_replay_is_refused_after_mark_run():
    meta = {APPROVED_CALL_KEY: _signed()}
    call, reason = approved_call_of(meta, now=NOW, key=KEY)
    assert call is not None and reason is None
    # Verifying alone doesn't consume it.
    assert approved_call_of(meta, now=NOW, key=KEY)[0] == call
    mark_run(call)
    assert approved_call_of(meta, now=NOW, key=KEY) == (None, "already run")
    # Another confirmation is unaffected.
    other = {APPROVED_CALL_KEY: _signed(confirmation_id="c-2")}
    assert approved_call_of(other, now=NOW, key=KEY)[1] is None


def test_a_bad_signature_is_reported_before_a_replay():
    mark_run(ApprovedCall("c-1", "refund", {}))
    raw = _signed()
    raw["signature"] = "0" * 64
    assert approved_call_of({APPROVED_CALL_KEY: raw}, now=NOW, key=KEY) == (None, "bad signature")


# ---------------------------------------------------------------------------
# flags, peer turn, exports
# ---------------------------------------------------------------------------

def test_sensitive_flags_keep_only_str_to_bool():
    meta = {SENSITIVE_TOOLS_KEY: {"a": True, "b": False, "": True, "c": "yes", "d": 1, "e": None, 7: True}}
    assert sensitive_flags_of(meta) == {"a": True, "b": False}


@pytest.mark.parametrize("raw", ["x", ["a"], 5, None, True])
def test_non_dict_flags_are_empty(raw):
    assert sensitive_flags_of({SENSITIVE_TOOLS_KEY: raw}) == {}


@pytest.mark.parametrize("meta", [None, {}, {"other": 1}])
def test_missing_flags_are_empty(meta):
    assert sensitive_flags_of(meta) == {}


def test_is_peer_turn_only_for_literal_true():
    assert is_peer_turn({PEER_TURN_KEY: True}) is True
    for value in (1, "true", False, None):
        assert is_peer_turn({PEER_TURN_KEY: value}) is False
    assert is_peer_turn(None) is False
    assert is_peer_turn({}) is False


def test_keys_and_exports():
    assert SENSITIVE_TOOLS_KEY == "genfleet.sensitive_tools"
    assert APPROVED_CALL_KEY == "genfleet.approved_call"
    assert PEER_TURN_KEY == "genfleet.peer_turn"
    assert CONFIRMATION_REQUESTS_KEY == "genfleet.confirmation_requests"
    assert APPROVAL_KEY_ENV == "GENFLEET_APPROVAL_KEY"


# ---------------------------------------------------------------------------
# @tool / ToolWrapper / manifest
# ---------------------------------------------------------------------------

def test_tool_sensitive_defaults_false_and_survives_with_slug():
    @tool
    def plain() -> str:
        return ""

    @tool(sensitive=True)
    def risky() -> str:
        return ""

    assert plain.sensitive is False
    assert risky.sensitive is True
    assert risky.with_slug("@a/risky").sensitive is True
    assert risky.with_slug("@a/risky").slug == "@a/risky"
    assert wrap_tool(risky, slug="@a/risky").sensitive is True


def test_manifest_parses_sensitive(tmp_path: Path):
    (tmp_path / "genfleet.toml").write_text(
        '[tool]\nslug = "@acme/pay"\nversion = "1.0.0"\nentry = "pay:run"\nsensitive = true\n'
    )
    assert load_tool_manifest(tmp_path).sensitive is True


def test_manifest_sensitive_defaults_false(tmp_path: Path):
    (tmp_path / "genfleet.toml").write_text('[tool]\nslug = "@acme/pay"\nversion = "1.0.0"\nentry = "pay:run"\n')
    assert load_tool_manifest(tmp_path).sensitive is False


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

class _Model:
    """A fake model: records what it is offered and shown, then replays rounds."""

    def __init__(self, *rounds: list[AgentOutput]) -> None:
        self.rounds = list(rounds)
        self.offered: list[str] = []
        self.seen: list[list[dict]] = []
        self.complete = self._complete

    async def _complete(self, messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        self.offered = [t.name for t in tools or []]
        self.seen.append([dict(m) for m in messages])
        for chunk in self.rounds.pop(0):
            yield chunk


_DONE = [AgentOutput(content="ok", done=True)]
ran: list[tuple[str, dict]] = []


@tool(sensitive=True)
def refund(order: str = "", amount: int = 0) -> str:
    ran.append(("refund", {"order": order, "amount": amount}))
    return "refunded"


@tool
def lookup(order: str = "") -> str:
    ran.append(("lookup", {"order": order}))
    return "shipped"


def _fn_search(q: str = "") -> str:
    ran.append(("search", {"q": q}))
    return "found"


def _fn_x(a: int = 0) -> str:
    ran.append(("x", {"a": a}))
    return "x-done"


def _agent(model: _Model, **kwargs) -> Agent:
    agent = Agent(role="r", model={"model": "openai/gpt-4o-mini", "api_key": "k"}, **kwargs)
    agent._provider = model
    return agent


async def _run(agent: Agent, metadata: dict) -> list[AgentOutput]:
    return [o async for o in agent.run(AgentInput(message="hi", metadata=metadata))]


def _call_round(name: str, **arguments) -> list[AgentOutput]:
    return [AgentOutput(tool_calls=[ToolCall(id="t1", name=name, arguments=arguments)])]


def _events(outputs: list[AgentOutput], kind: str) -> list[dict]:
    return [o.metadata["tool_event"] for o in outputs if o.metadata.get("tool_event", {}).get("type") == kind]


def _requests(outputs: list[AgentOutput]) -> list[dict]:
    return outputs[-1].metadata.get(CONFIRMATION_REQUESTS_KEY, [])


def _tool_messages(model: _Model) -> list[str]:
    return [m["content"] for m in model.seen[-1] if m.get("role") == "tool"]


@pytest.mark.asyncio
async def test_a_sensitive_tool_is_not_run_and_is_reported():
    model = _Model(_call_round("refund", order="A1"), _DONE)
    outputs = await _run(_agent(model, tools=[refund, lookup]), {})
    assert ran == []
    assert _tool_messages(model) == [PENDING_RESULT]
    result = _events(outputs, "tool_result")[0]
    assert result["ok"] is True
    assert result["output"] == PENDING_RESULT
    assert outputs[-1].done is True
    assert _requests(outputs) == [{"tool": "refund", "arguments": {"order": "A1"}}]


@pytest.mark.asyncio
async def test_an_ordinary_tool_still_runs_and_leaves_metadata_empty():
    model = _Model(_call_round("lookup", order="A1"), _DONE)
    outputs = await _run(_agent(model, tools=[refund, lookup]), {})
    assert ran == [("lookup", {"order": "A1"})]
    assert outputs[-1].done is True
    assert outputs[-1].metadata == {}


@pytest.mark.asyncio
async def test_no_tool_calls_means_empty_done_metadata():
    outputs = await _run(_agent(_Model(_DONE), tools=[refund]), {})
    assert outputs[-1].metadata == {}


@pytest.mark.asyncio
async def test_several_requests_are_all_collected_in_order():
    model = _Model(
        [AgentOutput(tool_calls=[
            ToolCall(id="1", name="refund", arguments={"order": "A"}),
            ToolCall(id="2", name="lookup", arguments={"order": "B"}),
            ToolCall(id="3", name="refund", arguments={"order": "C"}),
        ])],
        _DONE,
    )
    outputs = await _run(_agent(model, tools=[refund, lookup]), {})
    assert [r["arguments"]["order"] for r in _requests(outputs)] == ["A", "C"]
    assert ran == [("lookup", {"order": "B"})]


@pytest.mark.asyncio
async def test_request_key_for_a_slug_tagged_tool_is_its_slug():
    scoped = wrap_tool(_fn_x, slug="@scope/x")
    model = _Model(_call_round("_fn_x", a=1), _DONE)
    outputs = await _run(_agent(model, tools=[scoped]), {SENSITIVE_TOOLS_KEY: {"@scope/x": True}})
    assert ran == []
    assert _requests(outputs) == [{"tool": "@scope/x", "arguments": {"a": 1}}]


@pytest.mark.asyncio
async def test_request_key_for_a_plain_slug_gets_an_at():
    mounted = wrap_tool(_fn_search, slug="search")
    model = _Model(_call_round("_fn_search", q="z"), _DONE)
    outputs = await _run(_agent(model, tools=[mounted]), {SENSITIVE_TOOLS_KEY: {"@search": True}})
    assert ran == []
    assert _requests(outputs) == [{"tool": "@search", "arguments": {"q": "z"}}]


@pytest.mark.asyncio
async def test_a_slug_tagged_tool_ignores_a_name_keyed_flag():
    scoped = wrap_tool(_fn_x, slug="@scope/x")
    model = _Model(_call_round("_fn_x", a=1), _DONE)
    outputs = await _run(_agent(model, tools=[scoped]), {SENSITIVE_TOOLS_KEY: {"_fn_x": True}})
    assert ran == [("x", {"a": 1})]
    assert _requests(outputs) == []


@pytest.mark.asyncio
async def test_a_slug_tagged_author_marking_is_kept():
    @tool(sensitive=True)
    def pay() -> str:
        ran.append(("pay", {}))
        return "paid"

    model = _Model(_call_round("pay"), _DONE)
    outputs = await _run(_agent(model, tools=[pay.with_slug("@acme/pay")]), {})
    assert ran == []
    assert _requests(outputs) == [{"tool": "@acme/pay", "arguments": {}}]


@pytest.mark.asyncio
async def test_platform_flag_makes_an_unmarked_tool_sensitive():
    model = _Model(_call_round("lookup", order="A1"), _DONE)
    outputs = await _run(_agent(model, tools=[lookup]), {SENSITIVE_TOOLS_KEY: {"lookup": True}})
    assert ran == []
    assert _requests(outputs) == [{"tool": "lookup", "arguments": {"order": "A1"}}]


@pytest.mark.asyncio
async def test_platform_flag_false_unmarks_an_author_sensitive_tool():
    model = _Model(_call_round("refund", order="A1"), _DONE)
    outputs = await _run(_agent(model, tools=[refund]), {SENSITIVE_TOOLS_KEY: {"refund": False}})
    assert ran == [("refund", {"order": "A1", "amount": 0})]
    assert _requests(outputs) == []


@pytest.mark.asyncio
async def test_a_malformed_flag_falls_back_to_the_authors_marking():
    model = _Model(_call_round("refund", order="A1"), _DONE)
    outputs = await _run(_agent(model, tools=[refund]), {SENSITIVE_TOOLS_KEY: {"refund": "no"}})
    assert ran == []
    assert len(_requests(outputs)) == 1


@pytest.mark.asyncio
async def test_a_peer_turn_refuses_a_sensitive_tool():
    model = _Model(_call_round("refund", order="A1"), _DONE)
    outputs = await _run(_agent(model, tools=[refund]), {PEER_TURN_KEY: True})
    assert ran == []
    result = _events(outputs, "tool_result")[0]
    assert result["ok"] is False
    assert result["output"] == PEER_REFUSAL
    assert _tool_messages(model) == [PEER_REFUSAL]
    assert outputs[-1].metadata == {}


@pytest.mark.asyncio
async def test_a_peer_turn_still_runs_ordinary_tools():
    model = _Model(_call_round("lookup", order="A1"), _DONE)
    await _run(_agent(model, tools=[refund, lookup]), {PEER_TURN_KEY: True})
    assert ran == [("lookup", {"order": "A1"})]


@pytest.mark.asyncio
async def test_a_non_offered_sensitive_tool_is_not_found_and_not_requested():
    # Customer in private: refund (operator-only by default) is not offered.
    meta = {CALLER_METADATA_KEY: {"role": "customer", "private": True}}
    model = _Model(_call_round("refund", order="A1"), _DONE)
    outputs = await _run(_agent(model, tools=[refund, lookup]), meta)
    assert "refund" not in model.offered
    result = _events(outputs, "tool_result")[0]
    assert result["ok"] is False
    assert result["output"] == "Error: tool 'refund' not found"
    assert _requests(outputs) == []
    assert ran == []

    # The same on a peer turn: not-found, not the peer refusal (no leak).
    model = _Model(_call_round("refund", order="A1"), _DONE)
    outputs = await _run(_agent(model, tools=[refund, lookup]), {**meta, PEER_TURN_KEY: True})
    assert _events(outputs, "tool_result")[0]["output"] == "Error: tool 'refund' not found"


def _mcp_fetch(monkeypatch, names):
    async def fetch(cfg):
        return {n: {"description": "", "parameters": {}, "config": cfg} for n in names}

    monkeypatch.setattr(agent_module, "_fetch_mcp_tools", fetch)


@pytest.mark.asyncio
async def test_mcp_sensitive_tools_config_waits_for_approval(monkeypatch):
    config = {"type": "sse", "url": "http://mcp.test", "sensitive_tools": ["wire"]}
    _mcp_fetch(monkeypatch, ["wire", "read"])
    invoked: list[str] = []

    async def invoke(cfg, name, arguments):
        invoked.append(name)
        return "ok", True

    monkeypatch.setattr(agent_module, "_invoke_mcp_tool", invoke)

    model = _Model(
        [AgentOutput(tool_calls=[
            ToolCall(id="1", name="wire", arguments={"to": "x"}),
            ToolCall(id="2", name="read", arguments={}),
        ])],
        _DONE,
    )
    outputs = await _run(_agent(model, mcps=[config]), {})
    assert invoked == ["read"]
    assert _requests(outputs) == [{"tool": "wire", "arguments": {"to": "x"}}]


@pytest.mark.asyncio
async def test_mcp_platform_flag_overrides_the_config_by_name(monkeypatch):
    config = {"type": "sse", "url": "http://mcp.test", "sensitive_tools": ["wire"]}
    _mcp_fetch(monkeypatch, ["wire", "read"])
    invoked: list[str] = []

    async def invoke(cfg, name, arguments):
        invoked.append(name)
        return "ok", True

    monkeypatch.setattr(agent_module, "_invoke_mcp_tool", invoke)
    flags = {SENSITIVE_TOOLS_KEY: {"wire": False, "read": True}}
    model = _Model(
        [AgentOutput(tool_calls=[
            ToolCall(id="1", name="wire", arguments={}),
            ToolCall(id="2", name="read", arguments={}),
        ])],
        _DONE,
    )
    outputs = await _run(_agent(model, mcps=[config]), flags)
    assert invoked == ["wire"]
    assert _requests(outputs) == [{"tool": "read", "arguments": {}}]


@pytest.mark.asyncio
async def test_an_mcp_tool_named_like_a_slug_key_is_skipped(monkeypatch):
    config = {"type": "sse", "url": "http://mcp.test", "sensitive_tools": ["@acme/x"]}
    _mcp_fetch(monkeypatch, ["@acme/x", "fine"])
    agent = _agent(_Model(_DONE), mcps=[config])
    await agent._init_mcp()
    assert set(agent._mcp_tools) == {"fine"}


@pytest.mark.asyncio
async def test_an_mcp_tool_named_like_a_slug_key_cannot_borrow_a_manifest_tools_flag(monkeypatch):
    _mcp_fetch(monkeypatch, ["@scope/x"])
    scoped = wrap_tool(_fn_x, slug="@scope/x")
    model = _Model(_call_round("@scope/x", a=1), _DONE)
    outputs = await _run(
        _agent(model, tools=[scoped], mcps=[{"type": "sse", "url": "http://mcp.test"}]),
        {SENSITIVE_TOOLS_KEY: {"@scope/x": False}},
    )
    assert "@scope/x" not in model.offered
    assert _events(outputs, "tool_result")[0]["output"] == "Error: tool '@scope/x' not found"


# ---------------------------------------------------------------------------
# Agent: the approved call
# ---------------------------------------------------------------------------

def _approved_meta(**over) -> dict:
    return {APPROVED_CALL_KEY: _signed(expires_at=_future(), **over)}


def _future() -> str:
    return (datetime.now(UTC) + timedelta(minutes=10)).isoformat()


def _past() -> str:
    return (datetime.now(UTC) - timedelta(minutes=10)).isoformat()


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)


@pytest.mark.asyncio
async def test_an_approved_call_runs_once_before_the_model(key):
    model = _Model(_DONE)
    outputs = await _run(_agent(model, tools=[refund]), _approved_meta())
    assert ran == [("refund", {"order": "A1", "amount": 5})]
    calls, results = _events(outputs, "tool_call"), _events(outputs, "tool_result")
    assert [c["name"] for c in calls] == ["refund"]
    assert calls[0]["id"] == "approved-c-1"
    assert calls[0]["arguments"] == {"order": "A1", "amount": 5}
    assert results[0]["ok"] is True and results[0]["output"] == "refunded"
    # Events come before the model is asked; the model sees the call and its result.
    assert [o.metadata["tool_event"]["type"] for o in outputs if "tool_event" in o.metadata] == ["tool_call", "tool_result"]
    history = model.seen[0]
    assistant = next(m for m in history if m.get("tool_calls"))
    assert assistant["tool_calls"][0]["id"] == "approved-c-1"
    tool_msg = next(m for m in history if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "approved-c-1"
    assert tool_msg["content"] == "refunded"
    assert "approved `refund`" in history[0]["content"]
    assert outputs[-1].metadata == {}


@pytest.mark.asyncio
async def test_the_same_signed_call_redelivered_does_nothing(key):
    meta = _approved_meta()
    await _run(_agent(_Model(_DONE), tools=[refund]), meta)
    assert len(ran) == 1
    model = _Model(_DONE)
    outputs = await _run(_agent(model, tools=[refund]), meta)
    assert len(ran) == 1
    assert _events(outputs, "tool_call") == []
    assert "could not be verified" in model.seen[0][0]["content"]
    assert not any(m.get("role") == "tool" for m in model.seen[0])


@pytest.mark.asyncio
async def test_an_approved_call_for_a_sensitive_tool_runs_despite_sensitivity(key):
    meta = {**_approved_meta(), SENSITIVE_TOOLS_KEY: {"refund": True}}
    outputs = await _run(_agent(_Model(_DONE), tools=[refund]), meta)
    assert ran == [("refund", {"order": "A1", "amount": 5})]
    assert _requests(outputs) == []


@pytest.mark.asyncio
async def test_an_approved_call_is_consumed_even_if_the_tool_raises(key):
    @tool(sensitive=True)
    def boom() -> str:
        ran.append(("boom", {}))
        raise RuntimeError("nope")

    meta = _approved_meta(tool="boom", arguments={})
    # The turn goes on: the failure is the tool result, and the model tells the user.
    outputs = await _run(_agent(_Model(_DONE), tools=[boom]), meta)
    assert ran == [("boom", {})]
    [result] = _events(outputs, "tool_result")
    assert result["ok"] is False and "approved action failed (RuntimeError)" in result["output"]
    # Recorded as run: a redelivery does not try again.
    await _run(_agent(_Model(_DONE), tools=[boom]), meta)
    assert len(ran) == 1


@pytest.mark.asyncio
async def test_an_approved_call_by_slug_key(key):
    scoped = wrap_tool(_fn_x, slug="@scope/x")
    meta = _approved_meta(tool="@scope/x", arguments={"a": 3})
    await _run(_agent(_Model(_DONE), tools=[scoped]), meta)
    assert ran == [("x", {"a": 3})]


@pytest.mark.asyncio
async def test_an_approved_call_by_bare_name_does_not_reach_a_slug_tagged_tool(key):
    scoped = wrap_tool(_fn_x, slug="@scope/x")
    model = _Model(_DONE)
    await _run(_agent(model, tools=[scoped]), _approved_meta(tool="_fn_x", arguments={"a": 3}))
    assert ran == []
    assert "can't run for this caller" in model.seen[0][0]["content"]


@pytest.mark.asyncio
async def test_a_bad_signature_does_not_run(key):
    raw = _signed(expires_at=_future())
    raw["arguments"] = {"order": "EVIL"}
    model = _Model(_DONE)
    outputs = await _run(_agent(model, tools=[refund]), {APPROVED_CALL_KEY: raw})
    assert ran == []
    assert _events(outputs, "tool_call") == []
    assert "could not be verified" in model.seen[0][0]["content"]


@pytest.mark.asyncio
async def test_an_expired_call_does_not_run(key):
    model = _Model(_DONE)
    await _run(_agent(model, tools=[refund]), {APPROVED_CALL_KEY: _signed(expires_at=_past())})
    assert ran == []
    assert "could not be verified" in model.seen[0][0]["content"]


@pytest.mark.asyncio
async def test_no_key_in_the_environment_means_nothing_runs():
    model = _Model(_DONE)
    await _run(_agent(model, tools=[refund]), _approved_meta())
    assert ran == []
    assert "could not be verified" in model.seen[0][0]["content"]


@pytest.mark.asyncio
async def test_a_call_signed_with_another_key_does_not_run(key):
    model = _Model(_DONE)
    await _run(_agent(model, tools=[refund]), {APPROVED_CALL_KEY: _signed(key="other", expires_at=_future())})
    assert ran == []


@pytest.mark.asyncio
async def test_an_unknown_tool_does_not_run_and_is_still_consumed(key):
    meta = _approved_meta(tool="ghost", arguments={})
    model = _Model(_DONE)
    outputs = await _run(_agent(model, tools=[refund]), meta)
    assert ran == []
    assert _events(outputs, "tool_call") == []
    assert "can't run for this caller" in model.seen[0][0]["content"]
    # Used once: it can't run later either.
    assert "c-1" in confirmations._RUN


@pytest.mark.asyncio
async def test_a_tool_not_offered_to_this_caller_does_not_run(key):
    meta = {**_approved_meta(), CALLER_METADATA_KEY: {"role": "customer", "private": True}}
    model = _Model(_DONE)
    outputs = await _run(_agent(model, tools=[refund, lookup]), meta)
    assert "refund" not in model.offered
    assert ran == []
    assert _events(outputs, "tool_call") == []
    assert "can't run for this caller" in model.seen[0][0]["content"]
    assert "c-1" in confirmations._RUN


@pytest.mark.asyncio
async def test_an_audience_override_can_hide_the_tool_from_an_approved_call(key):
    meta = {
        **_approved_meta(),
        CALLER_METADATA_KEY: {"role": "operator", "private": True},
        TOOL_AUDIENCES_METADATA_KEY: {"refund": ["customer"]},
    }
    await _run(_agent(_Model(_DONE), tools=[refund]), meta)
    assert ran == []


@pytest.mark.asyncio
async def test_a_turn_with_no_approval_adds_no_note():
    model = _Model(_DONE)
    await _run(_agent(model, tools=[refund]), {})
    system = model.seen[0][0]["content"]
    assert "approval arrived" not in system
    assert "approved `" not in system
    assert "can't run for this caller" not in system


def test_json_dumps_is_what_canonical_json_wraps():
    value = {"b": [1, {"y": 1, "x": 2}], "a": "é"}
    assert canonical_json(value) == json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
