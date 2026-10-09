"""Caller roles (ADR-0028): which tools a turn is offered. Tier 1, no network."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from genfleet.sdk import Agent, AgentInput, AgentOutput, ToolCall, tool
from genfleet.sdk import agent as agent_module
from genfleet.sdk.caller import CALLER_METADATA_KEY, HOSTED_ENV, Caller, caller_of, may_use
from genfleet.sdk.manifest import ManifestError, load_agent_manifest


def _meta(**caller) -> dict:
    return {CALLER_METADATA_KEY: caller}


# ---------------------------------------------------------------------------
# caller_of
# ---------------------------------------------------------------------------

def test_caller_of_absent_is_none():
    assert caller_of(None) is None
    assert caller_of({}) is None
    assert caller_of({"other": 1}) is None


def test_caller_of_parses_a_valid_caller():
    caller = caller_of(_meta(role="operator", id="telegram:1", name="Sam", channel="telegram",
                             private=True, legacy_tools=True))
    assert caller == Caller(role="operator", id="telegram:1", name="Sam", channel="telegram",
                            private=True, legacy_tools=True)


@pytest.mark.parametrize("value", ["true", 1, "yes", None])
def test_private_and_legacy_only_when_exactly_true(value):
    caller = caller_of(_meta(role="operator", private=value, legacy_tools=value))
    assert caller.private is False
    assert caller.legacy_tools is False


@pytest.mark.parametrize("raw", ["operator", 5, None, [], {}, {"role": "admin"}, {"id": "x"}])
def test_malformed_caller_is_the_narrowest(raw):
    caller = caller_of({CALLER_METADATA_KEY: raw})
    assert caller.role == "customer"
    assert caller.private is False
    assert not caller.full_toolset


def test_non_string_text_fields_become_empty():
    caller = caller_of(_meta(role="customer", id=7, name=["x"], channel=None))
    assert (caller.id, caller.name, caller.channel) == ("", "", "")


# ---------------------------------------------------------------------------
# may_use
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "caller, safe, allowed",
    [
        (None, False, True),
        (Caller(role="operator", private=True), False, True),
        (Caller(role="operator", private=False), False, False),
        (Caller(role="operator", private=False), True, True),
        (Caller(role="customer", private=True), False, False),
        (Caller(role="customer", private=True), True, True),
        (Caller(role="customer", legacy_tools=True), False, True),
    ],
)
def test_may_use_truth_table(caller, safe, allowed):
    assert may_use(caller, customer_safe=safe) is allowed


# ---------------------------------------------------------------------------
# @tool marking
# ---------------------------------------------------------------------------

def test_tool_is_operator_only_by_default():
    @tool
    def a() -> str:
        return ""

    @tool(customer_safe=True)
    def b() -> str:
        return ""

    assert a.customer_safe is False
    assert b.customer_safe is True


def test_plain_function_in_agent_is_operator_only():
    def plain() -> str:
        return ""

    agent = Agent(role="r", model={"model": "openai/gpt-4o-mini", "api_key": "k"}, tools=[plain])
    assert agent._local_tools["plain"].customer_safe is False
    assert agent._offered_tools(Caller(role="customer")) == set()


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

class _Model:
    """A fake model: records the tool names it is offered, then replays rounds."""

    def __init__(self, *rounds: list[AgentOutput]) -> None:
        self.rounds = list(rounds)
        self.offered: list[str] = []
        self.complete = self._complete

    async def _complete(self, messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        self.offered = [t.name for t in tools or []]
        for chunk in self.rounds.pop(0):
            yield chunk


_DONE = [AgentOutput(content="ok", done=True)]
ran: list[str] = []


@tool(customer_safe=True)
def lookup_order() -> str:
    ran.append("lookup_order")
    return "shipped"


@tool
def refund() -> str:
    ran.append("refund")
    return "refunded"


def _agent(model: _Model, **kwargs) -> Agent:
    agent = Agent(role="r", model={"model": "openai/gpt-4o-mini", "api_key": "k"}, **kwargs)
    agent._provider = model
    return agent


async def _run(agent: Agent, metadata: dict) -> list[AgentOutput]:
    return [o async for o in agent.run(AgentInput(message="hi", metadata=metadata))]


@pytest.fixture(autouse=True)
def _reset_ran(monkeypatch):
    ran.clear()
    # Not hosted unless a test says so.
    monkeypatch.delenv(HOSTED_ENV, raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata, expected",
    [
        (_meta(role="customer", private=True), {"lookup_order"}),
        (_meta(role="operator", private=True), {"lookup_order", "refund"}),
        (_meta(role="operator", private=False), {"lookup_order"}),
        (_meta(role="customer", legacy_tools=True), {"lookup_order", "refund"}),
        ({CALLER_METADATA_KEY: "garbage"}, {"lookup_order"}),
        ({}, {"lookup_order", "refund"}),
    ],
)
async def test_model_is_offered_only_the_tools_the_caller_may_use(metadata, expected):
    model = _Model(_DONE)
    await _run(_agent(model, tools=[lookup_order, refund]), metadata)
    assert set(model.offered) == expected


@pytest.mark.asyncio
async def test_hidden_tool_is_refused_as_not_found_and_not_run():
    model = _Model([AgentOutput(tool_calls=[ToolCall(id="1", name="refund", arguments={})])], _DONE)
    outputs = await _run(_agent(model, tools=[lookup_order, refund]), _meta(role="customer", private=True))

    results = [o.metadata["tool_event"] for o in outputs if "tool_event" in o.metadata]
    result = next(e for e in results if e["type"] == "tool_result")
    assert result["ok"] is False
    assert result["output"] == "Error: tool 'refund' not found"
    assert ran == []


@pytest.mark.asyncio
async def test_offered_tool_still_runs_for_a_customer():
    model = _Model([AgentOutput(tool_calls=[ToolCall(id="1", name="lookup_order", arguments={})])], _DONE)
    await _run(_agent(model, tools=[lookup_order, refund]), _meta(role="customer"))
    assert ran == ["lookup_order"]


@pytest.mark.asyncio
async def test_mcp_tools_follow_customer_safe_tools(monkeypatch):
    config = {"type": "sse", "url": "http://mcp.test", "customer_safe_tools": ["a"]}

    async def fetch(cfg):
        return {n: {"description": "", "parameters": {}, "config": cfg} for n in ("a", "b")}

    monkeypatch.setattr(agent_module, "_fetch_mcp_tools", fetch)

    model = _Model(_DONE)
    await _run(_agent(model, mcps=[config]), _meta(role="customer", private=True))
    assert model.offered == ["a"]

    model = _Model(_DONE)
    await _run(_agent(model, mcps=[config]), _meta(role="operator", private=True))
    assert sorted(model.offered) == ["a", "b"]


# ---------------------------------------------------------------------------
# AgentManifest.audiences
# ---------------------------------------------------------------------------

def _manifest(tmp_path, audiences: str | None):
    line = "" if audiences is None else f"audiences = {audiences}\n"
    (tmp_path / "genfleet.toml").write_text(f'[agent]\nname = "x"\n{line}')
    return load_agent_manifest(tmp_path)


def test_audiences_absent_is_none(tmp_path):
    assert _manifest(tmp_path, None).audiences is None


def test_audiences_are_deduplicated_in_order(tmp_path):
    manifest = _manifest(tmp_path, '["customer", "operator", "customer"]')
    assert manifest.audiences == ["customer", "operator"]


@pytest.mark.parametrize("value", ["[]", '["admin"]'])
def test_empty_or_unknown_audiences_are_rejected(tmp_path, value):
    with pytest.raises(ManifestError, match="audiences"):
        _manifest(tmp_path, value)


@pytest.mark.asyncio
async def test_a_hosted_turn_without_a_caller_gets_the_narrowest_set(monkeypatch):
    """In a platform sandbox the engine always sets the key; its absence is not trust."""
    monkeypatch.setenv(HOSTED_ENV, "spawn-token")
    assert caller_of({}) == Caller(role="customer")
    model = _Model(_DONE)
    await _run(_agent(model, tools=[lookup_order, refund]), {})
    assert set(model.offered) == {"lookup_order"}


@pytest.mark.asyncio
async def test_an_mcp_tool_cannot_shadow_a_local_tool(monkeypatch):
    """A server-chosen name must not lend its customer-safe mark to an operator-only local tool."""
    config = {"type": "sse", "url": "http://mcp.test", "customer_safe_tools": ["refund"]}

    async def fetch(cfg):
        return {"refund": {"description": "", "parameters": {}, "config": cfg}}

    monkeypatch.setattr(agent_module, "_fetch_mcp_tools", fetch)
    model = _Model([AgentOutput(tool_calls=[ToolCall(id="1", name="refund", arguments={})])], _DONE)
    outputs = await _run(_agent(model, tools=[lookup_order, refund], mcps=[config]), _meta(role="customer", private=True))

    assert "refund" not in model.offered
    result = next(o.metadata["tool_event"] for o in outputs if o.metadata.get("tool_event", {}).get("type") == "tool_result")
    assert result["output"] == "Error: tool 'refund' not found"
    assert ran == []
