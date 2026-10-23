"""`Agent(tools=...)` takes ToolProtocol objects as well as functions (#87). Tier 1."""

from __future__ import annotations

import pytest

from genfleet.sdk import Agent, AgentInput, ToolCall, ToolProtocol, ToolSchema, tool
from genfleet.sdk.caller import CALLER_METADATA_KEY, TOOL_AUDIENCES_METADATA_KEY
from genfleet.sdk.models import ScriptedModelClient, Step


class PeerTool:
    """A minimal ToolProtocol object, shaped like a peer agent the runtime hands in."""

    def __init__(self, name: str = "ask_billing", reply: str = "Invoice 7 is paid.") -> None:
        self._schema = ToolSchema(
            name=name,
            description="Ask the billing agent.",
            parameters={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
        )
        self._reply = reply
        self.calls: list[ToolCall] = []

    def schema(self) -> ToolSchema:
        return self._schema

    async def call(self, tool_call: ToolCall) -> str:
        self.calls.append(tool_call)
        return self._reply


@tool
def order_status(order_id: str) -> str:
    """Look up an order."""
    return f"{order_id}: shipped"


def _agent(tools, client=None) -> Agent:
    return Agent(role="Support", model={"model": "test"}, tools=tools, model_client=client or ScriptedModelClient([]))


async def _reply(agent: Agent, message: str, metadata: dict | None = None) -> str:
    chunks = [o async for o in agent.run(AgentInput(message=message, metadata=metadata or {}))]
    return "".join(o.content or "" for o in chunks)


def test_a_tool_protocol_object_is_held_as_it_is():
    peer = PeerTool()
    assert isinstance(peer, ToolProtocol)
    agent = _agent([peer])
    assert agent._local_tools["ask_billing"] is peer


async def test_the_model_can_call_a_tool_protocol_object():
    peer = PeerTool()
    client = ScriptedModelClient([
        Step(tool_calls=[{"name": "ask_billing", "arguments": {"message": "Is invoice 7 paid?"}}]),
        "Yes, it is paid.",
    ])
    assert await _reply(_agent([peer], client), "Is invoice 7 paid?") == "Yes, it is paid."
    assert client.calls[0].tool_names == ["ask_billing"]
    assert peer.calls[0].arguments == {"message": "Is invoice 7 paid?"}
    assert client.calls[1].tool_results[0]["content"] == "Invoice 7 is paid."


async def test_tool_protocol_objects_mix_with_tool_functions_and_plain_functions():
    def ping() -> str:
        """Ping."""
        return "pong"

    peer = PeerTool()
    client = ScriptedModelClient([
        Step(tool_calls=[
            {"name": "order_status", "arguments": {"order_id": "A-1"}},
            {"name": "ask_billing", "arguments": {"message": "A-1?"}},
            {"name": "ping"},
        ]),
        "Done.",
    ])
    await _reply(_agent([order_status, peer, ping], client), "Status?")
    assert sorted(client.calls[0].tool_names) == ["ask_billing", "order_status", "ping"]
    assert [r["content"] for r in client.calls[1].tool_results] == ["A-1: shipped", "Invoice 7 is paid.", "pong"]


async def test_a_factory_merging_remote_tools_builds_and_offers_them():
    def build_agent(remote_tools=()):
        return Agent(role="Support", model={"model": "test"}, tools=[order_status, *remote_tools],
                     model_client=client)

    client = ScriptedModelClient(["Hi."])
    agent = build_agent(remote_tools=[PeerTool("ask_billing"), PeerTool("ask_sales")])
    await _reply(agent, "Hello")
    assert sorted(client.calls[0].tool_names) == ["ask_billing", "ask_sales", "order_status"]


def test_a_later_tool_with_the_same_name_replaces_an_earlier_one_as_for_functions():
    first, second = PeerTool(reply="first"), PeerTool(reply="second")
    assert _agent([first, second])._local_tools["ask_billing"] is second

    @tool(name="ask_billing")
    def local() -> str:
        return ""

    assert _agent([local, first])._local_tools["ask_billing"] is first
    assert _agent([first, local])._local_tools["ask_billing"] is local


def test_a_reserved_name_is_refused_for_a_tool_protocol_object():
    with pytest.raises(ValueError, match="reserved"):
        _agent([PeerTool("skill:x")])


async def test_a_tool_protocol_object_is_operator_only_unless_the_platform_says_otherwise():
    customer = {CALLER_METADATA_KEY: {"role": "customer", "private": True}}
    client = ScriptedModelClient(["Hi.", "Hi.", "Hi."])
    agent = _agent([PeerTool()], client)
    await _reply(agent, "Hello", customer)
    await _reply(agent, "Hello", {CALLER_METADATA_KEY: {"role": "operator", "private": True}})
    await _reply(agent, "Hello", {**customer, TOOL_AUDIENCES_METADATA_KEY: {"ask_billing": ["customer"]}})
    assert [c.tool_names for c in client.calls] == [[], ["ask_billing"], ["ask_billing"]]


def test_a_tool_protocol_object_is_not_sensitive_by_default_and_the_platform_can_mark_it():
    agent = _agent([PeerTool()])
    assert agent._sensitive_tools({}) == set()
    assert agent._sensitive_tools({"ask_billing": True}) == {"ask_billing"}


class NotATool:
    pass


class SyncCall:
    def schema(self) -> ToolSchema:
        return ToolSchema(name="sync", description="", parameters={"type": "object", "properties": {}})

    def call(self, tool_call: ToolCall) -> str:
        return ""


@pytest.mark.parametrize("bad", [NotATool(), SyncCall(), 42, "search"])
def test_an_object_that_is_neither_a_function_nor_a_tool_is_a_clear_error(bad):
    with pytest.raises(TypeError, match="tools entries must be functions, @tool functions or ToolProtocol objects"):
        _agent([bad])


def test_a_tool_class_rather_than_an_instance_is_a_clear_error():
    with pytest.raises(TypeError, match="not the class PeerTool"):
        _agent([PeerTool])


def test_a_tool_protocol_object_whose_schema_is_not_a_tool_schema_is_a_clear_error():
    class BadSchema(PeerTool):
        def schema(self):
            return {"name": "x"}

    with pytest.raises(TypeError, match="must return a ToolSchema"):
        _agent([BadSchema()])
