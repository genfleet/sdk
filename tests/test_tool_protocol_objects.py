"""`Agent(tools=...)` takes ToolProtocol objects as well as functions (#87). Tier 1."""

from __future__ import annotations

import pytest

from genfleet.sdk import (
    Agent,
    AgentInput,
    Skill,
    ToolCall,
    ToolProtocol,
    ToolSchema,
    tool,
)
from genfleet.sdk import agent as agent_module
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


def test_a_local_tool_keeps_its_name_whichever_order_a_peer_comes_in(caplog):
    @tool(name="ask_billing")
    def local() -> str:
        return ""

    peer = PeerTool()
    with caplog.at_level("WARNING", logger="genfleet.sdk.agent"):
        assert _agent([local, peer])._local_tools["ask_billing"] is local
        assert _agent([peer, local])._local_tools["ask_billing"] is local
    assert caplog.text.count("tool 'ask_billing' skipped: the name is already taken") == 2


def test_a_plain_function_also_keeps_its_name_over_a_peer_listed_first():
    def ask_billing(message: str) -> str:
        """Local billing."""
        return "local"

    agent = _agent([PeerTool(), ask_billing])
    assert agent._local_tools["ask_billing"].schema().description == "Local billing."


@pytest.mark.parametrize("order", ["local_first", "peer_first"])
def test_a_peer_cannot_shadow_a_sensitive_local_tool(order):
    @tool(name="ask_billing", sensitive=True)
    def refund() -> str:
        return ""

    tools = [refund, PeerTool()] if order == "local_first" else [PeerTool(), refund]
    agent = _agent(tools)
    assert agent._local_tools["ask_billing"] is refund
    assert agent._sensitive_tools({}) == {"ask_billing"}


def test_of_two_tool_protocol_objects_with_one_name_the_first_wins(caplog):
    first, second = PeerTool(reply="first"), PeerTool(reply="second")
    with caplog.at_level("WARNING", logger="genfleet.sdk.agent"):
        agent = _agent([first, second])
    assert agent._local_tools["ask_billing"] is first
    assert "tool 'ask_billing' skipped: the name is already taken" in caplog.text


def test_functions_with_one_name_still_replace_each_other_last_wins():
    @tool(name="dup")
    def a() -> str:
        return "a"

    @tool(name="dup")
    def b() -> str:
        return "b"

    assert _agent([a, b])._local_tools["dup"] is b


def test_a_tool_protocol_object_named_like_a_slug_key_is_skipped(caplog):
    with caplog.at_level("WARNING", logger="genfleet.sdk.agent"):
        agent = _agent([PeerTool("@acme/tickets")])
    assert agent._local_tools == {}
    assert "can't start with '@'" in caplog.text


def test_schema_is_read_once_per_tool_protocol_object():
    class Counting(PeerTool):
        reads = 0

        def schema(self) -> ToolSchema:
            Counting.reads += 1
            return super().schema()

    _agent([Counting()])
    assert Counting.reads == 1


def test_a_tool_protocol_object_with_a_reserved_name_is_skipped(caplog):
    with caplog.at_level("WARNING", logger="genfleet.sdk.agent"):
        agent = _agent([PeerTool("skill:x")])
    assert agent._local_tools == {}
    assert "tool 'skill:x' skipped" in caplog.text


def test_a_peer_named_read_skill_is_skipped_when_skills_are_configured(caplog):
    skill = Skill(name="returns", description="Returns.", instructions="Do it.")
    with caplog.at_level("WARNING", logger="genfleet.sdk.agent"):
        agent = Agent(role="r", model={"model": "test"}, model_client=ScriptedModelClient([]),
                      tools=[PeerTool("read_skill")], skills=[skill])
    assert agent._local_tools == {}
    assert "tool 'read_skill' skipped" in caplog.text


def test_a_function_with_a_reserved_name_still_raises():
    @tool(name="skill:x")
    def bad() -> str:
        return ""

    with pytest.raises(ValueError, match="reserved"):
        _agent([bad, PeerTool()])


def _mcp_serves(monkeypatch, *names):
    async def fetch(cfg):
        return {n: {"description": "", "parameters": {}, "config": cfg} for n in names}

    monkeypatch.setattr(agent_module, "_fetch_mcp_tools", fetch)


MCP_CONFIG = {"type": "sse", "url": "http://mcp.test", "sensitive_tools": ["ask_billing"]}


async def test_an_mcp_tool_keeps_its_name_over_a_peer_and_stays_sensitive(monkeypatch, caplog):
    _mcp_serves(monkeypatch, "ask_billing")
    agent = Agent(role="r", model={"model": "test"}, model_client=ScriptedModelClient([]),
                  tools=[PeerTool()], mcps=[MCP_CONFIG])
    with caplog.at_level("WARNING", logger="genfleet.sdk.agent"):
        await agent._init_mcp()
    assert "ask_billing" not in agent._local_tools
    assert "ask_billing" in agent._mcp_tools
    assert agent._sensitive_tools({}) == {"ask_billing"}
    assert "tool 'ask_billing' skipped: the name is taken by an MCP tool" in caplog.text


async def test_a_local_tool_still_keeps_its_name_over_an_mcp_tool(monkeypatch, caplog):
    @tool(name="ask_billing")
    def local() -> str:
        return ""

    _mcp_serves(monkeypatch, "ask_billing")
    agent = Agent(role="r", model={"model": "test"}, model_client=ScriptedModelClient([]),
                  tools=[local], mcps=[MCP_CONFIG])
    with caplog.at_level("WARNING", logger="genfleet.sdk.agent"):
        await agent._init_mcp()
    assert agent._local_tools["ask_billing"] is local
    assert agent._mcp_tools == {}
    assert "MCP tool 'ask_billing' skipped" in caplog.text


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
