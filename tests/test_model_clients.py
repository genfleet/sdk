"""The pluggable model client (genfleet.model_clients): discovery, precedence, the per-turn context."""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from genfleet.sdk import Agent, AgentInput
from genfleet.sdk import models as models_mod
from genfleet.sdk.models import FakeModelClient, ModelClient, discover_model_client
from genfleet.sdk.providers.base import Provider
from genfleet.sdk.turn import current_turn


def _entry_points(monkeypatch, factories: dict):
    eps = [
        EntryPoint(name=name, value=f"tests.test_model_clients:{name}", group=models_mod.ENTRY_POINT_GROUP)
        for name in factories
    ]
    for name, factory in factories.items():
        globals()[name] = factory
    monkeypatch.setattr(models_mod, "_entry_points", lambda: eps)


async def _collect(agent: Agent, message: str = "hi", **metadata) -> str:
    out = []
    async for chunk in agent.run(AgentInput(message=message, metadata=metadata)):
        out.append(chunk.content or "")
    return "".join(out)


def test_provider_is_the_model_client_protocol():
    assert ModelClient is Provider
    assert isinstance(FakeModelClient(["x"]), ModelClient)


async def test_an_explicit_client_wins():
    agent = Agent(role="r", model={"model": "general-fast"}, model_client=FakeModelClient(["from fake"]))
    assert await _collect(agent) == "from fake"


async def test_a_registered_client_is_used_when_it_activates(monkeypatch):
    seen = {}

    def inactive(config):
        return None

    def platform(config):
        seen["config"] = config
        return FakeModelClient(["from plugin"])

    _entry_points(monkeypatch, {"b_platform": platform, "a_inactive": inactive})
    agent = Agent(role="r", model={"model": "general-fast", "options": {"priority": "low"}})
    assert await _collect(agent) == "from plugin"
    # Options are opaque to the SDK: handed to the client unchanged.
    assert seen["config"]["options"] == {"priority": "low"}


def test_registered_clients_are_tried_in_name_order(monkeypatch):
    _entry_points(monkeypatch, {"z_last": lambda c: FakeModelClient(["z"]), "a_first": lambda c: FakeModelClient(["a"])})
    name, _ = discover_model_client({"model": "m"})
    assert name == "a_first"


def test_without_a_plugin_a_provider_model_id_uses_the_direct_provider(monkeypatch):
    from genfleet.sdk import agent as agent_mod

    _entry_points(monkeypatch, {})
    direct = FakeModelClient(["direct"])
    monkeypatch.setattr(agent_mod, "provider_for", lambda config: direct)
    agent = Agent(role="r", model={"model": "openai/gpt-4o-mini", "api_key": "sk-test"})
    assert agent._provider is direct


def test_a_plain_model_id_without_a_plugin_says_what_is_missing(monkeypatch):
    _entry_points(monkeypatch, {})
    with pytest.raises(ValueError, match="no model client is installed"):
        Agent(role="r", model={"model": "general-fast"})


class Peek(FakeModelClient):
    """Records what `current_turn()` says while the client runs."""

    async def complete(self, messages, tools, stream=True):
        self.seen = dict(current_turn())
        async for out in super().complete(messages, tools, stream):
            self.seen_mid = dict(current_turn())
            yield out


async def test_the_running_turn_is_visible_to_the_client():
    client = Peek(["ok"])
    agent = Agent(role="r", model={"model": "general-fast"}, model_client=client)
    await _collect(agent, session_id="s-1", channel="web")
    assert client.seen == client.seen_mid == {"session_id": "s-1", "channel": "web"}
    assert current_turn() == {}


async def test_the_turn_never_leaks_to_the_caller_even_when_it_stops_early():
    agent = Agent(role="r", model={"model": "general-fast"}, model_client=Peek(["a"]))
    stream = agent.run(AgentInput(message="hi", metadata={"session_id": "s-1"}))
    async for _ in stream:
        assert current_turn() == {}  # between yields, the consumer sees nothing
        break
    assert current_turn() == {}
    await stream.aclose()
    assert current_turn() == {}


async def test_interleaved_turns_see_only_their_own_metadata():
    a = Peek(["a1", "a2"])
    b = Peek(["b1", "b2"])
    ta = Agent(role="r", model={"model": "m"}, model_client=a).run(AgentInput(message="x", metadata={"session_id": "A"}))
    tb = Agent(role="r", model={"model": "m"}, model_client=b).run(AgentInput(message="x", metadata={"session_id": "B"}))
    await ta.__anext__()
    await tb.__anext__()
    assert (a.seen["session_id"], b.seen["session_id"]) == ("A", "B")
    await ta.aclose()
    await tb.aclose()


async def test_concurrent_turns_see_only_their_own_metadata():
    import asyncio

    clients = [Peek(["x"]) for _ in range(3)]
    agents = [Agent(role="r", model={"model": "m"}, model_client=c) for c in clients]
    await asyncio.gather(*(_collect(a, session_id=f"s-{i}") for i, a in enumerate(agents)))
    assert [c.seen["session_id"] for c in clients] == ["s-0", "s-1", "s-2"]


async def test_a_tool_sees_the_turn_too():
    from genfleet.sdk.schemas import AgentOutput, ToolCall

    seen = {}

    def lookup() -> str:
        """Look."""
        seen.update(current_turn())
        return "found"

    class CallsTool(FakeModelClient):
        async def complete(self, messages, tools, stream=True):
            if not any(m.get("role") == "tool" for m in messages):
                yield AgentOutput(tool_calls=[ToolCall(id="c1", name="lookup", arguments={})])
            else:
                yield AgentOutput(content="done", done=True)

    agent = Agent(role="r", model={"model": "m"}, model_client=CallsTool([]), tools=[lookup])
    assert await _collect(agent, session_id="s-9") == "done"
    assert seen == {"session_id": "s-9"}


def test_a_failing_plugin_names_its_entry_point(monkeypatch):
    def broken(config):
        raise RuntimeError("boom")

    _entry_points(monkeypatch, {"broken_plugin": broken})
    with pytest.raises(RuntimeError, match="broken_plugin"):
        discover_model_client({"model": "m"})


def test_only_a_missing_api_key_is_refused(monkeypatch):
    _entry_points(monkeypatch, {})
    with pytest.raises(ValueError, match="api_key"):
        Agent(role="r", model={"model": "openai/llama3"})
    # "" passes through as before (a keyless local OpenAI-compatible server).
    from genfleet.sdk.providers import openai as openai_provider

    seen = {}
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", lambda **kw: seen.update(kw))
    Agent(role="r", model={"model": "openai/llama3", "api_key": "", "base_url": "http://localhost:11434/v1"})
    assert seen["api_key"] == ""


def test_the_sdk_imports_no_platform_module():
    # The one-way dependency (ADR-0022 §5): no module of the open SDK imports
    # platform code, at import time or lazily inside a function.
    import ast
    from pathlib import Path

    import genfleet.sdk

    platform = ("genfleet.gateway", "genfleet.gateway_client", "genfleet.engine", "genfleet.adapter",
                "genfleet.loader", "genfleet.runner", "genfleet.protocols", "genfleet.core", "genfleet.backend")
    offenders = []
    for path in Path(genfleet.sdk.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            offenders += [f"{path.name}: {n}" for n in names if n.startswith(platform)]
    assert offenders == []
