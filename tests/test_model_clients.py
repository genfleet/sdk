"""The pluggable model client (genfleet.model_clients): discovery, precedence, the per-turn context."""

from __future__ import annotations

import subprocess
import sys
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


def test_a_direct_provider_without_a_key_says_so(monkeypatch):
    _entry_points(monkeypatch, {})
    with pytest.raises(ValueError, match="api_key"):
        Agent(role="r", model={"model": "openai/gpt-4o-mini"})


async def test_the_running_turn_is_visible_to_the_client():
    class Peek(FakeModelClient):
        async def complete(self, messages, tools, stream=True):
            self.seen = dict(current_turn())
            async for out in super().complete(messages, tools, stream):
                yield out

    client = Peek(["ok"])
    agent = Agent(role="r", model={"model": "general-fast"}, model_client=client)
    await _collect(agent, session_id="s-1", channel="web")
    assert client.seen == {"session_id": "s-1", "channel": "web"}
    assert current_turn() == {}  # nothing leaks past the turn


def test_the_sdk_imports_no_platform_module():
    # The one-way dependency (ADR-0022 §5): the open SDK never imports platform code.
    code = (
        "import sys, genfleet.sdk, genfleet.sdk.models, genfleet.sdk.turn, genfleet.sdk.agent;"
        "bad=[m for m in sys.modules if m.startswith(('genfleet.gateway','genfleet.engine','genfleet.adapter','genfleet.backend'))];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == ""
