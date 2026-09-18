"""Tests for the Agent class — Tier 1 (no network, no LLM)."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genfleet.sdk import Agent, AgentInput, AgentOutput, AgentProtocol, ToolCall, tool
from genfleet.sdk.agent import _wrap_tool
from genfleet.sdk.schemas import ToolSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(**kwargs) -> Agent:
    defaults = dict(role="You are helpful.", model={"model": "openai/gpt-4o-mini", "api_key": "sk-test"})
    defaults.update(kwargs)
    return Agent(**defaults)


def _mock_provider(chunks: list[AgentOutput]):
    """Return a mock provider whose complete() yields the given chunks."""
    async def _complete(messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        for chunk in chunks:
            yield chunk

    provider = MagicMock()
    provider.complete = _complete
    return provider


# ---------------------------------------------------------------------------
# Construction & protocol
# ---------------------------------------------------------------------------

def test_agent_satisfies_protocol():
    agent = _make_agent()
    assert isinstance(agent, AgentProtocol)


def test_agent_run_returns_async_gen():
    import inspect
    agent = _make_agent()
    result = agent.run(AgentInput(message="hi"))
    assert inspect.isasyncgen(result)


def test_agent_stores_role():
    agent = _make_agent(role="custom role")
    assert agent._role == "custom role"


def test_agent_context_optional():
    agent = _make_agent()
    assert agent._context is None

    agent_with_ctx = _make_agent(context="extra context")
    assert agent_with_ctx._context == "extra context"


def test_agent_data_parameter_is_gone():
    # Reserved for RAG since 0.1 and never implemented; ADR-0018 folds RAG
    # into memory. A silent no-op parameter is worse than a TypeError.
    with pytest.raises(TypeError):
        _make_agent(data={"type": "lightrag", "storage_dir": "/tmp"})


def test_agent_without_memory_stays_stateless_outside_a_sandbox(monkeypatch):
    monkeypatch.delenv("GENFLEET_MEMORY_URL", raising=False)
    assert _make_agent().memory is None


def test_agent_without_memory_gets_the_platform_store_inside_a_sandbox(monkeypatch):
    from genfleet.sdk.memory import PlatformMemory

    monkeypatch.setenv("GENFLEET_MEMORY_URL", "http://127.0.0.1:1/memory")
    monkeypatch.setenv("GENFLEET_MEMORY_TOKEN", "spawn-token")
    assert isinstance(_make_agent().memory, PlatformMemory)
    assert isinstance(_make_agent(memory="platform").memory, PlatformMemory)


def test_platform_memory_outside_a_sandbox_names_the_missing_variable(monkeypatch):
    monkeypatch.delenv("GENFLEET_MEMORY_URL", raising=False)
    with pytest.raises(RuntimeError, match="GENFLEET_MEMORY_URL"):
        _make_agent(memory="platform")


# ---------------------------------------------------------------------------
# Tool wrapping
# ---------------------------------------------------------------------------

def test_wrap_tool_plain_function():
    def add(a: float, b: float) -> str:
        """Adds two numbers."""
        return str(a + b)

    wrapper = _wrap_tool(add)
    schema = wrapper.schema()
    assert schema.name == "add"
    assert schema.description == "Adds two numbers."
    assert "a" in schema.parameters["required"]
    assert "b" in schema.parameters["required"]


def test_wrap_tool_already_wrapped():
    @tool(description="multiplies two numbers")
    def multiply(a: float, b: float) -> str:
        return str(a * b)

    wrapper = _wrap_tool(multiply)
    assert wrapper.schema().description == "multiplies two numbers"


def test_agent_registers_tools():
    def fn_a(x: str) -> str:
        return x

    def fn_b(y: int) -> str:
        return str(y)

    agent = _make_agent(tools=[fn_a, fn_b])
    assert "fn_a" in agent._local_tools
    assert "fn_b" in agent._local_tools


def test_agent_tool_schemas_correct():
    def search(query: str, limit: int = 10) -> str:
        """Searches the web."""
        return ""

    agent = _make_agent(tools=[search])
    schema = agent._local_tools["search"].schema()
    assert schema.name == "search"
    assert "query" in schema.parameters["required"]
    assert "limit" not in schema.parameters.get("required", [])


# ---------------------------------------------------------------------------
# run() — text response (no tool calls)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_run_text_response():
    agent = _make_agent()
    agent._provider = _mock_provider([
        AgentOutput(content="Hello", done=False),
        AgentOutput(content=" world", done=False),
        AgentOutput(content="", done=True),
    ])

    chunks = [c async for c in agent.run(AgentInput(message="hi"))]
    content = "".join(c.content or "" for c in chunks)
    assert "Hello world" in content


@pytest.mark.asyncio
async def test_agent_run_yields_done_chunk():
    agent = _make_agent()
    agent._provider = _mock_provider([
        AgentOutput(content="Answer", done=False),
        AgentOutput(content="", done=True),
    ])

    chunks = [c async for c in agent.run(AgentInput(message="hi"))]
    assert any(c.done for c in chunks)


# ---------------------------------------------------------------------------
# run() — tool dispatch loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_dispatches_tool_and_continues():
    called_with = {}

    def multiply(a: float, b: float) -> str:
        called_with["a"] = a
        called_with["b"] = b
        return str(a * b)

    agent = _make_agent(tools=[multiply])

    call_count = 0

    async def _complete(messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: request a tool call
            yield AgentOutput(
                tool_calls=[ToolCall(id="tc1", name="multiply", arguments={"a": 3, "b": 4})],
                done=False,
            )
        else:
            # Second call: return text after tool result is fed back
            yield AgentOutput(content="The answer is 12", done=False)
            yield AgentOutput(content="", done=True)

    agent._provider = MagicMock()
    agent._provider.complete = _complete

    chunks = [c async for c in agent.run(AgentInput(message="What is 3*4?"))]
    content = "".join(c.content or "" for c in chunks)

    assert called_with == {"a": 3, "b": 4}
    assert "12" in content
    assert call_count == 2


@pytest.mark.asyncio
async def test_agent_tool_not_found_returns_error_string():
    agent = _make_agent()

    call_count = 0

    async def _complete(messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield AgentOutput(
                tool_calls=[ToolCall(id="tc1", name="nonexistent", arguments={})],
                done=False,
            )
        else:
            yield AgentOutput(content="OK", done=True)

    agent._provider = MagicMock()
    agent._provider.complete = _complete

    chunks = [c async for c in agent.run(AgentInput(message="test"))]
    # Should complete without raising even for unknown tools
    assert call_count == 2


# ---------------------------------------------------------------------------
# run() — system prompt construction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_system_prompt_includes_role():
    agent = _make_agent(role="You are a pirate.")
    captured_messages = []

    async def _complete(messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        captured_messages.extend(messages)
        yield AgentOutput(content="Arrr", done=True)

    agent._provider = MagicMock()
    agent._provider.complete = _complete

    async for _ in agent.run(AgentInput(message="hello")):
        pass
    system = next(m for m in captured_messages if m["role"] == "system")
    assert "You are a pirate." in system["content"]


@pytest.mark.asyncio
async def test_agent_system_prompt_includes_context():
    agent = _make_agent(context="Always answer in French.")
    captured_messages = []

    async def _complete(messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        captured_messages.extend(messages)
        yield AgentOutput(content="Bonjour", done=True)

    agent._provider = MagicMock()
    agent._provider.complete = _complete

    async for _ in agent.run(AgentInput(message="hi")):
        pass

    system = next(m for m in captured_messages if m["role"] == "system")
    assert "Always answer in French." in system["content"]


# ---------------------------------------------------------------------------
# memory integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_loads_and_saves_memory():
    from genfleet.sdk.schemas import Message

    stored: dict[str, list] = {}

    class FakeMemory:
        async def load(self, session_id):
            return stored.get(session_id, [])

        async def save(self, session_id, history):
            stored[session_id] = history

        async def clear(self, session_id):
            stored.pop(session_id, None)

    agent = _make_agent()
    agent._memory = FakeMemory()
    agent._provider = _mock_provider([AgentOutput(content="Hi!", done=True)])

    async for _ in agent.run(AgentInput(message="hello", metadata={"session_id": "sess-1"})):
        pass

    # History should have been persisted
    assert "sess-1" in stored
    messages = stored["sess-1"]
    assert any(m.role == "user" for m in messages)


@pytest.mark.asyncio
async def test_agent_appends_one_turn_when_the_backend_can():
    """An appendable backend gets only this turn — never the thread it already holds."""
    from genfleet.sdk.schemas import Message

    class FakeAppendable:
        def __init__(self):
            self.loaded = [Message(role="user", content="earlier"), Message(role="assistant", content="ok")]
            self.appended = []
            self.saved = []

        async def load(self, session_id):
            return list(self.loaded)

        async def save(self, session_id, history):
            self.saved.append(history)

        async def clear(self, session_id):
            pass

        async def append(self, session_id, messages, *, subject=None, metadata=None, turn_id=None):
            self.appended.append((session_id, messages, subject, metadata, turn_id))

    mem = FakeAppendable()
    agent = _make_agent()
    agent._memory = mem
    agent._provider = _mock_provider([AgentOutput(content="Hi!", done=True)])

    async for _ in agent.run(AgentInput(
        message="hello",
        metadata={"session_id": "whatsapp:1", "subject": "cust-1", "channel": "whatsapp", "request_id": "r9"},
    )):
        pass

    assert mem.saved == []
    (session_id, messages, subject, metadata, turn_id), = mem.appended
    assert session_id == "whatsapp:1"
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "hello"
    assert subject == "cust-1"
    assert metadata == {"channel": "whatsapp"}  # request_id is per turn, not per session
    assert turn_id == "r9"  # the caller's id, so a retry of it is recognisable



@pytest.mark.asyncio
async def test_the_turn_is_persisted_even_when_the_caller_stops_at_done():
    """``serve`` breaks out of the ``async for`` on ``done``, which closes the
    generator. Anything written after that yield is never written at all."""

    class FakeAppendable:
        def __init__(self):
            self.appended = []

        async def load(self, session_id):
            return []

        async def save(self, session_id, history):  # pragma: no cover — appendable
            raise AssertionError("save must not be used on an appendable backend")

        async def clear(self, session_id):
            pass

        async def append(self, session_id, messages, *, subject=None, metadata=None, turn_id=None):
            self.appended.append(messages)

    mem = FakeAppendable()
    agent = _make_agent()
    agent._memory = mem
    agent._provider = _mock_provider([AgentOutput(content="Hi!", done=True)])

    async for output in agent.run(AgentInput(message="hello", metadata={"session_id": "s"})):
        if output.done:
            break

    assert [m.role for m in mem.appended[0]] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_a_turn_nobody_named_falls_back_to_the_invocation_id():
    from genfleet.sdk.agent import _turn_id

    assert _turn_id({"message_id": "m1"}, "inv") == "m1"
    assert _turn_id({"request_id": " "}, "inv") == "inv"


def test_a_contact_is_not_copied_into_the_session_metadata():
    # PII: the party is already named by the opaque ``subject``.
    from genfleet.sdk.agent import _session_metadata

    assert _session_metadata({"channel": "whatsapp", "contact": "+201234"}) == {"channel": "whatsapp"}


def test_a_redis_config_without_a_connection_names_the_missing_key():
    from genfleet.sdk.memory import memory_for

    with pytest.raises(ValueError, match="connection"):
        memory_for({"type": "redis"})


def test_a_token_without_a_url_runs_stateless_and_says_so(monkeypatch, caplog):
    monkeypatch.delenv("GENFLEET_MEMORY_URL", raising=False)
    monkeypatch.setenv("GENFLEET_MEMORY_TOKEN", "spawn-token")
    with caplog.at_level("WARNING", logger="genfleet.sdk.memory"):
        assert _make_agent().memory is None
    assert "GENFLEET_MEMORY_URL" in caplog.text
