"""Tests for the Agent class — Tier 1 (no network, no LLM)."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fleet.sdk import Agent, AgentInput, AgentOutput, AgentProtocol, ToolCall, tool
from fleet.sdk.agent import _wrap_tool
from fleet.sdk.schemas import ToolSchema


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


def test_agent_data_ignored_no_error():
    agent = _make_agent(data={"type": "lightrag", "storage_dir": "/tmp"})
    assert agent is not None


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
    from fleet.sdk.schemas import Message

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
