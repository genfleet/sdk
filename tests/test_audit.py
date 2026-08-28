"""Tests for the audit system — Tier 1 (no network, no LLM)."""

from __future__ import annotations

import json
import os
import tempfile
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest

from fleet.sdk import Agent, AgentInput, AgentOutput, AuditConfig, AuditEvent, TokenUsage, ToolCall
from fleet.sdk.audit import Auditor, audit_for, _truncate
from fleet.sdk.audit.backends import (
    CallbackAuditBackend,
    ConsoleAuditBackend,
    FileAuditBackend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(audit: AuditConfig | None = None, **kwargs) -> Agent:
    defaults = dict(role="You are helpful.", model={"model": "openai/gpt-4o-mini", "api_key": "sk-test"})
    defaults.update(kwargs)
    return Agent(**defaults, audit=audit)


def _mock_provider(chunks: list[AgentOutput]):
    async def _complete(messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        for chunk in chunks:
            yield chunk

    provider = MagicMock()
    provider.complete = _complete
    return provider


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def test_truncate_short_text():
    assert _truncate("hello", 500) == "hello"


def test_truncate_long_text():
    text = "a" * 600
    result = _truncate(text, 500)
    assert len(result) == 503  # 500 + "..."
    assert result.endswith("...")


def test_truncate_unlimited():
    text = "a" * 1000
    assert _truncate(text, 0) == text


# ---------------------------------------------------------------------------
# audit_for factory
# ---------------------------------------------------------------------------

def test_audit_for_none():
    assert audit_for(None) is None


def test_audit_for_console():
    auditor = audit_for({"backend": "console"})
    assert isinstance(auditor, Auditor)


def test_audit_for_file():
    auditor = audit_for({"backend": "file", "file_path": "/tmp/test.jsonl"})
    assert isinstance(auditor, Auditor)


def test_audit_for_callback():
    auditor = audit_for({"backend": "callback", "callback": lambda e: None})
    assert isinstance(auditor, Auditor)


def test_audit_for_callback_missing_raises():
    with pytest.raises(ValueError, match="callback"):
        audit_for({"backend": "callback"})


def test_audit_for_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown"):
        audit_for({"backend": "postgres"})


# ---------------------------------------------------------------------------
# ConsoleAuditBackend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_console_backend_emits(caplog):
    import logging
    logging.getLogger("fleet.audit").setLevel(logging.INFO)

    backend = ConsoleAuditBackend()
    event = AuditEvent(event_type="test.event", data={"key": "value"})

    with caplog.at_level(logging.INFO, logger="fleet.audit"):
        await backend.emit(event)

    assert "test.event" in caplog.text


# ---------------------------------------------------------------------------
# FileAuditBackend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_backend_writes_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        backend = FileAuditBackend(path)
        event = AuditEvent(event_type="test.file", data={"x": 1})
        await backend.emit(event)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event_type"] == "test.file"
        assert parsed["data"]["x"] == 1
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_file_backend_appends():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        backend = FileAuditBackend(path)
        await backend.emit(AuditEvent(event_type="event1"))
        await backend.emit(AuditEvent(event_type="event2"))

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# CallbackAuditBackend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_callback_backend_sync():
    received = []
    backend = CallbackAuditBackend(lambda e: received.append(e))
    event = AuditEvent(event_type="test.cb")
    await backend.emit(event)
    assert len(received) == 1
    assert received[0].event_type == "test.cb"


@pytest.mark.asyncio
async def test_callback_backend_async():
    received = []

    async def handler(e):
        received.append(e)

    backend = CallbackAuditBackend(handler)
    event = AuditEvent(event_type="test.async_cb")
    await backend.emit(event)
    assert len(received) == 1


# ---------------------------------------------------------------------------
# Auditor — event filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auditor_filters_events():
    received = []
    config: AuditConfig = {
        "backend": "callback",
        "callback": lambda e: received.append(e),
        "events": ["invocation.start", "invocation.end"],
    }
    auditor = audit_for(config)
    assert auditor is not None

    await auditor.emit("invocation.start", data={"msg": "hi"})
    await auditor.emit("llm.request", data={"model": "gpt-4o"})
    await auditor.emit("invocation.end", data={})

    assert len(received) == 2
    assert received[0].event_type == "invocation.start"
    assert received[1].event_type == "invocation.end"


@pytest.mark.asyncio
async def test_auditor_no_filter_emits_all():
    received = []
    config: AuditConfig = {
        "backend": "callback",
        "callback": lambda e: received.append(e),
    }
    auditor = audit_for(config)
    assert auditor is not None

    await auditor.emit("invocation.start")
    await auditor.emit("llm.request")
    await auditor.emit("tool.call")

    assert len(received) == 3


# ---------------------------------------------------------------------------
# Auditor — truncation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auditor_truncates_content():
    config: AuditConfig = {
        "backend": "callback",
        "callback": lambda e: None,
        "max_content_length": 10,
    }
    auditor = audit_for(config)
    assert auditor is not None
    assert auditor.truncate("short") == "short"
    assert auditor.truncate("a" * 20) == "a" * 10 + "..."


# ---------------------------------------------------------------------------
# Agent + audit integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_emits_audit_events_on_text_response():
    received: list[AuditEvent] = []

    agent = _make_agent(audit={
        "backend": "callback",
        "callback": lambda e: received.append(e),
        "agent_name": "test-agent",
    })
    agent._provider = _mock_provider([
        AgentOutput(content="Hello", done=False),
        AgentOutput(content="", done=True),
    ])

    async for _ in agent.run(AgentInput(message="hi")):
        pass

    types = [e.event_type for e in received]
    assert "invocation.start" in types
    assert "llm.request" in types
    assert "llm.response" in types
    assert "invocation.end" in types

    for e in received:
        assert e.agent_name == "test-agent"


@pytest.mark.asyncio
async def test_agent_emits_tool_audit_events():
    received: list[AuditEvent] = []

    def multiply(a: float, b: float) -> str:
        return str(a * b)

    agent = _make_agent(
        tools=[multiply],
        audit={"backend": "callback", "callback": lambda e: received.append(e)},
    )

    call_count = 0

    async def _complete(messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield AgentOutput(
                tool_calls=[ToolCall(id="tc1", name="multiply", arguments={"a": 3, "b": 4})],
                done=False,
            )
        else:
            yield AgentOutput(content="12", done=False)
            yield AgentOutput(content="", done=True)

    agent._provider = MagicMock()
    agent._provider.complete = _complete

    async for _ in agent.run(AgentInput(message="3*4?")):
        pass

    types = [e.event_type for e in received]
    assert "tool.call" in types
    assert "tool.result" in types

    tool_call_event = next(e for e in received if e.event_type == "tool.call")
    assert tool_call_event.data["tool_name"] == "multiply"

    tool_result_event = next(e for e in received if e.event_type == "tool.result")
    assert tool_result_event.data["tool_name"] == "multiply"
    assert tool_result_event.latency_ms is not None


@pytest.mark.asyncio
async def test_agent_emits_token_usage_in_audit():
    received: list[AuditEvent] = []

    agent = _make_agent(audit={
        "backend": "callback",
        "callback": lambda e: received.append(e),
    })

    usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)

    agent._provider = _mock_provider([
        AgentOutput(content="Hello", done=False),
        AgentOutput(content="", done=True, token_usage=usage),
    ])

    async for _ in agent.run(AgentInput(message="hi")):
        pass

    llm_response = next(e for e in received if e.event_type == "llm.response")
    assert llm_response.data["token_usage"]["input_tokens"] == 10
    assert llm_response.data["token_usage"]["output_tokens"] == 20

    invocation_end = next(e for e in received if e.event_type == "invocation.end")
    assert invocation_end.data["total_token_usage"]["input_tokens"] == 10
    assert invocation_end.data["total_token_usage"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_agent_audit_error_event():
    received: list[AuditEvent] = []

    agent = _make_agent(audit={
        "backend": "callback",
        "callback": lambda e: received.append(e),
    })

    async def _failing_complete(messages, tools, stream=True):
        raise RuntimeError("LLM exploded")
        yield  # noqa: unreachable — makes this an async generator

    agent._provider = MagicMock()
    agent._provider.complete = _failing_complete

    with pytest.raises(RuntimeError, match="LLM exploded"):
        async for _ in agent.run(AgentInput(message="hi")):
            pass

    types = [e.event_type for e in received]
    assert "invocation.error" in types
    error_event = next(e for e in received if e.event_type == "invocation.error")
    assert error_event.status == "error"
    assert error_event.data["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_agent_no_audit_no_overhead():
    agent = _make_agent(audit=None)
    assert agent._auditor is None

    agent._provider = _mock_provider([
        AgentOutput(content="Hello", done=True),
    ])

    chunks = [c async for c in agent.run(AgentInput(message="hi"))]
    assert any(c.done for c in chunks)


@pytest.mark.asyncio
async def test_agent_audit_latency_tracked():
    received: list[AuditEvent] = []

    agent = _make_agent(audit={
        "backend": "callback",
        "callback": lambda e: received.append(e),
    })
    agent._provider = _mock_provider([
        AgentOutput(content="ok", done=True),
    ])

    async for _ in agent.run(AgentInput(message="hi")):
        pass

    invocation_end = next(e for e in received if e.event_type == "invocation.end")
    assert invocation_end.latency_ms is not None
    assert invocation_end.latency_ms >= 0

    llm_response = next(e for e in received if e.event_type == "llm.response")
    assert llm_response.latency_ms is not None
