from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

from .schemas import AgentInput, AgentOutput, ToolCall, ToolSchema


@runtime_checkable
class ToolProtocol(Protocol):
    def schema(self) -> ToolSchema: ...
    async def call(self, tool_call: ToolCall) -> str: ...


@runtime_checkable
class AgentProtocol(Protocol):
    def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]: ...


AgentFactory = Any
