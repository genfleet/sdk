from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from ..schemas import AgentOutput, ToolSchema


@runtime_checkable
class Provider(Protocol):
    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolSchema],
        stream: bool = True,
    ) -> AsyncIterator[AgentOutput]: ...
