from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schemas import Message


@runtime_checkable
class Memory(Protocol):
    async def load(self, session_id: str) -> list[Message]: ...
    async def save(self, session_id: str, history: list[Message]) -> None: ...
    async def clear(self, session_id: str) -> None: ...
