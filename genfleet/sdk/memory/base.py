from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..schemas import Message


@runtime_checkable
class Memory(Protocol):
    """Episodic memory: one thread of messages per ``session_id``.

    ``load`` / ``save`` / ``clear`` are the whole-thread contract every
    backend has. A backend that can add to a thread without rewriting it
    also implements :class:`AppendableMemory`; the agent prefers that path
    when it exists, because a chat grows by one turn at a time and a
    thread that is re-sent whole on every turn does not survive a long one.
    """

    async def load(self, session_id: str) -> list[Message]: ...
    async def save(self, session_id: str, history: list[Message]) -> None: ...
    async def clear(self, session_id: str) -> None: ...


@runtime_checkable
class AppendableMemory(Protocol):
    async def append(
        self,
        session_id: str,
        messages: list[Message],
        *,
        subject: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
