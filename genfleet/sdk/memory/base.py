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

    A backend that can read just the recent end of a thread also implements
    :class:`WindowedMemory`.
    """

    async def load(self, session_id: str) -> list[Message]: ...
    async def save(self, session_id: str, history: list[Message]) -> None: ...
    async def clear(self, session_id: str) -> None: ...


@runtime_checkable
class AppendableMemory(Protocol):
    """Add one turn to a thread without rewriting it.

    ``turn_id`` is an idempotency key for that turn: a backend that honours it
    keeps the first append and drops a replay, so a retried invocation is not
    stored twice. A backend free to ignore it is free to duplicate.
    """

    async def append(
        self,
        session_id: str,
        messages: list[Message],
        *,
        subject: str | None = None,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> None: ...


@runtime_checkable
class WindowedMemory(Protocol):
    """Read only the newest messages of a thread.

    ``load_newest`` returns the newest ``limit`` messages, oldest first. It
    may leave out leading tool results whose call fell outside the window,
    so it can return fewer. An agent with ``max_history_turns`` uses it on a
    backend that also implements :class:`AppendableMemory`, so a long thread
    is not read whole on every turn. A backend that only saves is always
    loaded whole: ``save`` rewrites the thread from what was loaded.
    """

    async def load_newest(self, session_id: str, limit: int) -> list[Message]: ...
