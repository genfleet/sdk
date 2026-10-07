"""The running turn, for code the turn calls into (a model client, a tool).

`Agent.run` makes the turn's `AgentInput.metadata` visible only while its
model client or a tool is running: it is set around each step and reset before
control returns to the turn's consumer, so it never leaks to the caller, or
from one turn to another interleaved with it. The SDK gives the metadata's
keys no meaning; a platform documents the ones it sets.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Mapping
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any, TypeVar

T = TypeVar("T")

_EMPTY: Mapping[str, Any] = MappingProxyType({})
_current: ContextVar[Mapping[str, Any]] = ContextVar("genfleet_sdk_current_turn", default=_EMPTY)


def current_turn() -> Mapping[str, Any]:
    """The running turn's `AgentInput.metadata` (read-only), or an empty mapping outside one."""
    return _current.get()


async def within(metadata: Mapping[str, Any], stream: AsyncIterator[T]) -> AsyncIterator[T]:
    """Iterate `stream` with the turn visible during each step, and only then."""
    frozen = MappingProxyType(dict(metadata))
    iterator = stream.__aiter__()
    try:
        while True:
            token = _current.set(frozen)
            try:
                item = await iterator.__anext__()
            except StopAsyncIteration:
                return
            finally:
                _current.reset(token)
            yield item
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            token = _current.set(frozen)
            try:
                await aclose()
            finally:
                _current.reset(token)


async def call_within(metadata: Mapping[str, Any], call: Awaitable[T]) -> T:
    """Await `call` with the turn visible while it runs."""
    token = _current.set(MappingProxyType(dict(metadata)))
    try:
        return await call
    finally:
        _current.reset(token)
