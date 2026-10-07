"""The running turn, for code the turn calls into (a model client, a tool).

`Agent.run` sets it around each turn. The SDK gives the metadata's keys no
meaning; a platform documents the ones it sets.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any

_EMPTY: Mapping[str, Any] = MappingProxyType({})
_current: ContextVar[Mapping[str, Any]] = ContextVar("genfleet_sdk_current_turn", default=_EMPTY)


def current_turn() -> Mapping[str, Any]:
    """The running turn's `AgentInput.metadata` (read-only), or an empty mapping outside a turn."""
    return _current.get()


def _enter(metadata: Mapping[str, Any]):
    return _current.set(MappingProxyType(dict(metadata)))


def _leave(token) -> None:
    try:
        _current.reset(token)
    except ValueError:
        # A generator closed from another context (e.g. by garbage collection):
        # that context never saw the turn, so there is nothing to undo.
        pass
