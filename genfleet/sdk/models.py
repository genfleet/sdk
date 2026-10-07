"""The pluggable model client.

An `Agent` sends its model calls through a `ModelClient`. Which one:

1. the `model_client=` passed to `Agent` (tests, local overrides);
2. else the first installed client that is active here: packages register a
   factory under the entry-point group ``genfleet.model_clients``; each is
   asked in name order and returns a client, or ``None`` when it does not
   apply in this environment;
3. else the direct provider named by a ``provider/model`` id (the
   ``openai``/``anthropic``/``gemini`` extras, or OpenRouter).

A factory gets the agent's whole `ModelConfig`, including ``options``, which
the SDK passes through without reading.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Iterable
from importlib.metadata import EntryPoint, entry_points

from .providers.base import Provider
from .schemas import AgentOutput, ModelConfig, ToolSchema

log = logging.getLogger("genfleet.sdk.models")

ENTRY_POINT_GROUP = "genfleet.model_clients"

#: What an `Agent` calls to reach a model. The direct providers implement it.
ModelClient = Provider

ModelClientFactory = Callable[[ModelConfig], "ModelClient | None"]


def _entry_points() -> Iterable[EntryPoint]:
    return entry_points(group=ENTRY_POINT_GROUP)


def discover_model_client(config: ModelConfig) -> tuple[str, ModelClient] | None:
    """The first installed client, in entry-point name order, that is active for `config`."""
    for ep in sorted(_entry_points(), key=lambda e: e.name):
        factory: ModelClientFactory = ep.load()
        client = factory(config)
        if client is not None:
            return ep.name, client
    return None


class FakeModelClient:
    """An in-memory client for tests: answers each call with the next reply, as one text chunk."""

    def __init__(self, replies: Iterable[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    async def complete(
        self, messages: list[dict], tools: list[ToolSchema], stream: bool = True
    ) -> AsyncIterator[AgentOutput]:
        self.calls.append(list(messages))
        reply = self._replies.pop(0) if self._replies else ""
        yield AgentOutput(content=reply, done=True)


__all__ = ["ENTRY_POINT_GROUP", "FakeModelClient", "ModelClient", "ModelClientFactory", "discover_model_client"]
