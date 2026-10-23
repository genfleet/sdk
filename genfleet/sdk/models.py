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

import copy
import logging
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from .providers.base import Provider
from .schemas import AgentOutput, ModelConfig, TokenUsage, ToolCall, ToolSchema

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
        try:
            factory: ModelClientFactory = ep.load()
            client = factory(config)
        except Exception as exc:
            # Fail closed, but say which installed plugin failed.
            raise RuntimeError(f"model client {ep.name!r} ({ep.value}) failed: {exc}") from exc
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


@dataclass(frozen=True)
class Step:
    """One scripted model answer: text, tool calls, or both.

    A tool call is a `ToolCall`, or a mapping with ``name``, optional
    ``arguments`` and optional ``id``. A call with no id (or ``id=""``)
    gets ``call_<n>``, numbered across the client's script.
    """

    text: str = ""
    tool_calls: Sequence[ToolCall | Mapping[str, Any]] = ()
    #: Reported with the answer, as a provider does; the `Agent` adds it up.
    token_usage: TokenUsage | None = None


@dataclass(frozen=True)
class RecordedCall:
    """What a `ScriptedModelClient` was given on one call."""

    #: The whole conversation sent to the model, copied at call time.
    messages: list[dict]
    #: The tool schemas offered to the model.
    tools: list[ToolSchema]
    #: The tool results that came back since the model's last answer, in
    #: order: ``{"tool_call_id", "name", "content"}``.
    tool_results: list[dict] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        """The names of the tools offered on this call."""
        return [t.name for t in self.tools]


class ScriptedModelClient:
    """An in-memory client for tests that answers each call with the next `Step`.

    A plain string is a text-only step. A step with tool calls drives the
    `Agent`'s tool loop: the agent runs the calls, sends their results back,
    and the next step answers. Every call is recorded in `calls`. A call
    after the last step raises `RuntimeError`.
    """

    def __init__(self, steps: Iterable[Step | str]) -> None:
        self._steps = [Step(text=s) if isinstance(s, str) else s for s in steps]
        for step in self._steps:
            if not isinstance(step, Step):
                raise TypeError(f"a script step is a str or a Step, got {type(step).__name__}")
        self._next_id = 0
        self.calls: list[RecordedCall] = []

    async def complete(
        self, messages: list[dict], tools: list[ToolSchema], stream: bool = True
    ) -> AsyncIterator[AgentOutput]:
        self.calls.append(RecordedCall(
            messages=copy.deepcopy(messages), tools=list(tools), tool_results=_trailing_tool_results(messages)
        ))
        if len(self.calls) > len(self._steps):
            raise RuntimeError(
                f"ScriptedModelClient: the model was called {len(self.calls)} times but the script has "
                f"{len(self._steps)} step(s). Last message: {_describe(messages[-1]) if messages else 'none'}. "
                "Add a step for this call."
            )
        step = self._steps[len(self.calls) - 1]
        # The shape a streaming provider sends: text first, then either the
        # complete tool calls (the turn continues) or the end of the answer.
        if step.text:
            yield AgentOutput(content=step.text, done=False)
        if step.tool_calls:
            calls = [self._tool_call(tc) for tc in step.tool_calls]
            yield AgentOutput(tool_calls=calls, done=False, token_usage=step.token_usage)
        else:
            yield AgentOutput(content="", done=True, token_usage=step.token_usage)

    def _tool_call(self, tc: ToolCall | Mapping[str, Any]) -> ToolCall:
        call = tc if isinstance(tc, ToolCall) else ToolCall(
            id=tc.get("id") or "", name=tc["name"], arguments=dict(tc.get("arguments") or {})
        )
        if call.id:
            return call
        self._next_id += 1
        return call.model_copy(update={"id": f"call_{self._next_id}"})


def _trailing_tool_results(messages: list[dict]) -> list[dict]:
    """The tool messages at the end of `messages`, each with the name of the call it answers."""
    names = {
        tc["id"]: tc["function"]["name"]
        for m in messages if m.get("role") == "assistant"
        for tc in m.get("tool_calls") or []
    }
    results: list[dict] = []
    for m in reversed(messages):
        if m.get("role") != "tool":
            break
        results.append({
            "tool_call_id": m["tool_call_id"], "name": names.get(m["tool_call_id"], ""), "content": m["content"]
        })
    return results[::-1]


def _describe(message: dict) -> str:
    content = message.get("content")
    text = content if isinstance(content, str) else repr(content)
    if len(text) > 200:
        text = text[:200] + "..."
    return f"{message.get('role')}: {text!r}"


__all__ = [
    "ENTRY_POINT_GROUP",
    "FakeModelClient",
    "ModelClient",
    "ModelClientFactory",
    "RecordedCall",
    "ScriptedModelClient",
    "Step",
    "discover_model_client",
]
