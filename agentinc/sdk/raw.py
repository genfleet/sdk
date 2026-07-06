from __future__ import annotations

import asyncio
import inspect
import logging
import warnings
from typing import Any, AsyncIterator

from .schemas import AgentInput, AgentOutput, ToolCall

log = logging.getLogger("agentinc.sdk.raw")


def _is_str_param(p: inspect.Parameter) -> bool:
    ann = p.annotation
    return ann is str or ann == "str" or (ann is inspect.Parameter.empty and p.name == "message")


def _history_to_dicts(history: list) -> list:
    out = []
    for msg in history:
        d: dict = {"role": msg.role, "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in msg.tool_calls
            ]
        if getattr(msg, "tool_call_id", None):
            d["tool_call_id"] = msg.tool_call_id
        out.append(d)
    return out


def _schemas_to_dicts(schemas: list) -> list:
    return [
        {"name": s.name, "description": s.description, "parameters": s.parameters}
        for s in schemas
    ]


def _resolve_arg(fn: Any, input: AgentInput) -> Any:
    try:
        params = list(inspect.signature(fn).parameters.values())
        if params and _is_str_param(params[0]):
            if len(params) >= 3 and params[1].name == "history" and params[2].name == "tools":
                return fn(
                    input.message,
                    _history_to_dicts(input.history or []),
                    _schemas_to_dicts(input.tool_schemas or []),
                )
            if len(params) >= 2 and params[1].name == "history":
                return fn(input.message, _history_to_dicts(input.history or []))
            return fn(input.message)
    except (ValueError, TypeError):
        pass
    return fn(input)


def _chunk_to_output(chunk: Any) -> AgentOutput:
    tool_calls = [
        ToolCall(id=tc.get("id", ""), name=tc["name"], arguments=tc.get("arguments", {}))
        for tc in chunk.get("tool_calls", [])
        if tc.get("name")
    ]
    return AgentOutput(
        content=chunk.get("content", ""),
        tool_calls=tool_calls,
        done=chunk.get("done", False),
    )


class RawAdapter:
    """
    .. deprecated::
        RawAdapter is deprecated. Use :class:`agentinc.sdk.Agent` instead.
        RawAdapter will be removed in v0.4.
    """
    """
    Wraps any callable as an AgentProtocol.

    Supported signatures (no agentinc imports required in the wrapped fn):
    - async def fn(message: str) -> str
    - async def fn(message: str) -> AsyncIterator[str]
    - async def fn(message: str, history: list) -> AsyncIterator[str]
    - async def fn(message: str, history: list, tools: list) -> AsyncIterator[str | dict]
    - async def fn(input: AgentInput) -> AgentOutput
    - async def fn(input: AgentInput) -> AsyncIterator[AgentOutput]
    """

    def __init__(self, fn: Any) -> None:
        warnings.warn(
            "RawAdapter is deprecated and will be removed in v0.4. "
            "Use Agent() instead: from agentinc.sdk import Agent",
            DeprecationWarning,
            stacklevel=2,
        )
        self._fn = fn

    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]:
        fn_name = getattr(self._fn, "__name__", repr(self._fn))
        log.info("running %s", fn_name)

        result = _resolve_arg(self._fn, input)

        if asyncio.iscoroutine(result):
            result = await result

        if isinstance(result, str):
            yield AgentOutput(content=result, done=True)
        elif isinstance(result, AgentOutput):
            yield result
        elif inspect.isasyncgen(result):
            last_was_str = False
            content_buf: list[str] = []
            async for chunk in result:
                if isinstance(chunk, str):
                    last_was_str = True
                    content_buf.append(chunk)
                    yield AgentOutput(content=chunk, done=False)
                elif isinstance(chunk, dict):
                    last_was_str = False
                    out = _chunk_to_output(chunk)
                    yield out
                else:
                    last_was_str = False
                    yield chunk
            if last_was_str:
                yield AgentOutput(content="", done=True)
        else:
            yield AgentOutput(content=str(result), done=True)
