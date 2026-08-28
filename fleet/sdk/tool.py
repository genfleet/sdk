from __future__ import annotations

import asyncio
import inspect
import typing
from collections.abc import Callable
from typing import Any, get_args, get_origin

from .schemas import ToolCall, ToolSchema


_SIMPLE: dict[Any, dict[str, str]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
}


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_json_schema(non_none[0])
        return {}

    if origin is list:
        result: dict[str, Any] = {"type": "array"}
        if args:
            result["items"] = _annotation_to_json_schema(args[0])
        return result

    return _SIMPLE.get(annotation, {})


def _build_schema(fn: Callable[..., Any], name: str, description: str) -> ToolSchema:
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        annotation = hints.get(param_name, inspect.Parameter.empty)
        prop = _annotation_to_json_schema(annotation)
        properties[param_name] = prop if prop else {}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required

    return ToolSchema(name=name, description=description, parameters=parameters)


class ToolWrapper:
    """Wraps any sync or async callable as a ToolProtocol."""

    def __init__(self, fn: Callable[..., Any], schema: ToolSchema) -> None:
        self._fn = fn
        self._schema = schema

    def schema(self) -> ToolSchema:
        return self._schema

    async def call(self, tool_call: ToolCall) -> str:
        result = self._fn(**tool_call.arguments)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)

    async def __call__(self, **kwargs: Any) -> str:
        result = self._fn(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)

    def __repr__(self) -> str:
        return f"ToolWrapper(name={self._schema.name!r})"


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str = "",
) -> ToolWrapper | Callable[[Callable[..., Any]], ToolWrapper]:
    """
    Decorator that turns any function into a ToolProtocol.

    Usage:
        @tool
        async def search(query: str) -> str: ...

        @tool(description="adds two numbers")
        async def add(a: float, b: float) -> str: ...
    """

    def _wrap(f: Callable[..., Any]) -> ToolWrapper:
        resolved_name = name or f.__name__
        resolved_desc = description or (inspect.getdoc(f) or "")
        schema = _build_schema(f, name=resolved_name, description=resolved_desc)
        return ToolWrapper(fn=f, schema=schema)

    if fn is not None:
        return _wrap(fn)

    return _wrap
