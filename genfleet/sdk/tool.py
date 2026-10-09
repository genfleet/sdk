from __future__ import annotations

import asyncio
import inspect
import types
import typing
from collections.abc import Callable, Iterable
from typing import Any, get_args, get_origin

from .caller import EVERYONE, OPERATOR_ONLY, ROLES, Audience, audiences_of
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

    # `X | None` is a `types.UnionType`, not `typing.Union`; without both, every
    # optional parameter written in the modern spelling reached the model untyped.
    if origin is typing.Union or origin is types.UnionType:
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

    def __init__(
        self,
        fn: Callable[..., Any],
        schema: ToolSchema,
        *,
        customer_safe: bool = False,
        audiences: Iterable[str] | None = None,
        slug: str | None = None,
    ) -> None:
        self._fn = fn
        self._schema = schema
        #: Who the author offers the tool to (ADR-0028): the default the
        #: platform's per-tool setting replaces. ``[operator]`` unless marked.
        self.audiences: frozenset[Audience] = _resolve_audiences(audiences, customer_safe)
        #: The manifest slug a mounted tool came from (``load_tools``), so the
        #: platform's setting can name it before its function name is known.
        self.slug = slug or getattr(fn, TOOL_SLUG_ATTR, None)

    @property
    def customer_safe(self) -> bool:
        """Whether customers are in the audience (the 0.17 marking)."""
        return "customer" in self.audiences

    @customer_safe.setter
    def customer_safe(self, value: bool) -> None:
        self.audiences = EVERYONE if value else OPERATOR_ONLY

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
        return f"ToolWrapper(name={self._schema.name!r}, audiences={sorted(self.audiences)})"


#: Set by ``load_tools`` on a mounted tool: the manifest slug it came from.
TOOL_SLUG_ATTR = "__genfleet_tool_slug__"


def _resolve_audiences(audiences: Iterable[str] | None, customer_safe: bool) -> frozenset[Audience]:
    if audiences is None:
        return EVERYONE if customer_safe else OPERATOR_ONLY
    if customer_safe:
        raise ValueError("pass audiences= or customer_safe=, not both")
    resolved = audiences_of(list(audiences))
    if resolved is None:
        raise ValueError(f"audiences must be a non-empty list of {list(ROLES)}, got {audiences!r}")
    return resolved


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str = "",
    customer_safe: bool = False,
    audiences: Iterable[str] | None = None,
) -> ToolWrapper | Callable[[Callable[..., Any]], ToolWrapper]:
    """
    Decorator that turns any function into a ToolProtocol.

    Usage:
        @tool
        async def search(query: str) -> str: ...

        @tool(description="adds two numbers")
        async def add(a: float, b: float) -> str: ...

        @tool(audiences=["operator", "customer"])   # or customer_safe=True
        async def order_status(order_id: str) -> str: ...

        @tool(audiences=["customer"])
        async def start_return(order_id: str) -> str: ...

    Every tool has an audience (ADR-0028), ``["operator"]`` unless marked. On
    a hosted turn an operator in a private chat gets the tools for
    ``operator``; a customer, or anyone in a group chat, the tools for
    ``customer``. A ``["customer"]`` tool is one staff should not trigger.
    Offer a tool to customers only when any customer may see what it returns
    and trigger what it does.

    The marking is the author's **default**. On the platform the tenant owner
    sets each tool's audience, and that setting wins, wider or narrower.
    """

    def _wrap(f: Callable[..., Any]) -> ToolWrapper:
        resolved_name = name or f.__name__
        resolved_desc = description or (inspect.getdoc(f) or "")
        schema = _build_schema(f, name=resolved_name, description=resolved_desc)
        return ToolWrapper(fn=f, schema=schema, customer_safe=customer_safe, audiences=audiences)

    if fn is not None:
        return _wrap(fn)

    return _wrap
