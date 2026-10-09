from __future__ import annotations

import asyncio
import inspect
import types
import typing
import warnings
from collections.abc import Callable, Iterable
from typing import Any, get_args, get_origin

from .caller import EVERYONE, OPERATOR_ONLY, ROLES, Audience, _audiences_of, marked
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


def _type_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """``fn``'s resolved hints; a callable instance's from its ``__call__``.

    A builtin or an ``operator.itemgetter`` has none, and an unresolvable one
    is left untyped rather than failing the whole tool.
    """
    for target in (fn, getattr(type(fn), "__call__", None)):
        if target is None:
            continue
        try:
            return typing.get_type_hints(target)
        except (TypeError, NameError):
            continue
    return {}


def _build_schema(fn: Callable[..., Any], name: str, description: str) -> ToolSchema:
    sig = inspect.signature(fn)
    hints = _type_hints(fn)

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
        customer_safe: bool | None = None,
        audiences: Iterable[Audience] | None = None,
        slug: str | None = None,
        sensitive: bool = False,
    ) -> None:
        self._fn = fn
        self._schema = schema
        #: Who the author offers the tool to (ADR-0028): the default the
        #: platform's per-tool setting replaces. ``[operator]`` unless marked.
        self.audiences: frozenset[Audience] = _resolve_audiences(audiences, customer_safe, stacklevel=2)
        #: The manifest slug a mounted tool came from (``load_tools``), so the
        #: platform's setting can name it before its function name is known.
        self.slug = slug
        #: ADR-0028 §8a: the author's marking. A sensitive call doesn't run
        #: until an owner or admin approves it; the platform's flag wins.
        self.sensitive = sensitive

    @property
    def customer_safe(self) -> bool:
        """The 0.17 marking: offered to operators *and* customers. Read-only.

        False for a ``["customer"]``-only tool, which operators don't get.
        """
        return self.audiences == EVERYONE

    @customer_safe.setter
    def customer_safe(self, value: bool) -> None:
        """Deprecated (0.18), removed in 1.0: assign ``audiences`` instead."""
        _warn_customer_safe(stacklevel=2)
        self.audiences = marked(value)

    def with_slug(self, slug: str) -> ToolWrapper:
        """A copy carrying ``slug``; the original is left untouched."""
        return ToolWrapper(fn=self._fn, schema=self._schema, audiences=self.audiences, slug=slug, sensitive=self.sensitive)

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


#: Set by ``load_tools`` on the proxy it returns for a mounted tool (never on
#: the loaded object): the manifest slug it came from.
TOOL_SLUG_ATTR = "__genfleet_tool_slug__"


def wrap_tool(fn: Callable[..., Any], *, slug: str | None = None) -> ToolWrapper:
    """``fn`` as a ToolWrapper. A plain callable gets a schema from its
    signature, and the slug ``load_tools`` gave it; a ToolWrapper is returned
    as is, or copied to carry ``slug``."""
    if isinstance(fn, ToolWrapper):
        return fn.with_slug(slug) if slug else fn
    name = getattr(fn, "__name__", None) or (slug.rsplit("/", 1)[-1] if slug else type(fn).__name__)
    schema = _build_schema(fn, name=name, description=getattr(fn, "__doc__", None) or "")
    return ToolWrapper(fn=fn, schema=schema, slug=slug or getattr(fn, TOOL_SLUG_ATTR, None))


def _warn_customer_safe(*, stacklevel: int) -> None:
    warnings.warn(
        'customer_safe is deprecated since 0.18 and removed in 1.0; use audiences=["operator", "customer"]',
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def _resolve_audiences(
    audiences: Iterable[Audience] | None, customer_safe: bool | None, *, stacklevel: int
) -> frozenset[Audience]:
    """``stacklevel``: how far up the caller who wrote ``customer_safe=`` is."""
    if customer_safe is not None:
        if audiences is not None:
            raise ValueError("pass audiences= or customer_safe=, not both")
        _warn_customer_safe(stacklevel=stacklevel + 1)
        return marked(customer_safe)
    if audiences is None:
        return OPERATOR_ONLY
    resolved = _audiences_of(list(audiences))
    if resolved is None:
        raise ValueError(f"audiences must be a non-empty list of {list(ROLES)}, got {audiences!r}")
    return resolved


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str = "",
    customer_safe: bool | None = None,
    audiences: Iterable[Audience] | None = None,
    sensitive: bool = False,
) -> ToolWrapper | Callable[[Callable[..., Any]], ToolWrapper]:
    """
    Decorator that turns any function into a ToolProtocol.

    Usage:
        @tool
        async def search(query: str) -> str: ...

        @tool(description="adds two numbers")
        async def add(a: float, b: float) -> str: ...

        @tool(audiences=["operator", "customer"])
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

    ``customer_safe=True`` (0.17) still means both audiences; it is deprecated
    and removed in 1.0.

    ``sensitive=True`` (ADR-0028 §8a): the call doesn't run when the model
    makes it. It waits for a workspace owner or admin to approve it, then runs
    once in a follow-up turn. See :mod:`genfleet.sdk.confirmations`.
    """

    def _wrap(f: Callable[..., Any]) -> ToolWrapper:
        resolved_name = name or f.__name__
        resolved_desc = description or (inspect.getdoc(f) or "")
        schema = _build_schema(f, name=resolved_name, description=resolved_desc)
        # Resolved here, so a deprecation warning points at the decorated
        # definition (two frames up), not at this module.
        resolved = _resolve_audiences(audiences, customer_safe, stacklevel=2)
        return ToolWrapper(fn=f, schema=schema, audiences=resolved, sensitive=sensitive)

    if fn is not None:
        return _wrap(fn)

    return _wrap
