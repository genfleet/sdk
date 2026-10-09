"""Tool audiences (ADR-0028, SDK 0.18): who each tool is offered to. Tier 1, no network."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator

import pytest

from genfleet.sdk import EVERYONE as SDK_EVERYONE
from genfleet.sdk import OPERATOR_ONLY as SDK_OPERATOR_ONLY
from genfleet.sdk import Agent, AgentInput, AgentOutput, ToolCall, tool
from genfleet.sdk import agent as agent_module
from genfleet.sdk.caller import (
    _LEGACY_HOSTED_ENV,
    CALLER_METADATA_KEY,
    EVERYONE,
    HOSTED_ENV,
    OPERATOR_ONLY,
    TOOL_AUDIENCES_METADATA_KEY,
    Caller,
    effective_audiences,
    marked,
    may_offer,
    may_use,
    tool_audiences_of,
)
from genfleet.sdk.caller import (
    _audiences_of as audiences_of,
)
from genfleet.sdk.manifest import AgentManifest, ToolRef
from genfleet.sdk.schemas import ToolSchema
from genfleet.sdk.tool import ToolWrapper, wrap_tool
from genfleet.sdk.tools_loader import load_tools

OP = frozenset({"operator"})
CU = frozenset({"customer"})
BOTH = frozenset({"operator", "customer"})


def _meta(audiences=None, **caller) -> dict:
    meta: dict = {CALLER_METADATA_KEY: caller}
    if audiences is not None:
        meta[TOOL_AUDIENCES_METADATA_KEY] = audiences
    return meta


# ---------------------------------------------------------------------------
# may_offer
# ---------------------------------------------------------------------------

OPERATOR_PRIVATE = Caller(role="operator", private=True)
OPERATOR_GROUP = Caller(role="operator", private=False)
CUSTOMER_PRIVATE = Caller(role="customer", private=True)
CUSTOMER_GROUP = Caller(role="customer", private=False)


@pytest.mark.parametrize(
    "caller, audiences, allowed",
    [
        (None, OP, True),
        (None, CU, True),
        (Caller(role="customer", legacy_tools=True), OP, True),
        (Caller(role="customer", legacy_tools=True), CU, True),
        (Caller(role="operator", private=False, legacy_tools=True), CU, True),
        (OPERATOR_PRIVATE, OP, True),
        (OPERATOR_PRIVATE, CU, False),
        (OPERATOR_PRIVATE, BOTH, True),
        (OPERATOR_GROUP, OP, False),
        (OPERATOR_GROUP, CU, True),
        (OPERATOR_GROUP, BOTH, True),
        (CUSTOMER_PRIVATE, OP, False),
        (CUSTOMER_PRIVATE, CU, True),
        (CUSTOMER_PRIVATE, BOTH, True),
        (CUSTOMER_GROUP, OP, False),
        (CUSTOMER_GROUP, CU, True),
        (CUSTOMER_GROUP, BOTH, True),
    ],
)
def test_may_offer_truth_table(caller, audiences, allowed):
    assert may_offer(caller, audiences) is allowed


def test_constants():
    assert OPERATOR_ONLY == OP
    assert EVERYONE == BOTH


# ---------------------------------------------------------------------------
# audiences_of / tool_audiences_of
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (["operator"], OP),
        (["customer"], CU),
        (["customer", "operator", "customer"], BOTH),
        (("operator",), OP),
        ({"customer"}, CU),
        ([], None),
        (["admin"], None),
        (["operator", "admin"], None),
        ([1], None),
        ("operator", None),
        (None, None),
        (5, None),
        ({"operator": True}, None),
    ],
)
def test_audiences_of(value, expected):
    assert audiences_of(value) == expected


def test_tool_audiences_of_keeps_valid_entries():
    meta = {TOOL_AUDIENCES_METADATA_KEY: {"a": ["operator"], "@acme/b": ["customer", "operator"]}}
    assert tool_audiences_of(meta) == {"a": OP, "@acme/b": BOTH}


def test_tool_audiences_of_invalid_values_fail_closed():
    meta = {TOOL_AUDIENCES_METADATA_KEY: {
        "ok": ["customer"], "empty": [], "unknown": ["admin"], "text": "customer", "none": None,
        "": ["customer"], 7: ["customer"],
    }}
    assert tool_audiences_of(meta) == {
        "ok": CU, "empty": frozenset(), "unknown": frozenset(), "text": frozenset(), "none": frozenset(),
    }


def test_effective_audiences_precedence():
    overrides = {"n": CU, "@a/s": BOTH}
    assert effective_audiences("n", "@a/s", OP, overrides) == CU
    assert effective_audiences("x", "@a/s", OP, overrides) == BOTH
    assert effective_audiences("x", "@a/other", OP, overrides) == OP
    assert effective_audiences("x", None, OP, overrides) == OP
    assert effective_audiences("x", "", OP, {"": CU}) == OP
    assert effective_audiences("n", None, BOTH, {"n": frozenset()}) == frozenset()


def test_marked_and_sdk_exports():
    assert marked(True) == BOTH
    assert marked(False) == OP
    assert SDK_EVERYONE == BOTH
    assert SDK_OPERATOR_ONLY == OP


@pytest.mark.parametrize("raw", ["x", ["a"], 5, None, True])
def test_tool_audiences_of_non_dict_is_empty(raw):
    assert tool_audiences_of({TOOL_AUDIENCES_METADATA_KEY: raw}) == {}


@pytest.mark.parametrize("meta", [None, {}, {"other": 1}])
def test_tool_audiences_of_missing_is_empty(meta):
    assert tool_audiences_of(meta) == {}


# ---------------------------------------------------------------------------
# @tool / ToolWrapper
# ---------------------------------------------------------------------------

def test_tool_defaults_to_operator_only():
    @tool
    def a() -> str:
        return ""

    assert a.audiences == OP
    assert a.customer_safe is False
    assert a.slug is None


def test_customer_safe_means_both_audiences():
    with pytest.warns(DeprecationWarning):
        @tool(customer_safe=True)
        def a() -> str:
            return ""

    assert a.audiences == BOTH
    assert a.customer_safe is True


def test_customer_safe_false_warns_and_is_operator_only():
    with pytest.warns(DeprecationWarning):
        @tool(customer_safe=False)
        def a() -> str:
            return ""

    assert a.audiences == OP


def test_customer_only_audience():
    @tool(audiences=["customer"])
    def a() -> str:
        return ""

    assert a.audiences == CU
    assert a.customer_safe is False


def test_operator_audience_list_is_not_customer_safe():
    @tool(audiences=["operator"])
    def a() -> str:
        return ""

    assert a.audiences == OP
    assert a.customer_safe is False


def test_both_markings_are_rejected():
    with pytest.raises(ValueError, match="not both"):
        @tool(customer_safe=True, audiences=["customer"])
        def a() -> str:
            return ""

    with pytest.raises(ValueError, match="not both"):
        @tool(customer_safe=False, audiences=["customer"])
        def b() -> str:
            return ""


@pytest.mark.parametrize("bad", [[], ["admin"], ["operator", "root"]])
def test_bad_audiences_are_rejected(bad):
    with pytest.raises(ValueError, match="audiences"):
        @tool(audiences=bad)
        def a() -> str:
            return ""


def test_customer_safe_is_read_only():
    @tool(audiences=["operator", "customer"])
    def a() -> str:
        return ""

    assert a.customer_safe is True
    with pytest.raises(AttributeError):
        a.customer_safe = False


def test_repr_lists_sorted_audiences():
    @tool(audiences=["operator", "customer"])
    def a() -> str:
        return ""

    assert repr(a) == "ToolWrapper(name='a', audiences=['customer', 'operator'])"


def test_wrapper_slug_only_from_the_argument():
    def f() -> str:
        return ""

    f.__genfleet_tool_slug__ = "@b/f"  # the 0.18-dev attribute is no longer read
    schema = ToolSchema(name="f", description="", parameters={})
    assert ToolWrapper(f, schema, slug="@a/f").slug == "@a/f"
    assert ToolWrapper(f, schema).slug is None


def test_with_slug_returns_a_copy():
    @tool(audiences=["customer"])
    def f() -> str:
        return "x"

    copy = f.with_slug("@a/f")
    assert copy is not f
    assert copy.slug == "@a/f"
    assert f.slug is None
    assert copy.audiences == CU
    assert copy.schema() == f.schema()


def test_wrap_tool():
    def plain(order_id: str) -> str:
        """Look up."""
        return order_id

    wrapped = wrap_tool(plain, slug="@a/plain")
    assert isinstance(wrapped, ToolWrapper)
    assert wrapped.slug == "@a/plain"
    assert wrapped.schema().name == "plain"
    assert wrapped.schema().description == "Look up."
    assert wrapped.audiences == OP
    assert wrap_tool(plain).slug is None

    @tool
    def t() -> str:
        return ""

    assert wrap_tool(t) is t
    assert wrap_tool(t, slug="@a/t").slug == "@a/t"
    assert t.slug is None


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

class _Model:
    """A fake model: records the tool names it is offered, then replays rounds."""

    def __init__(self, *rounds: list[AgentOutput]) -> None:
        self.rounds = list(rounds)
        self.offered: list[str] = []
        self.complete = self._complete

    async def _complete(self, messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        self.offered = [t.name for t in tools or []]
        for chunk in self.rounds.pop(0):
            yield chunk


_DONE = [AgentOutput(content="ok", done=True)]
ran: list[str] = []


@tool(audiences=["operator", "customer"])
def lookup_order() -> str:
    ran.append("lookup_order")
    return "shipped"


@tool
def refund() -> str:
    ran.append("refund")
    return "refunded"


@tool(audiences=["customer"])
def start_return() -> str:
    ran.append("start_return")
    return "started"


def _agent(model: _Model, **kwargs) -> Agent:
    agent = Agent(role="r", model={"model": "openai/gpt-4o-mini", "api_key": "k"}, **kwargs)
    agent._provider = model
    return agent


async def _run(agent: Agent, metadata: dict) -> list[AgentOutput]:
    return [o async for o in agent.run(AgentInput(message="hi", metadata=metadata))]


async def _offered(metadata: dict, tools=None, **kwargs) -> set[str]:
    model = _Model(_DONE)
    await _run(_agent(model, tools=tools or [lookup_order, refund], **kwargs), metadata)
    return set(model.offered)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ran.clear()
    monkeypatch.delenv(HOSTED_ENV, raising=False)
    monkeypatch.delenv(_LEGACY_HOSTED_ENV, raising=False)


@pytest.mark.asyncio
async def test_override_by_name_widens_an_unmarked_tool():
    assert await _offered(_meta(role="customer", private=True)) == {"lookup_order"}
    got = await _offered(_meta({"refund": ["customer"]}, role="customer", private=True))
    assert got == {"lookup_order", "refund"}


@pytest.mark.asyncio
async def test_override_narrows_a_customer_safe_tool():
    got = await _offered(_meta({"lookup_order": ["operator"]}, role="customer", private=True))
    assert got == set()
    got = await _offered(_meta({"lookup_order": ["operator"]}, role="operator", private=True))
    assert got == {"lookup_order", "refund"}


@pytest.mark.asyncio
async def test_override_by_slug_applies_to_a_tagged_function():
    def plain() -> str:
        return ""

    plain = wrap_tool(plain, slug="@acme/plain")
    metadata = _meta({"@acme/plain": ["customer"]}, role="customer", private=True)
    assert await _offered(metadata, tools=[plain]) == {"plain"}
    assert await _offered(_meta(role="customer", private=True), tools=[plain]) == set()


@pytest.mark.asyncio
async def test_override_by_slug_applies_to_a_wrapper_with_a_slug():
    @tool
    def wrapped() -> str:
        return ""

    wrapped = wrapped.with_slug("@acme/wrapped")
    metadata = _meta({"@acme/wrapped": ["customer"]}, role="customer", private=True)
    assert await _offered(metadata, tools=[wrapped]) == {"wrapped"}


@pytest.mark.asyncio
async def test_a_slug_override_does_not_touch_an_untagged_tool():
    metadata = _meta({"@acme/refund": ["customer"]}, role="customer", private=True)
    assert await _offered(metadata) == {"lookup_order"}


@pytest.mark.asyncio
async def test_name_wins_over_slug():
    @tool
    def wrapped() -> str:
        return ""

    wrapped = wrapped.with_slug("@acme/wrapped")
    narrowed = _meta({"wrapped": ["operator"], "@acme/wrapped": ["customer"]}, role="customer", private=True)
    assert await _offered(narrowed, tools=[wrapped]) == set()
    widened = _meta({"wrapped": ["customer"], "@acme/wrapped": ["operator"]}, role="customer", private=True)
    assert await _offered(widened, tools=[wrapped]) == {"wrapped"}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [[], ["admin"], "customer", None])
async def test_a_malformed_override_hides_the_tool_from_everyone_restricted(bad):
    overrides = {"refund": bad, "lookup_order": bad}
    assert await _offered(_meta(overrides, role="customer", private=True)) == set()
    assert await _offered(_meta(overrides, role="operator", private=True)) == set()
    assert await _offered(_meta(overrides, role="operator", private=False)) == set()
    # Unrestricted turns still get everything.
    legacy = _meta(overrides, role="customer", legacy_tools=True)
    assert await _offered(legacy) == {"lookup_order", "refund"}
    assert await _offered({TOOL_AUDIENCES_METADATA_KEY: overrides}) == {"lookup_order", "refund"}


@pytest.mark.asyncio
async def test_a_malformed_override_for_one_tool_leaves_others_alone():
    got = await _offered(_meta({"refund": []}, role="operator", private=True))
    assert got == {"lookup_order"}


@pytest.mark.asyncio
async def test_a_non_dict_override_map_is_ignored():
    got = await _offered({CALLER_METADATA_KEY: {"role": "customer", "private": True},
                          TOOL_AUDIENCES_METADATA_KEY: ["refund"]})
    assert got == {"lookup_order"}


@pytest.mark.asyncio
async def test_a_customer_only_tool_is_hidden_from_an_operator_in_private():
    tools = [lookup_order, refund, start_return]
    assert await _offered(_meta(role="operator", private=True), tools) == {"lookup_order", "refund"}
    assert await _offered(_meta(role="operator", private=False), tools) == {"lookup_order", "start_return"}
    assert await _offered(_meta(role="customer", private=True), tools) == {"lookup_order", "start_return"}


@pytest.mark.asyncio
async def test_legacy_tools_and_no_caller_offer_a_customer_only_tool_too():
    tools = [lookup_order, refund, start_return]
    everything = {"lookup_order", "refund", "start_return"}
    assert await _offered(_meta(role="customer", legacy_tools=True), tools) == everything
    assert await _offered({}, tools) == everything


@pytest.mark.asyncio
async def test_override_cannot_widen_past_legacy_or_no_caller():
    # Overrides only matter when a caller is restricted; unrestricted turns see all.
    got = await _offered({TOOL_AUDIENCES_METADATA_KEY: {"refund": ["customer"]}})
    assert got == {"lookup_order", "refund"}


def _tool_result(outputs: list[AgentOutput]) -> dict:
    return next(
        o.metadata["tool_event"] for o in outputs if o.metadata.get("tool_event", {}).get("type") == "tool_result"
    )


@pytest.mark.asyncio
async def test_dispatch_of_a_tool_hidden_by_an_override_is_not_found():
    model = _Model([AgentOutput(tool_calls=[ToolCall(id="1", name="lookup_order", arguments={})])], _DONE)
    outputs = await _run(
        _agent(model, tools=[lookup_order, refund]),
        _meta({"lookup_order": ["operator"]}, role="customer", private=True),
    )
    result = _tool_result(outputs)
    assert result["ok"] is False
    assert result["output"] == "Error: tool 'lookup_order' not found"
    assert ran == []


@pytest.mark.asyncio
async def test_dispatch_of_a_customer_only_tool_to_a_private_operator_is_not_found():
    model = _Model([AgentOutput(tool_calls=[ToolCall(id="1", name="start_return", arguments={})])], _DONE)
    outputs = await _run(_agent(model, tools=[start_return]), _meta(role="operator", private=True))
    assert _tool_result(outputs)["output"] == "Error: tool 'start_return' not found"
    assert ran == []


@pytest.mark.asyncio
async def test_a_widened_tool_runs_for_a_customer():
    model = _Model([AgentOutput(tool_calls=[ToolCall(id="1", name="refund", arguments={})])], _DONE)
    await _run(_agent(model, tools=[refund]), _meta({"refund": ["customer"]}, role="customer"))
    assert ran == ["refund"]


@pytest.mark.asyncio
async def test_mcp_override_by_name(monkeypatch):
    config = {"type": "sse", "url": "http://mcp.test", "customer_safe_tools": ["a"]}

    async def fetch(cfg):
        return {n: {"description": "", "parameters": {}, "config": cfg} for n in ("a", "b")}

    monkeypatch.setattr(agent_module, "_fetch_mcp_tools", fetch)

    async def offered(audiences) -> list[str]:
        model = _Model(_DONE)
        await _run(_agent(model, mcps=[config]), _meta(audiences, role="customer", private=True))
        return sorted(model.offered)

    assert await offered(None) == ["a"]
    assert await offered({"b": ["customer"]}) == ["a", "b"]
    assert await offered({"a": ["operator"]}) == []
    assert await offered({"a": ["bogus"]}) == []


# ---------------------------------------------------------------------------
# load_tools tagging
# ---------------------------------------------------------------------------

class _Resolver:
    def __init__(self, by_slug: dict) -> None:
        self.by_slug = by_slug

    def resolve(self, ref: ToolRef):
        return self.by_slug[ref.slug]


def _load(by_slug: dict) -> list:
    manifest = AgentManifest(name="x", tools=[ToolRef(slug=s, version="1.0.0", digest="sha256:ab") for s in by_slug])
    return load_tools(manifest=manifest, resolver=_Resolver(by_slug))


def test_load_tools_returns_a_fresh_proxy_that_behaves_like_the_function():
    def plain(x: int) -> int:
        """Doubles."""
        return x * 2

    [loaded] = _load({"@acme/plain": plain})
    assert loaded is not plain
    assert not isinstance(loaded, ToolWrapper)
    # Called directly it is the function: sync, raw result (other SDKs call it).
    assert loaded(3) == 6
    assert loaded.__name__ == "plain" and loaded.__doc__ == "Doubles."
    assert loaded.__genfleet_tool_slug__ == "@acme/plain"
    assert not hasattr(plain, "__genfleet_tool_slug__")
    agent = Agent(role="r", model={"model": "openai/gpt-4o-mini", "api_key": "k"}, tools=[loaded])
    assert agent._local_tools["plain"].slug == "@acme/plain"
    assert agent._local_tools["plain"].schema().parameters["properties"]["x"]["type"] == "integer"


def test_load_tools_keeps_an_async_function_async():
    async def fetch(order_id: str) -> str:
        return order_id

    [loaded] = _load({"@acme/fetch": fetch})
    assert inspect.iscoroutinefunction(loaded)
    assert asyncio.run(loaded("o-1")) == "o-1"


def test_load_tools_does_not_mutate_a_resolved_wrapper():
    @tool(audiences=["customer"])
    def fresh() -> str:
        return ""

    @tool
    def tagged() -> str:
        return ""

    tagged = tagged.with_slug("@own/tagged")
    got_fresh, got_tagged = _load({"@acme/fresh": fresh, "@acme/tagged": tagged})
    assert got_fresh is not fresh and got_fresh.slug == "@acme/fresh"
    assert got_fresh.audiences == CU
    assert fresh.slug is None
    assert got_tagged.slug == "@acme/tagged"
    assert tagged.slug == "@own/tagged"


def test_one_object_under_two_slugs_gives_two_wrappers():
    @tool
    def shared() -> str:
        return ""

    [a] = _load({"@acme/one": shared})
    [b] = _load({"@acme/two": shared})
    assert a is not b
    assert (a.slug, b.slug) == ("@acme/one", "@acme/two")
    assert shared.slug is None


def test_load_tools_proxies_a_bound_method():
    class Holder:
        def lookup(self, order_id: str) -> str:
            """Look an order up."""
            return order_id

    [loaded] = _load({"@acme/lookup": Holder().lookup})
    assert loaded("o-9") == "o-9"
    agent = Agent(role="r", model={"model": "openai/gpt-4o-mini", "api_key": "k"}, tools=[loaded])
    wrapper = agent._local_tools["lookup"]
    assert wrapper.slug == "@acme/lookup"
    assert wrapper.schema().description == "Look an order up."
    assert "self" not in wrapper.schema().parameters["properties"]
    assert wrapper.audiences == OP
    assert agent._offered_tools(Caller(role="customer"), {"@acme/lookup": CU}) == {"lookup"}
    assert agent._offered_tools(Caller(role="customer"), {}) == set()


# ---------------------------------------------------------------------------
# Deprecated may_use still behaves as in 0.17
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "caller, safe, allowed",
    [
        (None, False, True),
        (Caller(role="operator", private=True), False, True),
        (Caller(role="operator", private=False), False, False),
        (Caller(role="operator", private=False), True, True),
        (Caller(role="customer", private=True), False, False),
        (Caller(role="customer", private=True), True, True),
        (Caller(role="customer", legacy_tools=True), False, True),
    ],
)
def test_may_use_is_unchanged(caller, safe, allowed):
    with pytest.warns(DeprecationWarning):
        assert may_use(caller, customer_safe=safe) is allowed


def test_full_toolset_is_deprecated_but_unchanged():
    with pytest.warns(DeprecationWarning):
        assert Caller(role="operator", private=True).full_toolset is True
    with pytest.warns(DeprecationWarning):
        assert Caller(role="operator", private=False).full_toolset is False
    with pytest.warns(DeprecationWarning):
        assert Caller(role="customer", legacy_tools=True).full_toolset is True
