"""Tool audiences (ADR-0028, SDK 0.18): who each tool is offered to. Tier 1, no network."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

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
    audiences_of,
    may_offer,
    may_use,
    tool_audiences_of,
)
from genfleet.sdk.manifest import AgentManifest, ToolRef
from genfleet.sdk.schemas import ToolSchema
from genfleet.sdk.tool import TOOL_SLUG_ATTR, ToolWrapper
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


def test_tool_audiences_of_drops_invalid_entries():
    meta = {TOOL_AUDIENCES_METADATA_KEY: {
        "ok": ["customer"], "empty": [], "unknown": ["admin"], "text": "customer", "none": None,
        "": ["customer"], 7: ["customer"],
    }}
    assert tool_audiences_of(meta) == {"ok": CU}


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
    @tool(customer_safe=True)
    def a() -> str:
        return ""

    assert a.audiences == BOTH
    assert a.customer_safe is True


def test_customer_only_audience():
    @tool(audiences=["customer"])
    def a() -> str:
        return ""

    assert a.audiences == CU
    assert a.customer_safe is True


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


@pytest.mark.parametrize("bad", [[], ["admin"], ["operator", "root"]])
def test_bad_audiences_are_rejected(bad):
    with pytest.raises(ValueError, match="audiences"):
        @tool(audiences=bad)
        def a() -> str:
            return ""


def test_customer_safe_setter_resets_audiences():
    @tool(audiences=["customer"])
    def a() -> str:
        return ""

    a.customer_safe = False
    assert a.audiences == OP
    a.customer_safe = True
    assert a.audiences == BOTH


def test_repr_lists_sorted_audiences():
    @tool(customer_safe=True)
    def a() -> str:
        return ""

    assert repr(a) == "ToolWrapper(name='a', audiences=['customer', 'operator'])"


def test_wrapper_slug_from_argument_or_function_attribute():
    def f() -> str:
        return ""

    schema = ToolSchema(name="f", description="", parameters={})
    assert ToolWrapper(f, schema, slug="@a/f").slug == "@a/f"
    setattr(f, TOOL_SLUG_ATTR, "@b/f")
    assert ToolWrapper(f, schema).slug == "@b/f"
    assert ToolWrapper(f, schema, slug="@a/f").slug == "@a/f"


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


@tool(customer_safe=True)
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

    setattr(plain, TOOL_SLUG_ATTR, "@acme/plain")
    metadata = _meta({"@acme/plain": ["customer"]}, role="customer", private=True)
    assert await _offered(metadata, tools=[plain]) == {"plain"}
    assert await _offered(_meta(role="customer", private=True), tools=[plain]) == set()


@pytest.mark.asyncio
async def test_override_by_slug_applies_to_a_wrapper_with_a_slug():
    @tool
    def wrapped() -> str:
        return ""

    wrapped.slug = "@acme/wrapped"
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

    wrapped.slug = "@acme/wrapped"
    narrowed = _meta({"wrapped": ["operator"], "@acme/wrapped": ["customer"]}, role="customer", private=True)
    assert await _offered(narrowed, tools=[wrapped]) == set()
    widened = _meta({"wrapped": ["customer"], "@acme/wrapped": ["operator"]}, role="customer", private=True)
    assert await _offered(widened, tools=[wrapped]) == {"wrapped"}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [[], ["admin"], "customer", None])
async def test_a_malformed_override_falls_back_to_the_default(bad):
    got = await _offered(_meta({"refund": bad, "lookup_order": bad}, role="customer", private=True))
    assert got == {"lookup_order"}
    got = await _offered(_meta({"refund": bad, "lookup_order": bad}, role="operator", private=True))
    assert got == {"lookup_order", "refund"}


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
    assert await offered({"a": ["bogus"]}) == ["a"]


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


def test_load_tools_tags_a_plain_function_and_agent_picks_up_the_slug():
    def plain() -> str:
        return ""

    [loaded] = _load({"@acme/plain": plain})
    assert loaded is plain
    assert getattr(plain, TOOL_SLUG_ATTR) == "@acme/plain"
    agent = Agent(role="r", model={"model": "openai/gpt-4o-mini", "api_key": "k"}, tools=[loaded])
    assert agent._local_tools["plain"].slug == "@acme/plain"


def test_load_tools_tags_a_wrapper_without_overwriting_an_existing_slug():
    @tool
    def fresh() -> str:
        return ""

    @tool
    def tagged() -> str:
        return ""

    tagged.slug = "@own/tagged"
    got_fresh, got_tagged = _load({"@acme/fresh": fresh, "@acme/tagged": tagged})
    assert got_fresh is fresh and fresh.slug == "@acme/fresh"
    assert got_tagged is tagged and tagged.slug == "@own/tagged"


def test_load_tools_wraps_a_callable_that_refuses_attributes():
    class Holder:
        def lookup(self, order_id: str) -> str:
            """Look an order up."""
            return order_id

    bound = Holder().lookup
    [loaded] = _load({"@acme/lookup": bound})
    assert isinstance(loaded, ToolWrapper)
    assert loaded.slug == "@acme/lookup"
    assert loaded.schema().name == "lookup"
    assert loaded.schema().description == "Look an order up."
    assert loaded.audiences == OP

    agent = Agent(role="r", model={"model": "openai/gpt-4o-mini", "api_key": "k"}, tools=[loaded])
    assert agent._offered_tools(Caller(role="customer"), {"@acme/lookup": CU}) == {"lookup"}
    assert agent._offered_tools(Caller(role="customer")) == set()


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
    assert may_use(caller, customer_safe=safe) is allowed
