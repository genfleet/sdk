"""Tool calls a model writes as text instead of as structured calls. Tier 1, no network."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from genfleet.sdk import (
    CONFIRMATION_REQUESTS_KEY,
    Agent,
    AgentInput,
    AgentOutput,
    ToolCall,
    tool,
)
from genfleet.sdk._text_calls import could_be_tool_call_text, tool_calls_in_text
from genfleet.sdk.caller import _LEGACY_HOSTED_ENV, CALLER_METADATA_KEY, HOSTED_ENV
from genfleet.sdk.confirmations import _PEER_REFUSAL, _PENDING_RESULT, PEER_TURN_KEY

OFFERED = {"lookup", "refund"}


def _one(text: str) -> list[tuple[str, dict]] | None:
    calls = tool_calls_in_text(text, OFFERED)
    return None if calls is None else [(c.name, c.arguments) for c in calls]


# ---------------------------------------------------------------------------
# tool_calls_in_text: the accepted shapes
# ---------------------------------------------------------------------------

def test_a_bare_json_list_of_calls():
    text = '[{"name": "lookup", "arguments": {"order": "A1"}}, {"name": "refund", "arguments": {}}]'
    assert _one(text) == [("lookup", {"order": "A1"}), ("refund", {})]


def test_a_single_call_object():
    assert _one('{"name": "lookup", "arguments": {"order": "A1"}}') == [("lookup", {"order": "A1"})]


def test_whitespace_around_the_call_is_trimmed():
    assert _one('\n\n  {"name": "lookup", "arguments": {}}  \n') == [("lookup", {})]


@pytest.mark.parametrize("fence", ["```json\n{}\n```", "```\n{}\n```", "```{}```", "  ```JSON\n{}\n```\n"])
def test_one_code_fence_around_the_call(fence):
    text = fence.replace("{}", '{"name": "lookup", "arguments": {"order": "A1"}}')
    assert _one(text) == [("lookup", {"order": "A1"})]


def test_arguments_may_be_the_json_of_an_object():
    assert _one('{"name": "lookup", "arguments": "{\\"order\\": \\"A1\\"}"}') == [("lookup", {"order": "A1"})]


def test_each_call_gets_its_own_id():
    calls = tool_calls_in_text('[{"name": "lookup", "arguments": {}}, {"name": "lookup", "arguments": {}}]', OFFERED)
    assert calls is not None and calls[0].id != calls[1].id


# ---------------------------------------------------------------------------
# tool_calls_in_text: everything else stays text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        # a tool that was not offered, alone or in a list with an offered one
        '{"name": "delete_everything", "arguments": {}}',
        '[{"name": "lookup", "arguments": {}}, {"name": "delete_everything", "arguments": {}}]',
        # malformed JSON
        '{"name": "lookup", "arguments": {"order": "A1"}',
        '[{"name": "lookup", "arguments": {}},]',
        # ordinary prose, and prose around a call
        "Your order A1 has shipped.",
        'I would call {"name": "lookup", "arguments": {}} here.',
        'Calling now: [{"name": "lookup", "arguments": {}}]',
        '{"name": "lookup", "arguments": {}} and that is it',
        '```json\n{"name": "lookup", "arguments": {}}\n```\nThat is the call I would make.',
        # extra or other keys
        '{"name": "lookup", "arguments": {}, "id": "call_1"}',
        '{"name": "lookup", "arguments": {}, "type": "function"}',
        '{"tool": "lookup", "args": {}}',
        '{"name": "lookup", "parameters": {}}',
        '{"name": "lookup"}',
        '{"function": {"name": "lookup", "arguments": {}}}',
        # a duplicated key, at the top or in the arguments
        '{"name": "refund", "name": "lookup", "arguments": {}}',
        '{"name": "lookup", "arguments": {"order": "A1", "order": "B2"}}',
        # empty, or not objects
        "[]",
        "{}",
        '["lookup"]',
        "[[]]",
        '[{"name": "lookup", "arguments": {}}, "lookup"]',
        # wrong types
        '{"name": ["lookup"], "arguments": {}}',
        '{"name": "lookup", "arguments": ["A1"]}',
        '{"name": "lookup", "arguments": "not json"}',
        '{"name": "lookup", "arguments": "[1]"}',
        '{"name": "lookup", "arguments": null}',
        # more than one fence, or a fence inside a fence
        '```\n{"name": "lookup", "arguments": {}}\n```\n```\n{"name": "lookup", "arguments": {}}\n```',
        '```\n```\n{"name": "lookup", "arguments": {}}\n```\n```',
        # an empty reply
        "",
        "   ",
        "``````",
    ],
)
def test_anything_else_stays_text(text):
    assert _one(text) is None


def test_nothing_offered_means_nothing_runs():
    assert tool_calls_in_text('{"name": "lookup", "arguments": {}}', set()) is None


def test_very_deep_json_stays_text():
    deep = '{"name": "lookup", "arguments": {"a": ' + "[" * 100_000 + "]" * 100_000 + "}}"
    assert _one(deep) is None


@pytest.mark.parametrize(("start", "expected"), [("{", True), (" [", True), ("\n```", True), ("Hi", False), ("1", False)])
def test_could_be_tool_call_text(start, expected):
    assert could_be_tool_call_text(start) is expected


# ---------------------------------------------------------------------------
# Agent: a call written as text runs through the same path as a structured one
# ---------------------------------------------------------------------------

class _Model:
    """A fake model: records the tools it is offered and the messages it sees, then replays rounds."""

    def __init__(self, *rounds: list[AgentOutput]) -> None:
        self.rounds = list(rounds)
        self.offered: list[list[str]] = []
        self.seen: list[list[dict]] = []
        self.complete = self._complete

    async def _complete(self, messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        self.offered.append([t.name for t in tools or []])
        self.seen.append([dict(m) for m in messages])
        for chunk in self.rounds.pop(0):
            yield chunk


ran: list[tuple[str, dict]] = []


@tool(audiences=["operator", "customer"])
def lookup(order: str = "") -> str:
    ran.append(("lookup", {"order": order}))
    return "shipped"


@tool(sensitive=True)
def refund(order: str = "") -> str:
    ran.append(("refund", {"order": order}))
    return "refunded"


@tool
def staff_note(text: str = "") -> str:
    ran.append(("staff_note", {"text": text}))
    return "noted"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ran.clear()
    monkeypatch.delenv(HOSTED_ENV, raising=False)
    monkeypatch.delenv(_LEGACY_HOSTED_ENV, raising=False)


def _agent(model: _Model, tools=None) -> Agent:
    agent = Agent(
        role="r", model={"model": "openai/gpt-4o-mini", "api_key": "k"}, tools=tools or [lookup, refund, staff_note]
    )
    agent._provider = model
    return agent


async def _run(agent: Agent, metadata: dict | None = None) -> list[AgentOutput]:
    return [o async for o in agent.run(AgentInput(message="hi", metadata=metadata or {}))]


def _text(*chunks: str) -> list[AgentOutput]:
    return [AgentOutput(content=c) for c in chunks[:-1]] + [AgentOutput(content=chunks[-1], done=True)]


def _written(*calls: tuple[str, dict]) -> list[AgentOutput]:
    body = json.dumps([{"name": n, "arguments": a} for n, a in calls])
    # Streamed in pieces, as a model would.
    return _text(body[:5], body[5:20], body[20:])


def _reply(outputs: list[AgentOutput]) -> str:
    return "".join(o.content for o in outputs if not o.metadata.get("tool_event"))


def _events(outputs: list[AgentOutput], kind: str) -> list[dict]:
    return [o.metadata["tool_event"] for o in outputs if o.metadata.get("tool_event", {}).get("type") == kind]


_DONE = _text("Your order has shipped.")


@pytest.mark.asyncio
async def test_a_call_list_written_as_text_runs_and_is_never_the_reply():
    model = _Model(_written(("lookup", {"order": "A1"})), _DONE)
    outputs = await _run(_agent(model))
    assert ran == [("lookup", {"order": "A1"})]
    assert _reply(outputs) == "Your order has shipped."
    assert [e["name"] for e in _events(outputs, "tool_call")] == ["lookup"]
    assert [e["ok"] for e in _events(outputs, "tool_result")] == [True]
    assert outputs[-1].done is True
    assert [o.done for o in outputs[:-1]] == [False] * (len(outputs) - 1)


@pytest.mark.asyncio
async def test_a_single_fenced_call_object_runs():
    body = '```json\n{"name": "lookup", "arguments": {"order": "B2"}}\n```'
    model = _Model(_text(body), _DONE)
    outputs = await _run(_agent(model))
    assert ran == [("lookup", {"order": "B2"})]
    assert "lookup" not in _reply(outputs)


@pytest.mark.asyncio
async def test_the_model_sees_a_structured_call_and_its_result():
    model = _Model(_written(("lookup", {"order": "A1"})), _DONE)
    await _run(_agent(model))
    assistant, result = model.seen[1][-2:]
    assert assistant["role"] == "assistant" and assistant["content"] == ""
    assert [tc["function"]["name"] for tc in assistant["tool_calls"]] == ["lookup"]
    assert result == {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": "shipped"}


@pytest.mark.asyncio
async def test_several_calls_written_as_text_all_run_in_order():
    model = _Model(_written(("lookup", {"order": "A"}), ("lookup", {"order": "B"})), _DONE)
    await _run(_agent(model))
    assert ran == [("lookup", {"order": "A"}), ("lookup", {"order": "B"})]


@pytest.mark.parametrize(
    "body",
    [
        '[{"name": "cancel_order", "arguments": {"order": "A1"}}]',   # not one of the agent's tools
        '[{"name": "lookup", "arguments": {"order": "A1"}]',           # malformed
        'Here it is: [{"name": "lookup", "arguments": {}}]',            # prose around it
        '[{"name": "lookup", "arguments": {}, "id": "c1"}]',            # an extra key
        "[]",                                                          # empty
        "Your order A1 has shipped.",                                  # ordinary prose
    ],
)
@pytest.mark.asyncio
async def test_other_text_is_the_reply_unchanged(body):
    model = _Model(_text(body[:3], body[3:]))
    outputs = await _run(_agent(model))
    assert ran == []
    assert _reply(outputs) == body
    assert _events(outputs, "tool_call") == []
    assert len(model.seen) == 1


@pytest.mark.asyncio
async def test_prose_still_streams_chunk_by_chunk():
    model = _Model(_text("Your ", "order ", "shipped."))
    outputs = await _run(_agent(model))
    assert [o.content for o in outputs] == ["Your ", "order ", "shipped.", ""]


@pytest.mark.asyncio
async def test_text_alongside_a_structured_call_is_left_alone():
    """Only a reply with no structured calls is read for written calls."""
    body = '{"name": "staff_note", "arguments": {"text": "x"}}'
    model = _Model(
        [AgentOutput(content=body), AgentOutput(tool_calls=[ToolCall(id="t1", name="lookup", arguments={})])],
        _DONE,
    )
    outputs = await _run(_agent(model))
    assert ran == [("lookup", {"order": ""})]
    assert body in _reply(outputs)


# -- safety: the same checks as a structured call ---------------------------

@pytest.mark.asyncio
async def test_a_sensitive_tool_written_as_text_waits_for_approval():
    model = _Model(_written(("refund", {"order": "A1"})), _DONE)
    outputs = await _run(_agent(model))
    assert ran == []
    assert outputs[-1].metadata[CONFIRMATION_REQUESTS_KEY] == [{"tool": "refund", "arguments": {"order": "A1"}}]
    assert _events(outputs, "tool_result")[0]["status"] == "pending"
    assert [m["content"] for m in model.seen[1] if m["role"] == "tool"] == [_PENDING_RESULT]


@pytest.mark.asyncio
async def test_a_sensitive_tool_written_as_text_is_refused_on_a_peer_turn():
    model = _Model(_written(("refund", {"order": "A1"})), _DONE)
    outputs = await _run(_agent(model), {PEER_TURN_KEY: True})
    assert ran == []
    assert CONFIRMATION_REQUESTS_KEY not in outputs[-1].metadata
    assert [m["content"] for m in model.seen[1] if m["role"] == "tool"] == [_PEER_REFUSAL]


@pytest.mark.asyncio
async def test_a_tool_not_offered_to_this_caller_stays_text():
    """A customer is not offered the operator-only tool, so writing its call does nothing."""
    body = '[{"name": "staff_note", "arguments": {"text": "x"}}]'
    model = _Model(_text(body))
    customer = {CALLER_METADATA_KEY: {"role": "customer", "private": True}}
    outputs = await _run(_agent(model), customer)
    assert "staff_note" not in model.offered[0]
    assert ran == []
    assert _reply(outputs) == body


@pytest.mark.asyncio
async def test_a_tool_offered_to_this_caller_runs_when_written():
    model = _Model(_written(("lookup", {"order": "A1"})), _DONE)
    customer = {CALLER_METADATA_KEY: {"role": "customer", "private": True}}
    await _run(_agent(model), customer)
    assert ran == [("lookup", {"order": "A1"})]


@pytest.mark.asyncio
async def test_written_calls_take_part_in_the_same_loop_as_structured_ones():
    """Each round counts as one loop round, whichever way the model made its call."""
    model = _Model(
        _written(("lookup", {"order": "1"})),
        [AgentOutput(tool_calls=[ToolCall(id="t2", name="lookup", arguments={"order": "2"})])],
        _written(("lookup", {"order": "3"})),
        _DONE,
    )
    outputs = await _run(_agent(model))
    assert ran == [("lookup", {"order": "1"}), ("lookup", {"order": "2"}), ("lookup", {"order": "3"})]
    assert len(model.seen) == 4
    assert len(_events(outputs, "tool_result")) == 3
    assert _reply(outputs) == "Your order has shipped."
