"""ScriptedModelClient drives the Agent's real tool loop. Tier 1, no network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from genfleet.sdk import (
    APPROVED_CALL_KEY,
    CONFIRMATION_REQUESTS_KEY,
    TOOL_EVENT_KEY,
    Agent,
    AgentInput,
    TokenUsage,
    ToolCall,
    confirmations,
    sign_approved_call,
    tool,
)
from genfleet.sdk.caller import CALLER_METADATA_KEY, HOSTED_ENV
from genfleet.sdk.confirmations import (
    _PEER_REFUSAL,
    _PENDING_RESULT,
    APPROVAL_KEY_ENV,
    PEER_TURN_KEY,
)
from genfleet.sdk.models import ModelClient, ScriptedModelClient, Step

ran: list[tuple[str, dict]] = []


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ran.clear()
    confirmations._RUN.clear()
    monkeypatch.delenv(HOSTED_ENV, raising=False)
    monkeypatch.delenv(APPROVAL_KEY_ENV, raising=False)


@tool(audiences=["operator", "customer"])
def order_status(order_id: str) -> str:
    """Look up an order."""
    ran.append(("order_status", {"order_id": order_id}))
    return f"{order_id}: shipped"


@tool
def refund(order_id: str, amount: float) -> str:
    """Refund an order. Operator only."""
    ran.append(("refund", {"order_id": order_id, "amount": amount}))
    return "refunded"


@tool(sensitive=True)
def delete_account(user: str) -> str:
    """Delete a user's account."""
    ran.append(("delete_account", {"user": user}))
    return "deleted"


def _agent(client: ScriptedModelClient, tools: list, **kwargs) -> Agent:
    return Agent(role="r", model={"model": "m"}, model_client=client, tools=tools, **kwargs)


async def _run(agent: Agent, metadata: dict | None = None) -> tuple[str, list[dict], dict]:
    text, events, final = "", [], {}
    async for chunk in agent.run(AgentInput(message="hi", metadata=metadata or {})):
        if TOOL_EVENT_KEY in chunk.metadata:
            events.append(chunk.metadata[TOOL_EVENT_KEY])
        else:
            text += chunk.content or ""
        if chunk.done:
            final = chunk.metadata
    return text, events, final


def test_is_a_model_client():
    assert isinstance(ScriptedModelClient([]), ModelClient)


def test_a_step_must_be_text_or_a_step():
    with pytest.raises(TypeError, match="script step 1: a step is a str or a Step"):
        ScriptedModelClient(["hi", {"name": "order_status"}])


def test_an_unknown_tool_call_key_is_refused_when_the_script_is_built():
    with pytest.raises(TypeError, match="script step 1: unknown tool call key 'args'"):
        ScriptedModelClient(["hi", Step(tool_calls=[{"name": "order_status", "args": {"order_id": "A1"}}])])


def test_a_tool_call_needs_a_name():
    with pytest.raises(TypeError, match="script step 0: a tool call needs a 'name'"):
        ScriptedModelClient([Step(tool_calls=[{"arguments": {}}])])


def test_a_badly_typed_tool_call_is_refused():
    with pytest.raises(TypeError, match="script step 0: invalid tool call"):
        ScriptedModelClient([Step(tool_calls=[{"name": "order_status", "arguments": "A1"}])])


def test_a_step_keeps_its_own_copy_of_the_calls():
    calls = [{"name": "order_status", "arguments": {"order_id": "A1"}}]
    step = Step(tool_calls=calls)
    calls.append({"name": "refund"})
    calls[0]["name"] = "changed"
    assert isinstance(step.tool_calls, tuple)
    assert [tc["name"] for tc in step.tool_calls] == ["order_status"]


async def test_auto_ids_skip_ids_the_script_uses():
    client = ScriptedModelClient([
        Step(tool_calls=[{"name": "order_status", "arguments": {"order_id": "A1"}}, {"id": "call_1", "name": "order_status", "arguments": {"order_id": "B2"}}]),
        Step(tool_calls=[{"name": "order_status", "arguments": {"order_id": "C3"}}, {"id": "call_3", "name": "order_status", "arguments": {"order_id": "D4"}}]),
        "ok",
    ])
    await _run(_agent(client, [order_status]))
    assert [r["tool_call_id"] for r in client.calls[1].tool_results] == ["call_2", "call_1"]
    assert [r["tool_call_id"] for r in client.calls[2].tool_results] == ["call_4", "call_3"]


def test_an_explicit_id_twice_in_one_step_is_refused():
    with pytest.raises(ValueError, match="script step 0: tool call id 'x' is used twice"):
        ScriptedModelClient([Step(tool_calls=[{"id": "x", "name": "a"}, {"id": "x", "name": "b"}])])


async def test_one_tool_call_then_text():
    client = ScriptedModelClient([
        Step(tool_calls=[{"name": "order_status", "arguments": {"order_id": "A1"}}]),
        "Your order shipped.",
    ])
    text, events, _ = await _run(_agent(client, [order_status]))

    assert text == "Your order shipped."
    assert ran == [("order_status", {"order_id": "A1"})]
    assert [e["type"] for e in events] == ["tool_call", "tool_result"]
    assert events[0]["id"] == "call_1"
    assert len(client.calls) == 2
    assert client.calls[0].tool_results == []
    assert client.calls[1].tool_results == [
        {"tool_call_id": "call_1", "name": "order_status", "content": "A1: shipped"}
    ]
    # The model saw its own call in the history before the result.
    assistant = client.calls[1].messages[-2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "order_status"


async def test_two_sequential_tool_calls():
    client = ScriptedModelClient([
        Step(tool_calls=[ToolCall(id="", name="order_status", arguments={"order_id": "A1"})]),
        Step(text="Refunding now.", tool_calls=[{"name": "refund", "arguments": {"order_id": "A1", "amount": 5}}]),
        "Done.",
    ])
    text, _, _ = await _run(_agent(client, [order_status, refund]))

    assert text == "Refunding now.Done."
    assert [name for name, _ in ran] == ["order_status", "refund"]
    assert [r["name"] for r in client.calls[1].tool_results] == ["order_status"]
    assert client.calls[2].tool_results == [{"tool_call_id": "call_2", "name": "refund", "content": "refunded"}]
    # Text sent alongside the calls is kept in the history.
    assert client.calls[2].messages[-2]["content"] == "Refunding now."


async def test_parallel_tool_calls_in_one_step():
    client = ScriptedModelClient([
        Step(tool_calls=[
            {"name": "order_status", "arguments": {"order_id": "A1"}},
            {"id": "mine", "name": "order_status", "arguments": {"order_id": "B2"}},
        ]),
        "Both shipped.",
    ])
    text, events, _ = await _run(_agent(client, [order_status]))

    assert text == "Both shipped."
    assert ran == [("order_status", {"order_id": "A1"}), ("order_status", {"order_id": "B2"})]
    # Every call of the round is reported before the results.
    assert [e["type"] for e in events] == ["tool_call", "tool_call", "tool_result", "tool_result"]
    assert [r["tool_call_id"] for r in client.calls[1].tool_results] == ["call_1", "mine"]
    assert [r["content"] for r in client.calls[1].tool_results] == ["A1: shipped", "B2: shipped"]


async def test_a_customer_is_not_offered_an_operator_only_tool():
    operator = ScriptedModelClient(["ok"])
    await _run(_agent(operator, [order_status, refund]), {CALLER_METADATA_KEY: {"role": "operator", "private": True}})
    assert operator.calls[0].tool_names == ["order_status", "refund"]

    customer = ScriptedModelClient(["ok"])
    await _run(_agent(customer, [order_status, refund]), {CALLER_METADATA_KEY: {"role": "customer"}})
    assert customer.calls[0].tool_names == ["order_status"]


async def test_a_sensitive_tool_does_not_run_and_the_model_sees_it_pending():
    client = ScriptedModelClient([
        Step(tool_calls=[{"name": "delete_account", "arguments": {"user": "u1"}}]),
        "Waiting for approval.",
    ])
    text, events, final = await _run(_agent(client, [delete_account]))

    assert text == "Waiting for approval."
    assert ran == []
    assert events[-1]["status"] == "pending"
    assert client.calls[1].tool_results == [
        {"tool_call_id": "call_1", "name": "delete_account", "content": _PENDING_RESULT}
    ]
    assert final[CONFIRMATION_REQUESTS_KEY] == [{"tool": "delete_account", "arguments": {"user": "u1"}}]


async def test_token_usage_is_reported_per_step_and_summed_by_the_agent():
    steps = [
        Step(tool_calls=[{"name": "order_status", "arguments": {"order_id": "A1"}}], token_usage=TokenUsage(total_tokens=3)),
        Step(text="ok", token_usage=TokenUsage(total_tokens=4)),
    ]
    client = ScriptedModelClient(steps)
    events = []
    await _run(_agent(client, [order_status], audit={"backend": "callback", "callback": events.append}))
    end = next(e for e in events if e.event_type == "invocation.end")
    assert end.data["total_token_usage"]["total_tokens"] == 7

    chunks = [c async for c in ScriptedModelClient(steps).complete([], [])]
    assert chunks[-1].tool_calls and not chunks[-1].done
    assert chunks[-1].token_usage == TokenUsage(total_tokens=3)


async def test_a_text_step_ends_the_answer():
    chunks = [c async for c in ScriptedModelClient([Step(text="ok")]).complete([], [])]
    assert [(c.content, c.done) for c in chunks] == [("ok", False), ("", True)]


async def test_a_tool_that_was_not_offered_is_refused_and_does_not_run():
    client = ScriptedModelClient([
        Step(tool_calls=[{"name": "refund", "arguments": {"order_id": "A1", "amount": 5}}]),
        "Sorry.",
    ])
    await _run(_agent(client, [order_status, refund]), {CALLER_METADATA_KEY: {"role": "customer"}})

    assert client.calls[0].tool_names == ["order_status"]
    assert ran == []
    assert client.calls[1].tool_results[0]["content"].startswith("Error:")
    assert "refund" in client.calls[1].tool_results[0]["content"]


async def test_a_peer_turn_is_refused_a_sensitive_tool():
    client = ScriptedModelClient([Step(tool_calls=[{"name": "delete_account", "arguments": {"user": "u1"}}]), "No."])
    _, _, final = await _run(_agent(client, [delete_account]), {PEER_TURN_KEY: True})

    assert ran == []
    assert client.calls[1].tool_results[0]["content"] == _PEER_REFUSAL
    assert CONFIRMATION_REQUESTS_KEY not in final


async def test_a_thought_signature_round_trips():
    client = ScriptedModelClient([
        Step(tool_calls=[{"name": "order_status", "arguments": {"order_id": "A1"}, "thought_signature": "c2ln"}]),
        "ok",
    ])
    await _run(_agent(client, [order_status]))
    assert client.calls[1].messages[-2]["tool_calls"][0]["thought_signature"] == "c2ln"


async def test_an_approved_call_result_is_the_first_recorded_result(monkeypatch):
    monkeypatch.setenv(APPROVAL_KEY_ENV, "k")
    call = {
        "confirmation_id": "c-1",
        "tool": "delete_account",
        "arguments": {"user": "u1"},
        "args_hash": "h",
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }
    call["signature"] = sign_approved_call(call, "k")
    client = ScriptedModelClient(["It is done."])
    text, _, _ = await _run(_agent(client, [delete_account]), {APPROVED_CALL_KEY: call})

    assert text == "It is done."
    assert ran == [("delete_account", {"user": "u1"})]
    assert client.calls[0].tool_results == [
        {"tool_call_id": "approved-c-1", "name": "delete_account", "content": "deleted"}
    ]


async def test_running_out_of_script_is_a_clear_error():
    client = ScriptedModelClient([Step(tool_calls=[{"name": "order_status", "arguments": {"order_id": "A1"}}])])
    with pytest.raises(RuntimeError, match=r"called 2 times but the script has 1 step\(s\).*tool: 'A1: shipped'"):
        await _run(_agent(client, [order_status]))
