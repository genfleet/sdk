"""ScriptedModelClient drives the Agent's real tool loop. Tier 1, no network."""

from __future__ import annotations

import pytest

from genfleet.sdk import (
    CONFIRMATION_REQUESTS_KEY,
    TOOL_EVENT_KEY,
    Agent,
    AgentInput,
    TokenUsage,
    ToolCall,
    tool,
)
from genfleet.sdk.caller import CALLER_METADATA_KEY, HOSTED_ENV
from genfleet.sdk.confirmations import _PENDING_RESULT
from genfleet.sdk.models import ModelClient, ScriptedModelClient, Step

ran: list[tuple[str, dict]] = []


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ran.clear()
    monkeypatch.delenv(HOSTED_ENV, raising=False)


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


def _agent(client: ScriptedModelClient, tools: list) -> Agent:
    return Agent(role="r", model={"model": "m"}, model_client=client, tools=tools)


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
    with pytest.raises(TypeError, match="str or a Step"):
        ScriptedModelClient([{"name": "order_status"}])


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


async def test_token_usage_is_reported_per_step():
    client = ScriptedModelClient([
        Step(tool_calls=[{"name": "order_status", "arguments": {"order_id": "A1"}}], token_usage=TokenUsage(total_tokens=3)),
        Step(text="ok", token_usage=TokenUsage(total_tokens=4)),
    ])
    chunks = [c async for c in client.complete([], [])]
    assert chunks[-1].tool_calls and not chunks[-1].done
    assert chunks[-1].token_usage == TokenUsage(total_tokens=3)
    chunks = [c async for c in client.complete([], [])]
    assert [(c.content, c.done) for c in chunks] == [("ok", False), ("", True)]
    assert chunks[-1].token_usage == TokenUsage(total_tokens=4)


async def test_running_out_of_script_is_a_clear_error():
    client = ScriptedModelClient([Step(tool_calls=[{"name": "order_status", "arguments": {"order_id": "A1"}}])])
    with pytest.raises(RuntimeError, match=r"called 2 times but the script has 1 step\(s\).*tool: 'A1: shipped'"):
        await _run(_agent(client, [order_status]))
