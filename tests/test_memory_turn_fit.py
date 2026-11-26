"""A turn saved to memory fits the platform's request cap, and a memory that
fails does not cost the person their reply."""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

from genfleet.sdk import Agent, AgentInput
from genfleet.sdk.agent import MEMORY_TOOL_RESULT_BUDGET, fit_tool_results
from genfleet.sdk.schemas import AgentOutput, ToolCall

PLATFORM_CAP = 32 * 1024


def _json_size(text: str) -> int:
    return len(json.dumps(text))


def test_results_that_fit_are_unchanged():
    results = ["a" * 100, "", '{"ok": true}']
    assert fit_tool_results(results) == results


def test_a_large_result_is_cut_and_says_so():
    big = "x" * 100_000
    (fitted,) = fit_tool_results([big])
    assert _json_size(fitted) <= MEMORY_TOOL_RESULT_BUDGET
    assert fitted.startswith("x" * 1000)
    assert "of 100000 characters kept]" in fitted


def test_a_small_result_is_not_cut_to_make_room_for_a_large_one():
    small, big = '{"status": "ok"}', "y" * 100_000
    fitted = fit_tool_results([big, small])
    assert fitted[1] == small
    assert fitted[0] != big
    assert sum(_json_size(f) for f in fitted) <= MEMORY_TOOL_RESULT_BUDGET


def test_several_large_results_share_the_budget():
    fitted = fit_tool_results(["a" * 50_000, "b" * 60_000, "c" * 70_000])
    assert sum(_json_size(f) for f in fitted) <= MEMORY_TOOL_RESULT_BUDGET
    assert all(len(f) > 5_000 for f in fitted)


def test_text_that_grows_when_encoded_still_fits():
    # A non-ASCII character takes six bytes in the request; a quote takes two.
    fitted = fit_tool_results(["م" * 30_000, '"' * 30_000])
    assert sum(_json_size(f) for f in fitted) <= MEMORY_TOOL_RESULT_BUDGET


def test_a_budget_smaller_than_the_marker_keeps_only_the_marker():
    (fitted,) = fit_tool_results(["z" * 500], budget=10)
    assert fitted.startswith("\n[tool result cut for memory: first 0 of 500")


class _Appendable:
    def __init__(self, fail: Exception | None = None):
        self.appended: list = []
        self.fail = fail

    async def load(self, session_id):
        return []

    async def save(self, session_id, history):
        raise AssertionError("an appendable memory is appended to")

    async def clear(self, session_id):
        pass

    async def append(self, session_id, messages, *, subject=None, metadata=None, turn_id=None):
        if self.fail:
            raise self.fail
        self.appended.append(messages)


def _agent_calling(tool, memory) -> Agent:
    agent = Agent(role="You are helpful.", model={"model": "openai/gpt-4o-mini", "api_key": "sk-test"}, tools=[tool])
    rounds = iter([
        [AgentOutput(tool_calls=[ToolCall(id="tc1", name=tool.__name__, arguments={})], done=False)],
        [AgentOutput(content="Here you go.", done=False), AgentOutput(content="", done=True)],
    ])

    async def _complete(messages, tools, stream=True):
        for chunk in next(rounds):
            yield chunk

    agent._provider = MagicMock()
    agent._provider.complete = _complete
    agent._memory = memory
    return agent


async def test_a_turn_with_a_large_tool_result_is_saved_within_the_cap():
    def list_products() -> str:
        return json.dumps({"products": ["p" * 40] * 2000})  # about 86 KB

    memory = _Appendable()
    agent = _agent_calling(list_products, memory)

    chunks = [c async for c in agent.run(AgentInput(message="list them", metadata={"session_id": "s1"}))]

    assert "Here you go." in "".join(c.content or "" for c in chunks)
    (messages,) = memory.appended
    body = json.dumps({"session_id": "s1", "messages": [m.model_dump(exclude_none=True) for m in messages]})
    assert len(body.encode()) < PLATFORM_CAP
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert "characters kept]" in messages[2].content
    assert messages[2].tool_call_id == "tc1"


async def test_a_memory_that_fails_does_not_fail_the_turn(caplog):
    def ping() -> str:
        return "pong"

    agent = _agent_calling(ping, _Appendable(fail=RuntimeError("the customer wrote: secret words")))

    with caplog.at_level(logging.ERROR, logger="genfleet.sdk.agent"):
        chunks = [c async for c in agent.run(AgentInput(message="ping", metadata={"session_id": "s1"}))]

    assert "Here you go." in "".join(c.content or "" for c in chunks)
    assert chunks[-1].done
    assert "not saved to memory (RuntimeError)" in caplog.text
    assert "secret words" not in caplog.text
