"""A turn saved to memory fits the platform's request cap, and a memory that
fails does not cost the person their reply."""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from genfleet.sdk import Agent, AgentInput
from genfleet.sdk.agent import MEMORY_TEXT_BUDGET, MEMORY_TOOL_RESULT_BUDGET, fit_texts
from genfleet.sdk.schemas import AgentOutput, ToolCall
from genfleet.sdk.storage_limit import StorageLimitExceeded

PLATFORM_CAP = 32 * 1024
BUDGET = MEMORY_TOOL_RESULT_BUDGET


def _size(texts: list[str]) -> int:
    """The bytes the texts take inside a JSON request, quotes not counted."""
    return sum(len(json.dumps(text)) - 2 for text in texts)


def test_texts_that_fit_are_unchanged():
    texts = ["a" * 100, "", '{"ok": true}']
    assert fit_texts(texts, BUDGET) == texts


def test_a_large_text_is_cut_and_says_how_much_was_kept():
    big = "x" * 100_000
    (fitted,) = fit_texts([big], BUDGET)
    assert _size([fitted]) <= BUDGET
    kept, _, marker = fitted.partition("\n")
    assert kept == "x" * len(kept) and len(kept) > 19_000
    assert marker == f"[cut for memory: first {len(kept)} of 100000 characters kept]"


def test_a_small_text_is_not_cut_to_make_room_for_a_large_one():
    small, big = '{"status": "ok"}', "y" * 100_000
    fitted = fit_texts([big, small], BUDGET)
    assert fitted[1] == small
    assert fitted[0] != big
    assert _size(fitted) <= BUDGET


def test_several_large_texts_share_the_budget():
    fitted = fit_texts(["a" * 50_000, "b" * 60_000, "c" * 70_000], BUDGET)
    assert _size(fitted) <= BUDGET
    assert all(len(f) > 5_000 for f in fitted)


def test_text_that_grows_when_encoded_still_fits():
    # A non-ASCII character takes six bytes in the request; a quote takes two.
    assert _size(fit_texts(["\u0645" * 30_000, '"' * 30_000], BUDGET)) <= BUDGET


def test_many_small_texts_still_fit_and_none_grows():
    for count, length in ((1000, 100), (400, 60), (5000, 30)):
        texts = ["r" * length] * count
        fitted = fit_texts(texts, BUDGET)
        assert _size(fitted) <= BUDGET
        assert all(len(f) <= length for f in fitted)


def test_with_no_room_for_the_full_marker_the_short_one_is_used():
    (fitted,) = fit_texts(["z" * 500], 10)
    assert fitted == "zzzzz[cut]"
    assert fit_texts(["z" * 500], 3) == [""]


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


async def test_a_turn_with_a_large_message_is_saved_within_the_cap():
    def ping() -> str:
        return "pong"

    memory = _Appendable()
    agent = _agent_calling(ping, memory)

    async for _ in agent.run(AgentInput(message="\u0645" * 20_000, metadata={"session_id": "s1"})):
        pass

    (messages,) = memory.appended
    body = json.dumps({"session_id": "s1", "messages": [m.model_dump(exclude_none=True) for m in messages]})
    assert len(body.encode()) < PLATFORM_CAP
    assert _size([m.content or "" for m in messages if m.role != "tool"]) <= MEMORY_TEXT_BUDGET
    assert messages[2].content == "pong"
    assert messages[3].content == "Here you go."


async def test_a_workspace_at_its_storage_limit_still_fails_the_turn():
    def ping() -> str:
        return "pong"

    agent = _agent_calling(ping, _Appendable(fail=StorageLimitExceeded("episodic/append")))

    with pytest.raises(StorageLimitExceeded):
        async for _ in agent.run(AgentInput(message="ping", metadata={"session_id": "s1"})):
            pass


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
