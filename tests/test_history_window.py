"""``max_history_turns``: how much of a session's thread goes to the model.

What is sent is cut on user messages, so a tool result never arrives without
the call that asked for it. What is stored is never cut.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from genfleet.sdk import Agent, AgentInput, AgentOutput
from genfleet.sdk.memory import PlatformMemory
from genfleet.sdk.memory.redis import RedisMemory
from genfleet.sdk.memory.window import (
    MAX_LOAD_MESSAGES,
    MESSAGES_PER_TURN,
    load_recent,
    recent_turns,
)
from genfleet.sdk.schemas import Message, ToolCall


def _turn(n: int, *, tool: bool = False) -> list[Message]:
    """One stored turn: the user's message, optionally a tool round, the reply."""
    out = [Message(role="user", content=f"u{n}")]
    if tool:
        out.append(Message(role="assistant", content="", tool_calls=[ToolCall(id=f"c{n}", name="lookup")]))
        out.append(Message(role="tool", content=f"r{n}", tool_call_id=f"c{n}"))
    out.append(Message(role="assistant", content=f"a{n}"))
    return out


def _thread(turns: int, *, tool: bool = False) -> list[Message]:
    return [m for n in range(1, turns + 1) for m in _turn(n, tool=tool)]


class _Recorder:
    """A model client that answers once and keeps what it was sent."""

    def __init__(self) -> None:
        self.sent: list[list[dict]] = []

    async def complete(self, messages, tools, stream=True):
        self.sent.append(list(messages))
        yield AgentOutput(content="ok", done=True)


def _agent(memory, **kwargs) -> tuple[Agent, _Recorder]:
    agent = Agent(
        role="You are helpful.",
        model={"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
        memory=memory,
        **kwargs,
    )
    recorder = _Recorder()
    agent._provider = MagicMock(complete=recorder.complete)
    return agent, recorder


async def _run(agent: Agent, message: str = "now", session_id: str = "s1", **input_kwargs) -> None:
    async for _ in agent.run(AgentInput(message=message, metadata={"session_id": session_id}, **input_kwargs)):
        pass


def _history_sent(recorder: _Recorder) -> list[dict]:
    """What went to the model between the system prompt and this turn's message."""
    return recorder.sent[0][1:-1]


def _assert_no_orphaned_tool_result(messages: list[dict]) -> None:
    called: set[str] = set()
    for m in messages:
        if m["role"] == "assistant":
            called |= {tc["id"] for tc in m.get("tool_calls", [])}
        elif m["role"] == "tool":
            assert m["tool_call_id"] in called, f"tool result {m['tool_call_id']} sent without its call"


class _SavingMemory:
    """A custom backend with only the whole-thread contract."""

    def __init__(self, thread: list[Message]) -> None:
        self.stored = {"s1": list(thread)}
        self.load_calls: list[tuple] = []

    async def load(self, session_id):
        self.load_calls.append((session_id,))
        return list(self.stored.get(session_id, []))

    async def save(self, session_id, history):
        self.stored[session_id] = list(history)

    async def clear(self, session_id):
        self.stored.pop(session_id, None)


# ---------------------------------------------------------------------------
# recent_turns
# ---------------------------------------------------------------------------

def test_recent_turns_keeps_a_short_history_as_it_is():
    thread = _thread(2, tool=True)
    assert recent_turns(thread, 5) == thread


def test_recent_turns_cuts_on_a_user_message_and_keeps_tool_rounds_whole():
    thread = _thread(5, tool=True)
    window = recent_turns(thread, 2)
    assert window == thread[-8:]
    assert window[0].role == "user" and window[0].content == "u4"


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_without_a_limit_the_whole_thread_goes_to_the_model():
    thread = _thread(30, tool=True)
    agent, recorder = _agent(_SavingMemory(thread))
    await _run(agent)
    assert len(_history_sent(recorder)) == len(thread)


@pytest.mark.asyncio
async def test_a_limit_sends_only_the_last_turns_and_stores_the_whole_thread():
    memory = _SavingMemory(_thread(10))
    agent, recorder = _agent(memory, max_history_turns=3)
    await _run(agent)

    sent = _history_sent(recorder)
    assert [m["content"] for m in sent] == ["u8", "a8", "u9", "a9", "u10", "a10"]
    # The backend saves the thread back whole: nothing it held is lost.
    assert memory.load_calls == [("s1",)]
    assert [m.content for m in memory.stored["s1"]][:2] == ["u1", "a1"]
    assert [m.content for m in memory.stored["s1"]][-2:] == ["now", "ok"]
    assert len(memory.stored["s1"]) == 22


@pytest.mark.asyncio
async def test_a_limit_never_sends_a_tool_result_without_its_call():
    # Turn 9 runs two tool rounds: a message-count cut would land inside it.
    thread = _thread(8, tool=True) + [
        Message(role="user", content="u9"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="x1", name="lookup")]),
        Message(role="tool", content="r", tool_call_id="x1"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="x2", name="lookup")]),
        Message(role="tool", content="r", tool_call_id="x2"),
        Message(role="assistant", content="a9"),
    ]
    agent, recorder = _agent(_SavingMemory(thread), max_history_turns=1)
    await _run(agent)

    sent = _history_sent(recorder)
    assert sent[0] == {"role": "user", "content": "u9"}
    assert len(sent) == 6
    _assert_no_orphaned_tool_result(sent)


@pytest.mark.asyncio
async def test_a_limit_applies_to_a_caller_supplied_history_too():
    agent, recorder = _agent(None, max_history_turns=1)
    await _run(agent, history=_thread(4))
    assert [m["content"] for m in _history_sent(recorder)] == ["u4", "a4"]


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "3"])
def test_a_limit_that_is_not_a_positive_int_is_refused(value):
    with pytest.raises(ValueError, match="max_history_turns"):
        _agent(None, max_history_turns=value)


@pytest.mark.asyncio
async def test_redis_memory_is_cut_before_the_model_and_saved_whole():
    class _FakeRedis:
        def __init__(self):
            self.data: dict[str, str] = {}

        async def get(self, key):
            return self.data.get(key)

        async def set(self, key, value, ex=None):
            self.data[key] = value

        async def delete(self, key):
            self.data.pop(key, None)

    memory = RedisMemory({"type": "redis", "connection": "redis://localhost:6379/0"})
    memory._redis = _FakeRedis()
    await memory.save("s1", _thread(6, tool=True))
    agent, recorder = _agent(memory, max_history_turns=2)
    await _run(agent)

    sent = _history_sent(recorder)
    assert sent[0]["content"] == "u5" and len(sent) == 8
    _assert_no_orphaned_tool_result(sent)
    assert len(await memory.load("s1")) == 6 * 4 + 2


@pytest.mark.asyncio
async def test_an_appending_backend_without_limit_is_loaded_whole_and_cut():
    class _Appending(_SavingMemory):
        def __init__(self, thread):
            super().__init__(thread)
            self.appended: list[list[Message]] = []

        async def append(self, session_id, messages, *, subject=None, metadata=None, turn_id=None):
            self.appended.append(messages)

    memory = _Appending(_thread(5))
    agent, recorder = _agent(memory, max_history_turns=2)
    await _run(agent)
    assert memory.load_calls == [("s1",)]
    assert [m["content"] for m in _history_sent(recorder)] == ["u4", "a4", "u5", "a5"]
    assert [m.content for m in memory.appended[0]] == ["now", "ok"]


# ---------------------------------------------------------------------------
# A store that reads the newest messages (WindowedMemory)
# ---------------------------------------------------------------------------

def _drop_leading_tool_results(page: list) -> list:
    """What the platform store does to a page cut inside a tool round."""
    while page and (page[0]["role"] if isinstance(page[0], dict) else page[0].role) == "tool":
        page = page[1:]
    return page


class _WindowedStore:
    """Answers the newest ``limit`` messages; optionally drops leading tool results."""

    def __init__(self, thread: list[Message], *, drops_orphans: bool = True) -> None:
        self.thread = thread
        self.drops_orphans = drops_orphans
        self.limits: list[int] = []
        self.whole_loads = 0
        self.appended: list[list[Message]] = []

    async def load(self, session_id):
        self.whole_loads += 1
        return list(self.thread)

    async def load_newest(self, session_id, limit):
        self.limits.append(limit)
        page = self.thread[-limit:]
        return _drop_leading_tool_results(page) if self.drops_orphans else page

    async def save(self, session_id, history):
        raise AssertionError("an appending store is never saved whole")

    async def clear(self, session_id):
        pass

    async def append(self, session_id, messages, *, subject=None, metadata=None, turn_id=None):
        self.appended.append(messages)


def _tool_heavy_turn(n: int, rounds: int) -> list[Message]:
    """A user message and ``rounds`` call/result pairs, with no final reply:
    the turn ends on a tool result, as one does when the model answers empty."""
    out = [Message(role="user", content=f"u{n}")]
    for k in range(rounds):
        out.append(Message(role="assistant", content="", tool_calls=[ToolCall(id=f"c{n}-{k}", name="t")]))
        out.append(Message(role="tool", content="r", tool_call_id=f"c{n}-{k}"))
    return out


def _dicts(messages: list[Message]) -> list[dict]:
    return [Agent._message_to_dict(m) for m in messages]


@pytest.mark.asyncio
async def test_load_recent_reads_one_bounded_page():
    store = _WindowedStore(_thread(50, tool=True))
    window = await load_recent(store, "s1", 3)
    assert store.limits == [3 * MESSAGES_PER_TURN]
    assert window[0].content == "u48" and len(window) == 12


@pytest.mark.asyncio
async def test_a_huge_limit_asks_for_at_most_the_largest_page():
    store = _WindowedStore(_thread(3))
    await load_recent(store, "s1", 10_000)
    assert store.limits == [MAX_LOAD_MESSAGES]


@pytest.mark.parametrize("drops_orphans", [True, False])
@pytest.mark.asyncio
async def test_a_cut_page_with_exactly_n_user_messages_starts_on_the_user_message(drops_orphans):
    # 19-message turns ending on a tool result; a 20-message page is one tool
    # result of the previous turn, then the last turn whole.
    thread = [m for n in range(1, 4) for m in _tool_heavy_turn(n, 9)]
    store = _WindowedStore(thread, drops_orphans=drops_orphans)
    window = await load_recent(store, "s1", 1)
    assert store.limits == [20]
    assert window[0].role == "user" and window[0].content == "u3"
    assert len(window) == 19
    _assert_no_orphaned_tool_result(_dicts(window))


@pytest.mark.asyncio
async def test_turns_longer_than_the_allowance_send_fewer_whole_turns():
    thread = [m for n in range(1, 6) for m in _tool_heavy_turn(n, 15)]  # 31 messages each
    store = _WindowedStore(thread, drops_orphans=False)
    window = await load_recent(store, "s1", 2)
    assert store.limits == [40]
    assert [m.content for m in window if m.role == "user"] == ["u5"]
    _assert_no_orphaned_tool_result(_dicts(window))


@pytest.mark.asyncio
async def test_a_thread_whose_visible_start_is_mid_turn_is_read_once():
    # The older part has expired: what is left starts inside a tool round.
    tail = [
        Message(role="tool", content="r", tool_call_id="gone"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="c0", name="t")]),
        Message(role="tool", content="r", tool_call_id="c0"),
        Message(role="assistant", content="a0"),
    ]
    store = _WindowedStore(tail + _thread(2), drops_orphans=False)
    window = await load_recent(store, "s1", 5)
    assert store.limits == [5 * MESSAGES_PER_TURN]
    assert [m.content for m in window] == ["u1", "a1", "u2", "a2"]


@pytest.mark.asyncio
async def test_a_thread_with_no_user_message_sends_nothing_on_both_paths():
    no_user = [Message(role="assistant", content="hello"), Message(role="assistant", content="anyone?")]
    store = _WindowedStore(no_user)
    assert await load_recent(store, "s1", 3) == []
    assert store.limits == [3 * MESSAGES_PER_TURN]
    assert recent_turns(no_user, 3) == []

    agent, recorder = _agent(None, max_history_turns=3)
    await _run(agent, history=no_user)
    assert _history_sent(recorder) == []


@pytest.mark.asyncio
async def test_the_agent_reads_a_windowed_appending_store_by_its_newest_messages():
    store = _WindowedStore([m for n in range(1, 4) for m in _tool_heavy_turn(n, 9)], drops_orphans=False)
    agent, recorder = _agent(store, max_history_turns=1)
    await _run(agent)
    assert store.limits == [20] and store.whole_loads == 0
    sent = _history_sent(recorder)
    assert sent[0] == {"role": "user", "content": "u3"}
    _assert_no_orphaned_tool_result(sent)
    assert [m.content for m in store.appended[0]] == ["now", "ok"]


@pytest.mark.asyncio
async def test_a_windowed_store_that_only_saves_is_loaded_whole():
    class _SaveOnly(_SavingMemory):
        async def load_newest(self, session_id, limit):
            raise AssertionError("a save-only store must be loaded whole")

    memory = _SaveOnly(_thread(5))
    agent, _ = _agent(memory, max_history_turns=2)
    await _run(agent)
    assert memory.load_calls == [("s1",)]
    assert len(memory.stored["s1"]) == 12


@pytest.mark.asyncio
async def test_a_caller_supplied_history_skips_the_stored_thread():
    store = _WindowedStore(_thread(5))
    agent, recorder = _agent(store, max_history_turns=2)
    await _run(agent, history=_thread(1))
    assert store.limits == [] and store.whole_loads == 0
    assert [m["content"] for m in _history_sent(recorder)] == ["u1", "a1"]


# ---------------------------------------------------------------------------
# The platform store, over HTTP
# ---------------------------------------------------------------------------

class _Proxy(BaseHTTPRequestHandler):
    calls: ClassVar[list[tuple[str, dict]]] = []
    thread: ClassVar[list[dict]] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        _Proxy.calls.append((self.path, body))
        payload: dict = {}
        if self.path.endswith("/episodic/load"):
            limit = body.get("limit")
            page = _Proxy.thread[-limit:] if limit else _Proxy.thread
            payload = {"messages": _drop_leading_tool_results(page)}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def proxy():
    _Proxy.calls = []
    _Proxy.thread = [m.model_dump(exclude_none=True) for m in _thread(40, tool=True)]
    server = HTTPServer(("127.0.0.1", 0), _Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/memory"
    server.shutdown()


@pytest.mark.asyncio
async def test_the_platform_store_is_asked_for_a_limited_page_and_the_turn_is_appended(proxy):
    agent, recorder = _agent(PlatformMemory(proxy, "t"), max_history_turns=2)
    await _run(agent, session_id="whatsapp:1")

    (load_path, load_body), (append_path, append_body) = _Proxy.calls
    assert load_path == "/memory/episodic/load"
    assert load_body == {"session_id": "whatsapp:1", "limit": 40}
    sent = _history_sent(recorder)
    assert sent[0]["content"] == "u39" and len(sent) == 8
    _assert_no_orphaned_tool_result(sent)
    # Appending is untouched: this turn only, whatever was sent.
    assert append_path == "/memory/episodic/append"
    assert [m["content"] for m in append_body["messages"]] == ["now", "ok"]


@pytest.mark.asyncio
async def test_a_platform_page_cut_inside_a_tool_round_starts_on_a_user_message(proxy):
    _Proxy.thread = [m.model_dump(exclude_none=True) for n in range(1, 4) for m in _tool_heavy_turn(n, 9)]
    agent, recorder = _agent(PlatformMemory(proxy, "t"), max_history_turns=1)
    await _run(agent)
    assert _Proxy.calls[0][1] == {"session_id": "s1", "limit": 20}
    sent = _history_sent(recorder)
    assert sent[0] == {"role": "user", "content": "u3"} and len(sent) == 19
    _assert_no_orphaned_tool_result(sent)


@pytest.mark.asyncio
async def test_without_a_limit_the_platform_store_is_loaded_whole(proxy):
    agent, recorder = _agent(PlatformMemory(proxy, "t"))
    await _run(agent)
    assert _Proxy.calls[0][1] == {"session_id": "s1"}
    assert len(_history_sent(recorder)) == 160
