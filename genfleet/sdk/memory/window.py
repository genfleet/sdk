"""The recent end of a thread, counted in turns.

A turn is one user message and everything after it up to the next user
message: the assistant's tool calls, their results, and its reply. Cutting on
a user message never separates a tool result from the call that asked for
it, which every provider rejects.
"""

from __future__ import annotations

import inspect

from ..schemas import Message
from .base import Memory

#: First guess at how many messages a turn takes (user, a few tool rounds,
#: the reply). Too small only costs another load; it never loses a turn.
_MESSAGES_PER_TURN = 10

#: The most messages one load asks a store for. A store may cap a page lower;
#: a thread whose last turns hold more than this is cut at a user message.
MAX_LOAD_MESSAGES = 2000


def recent_turns(messages: list[Message], turns: int) -> list[Message]:
    """The last ``turns`` turns of ``messages``; all of them when there are fewer.

    Anything before the oldest kept user message goes, so the window always
    starts on a user message once it cuts. A history with no more than
    ``turns`` user messages comes back unchanged.
    """
    starts = [i for i, m in enumerate(messages) if m.role == "user"]
    if len(starts) <= turns:
        return messages
    return messages[starts[-turns]:]


def load_takes_limit(memory: Memory) -> bool:
    """Whether ``memory.load`` accepts a ``limit`` keyword (the newest N messages)."""
    try:
        param = inspect.signature(memory.load).parameters.get("limit")
    except (TypeError, ValueError):
        return False
    return param is not None and param.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


async def load_recent(memory: Memory, session_id: str, turns: int) -> list[Message]:
    """The last ``turns`` turns of a thread, asking the store for no more than it needs.

    ``memory.load`` must accept ``limit``: the newest ``limit`` messages,
    oldest first. A store counts messages, not turns, so the first page is a
    guess; while it holds too few user messages and the thread clearly goes
    on before it, the page grows, up to :data:`MAX_LOAD_MESSAGES`.
    """
    budget = min(turns * _MESSAGES_PER_TURN, MAX_LOAD_MESSAGES)
    while True:
        loaded = await memory.load(session_id, limit=budget)
        users = sum(1 for m in loaded if m.role == "user")
        # A page that starts on a user message and came back short is the
        # whole thread. Anything else was cut by the limit.
        cut = len(loaded) >= budget or bool(loaded) and loaded[0].role != "user"
        if users >= turns or not cut:
            return recent_turns(loaded, turns)
        if budget >= MAX_LOAD_MESSAGES:
            # Fewer turns than asked fit in the largest page: keep the whole
            # turns it holds, from its first user message on.
            first = next((i for i, m in enumerate(loaded) if m.role == "user"), len(loaded))
            return loaded[first:]
        budget = min(budget * 4, MAX_LOAD_MESSAGES)
