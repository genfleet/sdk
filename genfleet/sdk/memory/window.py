"""The recent end of a thread, counted in turns.

A turn is one user message and everything after it up to the next user
message: the assistant's tool calls, their results, and its reply. Cutting on
a user message never separates a tool result from the call that asked for
it, which every provider rejects.
"""

from __future__ import annotations

from ..schemas import Message
from .base import WindowedMemory

#: Messages allowed per wanted turn when a store is asked for the newest
#: messages: the user message, a few tool rounds (a call and its result
#: each) and the reply. A turn longer than this leaves fewer whole turns in
#: the page, never a broken one.
MESSAGES_PER_TURN = 20

#: The most messages one load asks a store for.
MAX_LOAD_MESSAGES = 2000


def recent_turns(messages: list[Message], turns: int) -> list[Message]:
    """The last ``turns`` whole turns of ``messages``, starting on a user message.

    Anything before the first user message is dropped: it is the tail of a
    turn whose start is missing (a page that was cut, or a thread whose
    older part has expired), and sending it could orphan a tool result. A
    thread with no user message therefore gives ``[]``.
    """
    starts = [i for i, m in enumerate(messages) if m.role == "user"]
    if not starts:
        return []
    return messages[starts[max(len(starts) - turns, 0)]:]


async def load_recent(memory: WindowedMemory, session_id: str, turns: int) -> list[Message]:
    """The last ``turns`` turns of a thread, read in one bounded load.

    The store counts messages, not turns, so it is asked for
    ``turns * MESSAGES_PER_TURN`` of them (at most ``MAX_LOAD_MESSAGES``).
    The page may begin mid-turn, so it is cut to whole turns. When the last
    turns are longer than the allowance, fewer than ``turns`` come back.
    """
    budget = min(turns * MESSAGES_PER_TURN, MAX_LOAD_MESSAGES)
    return recent_turns(await memory.load_newest(session_id, budget), turns)
