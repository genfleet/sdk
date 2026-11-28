"""Tool calls a model wrote as plain text instead of as structured calls.

Some models, asked to call a tool, answer with the call written out as text,
for example ``[{"name": "lookup", "arguments": {"order": "A1"}}]``. Sent as
the reply, that text would reach the person the agent is talking to. `Agent`
asks :func:`tool_calls_in_text` whether a reply with no structured calls is
one of these, and if it is, runs the calls exactly as if the model had made
them as structured calls.

The rule is strict, because a false positive runs a tool the model was only
quoting. The whole reply, after trimming whitespace and at most one code
fence around it, must be JSON that is either one call object or a non-empty
list of call objects. A call object has exactly two keys:

- ``"name"``: a string naming a tool offered to the model on this turn;
- ``"arguments"``: an object, or a string that is the JSON of an object.

Anything else is left as text: prose around the JSON, a key other than these
two (``"tool"``/``"args"``, ``"parameters"``, an ``"id"``...), a duplicated
key, a tool that was not offered, an empty list, or malformed JSON.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from .schemas import ToolCall

__all__ = ["could_be_tool_call_text", "tool_calls_in_text"]

#: One code fence around the whole reply, with an optional language tag.
_FENCE = re.compile(r"\A```[A-Za-z0-9_+-]*[ \t]*\n?(.*?)\n?[ \t]*```\Z", re.DOTALL)
_CALL_KEYS = frozenset({"name", "arguments"})


def could_be_tool_call_text(text: str) -> bool:
    """Whether a reply starting with ``text`` may turn out to be a call written
    as text: its first non-blank character opens JSON or a code fence."""
    start = text.lstrip()[:1]
    return start in ("{", "[", "`")


def tool_calls_in_text(text: str, offered: set[str]) -> list[ToolCall] | None:
    """The calls ``text`` consists of, or ``None`` when it is not only calls
    to tools in ``offered`` (see the module doc for the exact rule)."""
    body = text.strip()
    if fenced := _FENCE.match(body):
        body = fenced.group(1).strip()
        if "```" in body:
            return None
    if body[:1] not in ("{", "["):
        return None
    try:
        parsed = json.loads(body, object_pairs_hook=_refuse_duplicate_keys)
    except (ValueError, RecursionError):
        return None
    items = parsed if isinstance(parsed, list) else [parsed]
    if not items:
        return None
    calls: list[ToolCall] = []
    for item in items:
        if not isinstance(item, dict) or item.keys() != _CALL_KEYS:
            return None
        name, arguments = item["name"], _arguments(item["arguments"])
        if not isinstance(name, str) or name not in offered or arguments is None:
            return None
        calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:24]}", name=name, arguments=arguments))
    return calls


def _arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value, object_pairs_hook=_refuse_duplicate_keys)
        except (ValueError, RecursionError):
            return None
    return value if isinstance(value, dict) else None


def _refuse_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate key")
    return result
