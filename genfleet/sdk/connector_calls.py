"""Connector tools: your workspace's verified API tools, and a channel's ``reply``.

A workspace owner or admin can define **tools** on a Custom API connection in
the dashboard: a name, a description, the endpoint and its inputs. Each one
is tested and verified there before any agent can use it. The agent is given
the verified tools of every connection the workspace lets it use, and of
each channel it is connected to, and hands the ones it wants to its model::

    from genfleet.sdk import Agent, connector_tools, reply_tool

    def build_agent(remote_tools=()):
        tools = [*connector_tools(), *remote_tools]
        reply = reply_tool()        # None unless the agent is on a channel
        if reply is not None:
            tools.append(reply)
        return Agent(role=ROLE, model=MODEL, tools=tools)

Nothing is added to an agent's tools unless its code adds it.

**Names.** A connection's tool reaches the model as ``<connection>__<tool>``
(``erp__create_lead``), and a channel's as its bare name (``reply``). The
name is used exactly as the platform lists it, and it is also the name the
workspace's per-tool audience and approval settings use.

**What the platform does.** The agent sends the tool's name and the model's
arguments. The platform checks that the tool is still the verified one, that
the person on this turn may use it, and that the arguments fit; then it
makes the call with the connection's key. The agent never sees the key, the
endpoint or the method.

**What the model reads.** Every answer is text; the tool never raises.

- An API answer: a JSON object with ``tool``, ``status``, ``ok``,
  ``contentType``, ``body`` (cut to the tool's size limit), ``truncated``,
  ``bytes``, ``bodyWithheld`` and ``note``.
- A channel send: ``sent``, and whether a quote was kept.
- A refusal: ``"<code>: <message>"``, the platform's message unchanged
  (:data:`REFUSAL_CODES`). Nothing was done.
- An unknown outcome (:data:`UNKNOWN_OUTCOME_CODES`, an API that did not
  answer in time or could not be read, or no answer at all): the call may
  have completed, so the model is told not to call the tool
  again in this turn and to tell the person what it tried to do.

**Retries.** Only ``backend_unreachable`` (the platform says the call never
left it) is retried, twice, after a short wait. Nothing else is retried:
any other failure may come after the call ran.

Like :func:`genfleet.sdk.send_message`, these need a platform sandbox and a
running turn: outside one a call answers ``not_in_sandbox``, and outside a
turn ``no_turn``, without sending anything. Arguments, bodies and answers are
never logged.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .caller import EVERYONE, OPERATOR_ONLY, Audience
from .channels import CHANNELS_TOKEN_ENV, CHANNELS_URL_ENV, ChannelSendError
from .config import ConnectorToolSpec, agent_config
from .schemas import ToolSchema
from .tool import ToolWrapper
from .turn import current_turn

__all__ = [
    "LOCAL_CODES",
    "REFUSAL_CODES",
    "RETRYABLE_CODES",
    "UNKNOWN_OUTCOME_CODES",
    "ConnectorToolError",
    "ReplyResult",
    "connector_tools",
    "reply",
    "reply_tool",
]

#: The codes the platform refuses a connector tool call with. Nothing was
#: done, and the message (passed to the model unchanged) says why.
REFUSAL_CODES: frozenset[str] = frozenset({
    # The request and the agent.
    "invalid_request",       # the call could not be read as sent
    "agent_not_allowed",     # this agent may not use the workspace's connections
    "turn_not_running",      # the turn the call names has ended
    "engine_rate_limited",   # too many tool calls from this agent; wait a moment
    "rejected",              # refused, with no more detail
    # The connection and the tool.
    "unknown_connection",    # no such connection in the workspace
    "connection_not_allowed",  # the workspace has not let this agent use it
    "connection_changed",    # its settings or key are being changed; try again shortly
    "unknown_tool",          # no such tool on the connection
    "tool_not_verified",     # the tool was edited and is not verified yet
    "tool_changed",          # the tool changed since the agent started
    "tool_not_allowed",      # not offered to the person on this turn
    "invalid_arguments",     # the arguments do not fit the tool's inputs
    "approval_required",     # it needs an owner's or admin's approval first
    "rate_limited",          # the connection's call limit is reached
    "host_refused",          # the connection's host is not a public address
    # Channel sends (``reply``, ``send_message``).
    "no_binding",            # no channel connection to send on
    "no_conversation",       # this turn is not answering a channel conversation
    "outside_window",        # the contact has not written in 24 hours
    "cold_send_disabled",    # the contact never wrote, and first contact is off
    "send_rejected",         # the channel refused the send
    "tool_call_text",        # the text looks like a tool call, not a message
    "unknown_recipient",     # (staff alerts) nobody by that name
    "already_escalated",     # (staff alerts) already alerted
})

#: The call was sent and its result is not known: it may have completed.
#: Never retried; the model is told not to call the tool again in this turn.
UNKNOWN_OUTCOME_CODES: frozenset[str] = frozenset({"backend_timeout", "backend_error"})

#: The call never left the platform: it did not run. The only code retried.
RETRYABLE_CODES: frozenset[str] = frozenset({"backend_unreachable"})

#: Answered by the SDK itself; nothing was sent.
LOCAL_CODES: frozenset[str] = frozenset({
    "not_in_sandbox",   # not running on the platform
    "no_turn",          # called outside a running turn
    "tool_not_listed",  # the platform does not offer this tool to the agent
})

#: Longer than the platform's own 30-second limit for one call.
_TIMEOUT_S = 35.0
#: Waits before each retry of ``backend_unreachable``.
_RETRY_DELAYS_S = (0.5, 1.0)
#: The turn reference the platform accepts: 1 to 256 characters each.
_MAX_TURN_REF = 256
_ROUTE = "connector-tool-call"
#: The answer fields of an API call, in the order the model reads them.
_RESULT_FIELDS = ("tool", "status", "ok", "contentType", "body", "truncated", "bytes", "bodyWithheld", "note")
_REPLY = "reply"


class ConnectorToolError(ChannelSendError):
    """A connector tool call the platform refused, or whose result is unknown.

    ``code`` is one of :data:`REFUSAL_CODES` (nothing was done),
    :data:`UNKNOWN_OUTCOME_CODES` (it may have been done),
    ``backend_unreachable`` (it was not done; already retried), one of
    :data:`LOCAL_CODES`, or None when no readable answer came. ``tool`` is the
    tool's model-facing name. ``str(error)`` is the platform's message.
    """

    def __init__(self, message: str, *, status: int, code: str | None = None, tool: str | None = None) -> None:
        super().__init__(message, status=status, code=code)
        self.tool = tool

    @property
    def outcome_unknown(self) -> bool:
        """True when the call may have completed: do not repeat it."""
        if self.code in UNKNOWN_OUTCOME_CODES:
            return True
        return self.code is None and not 400 <= self.status < 500

    @property
    def delivery_unknown(self) -> bool:
        return self.outcome_unknown

    @property
    def retryable(self) -> bool:
        """True only when the platform says the call never left it."""
        return self.code in RETRYABLE_CODES


@dataclass(frozen=True)
class ReplyResult:
    """A message sent into the conversation: its ids, and whether the quote was kept."""

    message_ids: tuple[str, ...] = ()
    #: None when no quote was asked for; False when the channel sends without one.
    quoted: bool | None = None


# --- The tools ---------------------------------------------------------------


def connector_tools(slot: str | None = None) -> list[ToolWrapper]:
    """The verified tools of the agent's connections, ready for ``Agent(tools=...)``.

    With no ``slot``, every connection the workspace lets the agent use; with
    one, only the connection filling that ``[connectors.<slot>]`` slot (an
    unfilled slot has none). Empty when the platform lists no tools (a local
    run, or an older platform). Each tool has the listed name, description
    and inputs, is offered to operators unless the workspace's setting says
    otherwise, and answers the model in text (see the module docs).
    """
    config = agent_config()
    found: list[tuple[str, ConnectorToolSpec]] = []
    if slot is None:
        for connection, grant in config.connections.items():
            found.extend((connection, spec) for spec in grant.tools)
        for binding in config.connectors.values():
            found.extend((binding.connection, spec) for spec in binding.tools)
    else:
        filled = config.connectors.get(slot)
        if filled is not None:
            found.extend((filled.connection, spec) for spec in filled.tools)
    tools: dict[str, ToolWrapper] = {}
    for connection, spec in found:
        if spec.name not in tools:
            tools[spec.name] = _as_tool(spec, connection, OPERATOR_ONLY)
    return list(tools.values())


def reply_tool() -> ToolWrapper | None:
    """The model-facing ``reply``, or None when the platform does not offer it.

    ``reply`` is offered when the agent is connected to a channel. It sends a
    message into the conversation the turn is answering, now, before the
    turn ends (a "one moment, checking" or a quote of one of the person's
    messages). The turn's answer is still sent at the end. It is offered to
    operators and customers unless the workspace's setting says otherwise.
    """
    spec = _reply_spec()
    return _as_tool(spec, None, EVERYONE) if spec is not None else None


async def reply(text: str, quote: str | None = None) -> ReplyResult:
    """Send ``text`` into the conversation the running turn is answering, now.

    ``quote`` is the id of one of the person's messages in this conversation
    to reply to; channels that cannot quote send without it
    (:attr:`ReplyResult.quoted` is False). The turn's own answer is still sent
    at the end. Raises :class:`ConnectorToolError`: ``no_conversation``
    outside a channel conversation, ``tool_not_listed`` when the platform
    does not offer ``reply`` to this agent, and the channel refusals of
    :func:`genfleet.sdk.send_message`.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if quote is not None and (not isinstance(quote, str) or not quote):
        raise ValueError("quote must be a message id")
    spec = _reply_spec()
    if spec is None:
        _require_sandbox(_REPLY)
        raise ConnectorToolError(
            "the platform does not offer reply to this agent: it has no live channel connection",
            status=0, code="tool_not_listed", tool=_REPLY,
        )
    arguments: dict[str, Any] = {"text": text}
    if quote is not None:
        arguments["quote"] = quote
    answer = await _invoke(spec, None, arguments)
    return _reply_result(answer, quote is not None)


# --- Calling the platform ----------------------------------------------------


def _reply_spec() -> ConnectorToolSpec | None:
    for tools in agent_config().channels.values():
        for spec in tools:
            if spec.name == _REPLY and spec.tool == _REPLY:
                return spec
    return None


def _as_tool(spec: ConnectorToolSpec, connection: str | None, audiences: frozenset[Audience]) -> ToolWrapper:
    async def handler(**arguments: Any) -> str:
        return await _model_text(spec, connection, arguments)

    handler.__name__ = spec.name
    schema = ToolSchema(name=spec.name, description=spec.description, parameters=copy.deepcopy(dict(spec.input_schema)))
    return ToolWrapper(fn=handler, schema=schema, audiences=audiences)


async def _model_text(spec: ConnectorToolSpec, connection: str | None, arguments: dict[str, Any]) -> str:
    """The call, as the text the model reads. Never raises for a platform answer."""
    try:
        answer = await _invoke(spec, connection, arguments)
    except ConnectorToolError as exc:
        return _error_text(spec.name, exc)
    return _answer_text(spec.name, answer)


async def _invoke(spec: ConnectorToolSpec, connection: str | None, arguments: dict[str, Any]) -> dict[str, Any]:
    _require_sandbox(spec.name)
    turn = current_turn()
    session_id, turn_id = turn.get("session_id"), turn.get("turn_id")
    if not all(isinstance(v, str) and 0 < len(v) <= _MAX_TURN_REF for v in (session_id, turn_id)):
        raise ConnectorToolError(
            f"{spec.name} can only be called while a turn is running",
            status=0, code="no_turn", tool=spec.name,
        )
    body: dict[str, Any] = {
        "session_id": session_id,
        "turn_id": turn_id,
        "tool": spec.tool,
        "arguments": dict(arguments),
        "definitionHash": spec.definition_hash,
    }
    if connection is not None:
        body["connection"] = connection
    return await asyncio.to_thread(_post, body, spec.name)


def _require_sandbox(name: str) -> None:
    if not os.environ.get(CHANNELS_URL_ENV) or not os.environ.get(CHANNELS_TOKEN_ENV):
        raise ConnectorToolError(
            f"{name} needs a platform sandbox: {CHANNELS_URL_ENV} and {CHANNELS_TOKEN_ENV} are not set",
            status=0, code="not_in_sandbox", tool=name,
        )


def _post(body: dict[str, Any], name: str) -> dict[str, Any]:
    """POST the call; retry only ``backend_unreachable``. Raises :class:`ConnectorToolError`."""
    for delay in (*_RETRY_DELAYS_S, None):
        try:
            return _post_once(body, name)
        except ConnectorToolError as exc:
            if not exc.retryable or delay is None:
                raise
        time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def _post_once(body: dict[str, Any], name: str) -> dict[str, Any]:
    url = os.environ.get(CHANNELS_URL_ENV, "")
    token = os.environ.get(CHANNELS_TOKEN_ENV, "")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/{_ROUTE}",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"authorization": f"Bearer {token}", "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # the URL is platform-provided
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise _refusal(exc.code, exc.read(), name) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        # Sent or not, no answer came: the call may have run.
        raise ConnectorToolError(f"{name} got no answer from the platform", status=0, tool=name) from None
    try:
        parsed = json.loads(raw) if raw else None
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict):
        raise ConnectorToolError(f"{name} got an answer it could not read", status=200, tool=name)
    return parsed


def _refusal(status: int, raw: bytes, name: str) -> ConnectorToolError:
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    payload = parsed if isinstance(parsed, dict) else {}
    code = payload.get("code") if isinstance(payload.get("code"), str) else None
    message = payload.get("message")
    if not isinstance(message, str):
        message = f"{name} was refused" if 400 <= status < 500 else f"{name} failed"
    return ConnectorToolError(message, status=status, code=code, tool=name)


# --- What the model reads ----------------------------------------------------


def _unknown_outcome_text(name: str, why: str) -> str:
    return (
        f"{name} may already have completed: {why}, so the result is unknown. "
        f"Do not call {name} again in this turn. "
        "Tell the person what you tried to do, so they can check whether it happened."
    )


_FAILURES = {
    "timeout": "the API did not answer in time",
    "transport": "the connection to the API failed",
    "too_large": "the API's answer was too large to read",
}


def _error_text(name: str, exc: ConnectorToolError) -> str:
    if exc.outcome_unknown:
        return _unknown_outcome_text(name, "the platform did not confirm the result")
    if exc.code is None:
        return f"{name} was refused ({exc.status}): {exc}"
    return f"{exc.code}: {exc}"


def _answer_text(name: str, answer: Mapping[str, Any]) -> str:
    failure = answer.get("failure")
    if isinstance(failure, str):
        return _unknown_outcome_text(name, _FAILURES.get(failure, f"the call failed ({failure})"))
    if "sent" in answer:
        if answer.get("sent") is not True:
            return _unknown_outcome_text(name, "the platform did not confirm the send")
        quoted = answer.get("quoted")
        if quoted is False:
            return "sent, without the quote: this channel cannot quote a message"
        return "sent"
    if isinstance(answer.get("status"), int):
        return json.dumps({k: answer[k] for k in _RESULT_FIELDS if k in answer}, ensure_ascii=False)
    return _unknown_outcome_text(name, "the platform's answer could not be read")


def _reply_result(answer: Mapping[str, Any], asked_quote: bool) -> ReplyResult:
    if answer.get("sent") is not True:
        raise ConnectorToolError("reply got an answer it could not read; delivery unknown", status=200, tool=_REPLY)
    ids = answer.get("messageIds")
    quoted = answer.get("quoted")
    return ReplyResult(
        message_ids=tuple(i for i in ids if isinstance(i, str)) if isinstance(ids, list) else (),
        quoted=(quoted if isinstance(quoted, bool) else None) if asked_quote else None,
    )

