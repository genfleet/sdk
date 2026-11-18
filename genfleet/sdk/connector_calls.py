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

**What the model reads.** Every answer is text; the tool never raises. A
refusal, an unknown outcome and an API answer that is not ``ok`` are reported
to the runtime as a failed tool call, with the same text.

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
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .caller import EVERYONE, OPERATOR_ONLY, Audience
from .channels import (
    CHANNELS_TOKEN_ENV,
    CHANNELS_URL_ENV,
    ChannelSendError,
    _NoAnswer,
    _send,
)
from .config import ConnectorToolSpec, agent_config
from .schemas import ToolCall, ToolSchema
from .tool import ToolWrapper
from .turn import current_turn

__all__ = [
    "LOCAL_CODES",
    "REFUSAL_CODES",
    "RETRYABLE_CODES",
    "UNKNOWN_OUTCOME_CODES",
    "ConnectorTool",
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
    "not_configured",   # the platform has no connector service for this agent (HTTP 503)
})

log = logging.getLogger("genfleet.sdk.connector_tools")

#: Longer than the platform's own 30-second limit for one call.
_TIMEOUT_S = 35.0
#: Waits before each retry of ``backend_unreachable``: three attempts in all.
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
        """The same as :attr:`outcome_unknown`, under `ChannelSendError`'s name."""
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


class ConnectorTool(ToolWrapper):
    """A tool the platform listed for the agent (:func:`connector_tools`, :func:`reply_tool`).

    ``source`` says where it came from (``connection erp``, ``channel``), for
    the warning when its name collides with another tool. ``Agent`` keeps the
    agent's own tool of that name and skips this one, in any order. A call
    reports a refusal or an unknown outcome as a failed tool call.
    """

    def __init__(self, spec: ConnectorToolSpec, connection: str | None, audiences: frozenset[Audience]) -> None:
        async def handler(**arguments: Any) -> str:
            return (await self.call_reporting_arguments(arguments))[0]

        handler.__name__ = spec.name
        schema = ToolSchema(name=spec.name, description=spec.description, parameters=copy.deepcopy(dict(spec.input_schema)))
        super().__init__(fn=handler, schema=schema, audiences=audiences)
        self.spec = spec
        self.connection = connection
        self.source = f"connection {connection}" if connection is not None else "channel"

    async def call_reporting(self, tool_call: ToolCall) -> tuple[str, bool]:
        """The model's text and whether the call succeeded."""
        return await self.call_reporting_arguments(tool_call.arguments)

    async def call_reporting_arguments(self, arguments: dict[str, Any]) -> tuple[str, bool]:
        try:
            answer = await _invoke(self.spec, self.connection, arguments)
        except ConnectorToolError as exc:
            return _error_text(self.spec.name, exc), False
        return _answer_text(self.spec.name, answer, asked_quote=bool(arguments.get("quote")))


# --- The tools ---------------------------------------------------------------


def connector_tools(slot: str | None = None) -> list[ConnectorTool]:
    """The verified tools of the agent's connections, ready for ``Agent(tools=...)``.

    With no ``slot``, every connection the workspace lets the agent use; with
    one, only the connection filling that ``[connectors.<slot>]`` slot (an
    unfilled slot has none). Empty when the platform lists no tools (a local
    run, or an older platform). Each tool has the listed name, description
    and inputs, is offered to operators unless the workspace's setting says
    otherwise, and answers the model in text (see the module docs). Two
    connections listing the same name keep the first, with a warning.
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
    tools: dict[str, ConnectorTool] = {}
    for connection, spec in found:
        kept = tools.get(spec.name)
        if kept is None:
            tools[spec.name] = ConnectorTool(spec, connection, OPERATOR_ONLY)
        elif kept.connection != connection:
            log.warning(
                "connector tool %r of connection %r skipped: connection %r already lists a tool of that name",
                spec.name, connection, kept.connection,
            )
    return list(tools.values())


def reply_tool() -> ConnectorTool | None:
    """The model-facing ``reply``, or None when the platform does not offer it.

    ``reply`` is offered when the agent is connected to a channel. It sends a
    message into the conversation the turn is answering, now, before the
    turn ends (a "one moment, checking" or a quote of one of the person's
    messages). The turn's answer is still sent at the end. It is offered to
    operators and customers unless the workspace's setting says otherwise.
    When the agent has its own tool named ``reply``, ``Agent`` keeps that one.
    """
    spec = _reply_spec()
    return ConnectorTool(spec, None, EVERYONE) if spec is not None else None


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


async def _invoke(spec: ConnectorToolSpec, connection: str | None, arguments: dict[str, Any]) -> dict[str, Any]:
    _require_sandbox(spec.name)
    turn = current_turn()
    session_id, turn_id = turn.get("session_id"), turn.get("turn_id")
    if not all(isinstance(v, str) and 0 < len(v) <= _MAX_TURN_REF for v in (session_id, turn_id)):
        raise ConnectorToolError(
            "this tool can only be called while a turn is running",
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
            f"this tool needs a platform sandbox: {CHANNELS_URL_ENV} and {CHANNELS_TOKEN_ENV} are not set",
            status=0, code="not_in_sandbox", tool=name,
        )


def _post(body: dict[str, Any], name: str) -> dict[str, Any]:
    """POST the call; retry only ``backend_unreachable``. Raises :class:`ConnectorToolError`."""
    for delay in _RETRY_DELAYS_S:
        try:
            return _post_once(body, name)
        except ConnectorToolError as exc:
            if not exc.retryable:
                raise
        time.sleep(delay)
    return _post_once(body, name)


def _post_once(body: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        status, raw = _send(_ROUTE, body, timeout=_TIMEOUT_S)
    except _NoAnswer:
        # Sent or not, no complete answer came: the call may have run.
        raise ConnectorToolError("no answer came from the platform", status=0, tool=name) from None
    try:
        parsed = json.loads(raw) if raw else None
    except ValueError:
        parsed = None
    if not 200 <= status < 300:
        raise _refusal(status, parsed, name)
    if not isinstance(parsed, dict):
        raise ConnectorToolError("the platform's answer could not be read", status=status, tool=name)
    return parsed


def _refusal(status: int, parsed: Any, name: str) -> ConnectorToolError:
    payload = parsed if isinstance(parsed, dict) else {}
    code = payload.get("code") if isinstance(payload.get("code"), str) else None
    message = next((payload[k] for k in ("message", "detail", "error") if isinstance(payload.get(k), str)), None)
    if code is None and status == 413:
        message = "the arguments are too large"
    elif code is None and status == 503:
        code = "not_configured"
        message = "the platform has no connector service for this agent; the tool did not run"
    if message is None:
        message = "the call was refused" if 400 <= status < 500 else "the call failed"
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


def _answer_text(name: str, answer: Mapping[str, Any], *, asked_quote: bool) -> tuple[str, bool]:
    """The model's text for a 2xx answer, and whether the call succeeded."""
    failure = answer.get("failure")
    if isinstance(failure, str):
        return _unknown_outcome_text(name, _FAILURES.get(failure, f"the call failed ({failure})")), False
    if "sent" in answer:
        if answer.get("sent") is not True:
            return _unknown_outcome_text(name, "the platform did not confirm the send"), False
        if asked_quote and answer.get("quoted") is False:
            return "sent, without the quote: this channel cannot quote a message", True
        return "sent", True
    if isinstance(answer.get("status"), int):
        text = json.dumps({k: answer[k] for k in _RESULT_FIELDS if k in answer}, ensure_ascii=False)
        return text, answer.get("ok") is True
    return _unknown_outcome_text(name, "the platform's answer could not be read"), False


def _reply_result(answer: Mapping[str, Any], asked_quote: bool) -> ReplyResult:
    if answer.get("sent") is not True:
        raise ConnectorToolError("the answer could not be read; delivery unknown", status=200, tool=_REPLY)
    ids = answer.get("messageIds")
    quoted = answer.get("quoted")
    return ReplyResult(
        message_ids=tuple(i for i in ids if isinstance(i, str)) if isinstance(ids, list) else (),
        quoted=(quoted if isinstance(quoted, bool) else None) if asked_quote else None,
    )
