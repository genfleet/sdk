"""Escalate a conversation to the workspace's staff.

The workspace keeps a staff list on the platform: each person has a name, a
role and whether they are reached by default, and the platform holds how to
reach them. An agent names who should know — a staff member's name, every
member with a role (``"role:sales"``), or ``"default"`` — and the words; the
platform delivers to your workspace's staff. The agent never sees or names an
address.

The platform also keeps the agent from paging staff twice about the same
conversation: once an alert was sent (or may have been), another one about
that conversation to the same person is refused for ten minutes, and a
failed one may be retried a few times.

Like :func:`genfleet.sdk.send_message`, this needs a platform sandbox; outside
one the call raises with code ``not_in_sandbox`` rather than pretending.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .channels import CHANNELS_TOKEN_ENV, CHANNELS_URL_ENV, ChannelSendError, _call
from .tool import tool
from .turn import current_turn

#: The most entries ``to`` may list.
MAX_RECIPIENTS = 5
#: The longest ``text`` the platform takes.
MAX_TEXT = 4_000

#: An escalation fans out to every way of reaching up to five people; give it longer than a send.
_TIMEOUT_S = 35.0

Outcome = Literal["sent", "unknown", "not_sent"]
RecipientOutcome = Literal["sent", "unknown", "not_sent", "already_escalated", "too_many_attempts"]


class EscalationError(ChannelSendError):
    """The platform refused or failed an escalation.

    ``code`` is ``unknown_recipient`` (a name, role or ``default`` that names
    nobody: the message lists the valid ones), ``already_escalated`` (every
    named person already got an alert about this conversation in the last
    ten minutes, or may have: do not resend), ``rate_limited``,
    ``invalid_request``, ``no_session`` (no conversation to escalate),
    ``not_in_sandbox``, or None for anything else. As for
    :class:`ChannelSendError`, :attr:`delivery_unknown` is True when the alert
    may have gone out (a timeout, a 5xx): do not resend it.
    """


@dataclass(frozen=True)
class EscalationRecipient:
    """One staff member the escalation named, and what happened for them."""

    name: str
    outcome: RecipientOutcome


@dataclass(frozen=True)
class EscalationResult:
    """``outcome`` is the best of the people reached: ``sent`` when any alert
    was delivered, ``unknown`` when one may have been (do not resend), else
    ``not_sent``. ``recipients`` lists everyone ``to`` named, including those
    skipped because they were already alerted."""

    outcome: Outcome
    recipients: tuple[EscalationRecipient, ...]

    @property
    def sent(self) -> bool:
        return self.outcome == "sent"


@dataclass(frozen=True)
class StaffMember:
    """A name an agent may escalate to, its role, and whether ``default`` reaches it."""

    name: str
    role: str
    is_default: bool


async def escalate(to: str | Sequence[str], text: str, *, session: str | None = None) -> EscalationResult:
    """Alert the workspace's staff about a conversation.

    ``to`` is one entry or a list of up to five: a staff member's name
    (matched without case), ``"role:<role>"`` or ``"default"``. Every entry
    must name someone, or nothing is sent (``unknown_recipient``). ``text``
    is plain text, at most 4000 characters. ``session`` is the conversation
    the alert is about; by default the running turn's ``session_id``.

    Returns an :class:`EscalationResult`; raises :class:`EscalationError` on
    a refusal or when the outcome could not be read.
    """
    targets = [to] if isinstance(to, str) else list(to)
    if not targets or len(targets) > MAX_RECIPIENTS or not all(isinstance(t, str) and t.strip() for t in targets):
        raise ValueError(f"to must name 1 to {MAX_RECIPIENTS} staff members, roles (role:<role>) or default")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise ValueError(f"text must be a non-empty string of at most {MAX_TEXT} characters")
    return await asyncio.to_thread(_escalate, targets, text, session)


async def staff_directory() -> list[StaffMember]:
    """The staff this agent can escalate to: names, roles and the default flag, never how they are reached."""
    answer = await asyncio.to_thread(_call, "staff-directory", None, what="staff_directory", error=EscalationError)
    staff = answer.get("staff")
    if not isinstance(staff, list):
        raise EscalationError("staff_directory expected a staff list", status=200)
    return [
        StaffMember(name=s["name"], role=s["role"], is_default=bool(s.get("isDefault")))
        for s in staff
        if isinstance(s, dict) and isinstance(s.get("name"), str) and isinstance(s.get("role"), str)
    ]


@tool(
    name="escalate",
    # Offered to customers too: the model only picks from the workspace's own
    # staff list, and the platform bounds how often staff can be alerted.
    audiences=["operator", "customer"],
    description=(
        "Alert the workspace's staff about this conversation, for something you cannot handle yourself "
        "(a person asks for a human, a complaint, an urgent problem). `to` is a list of up to 5: a staff "
        'member\'s name, "role:<role>" (e.g. "role:sales") or "default" when you do not know who. `text` '
        "is a short plain summary for staff: who, what they need, and what you already told them. "
        "Never put an address or phone number in `to`. Staff are alerted once per conversation; do not "
        "call this again for the same request."
    ),
)
async def escalate_tool(to: list[str], text: str) -> str:
    """The model-facing form: plain answers the model can act on, never an
    exception that ends its turn. An outcome the platform could not confirm
    says so, so the model does not page staff twice."""
    try:
        result = await escalate(to, text)
    except ValueError as exc:
        return f"not sent: {exc}"
    except EscalationError as exc:
        if exc.code == "already_escalated":
            return "already escalated: staff were already alerted about this conversation; do not resend"
        if exc.delivery_unknown:
            return "delivery unknown, do not resend"
        return f"not sent: {exc}"
    if result.outcome == "sent":
        skipped = [r.name for r in result.recipients if r.outcome == "already_escalated"]
        return "sent" + (f" (already escalated to {', '.join(skipped)})" if skipped else "")
    if result.outcome == "unknown":
        return "delivery unknown, do not resend"
    return "not sent: staff could not be reached; tell the person you could not pass it on right now"


def _escalate(targets: list[str], text: str, session: str | None) -> EscalationResult:
    session_key = session if session is not None else current_turn().get("session_id")
    body = {"to": targets, "text": text, "sessionKey": session_key}
    if not isinstance(session_key, str) or not session_key.strip():
        # The sandbox check first: a local run should say it is local, not that it has no session.
        _require_sandbox()
        raise EscalationError(
            "escalate needs the conversation it is about: pass session=, or call it during a turn",
            status=0,
            code="no_session",
        )
    answer = _call("escalate", body, what="escalate", timeout=_TIMEOUT_S, error=EscalationError)
    outcome = answer.get("outcome")
    recipients = answer.get("recipients")
    if outcome not in ("sent", "unknown", "not_sent") or not isinstance(recipients, list):
        raise EscalationError("escalate got an answer it could not read; delivery unknown", status=200)
    return EscalationResult(
        outcome=outcome,
        recipients=tuple(
            EscalationRecipient(name=r["name"], outcome=r["outcome"])
            for r in recipients
            if isinstance(r, dict) and isinstance(r.get("name"), str) and isinstance(r.get("outcome"), str)
        ),
    )


def _require_sandbox() -> None:
    if not os.environ.get(CHANNELS_URL_ENV) or not os.environ.get(CHANNELS_TOKEN_ENV):
        raise EscalationError(
            f"escalate needs a platform sandbox: {CHANNELS_URL_ENV} and {CHANNELS_TOKEN_ENV} are not set",
            status=0,
            code="not_in_sandbox",
        )
