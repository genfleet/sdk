"""Hand a piece of work to whoever the workspace chose to receive it.

An agent declares the hand-offs it can make in ``agent.toml``::

    [sends.lead]
    kind = "lead.v1"
    label = "Qualified lead"

    [sends.lead.options.full_details]
    type = "boolean"
    label = "Include every product asked about"

It never names the receiver. The workspace that installs the agent picks a
destination for each hand-off, or none. The agent asks whether one is set up
and sends::

    from genfleet.sdk.handoffs import handoff, is_wired, option

    if is_wired("lead"):
        result = await handoff("lead", envelope_json)
        if not result.delivered:
            ...  # tell staff; result.reason says why

``handoff`` answers one of three outcomes:

- ``delivered``: the destination took it.
- ``not_delivered``: it did not, or may not have. ``reason`` is a short slug.
- ``nowhere``: the workspace chose no destination. Nothing was sent.

When a hand-off goes nowhere, do not collect details for it and do not
promise a follow-up: nobody will receive it.

What counts as delivered depends on the hand-off's ``kind``. For ``lead.v1``
the destination answers a ``receipt.v1``, and the lead is delivered when the
receipt says it was filed (``created``, ``appended`` or ``duplicate``).

Outside the platform (a local run) every hand-off goes nowhere.
"""

from __future__ import annotations

import asyncio
import copy
import http.client
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .channels import _OPENER, _NoAnswer, _read_capped
from .config import agent_config
from .turn import current_turn, turn_id_of

__all__ = [
    "HANDOFFS_TOKEN_ENV",
    "HANDOFFS_URL_ENV",
    "LEAD_KIND",
    "MAX_PAYLOAD_CHARS",
    "RECEIPT_SCHEMA",
    "HandoffResult",
    "handoff",
    "is_wired",
    "option",
    "read_receipt",
]

HANDOFFS_URL_ENV = "GENFLEET_HANDOFFS_URL"
HANDOFFS_TOKEN_ENV = "GENFLEET_HANDOFFS_TOKEN"

#: The longest payload a hand-off carries.
MAX_PAYLOAD_CHARS = 64_000

LEAD_KIND = "lead.v1"
RECEIPT_SCHEMA = "receipt.v1"
#: The receipt statuses that mean the lead is with the destination.
FILED_STATUSES = frozenset({"created", "appended", "duplicate"})
#: A reply longer than this is not read as a receipt.
MAX_RECEIPT_CHARS = 4_096

#: A hand-off waits for the destination's whole turn, which may have to wake first.
_TIMEOUT_S = 120.0

_KEY = re.compile(r"[a-z][a-z0-9_]{0,31}")
_SLUG = re.compile(r"[a-z][a-z0-9_]{0,63}")

Outcome = Literal["delivered", "not_delivered", "nowhere"]


@dataclass(frozen=True)
class HandoffResult:
    """What happened to one hand-off.

    ``reason`` is a short slug. For ``delivered`` it is the destination's
    status when the kind has one (a lead's ``created``, ``appended`` or
    ``duplicate``). For ``not_delivered`` it says why: the destination's own
    reason, or ``no_receipt`` (its answer was not a receipt), ``refused``
    (the platform would not deliver: the destination is gone, or no longer
    accepts this kind), ``unavailable`` (the destination failed or timed
    out) or ``unreachable`` (the platform did not answer: it may or may not
    have been delivered).
    """

    outcome: Outcome
    reason: str | None = None

    @property
    def delivered(self) -> bool:
        return self.outcome == "delivered"


def read_receipt(reply: Any) -> tuple[bool, str]:
    """``(filed, detail)`` for a destination's answer to a ``lead.v1``.

    ``detail`` is a slug: the receipt's status when filed, its reason when
    not, else ``no_receipt``.
    """
    if not isinstance(reply, str) or len(reply) > MAX_RECEIPT_CHARS:
        return False, "no_receipt"
    try:
        data = json.loads(reply)
    except (ValueError, RecursionError):
        return False, "no_receipt"
    if not isinstance(data, dict) or data.get("schema") != RECEIPT_SCHEMA:
        return False, "no_receipt"
    status = data.get("status")
    if data.get("ok") is True and status in FILED_STATUSES:
        return True, status
    reason = data.get("reason")
    if isinstance(reason, str) and _SLUG.fullmatch(reason):
        return False, reason
    return False, "not_filed"


#: How each kind reads the destination's answer. A kind not listed here is
#: delivered once the destination's turn finished.
_REPLY_READERS: dict[str, Callable[[Any], tuple[bool, str]]] = {LEAD_KIND: read_receipt}


def is_wired(key: str) -> bool:
    """Whether the workspace chose a destination for the hand-off ``key``."""
    wiring = agent_config().handoffs.get(key)
    return wiring is not None and wiring.wired


def option(key: str, name: str, default: Any = None) -> Any:
    """One option of the hand-off ``key``: the workspace's value, else the manifest's default, else ``default``."""
    wiring = agent_config().handoffs.get(key)
    if wiring is None or name not in wiring.options:
        return default
    return copy.deepcopy(wiring.options[name])


async def handoff(key: str, payload: str | Mapping[str, Any]) -> HandoffResult:
    """Send the hand-off ``key`` to the destination the workspace chose.

    ``payload`` is what the hand-off's kind carries: JSON text, or a mapping
    that is sent as JSON. Returns a :class:`HandoffResult`; a failed delivery
    is an outcome, never an exception. Raises ``ValueError`` only for a bad
    ``key`` or ``payload``.
    """
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        raise ValueError("key must be the name of a [sends.<key>] table")
    if isinstance(payload, Mapping):
        try:
            payload = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
        except (TypeError, RecursionError) as exc:
            raise ValueError(f"payload must hold JSON values only: {exc}") from None
    if not isinstance(payload, str) or not payload.strip() or len(payload) > MAX_PAYLOAD_CHARS:
        raise ValueError(f"payload must be non-empty JSON text of at most {MAX_PAYLOAD_CHARS} characters, or a mapping")

    wiring = agent_config().handoffs.get(key)
    if wiring is None or not wiring.wired:
        return HandoffResult("nowhere")
    if not os.environ.get(HANDOFFS_URL_ENV) or not os.environ.get(HANDOFFS_TOKEN_ENV):
        return HandoffResult("nowhere")

    body: dict[str, Any] = {"key": key, "payload": payload}
    turn = current_turn()
    session_id, turn_id = turn.get("session_id"), turn_id_of(turn)
    if isinstance(session_id, str) and session_id and turn_id:
        body["session_id"], body["turn_id"] = session_id, turn_id
    answer = await asyncio.to_thread(_send, body)
    return _result(answer, wiring.kind)


def _send(body: dict[str, Any]) -> dict[str, Any] | str:
    """The platform's answer as an object, or the reason slug when there is none."""
    request = urllib.request.Request(
        f"{os.environ[HANDOFFS_URL_ENV].rstrip('/')}/send",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"authorization": f"Bearer {os.environ[HANDOFFS_TOKEN_ENV]}", "content-type": "application/json"},
    )
    try:
        with _OPENER.open(request, timeout=_TIMEOUT_S) as response:
            raw = _read_capped(response)
    except urllib.error.HTTPError as exc:
        # A 4xx is a refusal: nothing was delivered. Anything else is unknown.
        return "refused" if 400 <= exc.code < 500 else "unreachable"
    except (_NoAnswer, urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError):
        return "unreachable"
    try:
        parsed = json.loads(raw)
    except (ValueError, RecursionError):
        return "unreachable"
    return parsed if isinstance(parsed, dict) else "unreachable"


def _result(answer: dict[str, Any] | str, kind: str) -> HandoffResult:
    if isinstance(answer, str):
        return HandoffResult("not_delivered", answer)
    outcome = answer.get("outcome")
    if outcome == "nowhere":
        return HandoffResult("nowhere")
    if outcome == "not_delivered":
        reason = answer.get("reason")
        return HandoffResult("not_delivered", reason if isinstance(reason, str) and _SLUG.fullmatch(reason) else "refused")
    if outcome != "sent":
        # An answer this SDK does not know: it may have been delivered.
        return HandoffResult("not_delivered", "unreachable")
    reader = _REPLY_READERS.get(kind)
    if reader is None:
        return HandoffResult("delivered")
    filed, detail = reader(answer.get("content"))
    return HandoffResult("delivered" if filed else "not_delivered", detail)
