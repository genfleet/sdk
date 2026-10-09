"""Sensitive tools and their confirmations (ADR-0028 §8a).

A **sensitive** tool does not run when the model calls it. The ``Agent``
answers the model that the action needs approval by a workspace owner or
admin, and reports the request to the platform on the turn's final chunk,
under :data:`CONFIRMATION_REQUESTS_KEY`. The platform records it, an owner or
admin approves it in the dashboard (with step-up), and the platform then
starts a follow-up turn carrying the **approved call**
(:data:`APPROVED_CALL_KEY`). The ``Agent`` runs that one call first, exactly
once, before the model takes over again.

Which tools are sensitive:

- the author's marking, ``@tool(sensitive=True)``;
- the platform's per-tool setting on a hosted turn
  (:data:`SENSITIVE_TOOLS_KEY`, ``{key: bool}``), which wins either way. Keys
  are the same as for audiences: a manifest tool's slug key, any other tool's
  name (see :func:`genfleet.sdk.caller.effective_audiences`).

**The approved call is signed.** At spawn the engine gives each sandbox its
own key in :data:`APPROVAL_KEY_ENV`. It signs every approved call with it:
HMAC-SHA256 over :func:`canonical_json` of the call's fields, ``expires_at``
included. The ``Agent`` runs an approved call only when the signature
verifies, it has not expired, and its ``confirmation_id`` has not already run
in this process. The key sits in the agent's own environment, so the
signature stops **other agents and direct callers** from forging an
approval. It does not stop the agent's own code, which could run any tool
anyway. For code that does not use ``Agent``, all of this is advisory, as the
caller's role is (ADR-0028 §5).

**Peer turns fail closed.** On a turn another agent started
(:data:`PEER_TURN_KEY`), a sensitive tool is refused outright: nobody can
approve a peer's call.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

#: ``AgentInput.metadata``: the platform's per-tool sensitive flags, ``{key: bool}``.
SENSITIVE_TOOLS_KEY = "genfleet.sensitive_tools"
#: ``AgentInput.metadata``: one approved call to run first, signed by the engine.
APPROVED_CALL_KEY = "genfleet.approved_call"
#: ``AgentInput.metadata``: the turn was started by another agent (a peer call).
PEER_TURN_KEY = "genfleet.peer_turn"
#: The final ``AgentOutput.metadata``: this turn's requests for approval.
CONFIRMATION_REQUESTS_KEY = "genfleet.confirmation_requests"
#: Set by the engine in each sandbox: the per-spawn key approved calls are signed with.
APPROVAL_KEY_ENV = "GENFLEET_APPROVAL_KEY"

#: What the model is told instead of a result.
PENDING_RESULT = (
    "This action needs approval by a workspace owner or admin before it runs. "
    "The request was sent; it runs once approved. Tell the user so."
)
PEER_REFUSAL = "Error: this tool needs an owner's approval and can't be used on a call from another agent."

#: The fields an approved call is signed over, in this exact set.
_SIGNED_FIELDS = ("confirmation_id", "tool", "arguments", "args_hash", "expires_at")

#: ``confirmation_id`` s already run in this process (one spawn): an approved
#: call is run at most once, however often it is delivered.
_RUN: set[str] = set()


def canonical_json(value: Any) -> bytes:
    """The one byte form both sides sign: sorted keys, compact separators, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_approved_call(call: dict[str, Any], key: str | bytes) -> str:
    """The hex HMAC-SHA256 of ``call``'s signed fields (the engine signs with this)."""
    payload = canonical_json({field: call.get(field) for field in _SIGNED_FIELDS})
    secret = key.encode("utf-8") if isinstance(key, str) else key
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class ApprovedCall:
    confirmation_id: str
    #: The tool's key: a manifest tool's slug key, any other tool's name.
    tool: str
    arguments: dict[str, Any]


def approved_call_of(
    metadata: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    key: str | None = None,
) -> tuple[ApprovedCall | None, str | None]:
    """The turn's approved call, verified: ``(call, None)``.

    ``(None, reason)`` when one is present but must not run (no key in this
    environment, a bad signature, expired, already run, malformed).
    ``(None, None)`` when the turn carries none.
    """
    raw = (metadata or {}).get(APPROVED_CALL_KEY)
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "malformed"
    secret = key if key is not None else os.environ.get(APPROVAL_KEY_ENV)
    if not secret:
        return None, "no approval key in this environment"
    signature = raw.get("signature")
    if not isinstance(signature, str) or not hmac.compare_digest(signature, sign_approved_call(raw, secret)):
        return None, "bad signature"
    confirmation_id, tool, arguments = raw.get("confirmation_id"), raw.get("tool"), raw.get("arguments")
    if not (isinstance(confirmation_id, str) and confirmation_id and isinstance(tool, str) and tool):
        return None, "malformed"
    if not isinstance(arguments, dict):
        return None, "malformed"
    expires = _parse_time(raw.get("expires_at"))
    if expires is None:
        return None, "malformed"
    if (now or datetime.now(UTC)) >= expires:
        return None, "expired"
    if confirmation_id in _RUN:
        return None, "already run"
    return ApprovedCall(confirmation_id, tool, arguments), None


def mark_run(call: ApprovedCall) -> None:
    """Record that ``call`` ran, before it runs: a redelivery is refused."""
    _RUN.add(call.confirmation_id)


def sensitive_flags_of(metadata: dict[str, Any] | None) -> dict[str, bool]:
    """The platform's sensitive flags by key. Anything not a ``{str: bool}`` entry is ignored."""
    raw = (metadata or {}).get(SENSITIVE_TOOLS_KEY)
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and k and isinstance(v, bool)}


def is_peer_turn(metadata: dict[str, Any] | None) -> bool:
    return (metadata or {}).get(PEER_TURN_KEY) is True


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else None
