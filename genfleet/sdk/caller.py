"""Who a turn is for: an operator or a customer (ADR-0028).

On a hosted turn the platform resolves the caller from the provider-verified
sender (a Telegram user id, a WhatsApp number, a signed-in dashboard member)
and sets it on ``AgentInput.metadata`` under :data:`CALLER_METADATA_KEY`. The
engine overwrites any value a caller supplied, so agent code can trust it.

What it decides here is which tools the turn may use:

- An **operator** in a **private** chat gets every tool.
- Anyone else (a customer, or an operator writing in a group, where the reply
  is seen by everyone) gets only the tools marked customer-safe.
- A tool is operator-only unless it is marked customer-safe. Forgetting the
  mark keeps a tool away from customers; it never exposes one.

``name`` is the sender's own display name and is untrusted text: never base a
decision on it, and don't put it in a system prompt as if it were a fact.

A turn without the key is filtered by where the agent runs. Hosted (the
platform spawned it: ``GENFLEET_A2A_TOKEN`` is set), the engine always sets the
key, so a missing one is read as the narrowest caller. Not hosted (a local
``serve``, a test), there are no roles and nothing is filtered. A key that is
present but malformed is always the narrowest caller: a customer in a group.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

#: ``AgentInput.metadata`` key the platform sets on every hosted turn.
CALLER_METADATA_KEY = "genfleet.caller"

#: Set in every sandbox the platform spawns (the per-spawn A2A token): the
#: agent is hosted, and a turn without a caller is not trusted with every tool.
HOSTED_ENV = "GENFLEET_A2A_TOKEN"

Role = Literal["operator", "customer"]
ROLES: tuple[Role, ...] = ("operator", "customer")


@dataclass(frozen=True)
class Caller:
    role: Role
    #: The verified sender, e.g. ``telegram:123456``. Empty when unknown.
    id: str = ""
    #: The sender's display name. Untrusted text.
    name: str = ""
    #: The provider: ``telegram``, ``whatsapp``, ``web``, ``a2a``, …
    channel: str = ""
    #: A one-to-one chat with the agent. False for a group, where everyone sees the reply.
    private: bool = False
    #: The platform's grace flag for an agent migrated to serve customers
    #: before its tools were marked (``customerToolsLegacy``): every tool stays
    #: available, as before roles existed.
    legacy_tools: bool = False

    @property
    def full_toolset(self) -> bool:
        """Whether this turn may use operator-only tools."""
        return self.legacy_tools or (self.role == "operator" and self.private)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "id": self.id,
            "name": self.name,
            "channel": self.channel,
            "private": self.private,
            "legacy_tools": self.legacy_tools,
        }


#: What a malformed caller is read as: nothing more than a customer in a group.
_NARROWEST = Caller(role="customer")


def caller_of(metadata: dict[str, Any] | None) -> Caller | None:
    """The turn's caller; ``None`` (no roles) only for an agent the platform did not spawn."""
    if not metadata or CALLER_METADATA_KEY not in metadata:
        return _NARROWEST if os.environ.get(HOSTED_ENV) else None
    raw = metadata[CALLER_METADATA_KEY]
    if not isinstance(raw, dict) or raw.get("role") not in ROLES:
        return _NARROWEST
    return Caller(
        role=raw["role"],
        id=_text(raw.get("id")),
        name=_text(raw.get("name")),
        channel=_text(raw.get("channel")),
        private=raw.get("private") is True,
        legacy_tools=raw.get("legacy_tools") is True,
    )


def may_use(caller: Caller | None, *, customer_safe: bool) -> bool:
    """Whether a tool with this marking is offered on the caller's turn."""
    return caller is None or caller.full_toolset or customer_safe


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""
