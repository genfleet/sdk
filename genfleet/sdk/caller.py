"""Who a turn is for: an operator or a customer (ADR-0028).

On a hosted turn the platform resolves the caller from the provider-verified
sender (a Telegram user id, a WhatsApp number, a signed-in dashboard member)
and sets it on ``AgentInput.metadata`` under :data:`CALLER_METADATA_KEY`. The
engine overwrites any value a caller supplied, so agent code can trust it.

What it decides here is which tools the turn may use. Every tool has an
**audience**: ``operator``, ``customer``, or both.

- An **operator** in a **private** chat gets the tools whose audience
  includes ``operator``.
- Anyone else (a customer, or an operator writing in a group, where the reply
  is seen by everyone) gets the tools whose audience includes ``customer``.
- A tool's audience is ``[operator]`` unless its author marks it with
  ``@tool(audiences=…)``. Forgetting the mark keeps a tool away from
  customers; it never exposes one.
- **The platform's setting decides.** On a hosted turn the platform may send
  the tenant owner's audience for a tool, by tool name or by manifest slug,
  under :data:`TOOL_AUDIENCES_METADATA_KEY` (a manifest tool by its slug key
  only, any other tool by its name only). It replaces the author's marking,
  which is only the default, and may widen it as well as narrow it. The engine
  sets the key on every hosted turn and replaces any value a caller sent. An
  entry the SDK can't read is offered to no one (fail closed).
- This covers every tool an SDK ``Agent`` holds: its own functions,
  ``load_tools()`` tools, remote tools passed in as ``tools=``, and MCP tools.
  Tools the engine keeps itself (runner-held peers, ``remember``) are
  filtered by the engine.
- The platform's ``legacy_tools`` grace flag offers every tool.

``name`` is the sender's own display name and is untrusted text: never base a
decision on it, and don't put it in a system prompt as if it were a fact.

A turn without the caller key is filtered by where the agent runs. Hosted
(the platform spawned it: ``GENFLEET_HOSTED`` is set), the engine always sets
the key, so a missing one is read as a customer in a group: the tools for
customers, never the staff tools. Not hosted (a local ``serve``, a test),
there are no roles and nothing is filtered. A key that is present but
malformed is always a customer in a group.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Literal

#: ``AgentInput.metadata`` key the platform sets on every hosted turn.
CALLER_METADATA_KEY = "genfleet.caller"

#: Set in every sandbox the platform spawns, container or process: the agent is
#: hosted, and a turn without a caller is not trusted with every tool.
HOSTED_ENV = "GENFLEET_HOSTED"

#: Engines older than the ``GENFLEET_HOSTED`` marker set only the per-spawn A2A
#: token, and only in container sandboxes. It still counts as hosted.
_LEGACY_HOSTED_ENV = "GENFLEET_A2A_TOKEN"

#: ``AgentInput.metadata`` key for the tenant owner's per-tool audiences:
#: ``{tool name or manifest slug: ["operator", "customer"]}``. Platform-set.
TOOL_AUDIENCES_METADATA_KEY = "genfleet.tool_audiences"

Role = Literal["operator", "customer"]
ROLES: tuple[Role, ...] = ("operator", "customer")

#: Who a tool is for. The same two values as a caller's role.
Audience = Role
OPERATOR_ONLY: frozenset[Audience] = frozenset({"operator"})
EVERYONE: frozenset[Audience] = frozenset({"operator", "customer"})


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
        """Deprecated (0.18): ``may_offer(caller, OPERATOR_ONLY)``. Removed in 1.0 (backend#252)."""
        _deprecated("Caller.full_toolset", "may_offer(caller, OPERATOR_ONLY)")
        return may_offer(self, OPERATOR_ONLY)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "id": self.id,
            "name": self.name,
            "channel": self.channel,
            "private": self.private,
            "legacy_tools": self.legacy_tools,
        }


#: What a missing (hosted) or malformed caller is read as: a customer in a group.
_NARROWEST = Caller(role="customer")


def is_hosted() -> bool:
    """Whether the platform spawned this process (a hosted sandbox).

    Code that must behave differently on the platform (no durable local
    files, platform-scoped stores) reads this rather than the env itself.
    """
    return bool(os.environ.get(HOSTED_ENV) or os.environ.get(_LEGACY_HOSTED_ENV))


_hosted = is_hosted  # pre-0.19 private name


def caller_of(metadata: dict[str, Any] | None) -> Caller | None:
    """The turn's caller; ``None`` (no roles) only for an agent the platform did not spawn."""
    if not metadata or CALLER_METADATA_KEY not in metadata:
        return _NARROWEST if is_hosted() else None
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


def _audiences_of(value: Any) -> frozenset[Audience] | None:
    """A valid audience list as a set, or ``None`` (empty, unknown values, not a list)."""
    if not isinstance(value, (list, tuple, set, frozenset)) or not value:
        return None
    if any(v not in ROLES for v in value):
        return None
    return frozenset(value)


def tool_audiences_of(metadata: dict[str, Any] | None) -> dict[str, frozenset[Audience]]:
    """The platform's per-tool audiences on this turn, keyed by tool name or manifest slug.

    An entry whose value is not a valid audience list is read as **no
    audience**: the tool is offered to no one. A setting that arrives broken
    then fails closed instead of falling back to the author's (possibly
    wider) marking. A value that is not a mapping is no setting at all.
    """
    raw = metadata.get(TOOL_AUDIENCES_METADATA_KEY) if metadata else None
    if not isinstance(raw, dict):
        return {}
    return {
        key: _audiences_of(value) or frozenset()
        for key, value in raw.items()
        if isinstance(key, str) and key
    }


def effective_audiences(
    name: str,
    slug: str | None,
    default: frozenset[Audience],
    overrides: dict[str, frozenset[Audience]],
) -> frozenset[Audience]:
    """A tool's audience on this turn: the platform's setting, else its
    author's marking (``default``).

    Two key spaces that never mix (0.18.1). A tool that came from a manifest
    (it carries a ``slug``) is matched **only** by its slug key,
    :func:`slug_key`; a bare name in the setting never reaches it, even when
    the names are equal. Any other tool (code-defined, MCP, a peer) is matched
    only by its name. So the owner's setting for an installed tool ``search``
    (``@search``) can't open a code tool also called ``search``, nor the reverse.
    """
    key = slug_key(slug) if slug else name
    return overrides.get(key, default)


def slug_key(slug: str) -> str:
    """The platform's key for a manifest tool: its slug with a leading ``@``.

    ``@acme/crm`` stays as is; a platform tool ``search`` is ``@search``.
    Tool names never start with ``@``, so the two key spaces can't collide.
    """
    return slug if slug.startswith("@") else f"@{slug}"


def marked(customer_safe: bool) -> frozenset[Audience]:
    """The audience a 0.17 ``customer_safe`` marking stands for."""
    return EVERYONE if customer_safe else OPERATOR_ONLY


def may_offer(caller: Caller | None, audiences: frozenset[Audience]) -> bool:
    """Whether a tool with these audiences is offered on the caller's turn.

    No caller (an agent the platform did not spawn) and the ``legacy_tools``
    grace flag offer every tool. Otherwise an operator in a private chat needs
    ``operator`` in the audience; anyone else, ``customer``. So a
    ``[customer]``-only tool is not offered to staff in a private chat.
    """
    if caller is None or caller.legacy_tools:
        return True
    needed: Audience = "operator" if caller.role == "operator" and caller.private else "customer"
    return needed in audiences


def may_use(caller: Caller | None, *, customer_safe: bool) -> bool:
    """Deprecated (0.18): :func:`may_offer` with ``EVERYONE`` or ``OPERATOR_ONLY``.

    Removed in 1.0 (backend#252).
    """
    _deprecated("may_use(caller, customer_safe=…)", "may_offer(caller, EVERYONE | OPERATOR_ONLY)")
    return may_offer(caller, marked(customer_safe))


def _deprecated(old: str, new: str) -> None:
    warnings.warn(f"{old} is deprecated since 0.18 and removed in 1.0; use {new}", DeprecationWarning, stacklevel=3)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""
