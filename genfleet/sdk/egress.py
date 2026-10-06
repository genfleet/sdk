"""
`egress` — the hosts an agent or tool says it needs to reach (ADR-0021 §3).

A hosted agent's sandbox can only leave through the platform's egress proxy,
which allows a destination when some list names it. A manifest is one of those
lists, so the syntax is checked here, offline, where the developer writes it:

    egress = ["api.github.com", "*.googleapis.com", "hooks.example.com:8443"]

An entry is a hostname, optionally starting with `*.` (any subdomain, at any
depth, never the bare domain), optionally ending with `:port` (default 443).
IP literals (including shorthands a resolver accepts, such as `127.1`) and a
bare `*` are refused: a list is reviewed by people, and an address or a
wildcard says nothing they can check. Whether a host *resolves* somewhere
private is the proxy's decision at request time, not the syntax's.

The rules are pinned by tests/fixtures/egress-hosts.cases.json, which the
platform's own validators are tested against.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

DEFAULT_PORT = 443
MAX_EGRESS_ENTRIES = 100
MAX_HOST_LENGTH = 253

_ASCII_WHITESPACE = " \t\n\v\f\r"
_OUTSIDE_RAW = re.compile(r"[^\t\n\v\f\r\x20-\x7e]")
_OUTSIDE_TEXT = re.compile(r"[^\x21-\x7e]")
_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
_PORT = re.compile(r"^[1-9][0-9]{0,4}$")
#: A last label a resolver or a WHATWG URL parser reads as part of an IPv4
#: address: `127.1`, `2130706433`, and `0x` with or without digits (`1.2.3.0x`).
_NUMERIC_LABEL = re.compile(r"^([0-9]+|0x[0-9a-f]*)$")
_WILDCARD = "*."

Reason = Literal["invalid_host", "ip_not_allowed", "limit"]


class EgressEntryError(ValueError):
    """An entry, or a list, that is not valid `egress`; ``reason`` says why."""

    def __init__(self, entry: str, reason: Reason, problem: str) -> None:
        super().__init__(f"egress {entry!r}: {problem}")
        self.entry = entry
        self.reason = reason


@dataclass(frozen=True)
class EgressEntry:
    """One parsed entry. ``host`` keeps its leading ``*.`` when it has one."""

    host: str
    port: int = DEFAULT_PORT

    @property
    def is_wildcard(self) -> bool:
        return self.host.startswith(_WILDCARD)

    @property
    def review_warning(self) -> bool:
        """Whether marketplace review should look twice: every wildcard does.

        There is no public-suffix list here on purpose, so `*.github.io` is
        valid syntax that covers every GitHub Pages site. Whether that is
        acceptable is review's call, and this is what flags it.
        """
        return self.is_wildcard

    @property
    def canonical(self) -> str:
        return self.host if self.port == DEFAULT_PORT else f"{self.host}:{self.port}"

    def matches(self, host: str, port: int) -> bool:
        """Whether a request to ``host:port`` falls under this entry."""
        if port != self.port:
            return False
        name = host.lower().removesuffix(".")
        if self.is_wildcard:
            # The suffix keeps its leading dot, so `evilgoogleapis.com` and
            # the bare `googleapis.com` both miss `*.googleapis.com`.
            return name.endswith(self.host[1:])
        return name == self.host


def parse_egress_entry(raw: str) -> EgressEntry:
    """Validate and normalise one entry; `EgressEntryError` when it is not one."""
    # The order is part of the contract (the cases file): what is refused
    # first is never trimmed or case-folded into something valid, so a BOM, a
    # NEL or a KELVIN SIGN cannot become `k` or vanish, on any side.
    if _OUTSIDE_RAW.search(raw):
        raise EgressEntryError(raw, "invalid_host", "only printable ASCII is allowed; use punycode")
    text = raw.strip(_ASCII_WHITESPACE)
    if not text:
        raise EgressEntryError(raw, "invalid_host", "empty")
    if _OUTSIDE_TEXT.search(text):
        # An entry reaches the proxy's line protocol: no newline survives.
        raise EgressEntryError(raw, "invalid_host", "whitespace inside an entry")
    if text.startswith("[") or text.count(":") > 1:
        raise EgressEntryError(raw, "ip_not_allowed", "IP addresses are not allowed; name the host")
    host, port = _split_port(text)
    host = host.lower().removesuffix(".")
    wildcard = host.startswith(_WILDCARD)
    name = host[len(_WILDCARD):] if wildcard else host
    if _is_ip_literal(name) or _NUMERIC_LABEL.fullmatch(name.rsplit(".", 1)[-1]):
        raise EgressEntryError(raw, "ip_not_allowed", "IP addresses are not allowed; name the host")
    labels = name.split(".")
    if "*" in name or len(labels) < 2 or not all(_LABEL.fullmatch(label) for label in labels):
        raise EgressEntryError(
            raw,
            "invalid_host",
            "not a hostname: letters, digits and hyphens in at least two dot-separated "
            "labels, optionally starting with '*.'",
        )
    if len(name) > MAX_HOST_LENGTH:
        raise EgressEntryError(raw, "invalid_host", f"longer than {MAX_HOST_LENGTH} characters")
    return EgressEntry(host=host, port=port)


def parse_egress(entries: Iterable[str]) -> list[EgressEntry]:
    """Validate a whole list: capped, and deduplicated by canonical form."""
    parsed: dict[str, EgressEntry] = {}
    for raw in entries:
        entry = parse_egress_entry(raw)
        parsed.setdefault(entry.canonical, entry)
    if len(parsed) > MAX_EGRESS_ENTRIES:
        raise EgressEntryError(
            f"{len(parsed)} entries", "limit", f"at most {MAX_EGRESS_ENTRIES} entries"
        )
    return list(parsed.values())


def _split_port(raw: str) -> tuple[str, int]:
    host, sep, port = raw.rpartition(":")
    if not sep:
        return raw, DEFAULT_PORT
    if not _PORT.fullmatch(port) or int(port) > 65535:
        raise EgressEntryError(raw, "invalid_host", "the port must be a number from 1 to 65535")
    return host, int(port)


def _is_ip_literal(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


__all__ = [
    "DEFAULT_PORT",
    "MAX_EGRESS_ENTRIES",
    "MAX_HOST_LENGTH",
    "EgressEntry",
    "EgressEntryError",
    "parse_egress",
    "parse_egress_entry",
]
