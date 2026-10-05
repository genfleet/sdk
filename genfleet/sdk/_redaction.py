"""Tool-call arguments made safe to send: redaction v2, then an 8 KB cap.

``serve`` relays an agent's tool calls to whoever calls it, and a developer
can host ``serve`` with no platform engine in front of it, so it applies the
same rules the engine and backend apply (tool-call stream contract v1.1 §C).
Keep the three in step.

A value becomes ``"[redacted]"`` when:

1. its key names a credential: any word part of the key (split on camelCase,
   acronym runs, ``_ - .`` and whitespace) is in ``SECRET_KEY_PARTS``, or the
   whole key, lower-cased with ``_ - .`` removed, is in that set or ends with
   one of ``SECRET_KEY_SUFFIXES``;
2. it is the ``value`` of a header-style pair, an object whose ``name``,
   ``key`` or ``header`` field is a string that rule 1 would redact
   (``{"name": "Authorization", "value": …}``);
3. it is a string that looks like a credential whatever its key: ``Bearer `` /
   ``Basic `` schemes, common token prefixes, or a PEM block.

Kept as is: ``keyword``, ``monkey``, ``keys``, ``author``, ``tokenizer``,
``session_id``, ``session``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

REDACTED: Final[str] = "[redacted]"

SECRET_KEY_PARTS: Final[frozenset[str]] = frozenset({
    "key", "apikey", "token", "secret", "password", "passwd", "pwd",
    "authorization", "auth", "credential", "credentials", "cookie", "bearer",
    "privatekey", "accesstoken", "refreshtoken", "clientsecret",
})
SECRET_KEY_SUFFIXES: Final[tuple[str, ...]] = (
    "token", "secret", "password", "apikey", "privatekey",
)
#: The fields of a header-style pair that name it (rule 2).
_PAIR_NAME_FIELDS: Final[tuple[str, ...]] = ("name", "key", "header")

_LOWER_THEN_UPPER: Final = re.compile(r"([a-z0-9])([A-Z])")
_ACRONYM_THEN_WORD: Final = re.compile(r"([A-Z]+)([A-Z][a-z])")
_KEY_SEPARATORS: Final = re.compile(r"[\s_.\-]+")
_COMPACT_SEPARATORS: Final = re.compile(r"[_.\-]")
_CREDENTIAL_VALUE: Final = re.compile(
    r"^(?:Bearer |Basic |sk-|ghp_|github_pat_|xox[abp]-|AKIA[0-9A-Z]{16})"
    r"|-----BEGIN "
)

#: Arguments whose compact JSON is larger than this many UTF-8 bytes are
#: replaced by ``{"_truncated": true, "preview": <first N chars of it>}``.
ARGUMENTS_MAX_BYTES: Final[int] = 8 * 1024
ARGUMENTS_PREVIEW_CHARS: Final[int] = 2000


def _key_parts(key: str) -> list[str]:
    spaced = _ACRONYM_THEN_WORD.sub(r"\1 \2", _LOWER_THEN_UPPER.sub(r"\1 \2", key))
    return [p.lower() for p in _KEY_SEPARATORS.split(spaced) if p]


def key_looks_secret(key: Any) -> bool:
    """Rule 1: whether ``key`` names a credential."""
    if not isinstance(key, str):
        return False
    if any(p in SECRET_KEY_PARTS for p in _key_parts(key)):
        return True
    compact = _COMPACT_SEPARATORS.sub("", key.lower())
    return compact in SECRET_KEY_PARTS or compact.endswith(SECRET_KEY_SUFFIXES)


def redact(value: Any) -> Any:
    """``value`` with rules 1-3 applied at every depth."""
    if isinstance(value, dict):
        secret_pair = any(key_looks_secret(value.get(f)) for f in _PAIR_NAME_FIELDS)
        return {
            k: REDACTED if key_looks_secret(k) or (secret_pair and k == "value") else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str) and _CREDENTIAL_VALUE.search(value):
        return REDACTED
    return value


def safe_arguments(arguments: Any) -> dict[str, Any]:
    """Tool-call ``arguments`` redacted, then capped at ``ARGUMENTS_MAX_BYTES``.

    Size is the compact JSON (non-ASCII kept) in UTF-8 bytes, which is what
    ``JSON.stringify`` produces on the backend.
    """
    redacted = redact(arguments) if isinstance(arguments, dict) else {}
    serialized = json.dumps(redacted, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(serialized.encode("utf-8")) <= ARGUMENTS_MAX_BYTES:
        return redacted
    return {"_truncated": True, "preview": serialized[:ARGUMENTS_PREVIEW_CHARS]}
