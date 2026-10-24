"""The workspace's storage limit, as the platform reports it on memory and state calls.

When a memory or workflow-state write would take the workspace over its
storage limit, the platform answers HTTP 402 with ``code: "storage_limit"``.
The clients raise :class:`StorageLimitExceeded` for it. The limit belongs to
the whole workspace, not to the agent, and a retry cannot succeed until space
is freed or the plan changes, so the SDK never retries it.
"""

from __future__ import annotations

import json
from typing import Any

from .memory.platform import PlatformMemoryError
from .platform_state import PlatformStateError

STORAGE_LIMIT_CODE = "storage_limit"


class StorageLimitExceeded(PlatformStateError, PlatformMemoryError):
    """The workspace is at its storage limit, so the platform refused a write (HTTP 402).

    Catchable as :class:`~genfleet.sdk.platform_state.PlatformStateError` (so a
    :class:`~genfleet.sdk.state.StateError`) and as
    :class:`~genfleet.sdk.memory.PlatformMemoryError`, whichever client raised it.
    It is not retried.

    Attributes are ``None`` when the platform did not send them:

    - ``limit_bytes``: the workspace's limit.
    - ``current``: bytes in use.
    - ``required_bytes``: the size of the refused write.
    - ``categories``: bytes in use by category.
    """

    def __init__(
        self,
        op: str,
        status: int = 402,
        detail: str = "storage limit exceeded",
        *,
        limit_bytes: int | None = None,
        current: int | None = None,
        required_bytes: int | None = None,
        categories: dict[str, int] | None = None,
    ) -> None:
        RuntimeError.__init__(self, f"platform {op} failed ({status}): {detail}")
        self.op = op
        self.status = status
        self.detail = detail
        self.limit_bytes = limit_bytes
        self.current = current
        self.required_bytes = required_bytes
        self.categories = categories


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def storage_limit_from_body(op: str, status: int, raw: bytes) -> StorageLimitExceeded | None:
    """The error for a 402 body carrying ``code: "storage_limit"``, else ``None``."""
    if status != 402:
        return None
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("code") != STORAGE_LIMIT_CODE:
        return None
    categories = body.get("categories")
    if isinstance(categories, dict):
        categories = {str(k): v for k, v in categories.items() if _int(v) is not None}
    else:
        categories = None
    limit = body.get("limitMax", body.get("limit_bytes"))
    message = body.get("message")
    return StorageLimitExceeded(
        op,
        status,
        message if isinstance(message, str) and message else "storage limit exceeded",
        limit_bytes=_int(limit),
        current=_int(body.get("currentCount", body.get("current"))),
        required_bytes=_int(body.get("requiredBytes", body.get("required_bytes"))),
        categories=categories,
    )
