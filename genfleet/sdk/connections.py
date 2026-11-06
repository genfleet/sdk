"""Call one of your workspace's own APIs through a connection.

A workspace owner or admin sets up a **Custom API** connection in the
dashboard (Connections): a name, a base URL, the header the key goes in, and
the key. The agent names the connection and the request; the platform adds
the key and makes the call. The agent never sees the key, and the request
can only go to that connection's host.

    response = await call_connection("erp", "GET", "/integrations/sales_people")
    if response.ok:
        people = response.json()

``path`` is relative to the connection's base URL and may carry a query
string. ``json`` is the request body (any JSON value; not for ``GET`` or
``DELETE``). The agent sets no headers: the platform sends ``Accept`` and
``Content-Type`` as JSON, a ``User-Agent`` naming the agent, and the
connection's key. ``idempotency_key`` is sent as ``Idempotency-Key`` for an
API that dedupes repeated writes.

Any HTTP status the API answers comes back as a :class:`ConnectionResponse`,
redirects included (they are never followed). Anything else raises
:class:`ConnectionCallError`.

Like :func:`genfleet.sdk.send_message`, this needs a platform sandbox;
outside one the call raises with code ``not_in_sandbox``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json as _json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .channels import ChannelSendError, _call

#: The methods a connection call may use.
METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})
#: The default and the longest time the platform gives one call, in seconds.
DEFAULT_TIMEOUT_S = 10.0
MAX_TIMEOUT_S = 20.0
#: How long the SDK waits for the platform's answer: the call's limit plus headroom.
_PLATFORM_WAIT_S = MAX_TIMEOUT_S + 15.0


class ConnectionCallError(ChannelSendError):
    """A connection call that got no answer from the API.

    ``code`` is one of:

    - ``unknown_connection``: no connection of that name in this workspace;
    - ``agent_not_allowed``: this agent may not use the workspace's connections;
    - ``host_refused``: the connection's host is not a public address;
    - ``invalid_request``: the method, path, body or key was refused;
    - ``timeout``, ``transport``: the API did not answer in time, or could
      not be reached; a write may or may not have reached it;
    - ``too_large``: the API's answer was over the size limit (1 MB);
    - ``not_in_sandbox``: not running on the platform;
    - None: the platform itself failed; the request may or may not have
      reached the API.

    :attr:`may_have_reached_api` is True for ``timeout``, ``transport``,
    ``too_large`` and no code: resend a write only if the API dedupes it
    (``idempotency_key``).
    """

    @property
    def may_have_reached_api(self) -> bool:
        return self.code in (None, "timeout", "transport", "too_large")


@dataclass(frozen=True)
class ConnectionResponse:
    """The API's answer: its status, a few of its headers (``content-type``,
    ``cf-ray``, ``retry-after``, lowercased), and the body as bytes."""

    status: int
    headers: Mapping[str, str] = field(default_factory=dict)
    content: bytes = field(default=b"", repr=False)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """The body parsed as JSON; raises ``ValueError`` when it is not."""
        return _json.loads(self.content)


async def call_connection(
    connection: str,
    method: str,
    path: str,
    json: Any = None,
    *,
    idempotency_key: str | None = None,
    timeout: float | None = None,
) -> ConnectionResponse:
    """Call the workspace's Custom API connection ``connection``.

    ``method`` is GET, POST, PUT, PATCH or DELETE. ``path`` starts with ``/``,
    relative to the connection's base URL, query string included.
    ``timeout`` is seconds for the whole call, 1 to 20 (default 10).

    Returns a :class:`ConnectionResponse` for any status the API answers;
    raises :class:`ConnectionCallError` otherwise.
    """
    verb = method.upper() if isinstance(method, str) else ""
    if verb not in METHODS:
        raise ValueError(f"method must be one of {', '.join(sorted(METHODS))}")
    if not isinstance(connection, str) or not connection:
        raise ValueError("connection must name one of the workspace's connections")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("path must start with /")
    body: dict[str, Any] = {"connection": connection, "method": verb, "path": path}
    if json is not None:
        body["body"] = json
    if idempotency_key is not None:
        body["idempotencyKey"] = idempotency_key
    if timeout is not None:
        body["timeoutSeconds"] = float(timeout)
    return await asyncio.to_thread(_call_connection, body)


def _call_connection(body: dict[str, Any]) -> ConnectionResponse:
    answer = _call("connection-call", body, what="call_connection", timeout=_PLATFORM_WAIT_S, error=ConnectionCallError)
    failure = answer.get("failure")
    if isinstance(failure, str):
        raise ConnectionCallError(f"call_connection failed: {failure}", status=0, code=failure)
    status = answer.get("status")
    headers = answer.get("headers")
    encoded = answer.get("bodyBase64")
    if not isinstance(status, int) or not isinstance(encoded, str):
        raise ConnectionCallError("call_connection got an answer it could not read", status=200)
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ConnectionCallError("call_connection got an answer it could not read", status=200) from None
    kept = {str(k).lower(): v for k, v in headers.items() if isinstance(v, str)} if isinstance(headers, dict) else {}
    return ConnectionResponse(status=status, headers=kept, content=content)
