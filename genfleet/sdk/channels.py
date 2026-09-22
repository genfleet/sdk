"""Send a message on one of the agent's channels (ADR-0019 §5).

The reply to an inbound message needs nothing from here: whatever the agent
returns is sent to the contact by the platform. ``send_message`` is for the
rest — an alert to staff, a follow-up, a pre-approved template — and the
platform still does the sending. The agent names a recipient and the words;
it never sees the channel's token, and the platform decides which of the
agent's bindings the message goes out on.

The sandbox is handed ``GENFLEET_CHANNELS_URL`` and ``GENFLEET_CHANNELS_TOKEN``
at spawn. Outside a platform sandbox there are no channels, and the call says
so rather than pretending to send.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

from .tool import tool

CHANNELS_URL_ENV = "GENFLEET_CHANNELS_URL"
CHANNELS_TOKEN_ENV = "GENFLEET_CHANNELS_TOKEN"

_TIMEOUT_S = 25.0


class ChannelSendError(RuntimeError):
    """The platform refused or failed a send.

    ``code`` names a refusal the agent can act on — ``outside_window`` (the
    contact has not written in 24 hours: send a template), ``cold_send_disabled``
    (the contact never wrote and this binding does not allow first contact),
    ``no_binding``, ``invalid_request``, ``send_rejected`` — or is None for
    anything else. ``status`` is the HTTP status, 0 when nothing answered.
    """

    def __init__(self, message: str, *, status: int, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


async def send_message(
    to: str,
    text: str | None = None,
    *,
    template: str | None = None,
    params: list[str] | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """Send ``text``, or the approved ``template`` with ``params``, to ``to``.

    ``to`` is ``"<channel>:<address>"`` — ``"whatsapp:201001234567"``, the
    same shape as the ``session_id`` of an inbound turn, so replying out of
    band to the current contact is ``send_message(input.metadata["session_id"], ...)``.

    Free text reaches only a contact who wrote in the last 24 hours; a
    template reaches anyone the binding may contact. Returns the platform's
    answer (``{"sent": True, "messageIds": [...]}``); raises
    :class:`ChannelSendError` on a refusal.
    """
    if (text is None) == (template is None):
        raise ValueError("send_message needs exactly one of text or template")
    body: dict[str, Any] = {"to": to}
    if text is not None:
        body["text"] = text
    else:
        body["template"] = {"name": template, "language": language, "params": list(params or [])}
    return await asyncio.to_thread(_post, body)


@tool(
    name="send_message",
    description=(
        "Send a WhatsApp message to someone other than the person you are replying to, "
        "or later than the current reply — e.g. alert a staff member. `to` is "
        '"whatsapp:<phone number with country code>". Use `text` for someone who wrote '
        "in the last 24 hours; otherwise use an approved `template` with `params`. "
        "Your normal reply is sent automatically; do not use this for it."
    ),
)
async def send_message_tool(
    to: str,
    text: str | None = None,
    template: str | None = None,
    params: list[str] | None = None,
) -> str:
    """The model-facing form: a refusal is an answer the model can act on,
    not an exception that ends its turn."""
    try:
        await send_message(to, text, template=template, params=params)
    except (ChannelSendError, ValueError) as exc:
        return f"not sent: {exc}"
    return "sent"


def _post(body: dict[str, Any]) -> dict[str, Any]:
    url = os.environ.get(CHANNELS_URL_ENV)
    token = os.environ.get(CHANNELS_TOKEN_ENV)
    if not url or not token:
        raise ChannelSendError(
            "send_message needs a platform sandbox: "
            f"{CHANNELS_URL_ENV} and {CHANNELS_TOKEN_ENV} are not set",
            status=0,
            code="no_binding",
        )
    req = urllib.request.Request(
        f"{url.rstrip('/')}/send",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"content-type": "application/json", "authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310 — URL is platform-provided
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        message, code = _refusal(exc.read())
        raise ChannelSendError(message, status=exc.code, code=code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ChannelSendError(f"send_message could not reach the platform: {exc}", status=0) from None
    try:
        parsed = json.loads(raw) if raw else {}
    except ValueError:
        raise ChannelSendError("send_message got an undecodable answer", status=200) from None
    return parsed if isinstance(parsed, dict) else {}


def _refusal(raw: bytes) -> tuple[str, str | None]:
    try:
        parsed = json.loads(raw)
    except ValueError:
        return (raw[:200].decode(errors="replace") or "send_message failed"), None
    if not isinstance(parsed, dict):
        return "send_message failed", None
    code = parsed.get("code") if isinstance(parsed.get("code"), str) else None
    message = parsed.get("error") if isinstance(parsed.get("error"), str) else "send_message failed"
    return message, code
