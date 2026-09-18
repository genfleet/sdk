"""The platform's episodic memory, reached through the sandbox's memory proxy.

ADR-0018 §4 and §6. The sandbox is handed two environment variables at spawn:
``GENFLEET_MEMORY_URL`` (the engine's memory proxy) and
``GENFLEET_MEMORY_TOKEN`` (a per-spawn bearer token the engine maps to this
instance's ``(tenant, agent)``). This client sends a session id, a subject
and messages — never a tenant or agent id. There is nothing to forge: the
scope is whatever the engine minted the token for.

Stdlib HTTP on a worker thread, so the SDK's only hard dependency stays
pydantic and the call does not block the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from ..schemas import Message

log = logging.getLogger("genfleet.sdk.memory.platform")

MEMORY_URL_ENV = "GENFLEET_MEMORY_URL"
MEMORY_TOKEN_ENV = "GENFLEET_MEMORY_TOKEN"

_TIMEOUT_S = 10.0


class PlatformMemoryError(RuntimeError):
    """The proxy refused or failed a call. ``status`` is the HTTP status, 0 for no response."""

    def __init__(self, op: str, status: int, detail: str) -> None:
        super().__init__(f"platform memory {op} failed ({status}): {detail}")
        self.op = op
        self.status = status
        self.detail = detail


class PlatformMemory:
    def __init__(self, base_url: str, token: str) -> None:
        if not base_url or not token:
            raise ValueError("PlatformMemory needs a base URL and a token")
        self._base = base_url.rstrip("/")
        self._token = token

    @classmethod
    def from_env(cls) -> PlatformMemory:
        url = os.environ.get(MEMORY_URL_ENV)
        token = os.environ.get(MEMORY_TOKEN_ENV)
        if not url or not token:
            missing = MEMORY_URL_ENV if not url else MEMORY_TOKEN_ENV
            raise RuntimeError(
                f'memory="platform" needs {missing} — it is set by the platform sandbox. '
                "Outside a sandbox use memory={'type': 'redis', ...} or no memory."
            )
        return cls(url, token)

    # -- Memory ---------------------------------------------------------

    async def load(self, session_id: str, *, limit: int | None = None) -> list[Message]:
        body: dict[str, Any] = {"session_id": session_id}
        if limit:
            body["limit"] = limit
        data = await self._call("episodic/load", body)
        out: list[Message] = []
        for raw in data.get("messages", []):
            try:
                out.append(Message(**raw))
            except Exception:  # noqa: BLE001 — one bad row must not lose the thread
                log.warning("skipping an undecodable message in session %s", session_id)
        return out

    async def save(self, session_id: str, history: list[Message]) -> None:
        # Whole-thread semantics on an append-only store: clear, then append.
        # The agent never takes this path (it uses ``append``); it exists so
        # the ``Memory`` contract holds for callers that only know ``save``.
        await self.clear(session_id)
        if history:
            await self.append(session_id, history)

    async def clear(self, session_id: str) -> None:
        await self._call("episodic/clear", {"session_id": session_id})

    # -- AppendableMemory -------------------------------------------------

    async def append(
        self,
        session_id: str,
        messages: list[Message],
        *,
        subject: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not messages:
            return
        body: dict[str, Any] = {
            "session_id": session_id,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
        }
        if subject:
            body["subject"] = subject
        if metadata:
            body["metadata"] = metadata
        await self._call("episodic/append", body)

    # -- Beyond the protocol ---------------------------------------------

    async def search(self, subject: str, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Past messages of one party matching ``query``, newest first."""
        data = await self._call(
            "episodic/search", {"subject": subject, "query": query, "limit": limit}
        )
        return list(data) if isinstance(data, list) else list(data.get("results", []))

    async def purge(self, subject: str) -> dict[str, int]:
        """Erase everything about one party — every thread and every fact."""
        data = await self._call("episodic/purge", {"subject": subject})
        return {"sessions": int(data.get("sessions", 0)), "facts": int(data.get("facts", 0))}

    # -- transport --------------------------------------------------------

    async def _call(self, op: str, body: dict[str, Any]) -> Any:
        return await asyncio.to_thread(self._call_sync, op, body)

    def _call_sync(self, op: str, body: dict[str, Any]) -> Any:
        req = urllib.request.Request(
            f"{self._base}/{op}",
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self._token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310 — URL is platform-provided
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = _error_detail(exc.read())
            raise PlatformMemoryError(op, exc.code, detail) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PlatformMemoryError(op, 0, str(exc)) from None
        return json.loads(raw) if raw else {}


def _error_detail(raw: bytes) -> str:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
            return parsed["error"]
    except Exception:  # noqa: BLE001
        pass
    return raw[:200].decode(errors="replace") if raw else "no body"
