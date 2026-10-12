"""Workflow state through the engine's tenant-scoped sandbox proxy.

Scope is stamped by the platform, never sent by the agent:

- **tenant and agent** come from the sandbox's per-spawn token;
- **owner** comes from the running turn's verified caller (role + id), which
  the engine looks up from the turn this call names. The SDK names the turn
  with ``session_id`` and ``turn_id`` from :func:`genfleet.sdk.turn.current_turn`
  (set by ``Agent`` while a tool or the model client runs). A call made
  outside any turn uses the agent's own background scope, which no customer
  turn can read. A call naming a turn that has already ended is refused.

So the ``agent`` and ``owner`` arguments are not sent. ``owner`` is ignored
when hosted. ``agent`` must stay the same for one store: hosted, every call
shares the agent instance's scope, so a second ``agent`` value would silently
share runs (and ``purge``) with the first, and is refused instead.

Pause/resume is a continuation, not an approval: see the README.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

from .state import EffectState, RunState, StateError, StateQuotaExceeded, StateTooLarge
from .turn import current_turn

STATE_URL_ENV = "GENFLEET_STATE_URL"
STATE_TOKEN_ENV = "GENFLEET_STATE_TOKEN"


class PlatformStateError(StateError):
    """The proxy refused or failed an operation. ``status`` is 0 when it was unreachable."""

    def __init__(self, op: str, status: int, detail: str) -> None:
        super().__init__(f"platform state {op} failed ({status}): {detail}")
        self.op = op
        self.status = status


class _PlatformStateTooLarge(PlatformStateError, StateTooLarge):
    """HTTP 413 from the proxy."""


class _PlatformStateQuotaExceeded(PlatformStateError, StateQuotaExceeded):
    """HTTP 429 from the proxy: a backend quota or the proxy's rate cap."""


_BY_STATUS: dict[int, type[PlatformStateError]] = {
    413: _PlatformStateTooLarge,
    429: _PlatformStateQuotaExceeded,
}


class PlatformStateStore:
    """The hosted :class:`~genfleet.sdk.state_store.StateStore` (see the module doc)."""

    def __init__(self, base_url: str, token: str):
        if not base_url or not token:
            raise ValueError("PlatformStateStore needs a URL and token")
        self._base = base_url.rstrip("/")
        self._token = token
        self._agent: str | None = None

    @classmethod
    def from_env(cls) -> PlatformStateStore:
        url, token = os.environ.get(STATE_URL_ENV), os.environ.get(STATE_TOKEN_ENV)
        if not url or not token:
            raise RuntimeError(f"platform state needs {STATE_URL_ENV if not url else STATE_TOKEN_ENV}")
        return cls(url, token)

    async def create(self, agent: str, owner: str, state: Any, *, subject: str | None = None) -> RunState:
        return self._run("create", await self._call("create", agent, {"state": state, "subject": subject}))

    async def get(self, run_id: str, *, agent: str, owner: str) -> RunState:
        return self._run("get", await self._call("get", agent, {"run_id": run_id}))

    async def checkpoint(self, run_id: str, *, agent: str, owner: str, version: int, state: Any) -> RunState:
        body = {"run_id": run_id, "version": version, "state": state}
        return self._run("checkpoint", await self._call("checkpoint", agent, body))

    async def pause(self, run_id: str, *, agent: str, owner: str, version: int,
                    state: Any, ttl_seconds: float = 3600) -> tuple[RunState, str]:
        data = await self._call("pause", agent, {"run_id": run_id, "version": version,
                                                 "state": state, "ttl_seconds": ttl_seconds})
        token = data.get("token")
        if not isinstance(token, str) or not token:
            raise PlatformStateError("pause", 200, "invalid response")
        return self._run("pause", data), token

    async def resume(self, run_id: str, *, agent: str, owner: str, token: str) -> RunState:
        return self._run("resume", await self._call("resume", agent, {"run_id": run_id, "token": token}))

    async def complete(self, run_id: str, *, agent: str, owner: str, version: int, state: Any) -> RunState:
        body = {"run_id": run_id, "version": version, "state": state}
        return self._run("complete", await self._call("complete", agent, body))

    async def cancel(self, run_id: str, *, agent: str, owner: str, version: int) -> RunState:
        return self._run("cancel", await self._call("cancel", agent, {"run_id": run_id, "version": version}))

    async def purge(self, *, agent: str, owner: str) -> int:
        """Delete the current scope's runs (the turn's caller, or the background scope)."""
        return self._count("purge", await self._call("purge", agent, {}))

    async def purge_subject(self, *, agent: str, subject: str) -> int:
        """Refused on a customer's turn: it would reach other callers' runs."""
        return self._count("purge_subject", await self._call("purge_subject", agent, {"subject": subject}))

    async def claim_effect(self, run_id: str, effect_id: str, *, agent: str, owner: str) -> EffectState:
        body = {"run_id": run_id, "effect_id": effect_id}
        return self._effect("claim_effect", await self._call("claim_effect", agent, body))

    async def get_effect(self, run_id: str, effect_id: str, *, agent: str, owner: str) -> EffectState:
        body = {"run_id": run_id, "effect_id": effect_id}
        return self._effect("get_effect", await self._call("get_effect", agent, body))

    async def complete_effect(self, run_id: str, effect_id: str, *, agent: str,
                              owner: str, result: Any) -> EffectState:
        body = {"run_id": run_id, "effect_id": effect_id, "result": result}
        return self._effect("complete_effect", await self._call("complete_effect", agent, body))

    @staticmethod
    def _run(op: str, data: dict[str, Any]) -> RunState:
        try:
            return RunState(str(data["run_id"]), data["status"], data["state"], int(data["version"]))
        except (KeyError, TypeError, ValueError):
            raise PlatformStateError(op, 200, "invalid response") from None

    @staticmethod
    def _effect(op: str, data: dict[str, Any]) -> EffectState:
        try:
            return EffectState(str(data["effect_id"]), str(data["idempotency_key"]),
                               data["status"], data.get("result"))
        except (KeyError, TypeError):
            raise PlatformStateError(op, 200, "invalid response") from None

    @staticmethod
    def _count(op: str, data: dict[str, Any]) -> int:
        try:
            return int(data["runs"])
        except (KeyError, TypeError, ValueError):
            raise PlatformStateError(op, 200, "invalid response") from None

    def _scope(self, agent: str) -> None:
        if not agent:
            raise ValueError("agent is required")
        if self._agent is None:
            self._agent = agent
        elif agent != self._agent:
            raise StateError(
                f"hosted state has one scope per agent instance; this store already uses "
                f"agent={self._agent!r}, so agent={agent!r} would share its runs"
            )

    async def _call(self, op: str, agent: str, body: dict[str, Any]) -> dict[str, Any]:
        self._scope(agent)
        return await asyncio.to_thread(self._call_sync, op, {**body, **_turn_ref()})

    def _call_sync(self, op: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self._base}/{op}", data=json.dumps(body, allow_nan=False).encode(), method="POST",
            headers={"content-type": "application/json", "authorization": f"Bearer {self._token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:  # noqa: S310 — platform URL
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raise _BY_STATUS.get(exc.code, PlatformStateError)(op, exc.code, "request rejected") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PlatformStateError(op, 0, str(exc)) from None
        try:
            data = json.loads(payload)
        except ValueError:
            raise PlatformStateError(op, 200, "invalid response") from None
        if not isinstance(data, dict):
            raise PlatformStateError(op, 200, "expected an object")
        return data


def _turn_ref() -> dict[str, Any]:
    """The running turn's ``session_id`` and ``turn_id``, or nothing outside a turn.

    The engine stamps the owner from the caller of exactly this turn. A turn
    that carries only one of the two still sends what it has, so the engine
    refuses it rather than treating a customer's turn as background work.
    """
    turn = current_turn()
    ref = {key: turn[key] for key in ("session_id", "turn_id") if turn.get(key) is not None}
    return ref if ref.get("turn_id") or ref.get("session_id") else {}
