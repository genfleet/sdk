"""Workflow state through the engine's tenant-scoped sandbox proxy."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

from .state import EffectState, RunState, StateError

STATE_URL_ENV = "GENFLEET_STATE_URL"
STATE_TOKEN_ENV = "GENFLEET_STATE_TOKEN"


class PlatformStateError(StateError):
    def __init__(self, op: str, status: int, detail: str) -> None:
        super().__init__(f"platform state {op} failed ({status}): {detail}")
        self.op = op
        self.status = status


class PlatformStateStore:
    """The platform resolves tenant and agent from a per-spawn token.

    ``agent`` keeps the same SDK API as the local store; it is never sent on
    the wire. The proxy stamps its own agent scope. ``owner`` is a run key,
    not authorization to approve another caller's action.
    """

    def __init__(self, base_url: str, token: str):
        if not base_url or not token:
            raise ValueError("PlatformStateStore needs a URL and token")
        self._base = base_url.rstrip("/")
        self._token = token

    @classmethod
    def from_env(cls) -> PlatformStateStore:
        url, token = os.environ.get(STATE_URL_ENV), os.environ.get(STATE_TOKEN_ENV)
        if not url or not token:
            raise RuntimeError(f"platform state needs {STATE_URL_ENV if not url else STATE_TOKEN_ENV}")
        return cls(url, token)

    async def create(self, agent: str, owner: str, state: Any, *, subject: str | None = None) -> RunState:
        return self._run(await self._call("create", {"owner": owner, "state": state, "subject": subject}))

    async def get(self, run_id: str, *, agent: str, owner: str) -> RunState:
        return self._run(await self._call("get", {"run_id": run_id, "owner": owner}))

    async def checkpoint(self, run_id: str, *, agent: str, owner: str, version: int, state: Any) -> RunState:
        return self._run(await self._call("checkpoint", {"run_id": run_id, "owner": owner, "version": version, "state": state}))

    async def pause(self, run_id: str, *, agent: str, owner: str, version: int,
                    state: Any, ttl_seconds: float = 3600) -> tuple[RunState, str]:
        data = await self._call("pause", {"run_id": run_id, "owner": owner, "version": version,
                                          "state": state, "ttl_seconds": ttl_seconds})
        token = data.pop("token")
        return self._run(data), token

    async def resume(self, run_id: str, *, agent: str, owner: str, token: str) -> RunState:
        return self._run(await self._call("resume", {"run_id": run_id, "owner": owner, "token": token}))

    async def complete(self, run_id: str, *, agent: str, owner: str, version: int, state: Any) -> RunState:
        return self._run(await self._call("complete", {"run_id": run_id, "owner": owner, "version": version, "state": state}))

    async def cancel(self, run_id: str, *, agent: str, owner: str, version: int) -> RunState:
        return self._run(await self._call("cancel", {"run_id": run_id, "owner": owner, "version": version}))

    async def purge(self, *, agent: str, owner: str) -> int:
        return int((await self._call("purge", {"owner": owner}))["runs"])

    async def purge_subject(self, *, agent: str, subject: str) -> int:
        return int((await self._call("purge_subject", {"subject": subject}))["runs"])

    async def claim_effect(self, run_id: str, effect_id: str, *, agent: str, owner: str) -> EffectState:
        return self._effect(await self._call("claim_effect", {"run_id": run_id, "effect_id": effect_id, "owner": owner}))

    async def get_effect(self, run_id: str, effect_id: str, *, agent: str, owner: str) -> EffectState:
        return self._effect(await self._call("get_effect", {"run_id": run_id, "effect_id": effect_id, "owner": owner}))

    async def complete_effect(self, run_id: str, effect_id: str, *, agent: str,
                              owner: str, result: Any) -> EffectState:
        return self._effect(await self._call("complete_effect", {"run_id": run_id,
                            "effect_id": effect_id, "owner": owner, "result": result}))

    @staticmethod
    def _run(data: dict[str, Any]) -> RunState:
        return RunState(data["run_id"], data["status"], data["state"], data["version"])

    @staticmethod
    def _effect(data: dict[str, Any]) -> EffectState:
        return EffectState(data["effect_id"], data["idempotency_key"], data["status"], data.get("result"))

    async def _call(self, op: str, body: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._call_sync, op, body)

    def _call_sync(self, op: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self._base}/{op}", data=json.dumps(body, allow_nan=False).encode(), method="POST",
            headers={"content-type": "application/json", "authorization": f"Bearer {self._token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:  # noqa: S310 — platform URL
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raise PlatformStateError(op, exc.code, "request rejected") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PlatformStateError(op, 0, str(exc)) from None
        try:
            data = json.loads(payload)
        except ValueError:
            raise PlatformStateError(op, 200, "invalid response") from None
        if not isinstance(data, dict):
            raise PlatformStateError(op, 200, "expected an object")
        return data
