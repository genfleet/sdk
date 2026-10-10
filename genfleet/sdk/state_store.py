"""Async workflow-state interface shared by hosted and local agents.

``agent`` and ``owner`` scope every run. Locally (:class:`LocalStateStore`)
both are whatever the caller passes: pass the verified caller id from your
ingress as ``owner``. Hosted (:class:`PlatformStateStore`) the platform stamps
the scope itself (tenant and agent instance from the sandbox, owner from the
running turn's verified caller), so ``owner`` is ignored there.

Pause/resume is a continuation, not an approval (ADR-0028 §8a, sdk#59).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Literal, Protocol

from .platform_state import STATE_TOKEN_ENV, STATE_URL_ENV, PlatformStateStore
from .state import EffectState, RunState, SQLiteStateStore


class StateStore(Protocol):
    async def create(self, agent: str, owner: str, state: Any, *, subject: str | None = None) -> RunState: ...
    async def get(self, run_id: str, *, agent: str, owner: str) -> RunState: ...
    async def checkpoint(self, run_id: str, *, agent: str, owner: str, version: int, state: Any) -> RunState: ...
    async def pause(self, run_id: str, *, agent: str, owner: str, version: int,
                    state: Any, ttl_seconds: float = 3600) -> tuple[RunState, str]: ...
    async def resume(self, run_id: str, *, agent: str, owner: str, token: str) -> RunState: ...
    async def complete(self, run_id: str, *, agent: str, owner: str, version: int, state: Any) -> RunState: ...
    async def cancel(self, run_id: str, *, agent: str, owner: str, version: int) -> RunState: ...
    async def purge(self, *, agent: str, owner: str) -> int: ...
    async def purge_subject(self, *, agent: str, subject: str) -> int: ...
    async def claim_effect(self, run_id: str, effect_id: str, *, agent: str, owner: str) -> EffectState: ...
    async def get_effect(self, run_id: str, effect_id: str, *, agent: str, owner: str) -> EffectState: ...
    async def complete_effect(self, run_id: str, effect_id: str, *, agent: str,
                              owner: str, result: Any) -> EffectState: ...


class LocalStateStore:
    """Async wrapper for the local SQLite implementation."""

    def __init__(self, path: str | Path):
        self._store = SQLiteStateStore(path)

    async def create(self, agent: str, owner: str, state: Any, *, subject: str | None = None) -> RunState:
        return await asyncio.to_thread(self._store.create, agent, owner, state, subject=subject)

    async def get(self, run_id: str, *, agent: str, owner: str) -> RunState:
        return await asyncio.to_thread(self._store.get, run_id, agent=agent, owner=owner)

    async def checkpoint(self, run_id: str, *, agent: str, owner: str, version: int, state: Any) -> RunState:
        return await asyncio.to_thread(self._store.checkpoint, run_id, agent=agent, owner=owner, version=version, state=state)

    async def pause(self, run_id: str, *, agent: str, owner: str, version: int,
                    state: Any, ttl_seconds: float = 3600) -> tuple[RunState, str]:
        return await asyncio.to_thread(self._store.pause, run_id, agent=agent, owner=owner,
                                       version=version, state=state, ttl_seconds=ttl_seconds)

    async def resume(self, run_id: str, *, agent: str, owner: str, token: str) -> RunState:
        return await asyncio.to_thread(self._store.resume, run_id, agent=agent, owner=owner, token=token)

    async def complete(self, run_id: str, *, agent: str, owner: str, version: int, state: Any) -> RunState:
        return await asyncio.to_thread(self._store.complete, run_id, agent=agent, owner=owner, version=version, state=state)

    async def cancel(self, run_id: str, *, agent: str, owner: str, version: int) -> RunState:
        return await asyncio.to_thread(self._store.cancel, run_id, agent=agent, owner=owner, version=version)

    async def purge(self, *, agent: str, owner: str) -> int:
        return await asyncio.to_thread(self._store.purge, agent=agent, owner=owner)

    async def purge_subject(self, *, agent: str, subject: str) -> int:
        return await asyncio.to_thread(self._store.purge_subject, agent=agent, subject=subject)

    async def claim_effect(self, run_id: str, effect_id: str, *, agent: str, owner: str) -> EffectState:
        return await asyncio.to_thread(self._store.claim_effect, run_id, effect_id, agent=agent, owner=owner)

    async def get_effect(self, run_id: str, effect_id: str, *, agent: str, owner: str) -> EffectState:
        return await asyncio.to_thread(self._store.get_effect, run_id, effect_id, agent=agent, owner=owner)

    async def complete_effect(self, run_id: str, effect_id: str, *, agent: str,
                              owner: str, result: Any) -> EffectState:
        return await asyncio.to_thread(self._store.complete_effect, run_id, effect_id,
                                       agent=agent, owner=owner, result=result)


def state_for(explicit: StateStore | Literal["platform"] | None = None) -> StateStore | None:
    """Use the platform proxy when provided, else an explicit local store."""
    if explicit == "platform":
        return PlatformStateStore.from_env()
    if explicit is not None:
        return explicit
    if os.environ.get(STATE_URL_ENV) or os.environ.get(STATE_TOKEN_ENV):
        return PlatformStateStore.from_env()
    return None
