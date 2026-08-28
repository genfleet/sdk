from __future__ import annotations

import json
import logging

from ..schemas import MemoryConfig, Message

log = logging.getLogger("fleet.sdk.memory.redis")

_TTL = 86400  # 24 hours
_KEY_PREFIX = "fleet:session"


class RedisMemory:
    def __init__(self, config: MemoryConfig) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "Redis memory requires the memory extra: "
                "pip install 'withfleet-sdk[memory]'"
            )
        self._redis = aioredis.from_url(
            config["connection"],
            username=config.get("user"),
            password=config.get("password"),
            decode_responses=True,
        )

    def _key(self, session_id: str) -> str:
        return f"{_KEY_PREFIX}:{session_id}:history"

    async def load(self, session_id: str) -> list[Message]:
        raw = await self._redis.get(self._key(session_id))
        if not raw:
            return []
        try:
            return [Message(**m) for m in json.loads(raw)]
        except Exception:
            log.warning("Failed to deserialise history for session %s", session_id)
            return []

    async def save(self, session_id: str, history: list[Message]) -> None:
        payload = json.dumps([m.model_dump() for m in history])
        await self._redis.set(self._key(session_id), payload, ex=_TTL)

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
