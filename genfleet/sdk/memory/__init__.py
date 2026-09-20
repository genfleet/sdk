from __future__ import annotations

import logging
import os
from typing import Literal

from ..schemas import MemoryConfig
from .base import AppendableMemory, Memory
from .platform import MEMORY_TOKEN_ENV, MEMORY_URL_ENV, PlatformMemory, PlatformMemoryError

log = logging.getLogger("genfleet.sdk.memory")


def memory_for(config: MemoryConfig | Literal["platform"] | None) -> Memory | None:
    """Pick a backend from what the author wrote and where the agent runs.

    * ``{"type": "redis", ...}`` — the author's own Redis; unchanged.
    * ``"platform"`` / ``{"type": "platform"}`` — the platform's episodic
      store, reached through the sandbox's memory proxy. Outside a sandbox
      this raises, and the message says which variable is missing.
    * ``None`` — the platform store *if* a sandbox provides it, else no
      memory at all. An agent published to the platform gets memory without
      naming a backend it cannot see; the same code on a laptop stays
      stateless rather than failing.

    The sandbox sets ``GENFLEET_MEMORY_URL`` and ``GENFLEET_MEMORY_TOKEN``
    together. Half a pair is a broken sandbox, and which half decides what
    happens: the URL alone raises even though ``memory`` was omitted, because
    a store was clearly meant; the token alone cannot be dialled, so the agent
    runs stateless and says so in the log rather than failing a deploy over a
    variable the author never set.
    """
    if config is None:
        if os.environ.get(MEMORY_URL_ENV):
            return PlatformMemory.from_env()
        if os.environ.get(MEMORY_TOKEN_ENV):
            log.warning(
                "%s is set but %s is not — running without memory.",
                MEMORY_TOKEN_ENV, MEMORY_URL_ENV,
            )
        return None
    if isinstance(config, str):
        config = MemoryConfig(type=config)
    kind = config["type"]
    if kind == "redis":
        if not config.get("connection"):
            raise ValueError('memory={"type": "redis"} needs a "connection" URL, e.g. "redis://localhost:6379/0"')
        from .redis import RedisMemory
        return RedisMemory(config)
    if kind == "platform":
        return PlatformMemory.from_env()
    raise ValueError(f"Unsupported memory type: {kind!r}")


__all__ = [
    "AppendableMemory",
    "Memory",
    "PlatformMemory",
    "PlatformMemoryError",
    "memory_for",
]
