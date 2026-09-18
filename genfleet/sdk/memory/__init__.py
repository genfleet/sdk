from __future__ import annotations

import os

from ..schemas import MemoryConfig
from .base import AppendableMemory, Memory
from .platform import MEMORY_URL_ENV, PlatformMemory


def memory_for(config: MemoryConfig | str | None) -> Memory | None:
    """Pick a backend from what the author wrote and where the agent runs.

    * ``{"type": "redis", ...}`` — the author's own Redis; unchanged.
    * ``"platform"`` / ``{"type": "platform"}`` — the platform's episodic
      store, reached through the sandbox's memory proxy. Outside a sandbox
      this raises, and the message says which variable is missing.
    * ``None`` — the platform store *if* a sandbox provides it, else no
      memory at all. An agent published to the platform gets memory without
      naming a backend it cannot see; the same code on a laptop stays
      stateless rather than failing.
    """
    if config is None:
        return PlatformMemory.from_env() if os.environ.get(MEMORY_URL_ENV) else None
    if isinstance(config, str):
        config = {"type": config}  # type: ignore[assignment]
    kind = config["type"]
    if kind == "redis":
        from .redis import RedisMemory
        return RedisMemory(config)
    if kind == "platform":
        return PlatformMemory.from_env()
    raise ValueError(f"Unsupported memory type: {kind!r}")


__all__ = ["AppendableMemory", "Memory", "PlatformMemory", "memory_for"]
