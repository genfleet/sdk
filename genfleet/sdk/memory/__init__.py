from __future__ import annotations

from ..schemas import MemoryConfig
from .base import Memory


def memory_for(config: MemoryConfig) -> Memory:
    if config["type"] == "redis":
        from .redis import RedisMemory
        return RedisMemory(config)
    raise ValueError(f"Unsupported memory type: {config['type']!r}")


__all__ = ["Memory", "memory_for"]
