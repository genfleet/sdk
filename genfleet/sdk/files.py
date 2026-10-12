"""Reading a file part's bytes: the ``FileResolver`` seam.

A media part either carries its bytes inline (``data``) or names a
platform-issued reference (``uri``). ``read_part`` returns the bytes either
way; a reference goes through a ``FileResolver``. The platform implements one
that asks it for the file with the running turn's credentials, so it can
check that this caller, in this tenant, may read it and that it has not
expired. ``InMemoryFileResolver`` is a local one for tests and examples.

The SDK ships no resolver that fetches arbitrary URLs: a reference is read
only by a resolver that recognises it, so a caller can't make an agent fetch
an internal address (SSRF) by sending a ``uri``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .parts import MediaPart, part_for


class FileAccessError(Exception):
    """A file part's bytes could not be read. ``reason`` is safe to show a model or a caller."""

    reason = "unavailable"


class FileNotFound(FileAccessError):
    """The reference names no file this resolver knows, or the caller may not read it.

    The two are one error on purpose: a caller learns nothing about files it
    may not see.
    """

    reason = "not found or not accessible"


class FileExpired(FileAccessError):
    reason = "expired"


class FileTooLarge(FileAccessError):
    reason = "too large"


class UnsupportedReference(FileAccessError):
    """No resolver handles this reference's scheme."""

    reason = "not readable here"


@runtime_checkable
class FileResolver(Protocol):
    """Reads the file a reference names, for the running turn.

    **Access is decided by the platform, never here.** A resolver forwards
    the turn's platform credential (the per-spawn token, the engine-verified
    caller) to the platform, which checks the file against the caller and
    tenant it authenticated, and its expiry. ``metadata`` is the turn's
    ``AgentInput.metadata``, and much of it is caller-supplied: under
    ``serve``, request metadata passes through and ``contextId`` becomes
    ``session_id``. Never grant access from ids in it.

    Raise ``FileNotFound`` (also for no access), ``FileExpired`` or
    ``FileTooLarge``; never return more than ``max_bytes`` bytes, and stop
    reading past them rather than reading and then checking.
    """

    def handles(self, uri: str) -> bool: ...

    async def read(self, uri: str, *, max_bytes: int, metadata: Mapping[str, Any]) -> bytes: ...


async def read_part(
    part: MediaPart,
    *,
    max_bytes: int,
    metadata: Mapping[str, Any],
    resolver: FileResolver | None,
) -> bytes:
    """The part's bytes: inline ``data``, else its ``uri`` through ``resolver``."""
    if part.data is not None:
        if len(part.data) > max_bytes:
            raise FileTooLarge()
        return part.data
    if part.size is not None and part.size > max_bytes:
        raise FileTooLarge()
    if resolver is None or not part.uri or not resolver.handles(part.uri):
        raise UnsupportedReference()
    data = await resolver.read(part.uri, max_bytes=max_bytes, metadata=metadata)
    if len(data) > max_bytes:
        raise FileTooLarge()
    return data


@dataclass
class _Stored:
    data: bytes
    expires_at: float | None
    session_id: str | None


class InMemoryFileResolver:
    """Files kept in this process, under ``mem://`` references. **For tests only.**

    ``put`` stores bytes and returns a part that references them. A file put
    with a ``ttl`` (seconds) expires after it. A file put with a
    ``session_id`` is readable only on a turn whose ``metadata["session_id"]``
    matches: that exercises the "not accessible" path in tests, and is not
    access control (a ``serve`` caller sets ``session_id`` via ``contextId``).
    """

    SCHEME = "mem://"

    def __init__(self, *, clock: Any = time.monotonic) -> None:
        self._files: dict[str, _Stored] = {}
        self._clock = clock

    def put(
        self,
        data: bytes,
        *,
        mime_type: str,
        name: str | None = None,
        ttl: float | None = None,
        session_id: str | None = None,
    ) -> MediaPart:
        uri = f"{self.SCHEME}{uuid.uuid4().hex}"
        expires = self._clock() + ttl if ttl is not None else None
        self._files[uri] = _Stored(bytes(data), expires, session_id)
        return part_for(mime_type, uri=uri, name=name, size=len(data))

    def handles(self, uri: str) -> bool:
        return uri.startswith(self.SCHEME)

    async def read(self, uri: str, *, max_bytes: int, metadata: Mapping[str, Any]) -> bytes:
        stored = self._files.get(uri)
        if stored is None or (stored.session_id is not None and metadata.get("session_id") != stored.session_id):
            raise FileNotFound()
        if stored.expires_at is not None and self._clock() >= stored.expires_at:
            raise FileExpired()
        if len(stored.data) > max_bytes:
            raise FileTooLarge()
        return stored.data


__all__ = [
    "FileAccessError",
    "FileExpired",
    "FileNotFound",
    "FileResolver",
    "FileTooLarge",
    "InMemoryFileResolver",
    "UnsupportedReference",
    "read_part",
]
