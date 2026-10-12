"""Typed parts of a turn: text, image, audio, video and file.

``AgentInput.parts`` and ``AgentOutput.parts`` carry a turn's content as a
list of parts. Text agents can ignore them: ``AgentInput.message`` is the
turn's text and ``AgentOutput.content`` the reply's, as before.

A media part (image, audio, video, file) has one shape, whatever its kind:
``mime_type``, an optional ``name`` and ``size``, and its bytes either inline
(``data``, capped at ``MAX_INLINE_BYTES``) or by reference (``uri``). Large
media travel by reference: the platform stores the file and issues a
reference, and an agent reads it through a :class:`~genfleet.sdk.files.FileResolver`
that checks access and expiry. The SDK never fetches a reference itself.

The kind follows the MIME type (``part_for``): ``image/*`` is an
``ImagePart``, ``audio/*`` an ``AudioPart``, ``video/*`` a ``VideoPart``, and
anything else a ``FilePart``. Processors for a kind (an image model, a
transcriber) read the same fields, so adding one never changes the contract.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Annotated, Any, Iterable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: The largest file a part may carry inline, in bytes (1 MiB). Anything larger
#: goes by reference: inline bytes are read into memory, JSON-encoded and
#: copied through every hop of a request.
MAX_INLINE_BYTES = 1024 * 1024

#: The longest ``name`` a part keeps; a longer one is cut.
MAX_NAME_CHARS = 255

#: The MIME type of a file that arrives without one.
DEFAULT_MIME_TYPE = "application/octet-stream"

_MIME = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,126}/[a-z0-9][a-z0-9!#$&^_.+-]{0,126}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:[^\s\x00-\x1f\x7f]+$")

#: The longest ``uri`` a part may carry.
MAX_URI_CHARS = 2048


def normalize_mime_type(value: str | None) -> str:
    """``value`` lowercased without parameters (``; charset=…``), or ``DEFAULT_MIME_TYPE`` if empty.

    Raises ``ValueError`` when it is not ``type/subtype``.
    """
    if not value or not value.strip():
        return DEFAULT_MIME_TYPE
    mime = value.split(";", 1)[0].strip().lower()
    if not _MIME.match(mime):
        raise ValueError("mime_type must be 'type/subtype'")
    return mime


def safe_name(value: str | None) -> str | None:
    """A display name for a file: the last path segment, without control characters.

    A part's name comes from whoever sent the file. It is shown to models and
    people, never used as a path: this strips directories (``../../etc/passwd``
    becomes ``passwd``) so a consumer that does write it to disk cannot be
    steered out of its directory by it.
    """
    if value is None:
        return None
    name = _CONTROL.sub("", value).replace("\\", "/").rsplit("/", 1)[-1].strip()
    if name in ("", ".", ".."):
        return None
    return name[:MAX_NAME_CHARS]


def mime_matches(mime_type: str, pattern: str) -> bool:
    """Whether ``mime_type`` matches ``pattern``: exact, ``type/*`` or ``*/*``."""
    pattern = pattern.strip().lower()
    if pattern in ("*", "*/*"):
        return True
    if pattern.endswith("/*"):
        return mime_type.startswith(pattern[:-1])
    return mime_type == pattern


def mime_patterns(patterns: Iterable[str] | None) -> tuple[str, ...]:
    """``patterns`` validated and normalised: each ``type/subtype``, ``type/*`` or ``*/*``."""
    if isinstance(patterns, str):
        raise TypeError("MIME types must be a list, not a string")
    normalised = []
    for pattern in patterns or ():
        if not isinstance(pattern, str):
            raise TypeError("MIME types must be strings")
        value = pattern.strip().lower()
        if value in ("*", "*/*"):
            normalised.append("*/*")
        elif value.endswith("/*"):
            normalize_mime_type(value[:-1] + "x")
            normalised.append(value)
        else:
            normalised.append(normalize_mime_type(value))
    return tuple(normalised)


_IMAGE_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
}


def looks_like(mime_type: str, data: bytes) -> bool:
    """Whether ``data`` starts like a ``mime_type`` image (PNG, JPEG, GIF, WebP).

    ``True`` for any other type: only these are checked.
    """
    if mime_type == "image/webp":
        return data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    signatures = _IMAGE_SIGNATURES.get(mime_type)
    return signatures is None or data.startswith(signatures)


def accepts(mime_type: str, patterns: Iterable[str]) -> bool:
    """Whether any of ``patterns`` matches ``mime_type``."""
    return any(mime_matches(mime_type, p) for p in patterns)


class UnsupportedInput(ValueError):
    """A turn carries a file of a type the agent does not accept."""

    def __init__(self, unsupported: Iterable[str], accepted: Iterable[str]) -> None:
        self.unsupported = sorted(set(unsupported))
        self.accepted = ["text/plain", *accepted]
        super().__init__(
            f"Unsupported input type: {', '.join(self.unsupported)}. "
            f"This agent accepts: {', '.join(self.accepted)}."
        )


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class _MediaPart(BaseModel):
    """The fields every non-text part shares. See the module docstring."""

    model_config = ConfigDict(ser_json_bytes="base64")

    mime_type: str = DEFAULT_MIME_TYPE
    #: The sender's file name, for display only (``safe_name``).
    name: str | None = None
    #: Size in bytes: ``len(data)`` when inline; for a reference, what the
    #: issuer declared (a resolver enforces its own limit when reading).
    size: int | None = Field(default=None, ge=0)
    #: A platform-issued reference, read through a ``FileResolver``.
    uri: str | None = None
    #: Inline bytes, at most ``MAX_INLINE_BYTES``. JSON carries them base64.
    data: bytes | None = None

    @field_validator("mime_type", mode="before")
    @classmethod
    def _mime(cls, value: Any) -> str:
        return normalize_mime_type(value if isinstance(value, str) else None)

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, value: Any) -> str | None:
        return safe_name(value) if isinstance(value, str) else None

    @field_validator("uri")
    @classmethod
    def _uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) > MAX_URI_CHARS or not _URI.match(value):
            raise ValueError("uri must be an absolute 'scheme:…' reference without whitespace")
        return value

    @field_validator("data", mode="before")
    @classmethod
    def _data(cls, value: Any) -> Any:
        # A string is base64 (the JSON form); bytes are the bytes.
        if isinstance(value, str):
            if len(value) > (MAX_INLINE_BYTES * 4) // 3 + 4:
                raise ValueError(f"inline data exceeds {MAX_INLINE_BYTES} bytes; send the file by reference")
            try:
                return base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("inline data must be base64") from exc
        return value

    @model_validator(mode="after")
    def _content(self) -> "_MediaPart":
        if self.data is None and not self.uri:
            raise ValueError("a file part needs data or a uri")
        if self.data is not None:
            if len(self.data) > MAX_INLINE_BYTES:
                raise ValueError(f"inline data exceeds {MAX_INLINE_BYTES} bytes; send the file by reference")
            self.size = len(self.data)
        return self

    def describe(self) -> str:
        """A one-line description for a model: kind, name, type and size; never the bytes."""
        fields = []
        if self.name:
            # Quoted: the name is the sender's text, not an instruction.
            fields.append(f"name={_quote(self.name)}")
        fields.append(f"type={self.mime_type}")
        if self.size is not None:
            fields.append(f"size={self.size} bytes")
        return f"[Attached {self.type}: {', '.join(fields)}]"  # type: ignore[attr-defined]


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class ImagePart(_MediaPart):
    type: Literal["image"] = "image"


class AudioPart(_MediaPart):
    type: Literal["audio"] = "audio"


class VideoPart(_MediaPart):
    type: Literal["video"] = "video"


class FilePart(_MediaPart):
    type: Literal["file"] = "file"


MediaPart = Union[ImagePart, AudioPart, VideoPart, FilePart]
Part = Annotated[Union[TextPart, ImagePart, AudioPart, VideoPart, FilePart], Field(discriminator="type")]

_KINDS: dict[str, type[_MediaPart]] = {"image/": ImagePart, "audio/": AudioPart, "video/": VideoPart}


def part_for(
    mime_type: str | None,
    *,
    data: bytes | str | None = None,
    uri: str | None = None,
    name: str | None = None,
    size: int | None = None,
) -> MediaPart:
    """The part of the kind ``mime_type`` names, e.g. ``ImagePart`` for ``image/png``."""
    mime = normalize_mime_type(mime_type)
    cls = next((c for prefix, c in _KINDS.items() if mime.startswith(prefix)), FilePart)
    return cls(mime_type=mime, data=data, uri=uri, name=name, size=size)  # type: ignore[return-value]


def text_of(parts: Iterable[Any]) -> str:
    """The text parts, concatenated."""
    return "".join(p.text for p in parts if isinstance(p, TextPart))


def media_of(parts: Iterable[Any]) -> list[MediaPart]:
    """The non-text parts, in order."""
    return [p for p in parts if isinstance(p, _MediaPart)]  # type: ignore[misc]


__all__ = [
    "DEFAULT_MIME_TYPE",
    "MAX_INLINE_BYTES",
    "AudioPart",
    "FilePart",
    "ImagePart",
    "MediaPart",
    "Part",
    "TextPart",
    "UnsupportedInput",
    "VideoPart",
    "accepts",
    "media_of",
    "looks_like",
    "mime_matches",
    "mime_patterns",
    "normalize_mime_type",
    "part_for",
    "safe_name",
    "text_of",
]
