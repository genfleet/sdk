"""Typed parts on AgentInput/AgentOutput, and the file-reference resolver."""

from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from genfleet.sdk import (
    MAX_INLINE_BYTES,
    AgentInput,
    AgentOutput,
    AudioPart,
    FileExpired,
    FileNotFound,
    FilePart,
    FileTooLarge,
    ImagePart,
    InMemoryFileResolver,
    TextPart,
    UnsupportedReference,
    VideoPart,
    part_for,
    read_part,
)
from genfleet.sdk.parts import accepts, safe_name


def test_a_text_only_input_is_unchanged_and_gains_a_text_part() -> None:
    turn = AgentInput(message="hi")
    assert turn.message == "hi"
    assert turn.parts == [TextPart(text="hi")]


def test_message_is_the_text_parts_joined() -> None:
    turn = AgentInput(parts=[TextPart(text="look "), part_for("image/png", data=b"x"), TextPart(text="here")])
    assert turn.message == "look here"


def test_a_message_beside_file_parts_becomes_the_first_part() -> None:
    image = part_for("image/png", data=b"x")
    turn = AgentInput(message="what is this?", parts=[image])
    assert turn.parts == [TextPart(text="what is this?"), image]


def test_a_file_only_input_has_an_empty_message() -> None:
    turn = AgentInput(parts=[part_for("application/pdf", uri="mem://a")])
    assert turn.message == ""
    assert len(turn.parts) == 1


@pytest.mark.parametrize(
    ("mime", "kind"),
    [("image/png", ImagePart), ("audio/ogg", AudioPart), ("video/mp4", VideoPart), ("application/pdf", FilePart)],
)
def test_the_kind_follows_the_mime_type(mime: str, kind: type) -> None:
    assert type(part_for(mime, data=b"1")) is kind


def test_mime_types_are_normalised() -> None:
    assert part_for("Image/PNG; charset=binary", data=b"1").mime_type == "image/png"
    assert part_for(None, data=b"1").mime_type == "application/octet-stream"
    with pytest.raises(ValueError):
        part_for("not a mime", data=b"1")


def test_a_part_needs_data_or_a_uri() -> None:
    with pytest.raises(ValidationError):
        FilePart(mime_type="application/pdf")


def test_inline_size_is_the_data_length() -> None:
    assert part_for("text/csv", data=b"abc", size=999).size == 3


def test_inline_data_is_capped() -> None:
    with pytest.raises(ValidationError, match="by reference"):
        part_for("application/octet-stream", data=b"x" * (MAX_INLINE_BYTES + 1))
    with pytest.raises(ValidationError, match="by reference"):
        FilePart(data=base64.b64encode(b"x" * (MAX_INLINE_BYTES + 3)).decode())


@pytest.mark.parametrize("uri", ["no-scheme", "mem://a b", "x" * 3000, "mem://\nx"])
def test_a_uri_must_be_an_absolute_reference(uri: str) -> None:
    with pytest.raises(ValidationError):
        part_for("application/pdf", uri=uri)


@pytest.mark.parametrize(
    ("raw", "safe"),
    [("../../etc/passwd", "passwd"), ("C:\\Users\\a\\b.pdf", "b.pdf"), ("a\x00b\n.txt", "ab.txt"), ("..", None), ("/", None)],
)
def test_names_are_display_names_only(raw: str, safe: str | None) -> None:
    assert safe_name(raw) == safe
    assert part_for("text/plain", data=b"1", name=raw).name == safe


def test_parts_round_trip_through_json_with_bytes_as_base64() -> None:
    turn = AgentInput(parts=[TextPart(text="hi"), part_for("image/png", data=b"\x89PNG", name="a.png")])
    wire = turn.model_dump_json()
    assert base64.b64encode(b"\x89PNG").decode() in wire
    again = AgentInput.model_validate_json(wire)
    assert again == turn
    assert AgentInput.model_validate(turn.model_dump(mode="json")) == turn


def test_output_content_is_the_text_parts_when_left_out() -> None:
    out = AgentOutput(parts=[TextPart(text="see "), part_for("text/csv", data=b"a,b"), TextPart(text="attached")])
    assert out.content == "see attached"
    assert AgentOutput(content="set", parts=[TextPart(text="other")]).content == "set"
    assert AgentOutput(parts=[part_for("text/csv", data=b"a")]).content is None


def test_describe_never_includes_the_bytes() -> None:
    part = part_for("application/pdf", data=b"SECRET-BYTES", name='q"1.pdf')
    text = part.describe()
    assert "SECRET" not in text
    assert text == '[Attached file: name="q\\"1.pdf", type=application/pdf, size=12 bytes]'


def test_mime_patterns() -> None:
    assert accepts("image/png", ["image/*"])
    assert accepts("application/pdf", ["*/*"])
    assert accepts("application/pdf", ["application/pdf"])
    assert not accepts("image/png", ["application/pdf", "audio/*"])
    assert not accepts("image/png", [])


# ---------------------------------------------------------------------------
# FileResolver
# ---------------------------------------------------------------------------


async def test_inline_bytes_need_no_resolver() -> None:
    part = part_for("image/png", data=b"abc")
    assert await read_part(part, max_bytes=10, metadata={}, resolver=None) == b"abc"
    with pytest.raises(FileTooLarge):
        await read_part(part, max_bytes=2, metadata={}, resolver=None)


async def test_a_reference_is_read_through_the_resolver() -> None:
    files = InMemoryFileResolver()
    part = files.put(b"%PDF", mime_type="application/pdf", name="r.pdf")
    assert isinstance(part, FilePart) and part.uri.startswith("mem://") and part.size == 4
    assert await read_part(part, max_bytes=10, metadata={}, resolver=files) == b"%PDF"


async def test_an_unknown_reference_is_not_found() -> None:
    files = InMemoryFileResolver()
    with pytest.raises(FileNotFound):
        await read_part(part_for("image/png", uri="mem://nope"), max_bytes=10, metadata={}, resolver=files)


async def test_a_reference_no_resolver_handles_is_never_fetched() -> None:
    files = InMemoryFileResolver()
    for resolver in (None, files):
        with pytest.raises(UnsupportedReference):
            await read_part(
                part_for("image/png", uri="http://169.254.169.254/latest"), max_bytes=10, metadata={}, resolver=resolver
            )


async def test_a_file_of_another_session_is_not_found() -> None:
    files = InMemoryFileResolver()
    part = files.put(b"x", mime_type="image/png", session_id="s1")
    assert await read_part(part, max_bytes=10, metadata={"session_id": "s1"}, resolver=files) == b"x"
    with pytest.raises(FileNotFound):
        await read_part(part, max_bytes=10, metadata={"session_id": "s2"}, resolver=files)


async def test_an_expired_file_is_refused() -> None:
    now = [100.0]
    files = InMemoryFileResolver(clock=lambda: now[0])
    part = files.put(b"x", mime_type="image/png", ttl=60)
    assert await read_part(part, max_bytes=10, metadata={}, resolver=files) == b"x"
    now[0] = 160.0
    with pytest.raises(FileExpired):
        await read_part(part, max_bytes=10, metadata={}, resolver=files)


async def test_a_reference_larger_than_the_limit_is_refused_before_reading() -> None:
    files = InMemoryFileResolver()
    part = files.put(b"x" * 100, mime_type="video/mp4")
    with pytest.raises(FileTooLarge):
        await read_part(part, max_bytes=10, metadata={}, resolver=files)
