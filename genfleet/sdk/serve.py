from __future__ import annotations

import base64
import json
import logging
import uuid
from urllib.parse import urlsplit
from typing import Any, AsyncIterator, Iterable

from pydantic import ValidationError

from ._redaction import safe_arguments
from .parts import (
    MAX_INLINE_BYTES,
    MediaPart,
    Part,
    TextPart,
    UnsupportedInput,
    accepts,
    media_of,
    mime_patterns,
    part_for,
)
from .protocol import AgentProtocol
from .schemas import TOOL_EVENT_KEY, AgentInput, AgentOutput, Message

log = logging.getLogger("genfleet.sdk.serve")

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from sse_starlette.sse import EventSourceResponse
except ImportError as exc:
    raise ImportError(
        "The serve helper requires the serve extra. "
        "Install it with: pip install 'genfleet-sdk[serve]'"
    ) from exc


#: A2A v1.0 (the JSON-RPC binding, A2A v1.0.1 proto in ProtoJSON form): the
#: card path, version header and the error codes this server answers with.
PROTOCOL_VERSION = "1.0"
VERSION_HEADER = "A2A-Version"
CARD_PATH = "/.well-known/agent-card.json"
#: The 0.x card path, served as an alias for one minor release.
LEGACY_CARD_PATH = "/.well-known/agent.json"
#: Request ``metadata`` key for the turn's prior messages (``Message`` dicts).
#: Not an A2A field; a conforming peer ignores it. Removed from the metadata
#: the agent sees.
HISTORY_METADATA_KEY = "genfleet.history"
_RESULT_ARTIFACT = "result"

_INVALID_REQUEST = -32600
_INVALID_PARAMS = -32602
_CONTENT_TYPE_NOT_SUPPORTED = -32005
_UNSUPPORTED_OPERATION = -32004
_PUSH_NOT_SUPPORTED = -32003
_EXTENDED_CARD_NOT_CONFIGURED = -32007
_VERSION_NOT_SUPPORTED = -32009
_TASK_STORE_METHODS = frozenset({"GetTask", "ListTasks", "CancelTask", "SubscribeToTask"})
_PUSH_METHODS = frozenset({
    "CreateTaskPushNotificationConfig",
    "GetTaskPushNotificationConfig",
    "ListTaskPushNotificationConfigs",
    "DeleteTaskPushNotificationConfig",
})


def _history(raw: Any) -> list[Message]:
    if not isinstance(raw, list):
        return []
    try:
        return [Message(**m) for m in raw]
    except Exception:
        log.warning("Ignoring invalid history in A2A request")
        return []


class _PartError(Exception):
    """A request part the agent can't take: answered as a JSON-RPC error."""

    def __init__(self, code: int, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


#: The attribute an agent declares the MIME types of the files it accepts on
#: (``"image/*"``, ``"application/pdf"``). Text is always accepted.
ACCEPTED_INPUT_TYPES_ATTR = "accepted_input_types"
#: The attribute an agent declares the MIME types of the files it may return
#: on; the card lists them as ``defaultOutputModes`` beside text.
OUTPUT_TYPES_ATTR = "output_types"
_JSON_MIME = "application/json"


def _part_kind(raw: dict[str, Any]) -> str | None:
    """The 0.x discriminator (``type``, or ``kind`` in A2A 0.3), if any."""
    kind = raw.get("type", raw.get("kind"))
    return kind if isinstance(kind, str) else None


def _media_part(mime: Any, *, data: Any = None, uri: Any = None, name: Any = None, size: Any = None) -> MediaPart:
    if isinstance(data, str) and len(data) > (MAX_INLINE_BYTES * 4) // 3 + 4:
        raise _PartError(
            _INVALID_PARAMS,
            f"An inline file part is larger than {MAX_INLINE_BYTES} bytes; send it by reference (url).",
        )
    try:
        return part_for(
            mime if isinstance(mime, str) else None,
            data=data,
            uri=uri if isinstance(uri, str) else None,
            name=name if isinstance(name, str) else None,
            size=size if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else None,
        )
    except (ValidationError, ValueError) as exc:
        detail = "larger than the inline limit; send it by reference (url)" if "exceeds" in str(exc) else "malformed"
        raise _PartError(_INVALID_PARAMS, f"Invalid file part: {detail}.") from exc


def _json_part(value: Any, mime: Any) -> MediaPart:
    encoded = json.dumps(value, separators=(",", ":")).encode()
    if len(encoded) > MAX_INLINE_BYTES:
        raise _PartError(_INVALID_PARAMS, f"A data part is larger than {MAX_INLINE_BYTES} bytes.")
    return _media_part(mime if isinstance(mime, str) and mime else _JSON_MIME, data=encoded)


def _part_in(raw: Any) -> Part | None:
    """One A2A part (v1.0 or 0.x) as a typed part; ``None`` for one with no content."""
    if not isinstance(raw, dict):
        raise _PartError(_INVALID_PARAMS, "Invalid request parameters: a part must be an object.")
    kind = _part_kind(raw)
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    if kind == "file":  # 0.x FilePart
        file = raw.get("file")
        if not isinstance(file, dict):
            raise _PartError(_INVALID_PARAMS, "Invalid file part: malformed.")
        return _media_part(
            file.get("mimeType"), data=file.get("bytes"), uri=file.get("uri"),
            name=file.get("name"), size=metadata.get("size"),
        )
    if kind == "data":  # 0.x DataPart
        return _json_part(raw.get("data"), raw.get("mediaType"))
    if isinstance(raw.get("text"), str):
        return TextPart(text=raw["text"])
    if kind == "text":
        return None
    # v1.0: exactly one of raw, url, data, with mediaType and filename beside it.
    if "raw" in raw or "url" in raw:
        return _media_part(
            raw.get("mediaType"), data=raw.get("raw"), uri=raw.get("url"),
            name=raw.get("filename"), size=metadata.get("size"),
        )
    if "data" in raw:
        return _json_part(raw["data"], raw.get("mediaType"))
    raise _PartError(_INVALID_PARAMS, "Invalid request parameters: unrecognised part.")


def _parts_in(raw_parts: list[Any], accepted: tuple[str, ...]) -> list[Part]:
    """The request's parts, typed; raises ``_PartError`` for one the agent does not accept."""
    parts = [p for p in (_part_in(raw) for raw in raw_parts) if p is not None]
    refused = sorted({p.mime_type for p in media_of(parts) if not accepts(p.mime_type, accepted)})
    if refused:
        error = UnsupportedInput(refused, accepted)
        raise _PartError(
            _CONTENT_TYPE_NOT_SUPPORTED, str(error), {"unsupported": error.unsupported, "accepted": error.accepted}
        )
    return parts


def _build_v1_input(
    params: dict[str, Any], *, accepted: tuple[str, ...] = (), accept_history: bool = False
) -> AgentInput | None:
    """A v1.0 ``SendMessageRequest`` as the agent's turn, or ``None`` if malformed.

    The message's parts are the turn's parts (text, and the files the agent
    accepts; ``_PartError`` for one it does not); request ``metadata`` passes
    through; the message's ``contextId`` is ``metadata["session_id"]``, over
    any ``session_id`` in the metadata (a gateway in front may scope
    ``contextId`` per caller; metadata must not route around it), which is
    used only when there is no ``contextId``. ``metadata["genfleet.history"]``
    is always removed from the metadata, and forwarded as the turn's history
    only with ``accept_history``.
    """
    message = params.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("parts"), list):
        return None
    parts = _parts_in(message["parts"], accepted)
    metadata = params.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    raw_history = metadata.pop(HISTORY_METADATA_KEY, None)
    history = _history(raw_history) if accept_history else []
    context_id = message.get("contextId")
    if isinstance(context_id, str) and context_id:
        metadata["session_id"] = context_id
    turn = AgentInput(parts=parts, history=history, metadata=metadata)
    turn._served_input_types = accepted
    return turn


def _v1_part_out(part: MediaPart) -> dict[str, Any]:
    out: dict[str, Any] = {"mediaType": part.mime_type}
    if part.data is not None:
        out["raw"] = base64.b64encode(part.data).decode()
    else:
        out["url"] = part.uri
    if part.name:
        out["filename"] = part.name
    if part.size is not None:
        out["metadata"] = {"size": part.size}
    return out


def _v0_part_out(part: MediaPart) -> dict[str, Any]:
    file: dict[str, Any] = {"mimeType": part.mime_type}
    if part.data is not None:
        file["bytes"] = base64.b64encode(part.data).decode()
    else:
        file["uri"] = part.uri
    if part.name:
        file["name"] = part.name
    return {"type": "file", "file": file}


def _v1_parts_out(text: str | None, media: list[MediaPart]) -> list[dict[str, Any]]:
    return ([{"text": text}] if text else []) + [_v1_part_out(p) for p in media]


def _v0_parts_out(text: str | None, media: list[MediaPart]) -> list[dict[str, Any]]:
    return ([{"type": "text", "text": text}] if text else []) + [_v0_part_out(p) for p in media]


def _v1_ids(params: dict[str, Any]) -> tuple[str, str]:
    message = params.get("message") or {}
    task_id, context_id = message.get("taskId"), message.get("contextId")
    return (
        task_id if isinstance(task_id, str) and task_id else str(uuid.uuid4()),
        context_id if isinstance(context_id, str) and context_id else str(uuid.uuid4()),
    )


def _supported_version(value: str | None) -> bool:
    """1.x, or empty/0.x (a 0.3 client sends none; the 0.x methods stay one release)."""
    return not value or value.strip().split(".", 1)[0] in ("0", "1")


def _build_agent_input(
    message_text: str, parts: list[Part], params: dict[str, Any], *, accept_history: bool = False
) -> AgentInput:
    """Build an AgentInput from A2A params, forwarding session and state.

    - ``params.metadata`` (dict) is passed through as ``AgentInput.metadata``.
    - ``params.sessionId`` (A2A-spec field) populates ``metadata["session_id"]``
      unless the caller already set that key explicitly.
    - ``params.history`` is forwarded, with ``accept_history``, when it
      validates as Message models; invalid history is ignored rather than
      failing the request.
    """
    metadata = params.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata.pop(HISTORY_METADATA_KEY, None)  # never history by another name

    session_id = params.get("sessionId")
    if session_id and "session_id" not in metadata:
        metadata["session_id"] = session_id

    history = _history(params.get("history")) if accept_history else []
    return AgentInput(message=message_text, parts=parts, history=history, metadata=metadata)


#: What a caller is told when the agent raised. Deliberately fixed text: the
#: alternative — scanning the exception for provider markers and redacting
#: matches — is a denylist, and a denylist is a list of the leaks someone
#: remembered. The next provider's error format is exposed by default.
OPAQUE_FAILURE_MESSAGE = "The agent failed to handle this request."


def _opaque_failure(context: str) -> str:
    """Log the active exception; return what the caller is told instead.

    `serve` is the public package's network surface, running on machines the
    platform does not operate, so the exception text never crosses it: provider
    errors carry request payloads, file paths, console URLs, and — observed on
    OpenAI — an echo of the submitted API key (sdk#20). The detail stays in the
    operator's own log, reachable from the id quoted back to the caller.

    Both failure paths go through here so the two cannot drift into saying
    different amounts about the same failure.
    """
    error_id = uuid.uuid4().hex[:12]
    log.exception("%s [error_id=%s]", context, error_id)
    return f"{OPAQUE_FAILURE_MESSAGE} (error id: {error_id})"


def _jsonrpc_error(req_id: Any, code: int, message: str, data: dict[str, Any] | None = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if data:
        error["data"] = data
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "error": error,
    })


#: The largest request body ``POST /`` reads, as the platform's A2A server.
#: Inline file parts make bodies large; past this, files go by reference.
DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024


async def _read_body(request: Request, limit: int) -> bytes | None:
    """The request body, or ``None`` once it is longer than ``limit`` (read no further)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        return None
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            return None
    return bytes(body)


def _accepted_types(agent: AgentProtocol, declared: Iterable[str] | None) -> tuple[str, ...]:
    """The file types this app admits: ``declared``, else the agent's attribute, validated.

    The agent is never changed: each turn carries the app's types to it
    (``AgentInput._served_input_types``), so two apps over one agent each
    keep their own.
    """
    if declared is None:
        declared = getattr(agent, ACCEPTED_INPUT_TYPES_ATTR, None)
    return mime_patterns(declared)


def _output_types(agent: AgentProtocol, declared: Iterable[str] | None) -> tuple[str, ...]:
    """The file types ``agent`` may return: ``declared``, else its ``output_types`` attribute."""
    return mime_patterns(declared if declared is not None else getattr(agent, OUTPUT_TYPES_ATTR, None))


def _jsonrpc_result(req_id: Any, result: Any) -> JSONResponse:
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    })


#: Cap on a relayed tool result's ``output``, matching the platform's A2A
#: server: enough for the receiver to see a long output was cut and shape its
#: own preview, while a megabyte result stays off the wire.
TOOL_EVENT_OUTPUT_MAX_CHARS = 4000


#: What a failed tool result says on the wire. A failed call's output is
#: error text (an MCP failure carries the exception's message), and `serve`
#: never sends exception text to its caller (sdk#20).
TOOL_FAILED_OUTPUT = "The tool failed."


def _relayable_tool_event(output: AgentOutput) -> dict[str, Any] | None:
    """The tool event on ``output`` (``TOOL_EVENT_KEY``), made safe for the wire.

    A call's ``arguments`` are redacted and capped at 8 KB (``_redaction``),
    a failed result's ``output`` is replaced by ``TOOL_FAILED_OUTPUT``, and a
    successful result's ``output`` is sent as the tool returned it, cut to
    ``TOOL_EVENT_OUTPUT_MAX_CHARS``.
    """
    event = output.metadata.get(TOOL_EVENT_KEY)
    if not isinstance(event, dict) or event.get("type") not in ("tool_call", "tool_result"):
        return None
    relayed = dict(event)
    if relayed["type"] == "tool_call":
        relayed["arguments"] = safe_arguments(relayed.get("arguments"))
    elif relayed.get("ok") is not True:
        relayed["output"] = TOOL_FAILED_OUTPUT
    elif isinstance(text := relayed.get("output"), str):
        relayed["output"] = text[:TOOL_EVENT_OUTPUT_MAX_CHARS]
    return relayed


async def _collect_output(agent: AgentProtocol, agent_input: AgentInput) -> AgentOutput:
    """The whole reply: every chunk's text joined, and every chunk's files in order."""
    content_parts: list[str] = []
    media: list[MediaPart] = []
    last_output: AgentOutput | None = None
    async for output in agent.run(agent_input):
        last_output = output
        if output.content:
            content_parts.append(output.content)
        media.extend(media_of(output.parts))
        if output.done:
            break
    if last_output is None:
        return AgentOutput(content="", done=True)
    return AgentOutput(
        content="".join(content_parts),
        parts=media,
        tool_calls=last_output.tool_calls,
        done=True,
        metadata=last_output.metadata,
    )


async def _stream_output(
    agent: AgentProtocol, agent_input: AgentInput, req_id: Any, task_id: str
) -> AsyncIterator[dict[str, str]]:
    def _event(payload: dict[str, Any]) -> dict[str, str]:
        return {"data": json.dumps({"jsonrpc": "2.0", "id": req_id, "result": payload})}

    yield _event({"id": task_id, "status": {"state": "working"}, "final": False})

    try:
        async for output in agent.run(agent_input):
            # A tool event rides on a non-final ``working`` status update under
            # ``metadata.tool_event``, as the platform's A2A server sends it;
            # a reader that does not know the key sees one more progress event.
            tool_event = _relayable_tool_event(output)
            if tool_event is not None:
                yield _event({
                    "id": task_id,
                    "status": {"state": "working"},
                    "final": False,
                    "metadata": {TOOL_EVENT_KEY: tool_event},
                })
                continue
            if output.content or media_of(output.parts):
                yield _event({
                    "id": task_id,
                    "artifact": {"parts": _v0_parts_out(output.content, media_of(output.parts))},
                    "final": False,
                })
            if output.done:
                yield _event({"id": task_id, "status": {"state": "completed"}, "final": True})
                return
    except Exception:
        # The same leak as `tasks/send`, on the streaming path. The flag
        # (sdk#20) named only the other one; both reach the network.
        message = _opaque_failure("streaming failed")
        yield _event({
            "id": task_id,
            "status": {
                "state": "failed",
                # A2A types `TaskStatus.message` as a Message, not a string.
                # It was emitted as a bare string here, so a spec-conforming
                # reader — including the platform's own A2A client, which does
                # `status_msg.get("parts")` — raised AttributeError instead of
                # reporting the failure. That made the error id unreachable on
                # exactly the path this fix exists to cover.
                "message": {
                    "role": "agent",
                    "parts": [{
                        "type": "text",
                        "text": message,
                    }],
                },
            },
            "final": True,
        })


async def _v1_stream(
    agent: AgentProtocol, agent_input: AgentInput, req_id: Any, task_id: str, context_id: str
) -> AsyncIterator[dict[str, str]]:
    """``SendStreamingMessage``: working, appended ``result`` chunks, then completed or failed."""

    def _event(payload: dict[str, Any]) -> dict[str, str]:
        return {"data": json.dumps({"jsonrpc": "2.0", "id": req_id, "result": payload})}

    def _status(state: str, *, text: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, str]:
        status: dict[str, Any] = {"state": state}
        if text:
            status["message"] = {"messageId": str(uuid.uuid4()), "role": "ROLE_AGENT", "parts": [{"text": text}]}
        update: dict[str, Any] = {"taskId": task_id, "contextId": context_id, "status": status}
        if metadata:
            update["metadata"] = metadata
        return _event({"statusUpdate": update})

    chunks = 0

    def _chunk(parts: list[dict[str, Any]], *, last: bool) -> dict[str, str]:
        nonlocal chunks
        chunks += 1
        return _event({"artifactUpdate": {
            "taskId": task_id,
            "contextId": context_id,
            "artifact": {"artifactId": _RESULT_ARTIFACT, "parts": parts},
            "append": chunks > 1,
            "lastChunk": last,
        }})

    yield _status("TASK_STATE_WORKING")
    try:
        async for output in agent.run(agent_input):
            tool_event = _relayable_tool_event(output)
            if tool_event is not None:
                yield _status("TASK_STATE_WORKING", metadata={TOOL_EVENT_KEY: tool_event})
                continue
            parts = _v1_parts_out(output.content, media_of(output.parts))
            if parts:
                yield _chunk(parts, last=output.done)
            if output.done:
                yield _status("TASK_STATE_COMPLETED")
                return
        yield _status("TASK_STATE_COMPLETED")
    except Exception:
        yield _status("TASK_STATE_FAILED", text=_opaque_failure("streaming failed"))


def create_app(
    agent: AgentProtocol,
    *,
    name: str = "agent",
    description: str = "",
    public_url: str | None = None,
    accept_history: bool = False,
    accepted_input_types: Iterable[str] | None = None,
    output_types: Iterable[str] | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> FastAPI:
    """Build a FastAPI app that serves an agent over A2A v1.0 (JSON-RPC binding).

    The card is at ``/.well-known/agent-card.json`` (and, for one minor
    release, ``/.well-known/agent.json``). ``SendMessage`` returns a completed
    task with the reply as artifact ``result``; ``SendStreamingMessage``
    streams it. The server keeps no tasks, so ``GetTask``, ``ListTasks``,
    ``CancelTask`` and ``SubscribeToTask`` answer UnsupportedOperation, and
    push notifications are not supported. The 0.x ``tasks/send`` and
    ``tasks/sendSubscribe`` still work for one minor release.

    ``SendStreamingMessage`` (and ``tasks/sendSubscribe``) relays each tool call and result the agent reports
    (``metadata.tool_event``) on a ``working`` status update. Tool calls go out
    with their arguments redacted and capped at 8 KB, and a failed result says
    only "The tool failed.". **A successful tool's output is sent raw**, up to
    4000 characters, to whoever calls this app: a tool that reads a database
    row, a file or an internal API exposes that data to a direct caller. On the
    platform the engine scrubs it before a user sees it; when you host ``serve``
    yourself, nothing does. ``SendMessage`` returns only the agent's text.

    ``public_url``: the URL the card advertises (``supportedInterfaces[0].url``).
    Set it when the app is reachable from outside: without it the card names
    the URL of each request, which is built from the caller's ``Host`` header,
    so a caller can make the card advertise any host.

    ``accept_history``: forward caller-supplied prior turns
    (``metadata["genfleet.history"]``, or 0.x ``params.history``) to the agent
    as its history. Off by default: whoever can call the app could otherwise
    put words in the agent's earlier turns. Turn it on only for a trusted
    caller that keeps the conversation itself.

    ``accepted_input_types``: the MIME types (``"image/*"``,
    ``"application/pdf"``, ``"*/*"``) of the files the agent takes, besides
    text. Defaults to the agent's own ``accepted_input_types`` attribute
    (``Agent(accepts=…)`` sets it), else none. When given, it applies to this
    app only: the agent is not changed, and ``Agent.run`` accepts what this
    app admitted. The card lists them as
    ``defaultInputModes``; a request with a file of any other type is
    answered ContentTypeNotSupported (-32005), naming the type and what is
    accepted. Files arrive as ``AgentInput.parts``: inline (``raw``, at most
    ``MAX_INLINE_BYTES``) or by reference (``url``), which the agent reads
    through a ``FileResolver``; ``serve`` never fetches one. An agent's
    file results (``AgentOutput.parts``) go out as file parts of the
    ``result`` artifact.

    ``output_types``: the MIME types of files the agent may return, listed in
    the card's ``defaultOutputModes`` after ``text/plain``. Defaults to the
    agent's ``output_types`` attribute, else none.

    ``max_body_bytes``: the largest request body read (8 MiB, as the
    platform's A2A server); a larger one is refused before it is parsed.
    """
    accepted = _accepted_types(agent, accepted_input_types)
    produced = _output_types(agent, output_types)
    if public_url is not None:
        parts = urlsplit(public_url)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.query or parts.fragment:
            raise ValueError("public_url must be an absolute http(s) URL with no query or fragment")
    app = FastAPI(title=name, docs_url=None, redoc_url=None)

    def card(base_url: str) -> dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "supportedInterfaces": [
                {"url": base_url, "protocolBinding": "JSONRPC", "protocolVersion": PROTOCOL_VERSION}
            ],
            "version": "1.0.0",
            "capabilities": {"streaming": True, "pushNotifications": False, "extendedAgentCard": False},
            "defaultInputModes": ["text/plain", *accepted],
            "defaultOutputModes": ["text/plain", *produced],
            "skills": [],
        }

    @app.get(CARD_PATH)
    async def agent_card(request: Request) -> JSONResponse:
        # A card built from the request's Host must not be cached for others.
        cache = "max-age=300" if public_url else "no-store"
        return JSONResponse(card(public_url or str(request.base_url)), headers={"Cache-Control": cache})

    # The 0.x path, for one minor release: the same card.
    app.get(LEGACY_CARD_PATH)(agent_card)

    @app.post("/")
    async def jsonrpc(request: Request):  # type: ignore[return]
        raw = await _read_body(request, max_body_bytes)
        if raw is None:
            return _jsonrpc_error(
                None, _INVALID_REQUEST, f"Request body larger than {max_body_bytes} bytes; send files by reference."
            )
        try:
            body = json.loads(raw)
        except Exception:
            return _jsonrpc_error(None, -32700, "Parse error")
        if not isinstance(body, dict):
            return _jsonrpc_error(None, _INVALID_REQUEST, "Invalid Request")

        req_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})
        if not isinstance(params, dict):
            params = {}

        if not _supported_version(request.headers.get(VERSION_HEADER)):
            return _jsonrpc_error(req_id, _VERSION_NOT_SUPPORTED, "Supported A2A versions: 1.0.")
        if method in ("SendMessage", "SendStreamingMessage"):
            try:
                v1_input = _build_v1_input(params, accepted=accepted, accept_history=accept_history)
            except _PartError as exc:
                return _jsonrpc_error(req_id, exc.code, exc.message, exc.data)
            if v1_input is None:
                return _jsonrpc_error(req_id, _INVALID_PARAMS, "Invalid request parameters: check message.")
            task_id, context_id = _v1_ids(params)
            if method == "SendStreamingMessage":
                return EventSourceResponse(_v1_stream(agent, v1_input, req_id, task_id, context_id))
            try:
                output = await _collect_output(agent, v1_input)
            except Exception:
                return _jsonrpc_error(req_id, -32603, _opaque_failure("SendMessage failed"))
            return _jsonrpc_result(req_id, {"task": {
                "id": task_id,
                "contextId": context_id,
                "status": {"state": "TASK_STATE_COMPLETED"},
                "artifacts": (
                    [{"artifactId": _RESULT_ARTIFACT, "parts": result_parts}]
                    if (result_parts := _v1_parts_out(output.content, media_of(output.parts)))
                    else []
                ),
            }})
        if method in _TASK_STORE_METHODS:
            return _jsonrpc_error(req_id, _UNSUPPORTED_OPERATION, "This agent keeps no tasks.")
        if method in _PUSH_METHODS:
            return _jsonrpc_error(req_id, _PUSH_NOT_SUPPORTED, "Push notifications are not supported.")
        if method == "GetExtendedAgentCard":
            return _jsonrpc_error(req_id, _EXTENDED_CARD_NOT_CONFIGURED, "No extended agent card.")

        # 0.x, for one minor release.
        message_text = ""
        v0_parts: list[Part] = []
        msg = params.get("message", {})
        if isinstance(msg, dict):
            raw_parts = msg.get("parts", [])
            raw_parts = raw_parts if isinstance(raw_parts, list) else []
            try:
                v0_parts = _parts_in(raw_parts, accepted)
            except _PartError as exc:
                return _jsonrpc_error(req_id, exc.code, exc.message, exc.data)
            message_text = " ".join(p.text for p in v0_parts if isinstance(p, TextPart))
        elif isinstance(msg, str):
            message_text = msg

        agent_input = _build_agent_input(message_text, v0_parts, params, accept_history=accept_history)
        agent_input._served_input_types = accepted
        task_id = params.get("id", str(uuid.uuid4()))

        if method == "tasks/send":
            try:
                output = await _collect_output(agent, agent_input)
            except Exception:
                message = _opaque_failure("tasks/send failed")
                return _jsonrpc_error(req_id, -32603, message)

            return _jsonrpc_result(req_id, {
                "id": task_id,
                "status": {"state": "completed"},
                "artifacts": [{"parts": result_parts}]
                if (result_parts := _v0_parts_out(output.content, media_of(output.parts))) else [],
            })

        if method == "tasks/sendSubscribe":
            return EventSourceResponse(_stream_output(agent, agent_input, req_id, task_id))

        return _jsonrpc_error(req_id, -32601, f"Unknown method: {method}")

    return app


def serve(
    agent: AgentProtocol,
    *,
    name: str = "agent",
    description: str = "",
    host: str = "0.0.0.0",
    port: int = 8000,
    public_url: str | None = None,
    accept_history: bool = False,
    accepted_input_types: Iterable[str] | None = None,
    output_types: Iterable[str] | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> None:
    """Run an agent as an A2A server (blocking).

    The app is ``create_app``'s; its docstring says what tool events a caller
    sees, including that a successful tool's output is sent raw.
    """
    import uvicorn

    app = create_app(
        agent,
        name=name,
        description=description,
        public_url=public_url,
        accept_history=accept_history,
        accepted_input_types=accepted_input_types,
        output_types=output_types,
        max_body_bytes=max_body_bytes,
    )
    uvicorn.run(app, host=host, port=port)
