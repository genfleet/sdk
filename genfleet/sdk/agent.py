from __future__ import annotations

import base64
import hashlib
import inspect
import json
import logging
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, AsyncIterator, Literal, NamedTuple

from .audit import Auditor, audit_for
from .audit.schemas import AuditConfig
from .memory import memory_for
from .state_store import StateStore, state_for

#: ``AgentInput.metadata`` key the platform sets per turn: ``"off"`` means the
#: Agent neither loads nor writes its session memory for that turn.
MEMORY_METADATA_KEY = "genfleet.memory"
from .memory.base import AppendableMemory, Memory, WindowedMemory
from .memory.window import load_recent, recent_turns
from . import turn as _turn
from .models import ModelClient, discover_model_client
from .providers import provider_for
from .schemas import (
    TOOL_EVENT_KEY,
    AgentInput,
    AgentOutput,
    MCPConfig,
    MemoryConfig,
    Message,
    ModelConfig,
    TokenUsage,
    ToolCall,
    ToolSchema,
)
from .caller import (
    Audience,
    Caller,
    EVERYONE,
    OPERATOR_ONLY,
    caller_of,
    effective_audiences,
    marked,
    slug_key,
    may_offer,
    tool_audiences_of,
)
from .confirmations import (
    _APPROVAL_UNAVAILABLE_NOTE,
    _APPROVAL_UNVERIFIED_NOTE,
    _PEER_REFUSAL,
    _PENDING_RESULT,
    APPROVED_CALL_RESULT_KEY,
    CONFIRMATION_REQUESTS_KEY,
    ApprovedCallResult,
    _approved_call_note,
    _claim_approved_call,
    _is_peer_turn,
    _is_sensitive,
    _sensitive_flags_of,
)
from .files import FileAccessError, FileResolver, read_part
from .parts import ImagePart, UnsupportedInput, looks_like, media_of, mime_patterns
from .parts import accepts as _accepts
from .protocol import ToolProtocol
from .tool import ToolWrapper, wrap_tool
from .skills import Skill, load_skills, pinned_revision

log = logging.getLogger("genfleet.sdk.agent")
SKILL_TOOL_NAME = "read_skill"
SKILL_MEMORY_MARKER = "[skill body omitted; call read_skill again to use it]"

#: Image types sent to a vision model as images; every provider takes these.
MODEL_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
#: The largest image sent to a model (Anthropic's per-image limit). A larger
#: one is described to the model instead.
MAX_MODEL_IMAGE_BYTES = 5 * 1024 * 1024
#: Default limits on the images one turn sends to a model (``Agent`` args).
DEFAULT_MAX_IMAGES_PER_TURN = 8
DEFAULT_MAX_IMAGE_BYTES_PER_TURN = 20 * 1024 * 1024


def _is_tool_object(obj: object) -> bool:
    """Whether ``obj`` is an instance that already implements ToolProtocol:
    a callable ``schema()`` and a coroutine ``call()``. A class is not one."""
    if isinstance(obj, type):
        return False
    return callable(getattr(obj, "schema", None)) and inspect.iscoroutinefunction(getattr(obj, "call", None))


def _as_tool(entry: Callable | ToolProtocol) -> tuple[ToolProtocol, str]:
    """A ``tools=`` entry as the tool the agent holds, and its name.

    A function (plain or ``@tool``) becomes a `ToolWrapper`; an object that
    already implements `ToolProtocol` (``schema()`` and an async ``call()``,
    such as a peer agent in ``remote_tools``) is held as it is.
    """
    if isinstance(entry, ToolWrapper):
        return entry, entry.schema().name
    if _is_tool_object(entry):
        schema = entry.schema()
        if not isinstance(schema, ToolSchema):
            raise TypeError(f"{entry!r}.schema() must return a ToolSchema, got {schema!r}")
        return entry, schema.name
    if isinstance(entry, type) and issubclass(entry, ToolProtocol):
        raise TypeError(f"tools entries take a ToolProtocol instance, not the class {entry.__name__}")
    if callable(entry):
        wrapper = wrap_tool(entry)
        return wrapper, wrapper.schema().name
    raise TypeError(
        "tools entries must be functions, @tool functions or ToolProtocol objects "
        f"(schema() and async call()), got {entry!r}"
    )


def _reserved_name(name: str, skills_configured: bool) -> str | None:
    """Why ``name`` is reserved, or ``None`` when a tool may take it."""
    if name.startswith("skill:"):
        return "tool names starting with 'skill:' are reserved"
    if skills_configured and name == SKILL_TOOL_NAME:
        return f"tool name {SKILL_TOOL_NAME!r} is reserved when skills are configured"
    return None


def _local_tools_of(
    entries: Iterable[Callable | ToolProtocol], *, skills_configured: bool
) -> dict[str, ToolProtocol]:
    """The agent's tools by name.

    Among functions a later one replaces an earlier one with the same name,
    and a reserved name is a `ValueError`. Unwrapped ToolProtocol objects are
    placed after every function: one never takes a name a function has (in
    either order) or an earlier object took, since dispatch goes by name and
    it would otherwise drop that tool's sensitive marking and audience. Like
    an MCP tool, it can't take a reserved name or one starting with ``@``
    (a slug key's form). Each one skipped is logged.
    """
    tools: dict[str, ToolProtocol] = {}
    unwrapped: list[tuple[str, ToolProtocol]] = []
    for entry in entries:
        tool, name = _as_tool(entry)
        if not isinstance(tool, ToolWrapper):
            unwrapped.append((name, tool))
            continue
        reason = _reserved_name(name, skills_configured)
        if reason:
            raise ValueError(reason)
        tools[name] = tool
    for name, tool in unwrapped:
        if name in tools:
            log.warning("tool %r skipped: the name is already taken by another tool", name)
        elif name.startswith("@"):
            log.warning("tool %r skipped: a tool name can't start with '@'", name)
        elif reason := _reserved_name(name, skills_configured):
            log.warning("tool %r skipped: %s", name, reason)
        else:
            tools[name] = tool
    return tools


class _Marking(NamedTuple):
    """What the author marked on a tool; the platform's settings still win."""

    slug: str | None
    audiences: frozenset[Audience]
    sensitive: bool


#: A ToolProtocol object that isn't a `ToolWrapper` carries no marking: like
#: an unmarked function, it is operator-only, not sensitive, keyed by name.
_UNMARKED = _Marking(slug=None, audiences=OPERATOR_ONLY, sensitive=False)


def _marking_of(tool: ToolProtocol) -> _Marking:
    if isinstance(tool, ToolWrapper):
        return _Marking(slug=tool.slug, audiences=tool.audiences, sensitive=tool.sensitive)
    return _UNMARKED


def _tool_call_to_dict(tc: ToolCall) -> dict:
    """Serialize a ToolCall into the OpenAI-style message-history shape.

    ``thought_signature`` is only emitted when set (Gemini 3.x), so OpenAI and
    Anthropic payloads are byte-for-byte unchanged. It rides alongside the
    ``function`` block rather than inside it so providers that pass the block
    straight through never see an unexpected key.
    """
    d: dict = {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
    }
    if tc.thought_signature:
        d["thought_signature"] = tc.thought_signature
    return d


def _tool_event(payload: dict[str, Any]) -> AgentOutput:
    """An output reporting one tool event (see ``schemas.TOOL_EVENT_KEY``)."""
    return AgentOutput(content="", done=False, metadata={TOOL_EVENT_KEY: payload})


def _model_client_for(config: ModelConfig, explicit: ModelClient | None) -> ModelClient:
    """An explicit client, else an installed one that is active, else the direct provider."""
    if explicit is not None:
        return explicit
    found = discover_model_client(config)
    if found is not None:
        name, client = found
        log.info("model client %s active for %s", name, config["model"])
        return client
    return provider_for(config)


def _resolve_skills(sources: list[Skill | str | Path]) -> list[Skill]:
    resolved: list[Skill] = []
    for source in sources:
        if isinstance(source, Skill):
            resolved.append(source)
        elif isinstance(source, (str, Path)):
            if isinstance(source, str) and not pinned_revision(source):
                raise ValueError("Agent requires a pinned skill repository (@40-character commit); use load_skills for interactive discovery")
            resolved.extend(load_skills(source))
        else:
            raise TypeError("skills entries must be Skill or repository source")
    return resolved


class Agent:
    """
    High-level agent that wires together a provider, tools, MCP servers,
    episodic memory, and optional context into a single AgentProtocol-compatible object.

    Usage::

        agent = Agent(
            role="You are a helpful assistant.",
            model={"model": "gpt-4o-mini", "api_key": "sk-…"},
            tools=[my_function],
            memory="platform",   # the platform's store when hosted; or {"type": "redis", ...}
        )
        serve(agent, name="my-agent", port=8000)

    ``tools`` takes plain functions, ``@tool`` functions and objects that
    implement `ToolProtocol` (``schema()`` and an async ``call()``), such as
    the peer agents a factory receives in ``remote_tools``. Functions are
    wrapped; a ToolProtocol object is used as it is, operator-only and not
    sensitive unless the platform's per-tool settings say otherwise. Tools
    are keyed by schema name, and ToolProtocol objects are placed after
    every function. A function always keeps its name: an object whose name
    a function has (in either order), an earlier object took, an MCP tool
    has, or that is reserved or starts with ``@``, is skipped with a logged
    warning. Among functions a later one replaces an earlier one, and a
    function with a reserved name is a `ValueError`.

    ``memory`` may be left out: hosted on the platform the agent then gets
    the platform store anyway; on a laptop it runs without memory.

    ``max_history_turns`` caps how many past turns go to the model. A turn is
    a user message and everything up to the next one, so a tool call is never
    sent without its result. Memory still stores every turn; ``None`` (the
    default) sends the whole thread.
    """

    def __init__(
        self,
        role: str,
        model: ModelConfig,
        tools: list[Callable | ToolProtocol] | None = None,
        mcps: list[MCPConfig] | None = None,
        memory: MemoryConfig | Literal["platform"] | Memory | None = None,
        context: str | None = None,
        audit: AuditConfig | None = None,
        model_client: ModelClient | None = None,
        skills: list[Skill | str | Path] | None = None,
        state: StateStore | Literal["platform"] | None = None,
        accepts: list[str] | None = None,
        files: FileResolver | None = None,
        max_images_per_turn: int = DEFAULT_MAX_IMAGES_PER_TURN,
        max_image_bytes_per_turn: int = DEFAULT_MAX_IMAGE_BYTES_PER_TURN,
        max_history_turns: int | None = None,
    ) -> None:
        self._role = role
        # How many past turns go to the model; ``None`` sends them all. Only
        # what the model sees is cut: memory still keeps every turn.
        if max_history_turns is not None and (
            not isinstance(max_history_turns, int)
            or isinstance(max_history_turns, bool)
            or max_history_turns < 1
        ):
            raise ValueError("max_history_turns must be an int of at least 1, or None for no limit")
        self._max_history_turns = max_history_turns
        # The file types this agent takes besides text (`serve` advertises
        # and enforces them). Images go to the model as images when the model
        # config says ``vision``; every other file is described, never sent.
        # A `serve(accepted_input_types=…)` override doesn't change this: it
        # rides on each turn it admits (`AgentInput._served_input_types`).
        self.accepted_input_types: tuple[str, ...] = mime_patterns(accepts)
        self._files = files
        if max_images_per_turn < 0 or max_image_bytes_per_turn < 0:
            raise ValueError("image limits must not be negative")
        self._max_images_per_turn = max_images_per_turn
        self._max_image_bytes_per_turn = max_image_bytes_per_turn
        self._context = context
        self._model_config = model

        self._provider: ModelClient = _model_client_for(model, model_client)
        self._memory: Memory | None = memory_for(memory)
        self._state: StateStore | None = state_for(state)
        self._auditor: Auditor | None = audit_for(audit)
        self._mcp_configs: list[MCPConfig] = mcps or []

        resolved_skills = _resolve_skills(skills or [])
        self._skills = {skill.name: skill for skill in resolved_skills}
        if len(self._skills) != len(resolved_skills):
            raise ValueError("skill names must be unique")
        # Functions become ToolWrappers, ToolProtocol objects are held as is,
        # and on a name collision a function wins (`_local_tools_of`).
        self._local_tools = _local_tools_of(tools or [], skills_configured=bool(self._skills))

        # MCP tool registry — populated on first run()
        self._mcp_tools: dict[str, Any] = {}
        self._mcp_ready = False

    @property
    def memory(self) -> Memory | None:
        """The episodic backend, for calls beyond the turn loop — a
        :class:`~genfleet.sdk.memory.PlatformMemory` exposes ``search`` and
        ``purge``. ``None`` when the agent runs without memory."""
        return self._memory

    @property
    def state(self) -> StateStore | None:
        """Workflow checkpoints; platform-backed when hosted, if configured."""
        return self._state

    @property
    def skills(self) -> tuple[Skill, ...]:
        return tuple(self._skills.values())

    # ------------------------------------------------------------------
    # AgentProtocol
    # ------------------------------------------------------------------

    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]:
        invocation_id = str(uuid.uuid4())
        invocation_start = time.monotonic()
        auditor = self._auditor
        total_usage = TokenUsage()

        # `serve` stamps the types its app admitted; a direct call uses ours.
        accepted = input._served_input_types
        if accepted is None:
            accepted = self.accepted_input_types
        refused = [p.mime_type for p in media_of(input.parts) if not _accepts(p.mime_type, accepted)]
        if refused:
            raise UnsupportedInput(refused, accepted)

        if self._mcp_configs and not self._mcp_ready:
            await self._init_mcp()

        session_id = input.metadata.get("session_id") or str(uuid.uuid4())
        # The platform turns memory off for a turn from its public A2A edge:
        # the session is then neither loaded nor written.
        use_memory = self._memory is not None and input.metadata.get(MEMORY_METADATA_KEY) != "off"

        if auditor:
            await auditor.emit(
                "invocation.start",
                invocation_id=invocation_id,
                session_id=session_id,
                data={"message": auditor.truncate(input.message)},
            )

        try:
            # Load history from memory backend (empty list if no memory configured).
            # A caller-supplied history replaces the stored one for this turn
            # (it is never merged), so the store is not read at all.
            persisted: list[Message] = []
            if use_memory and not input.history:
                persisted = await _turn.call_within(input.metadata, self._load_history(session_id))

            history = persisted if not input.history else list(input.history)
            # `history` stays whole: a backend without `append` saves it back.
            sent_history = (
                history if self._max_history_turns is None
                else recent_turns(history, self._max_history_turns)
            )

            # Build system prompt
            system_parts = [self._role]
            if self._context:
                system_parts.append(self._context)
            caller = caller_of(input.metadata)
            audience_overrides = tool_audiences_of(input.metadata)
            allowed_skills = {
                name: skill for name, skill in self._skills.items()
                if may_offer(caller, effective_audiences(f"skill:{name}", None, skill.audiences, audience_overrides))
            }
            if allowed_skills:
                index = "\n".join(
                    f"- {skill.name} (v{skill.version}): {skill.description}"
                    for skill in allowed_skills.values()
                )
                system_parts.append(
                    "Available skills (read a skill before following its instructions):\n"
                    + index + "\nUse read_skill(name) to load one. Skills do not grant tool access."
                )

            messages: list[dict] = [{"role": "system", "content": "\n\n".join(system_parts)}]
            for msg in sent_history:
                messages.append(self._message_to_dict(msg))
            turn_text, user_content = await self._user_content(input)
            messages.append({"role": "user", "content": user_content})

            # The tools this caller may use: by each tool's
            # audience, the platform's per-tool setting first. A tool left out
            # here is also refused at dispatch, whatever the model asks for.
            offered = self._offered_tools(caller, audience_overrides)
            tool_schemas = [w.schema() for name, w in self._local_tools.items() if name in offered]
            tool_schemas += [
                ToolSchema(name=name, description=spec["description"], parameters=spec["parameters"])
                for name, spec in self._mcp_tools.items()
                if name in offered
            ]
            if allowed_skills and may_offer(caller, effective_audiences(SKILL_TOOL_NAME, None, EVERYONE, audience_overrides)):
                tool_schemas.append(ToolSchema(
                    name=SKILL_TOOL_NAME,
                    description="Read the full instructions for an available skill by name.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "enum": list(allowed_skills)},
                            "path": {"type": "string", "description": "Optional reference path from the skill."},
                        },
                        "required": ["name"],
                    },
                ))
                offered.add(SKILL_TOOL_NAME)

            # Sensitive tools wait for an owner's approval; an
            # approved call (signed by the engine) runs first, exactly once.
            sensitive = self._sensitive_tools(_sensitive_flags_of(input.metadata))
            peer_turn = _is_peer_turn(input.metadata)
            requests: list[dict[str, Any]] = []
            new_messages: list[dict] = []
            approved_result: ApprovedCallResult | None = None
            # Verified and claimed in one step: it can't run twice, nor later.
            approved, refused = _claim_approved_call(input.metadata)
            if refused:
                approved_result = "refused"
                log.warning("approved call not run: %s", refused)
                messages[0]["content"] += "\n\n" + _APPROVAL_UNVERIFIED_NOTE
            elif approved:
                name = self._tool_by_key(approved.tool)
                if name is None or name not in offered:
                    approved_result = "refused"
                    messages[0]["content"] += "\n\n" + _APPROVAL_UNAVAILABLE_NOTE
                else:
                    tc = ToolCall(id=f"approved-{approved.confirmation_id}", name=name, arguments=approved.arguments)
                    yield _tool_event({"type": "tool_call", "id": tc.id, "name": tc.name, "arguments": tc.arguments})
                    # Only the tool's own exception is contained: one from the
                    # audit sink after the tool ran must not report "failed".
                    result, ok = await self._audited_dispatch(
                        tc, offered, input, invocation_id, session_id, contain_tool_errors=True
                    )
                    approved_result = "ran" if ok else "failed"
                    new_messages.append({"role": "assistant", "content": "", "tool_calls": [_tool_call_to_dict(tc)]})
                    new_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    yield _tool_event({"type": "tool_result", "id": tc.id, "name": tc.name, "ok": ok, "output": self._public_tool_result(tc.name, result, ok)})
                    messages[0]["content"] += "\n\n" + _approved_call_note(name, ok)

            # Agentic loop
            while True:
                tool_calls_batch: list[ToolCall] = []

                if auditor:
                    await auditor.emit(
                        "llm.request",
                        invocation_id=invocation_id,
                        session_id=session_id,
                        data={
                            "model": self._model_config["model"],
                            "message_count": len(messages) + len(new_messages),
                            "tool_count": len(tool_schemas),
                        },
                    )

                llm_start = time.monotonic()
                last_token_usage: TokenUsage | None = None
                response_content_parts: list[str] = []

                # The model client reads this turn's metadata through
                # `current_turn()`, set only while it runs (never across a yield).
                async for chunk in _turn.within(
                    input.metadata, self._provider.complete(messages + new_messages, tool_schemas)
                ):
                    if chunk.token_usage:
                        last_token_usage = chunk.token_usage
                    if chunk.tool_calls:
                        tool_calls_batch.extend(chunk.tool_calls)
                    elif chunk.content:
                        response_content_parts.append(chunk.content)
                        # ``done`` describes the turn, not a provider chunk: the
                        # loop may still have a tool round to run, and a consumer
                        # stops reading at the first ``done`` it sees. Only the
                        # yield at the bottom of the loop ends the turn.
                        yield chunk.model_copy(update={"done": False}) if chunk.done else chunk
                    if chunk.done:
                        break

                llm_latency = (time.monotonic() - llm_start) * 1000

                if last_token_usage:
                    total_usage.input_tokens += last_token_usage.input_tokens
                    total_usage.output_tokens += last_token_usage.output_tokens
                    total_usage.total_tokens += last_token_usage.total_tokens

                if auditor:
                    response_text = "".join(response_content_parts)
                    await auditor.emit(
                        "llm.response",
                        invocation_id=invocation_id,
                        session_id=session_id,
                        data={
                            "content": auditor.truncate(response_text) if response_text else None,
                            "tool_calls": [
                                {"name": tc.name, "arguments": tc.arguments}
                                for tc in tool_calls_batch
                            ] if tool_calls_batch else None,
                            "token_usage": last_token_usage.model_dump() if last_token_usage else None,
                        },
                        latency_ms=llm_latency,
                    )

                if not tool_calls_batch:
                    final_text = "".join(response_content_parts)
                    if final_text:
                        new_messages.append({"role": "assistant", "content": final_text})
                    # Persist *before* announcing the turn is done. A consumer
                    # stops reading at ``done`` — ``serve`` breaks out of the
                    # ``async for`` — which closes this generator at the yield
                    # below, so anything after it never runs.
                    await _turn.call_within(input.metadata, self._persist_turn(
                        session_id, input, turn_text, history, new_messages, invocation_id
                    ))
                    # Same reason as the persist: after the ``done`` yield this
                    # generator may already be closed, and the audit trail
                    # would have a start with no end for every served turn.
                    if auditor:
                        await auditor.emit(
                            "invocation.end",
                            invocation_id=invocation_id,
                            session_id=session_id,
                            data={"total_token_usage": total_usage.model_dump()},
                            latency_ms=(time.monotonic() - invocation_start) * 1000,
                        )
                    # The requests and the approved call's result ride on the
                    # final chunk: a non-streaming invoke (every channel turn)
                    # returns only its metadata.
                    final: dict[str, Any] = {}
                    if requests:
                        final[CONFIRMATION_REQUESTS_KEY] = requests
                    if approved_result:
                        final[APPROVED_CALL_RESULT_KEY] = approved_result
                    yield AgentOutput(content="", done=True, metadata=final)
                    break

                # Append assistant tool-call turn (keep any text streamed
                # alongside the tool calls — the user already saw it)
                new_messages.append({
                    "role": "assistant",
                    "content": "".join(response_content_parts),
                    "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls_batch],
                })

                # Report each call once its input is complete, then run them.
                # Same order as the platform runner: every call of the round
                # first, then each result as its tool returns.
                for tc in tool_calls_batch:
                    yield _tool_event({
                        "type": "tool_call",
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    })

                # Dispatch all tool calls and append results
                for tc in tool_calls_batch:
                    status: str | None = None
                    if tc.name in offered and tc.name in sensitive:
                        # Not run: a peer can't be approved for; anyone else waits for an owner.
                        if peer_turn:
                            result, ok = _PEER_REFUSAL, False
                        else:
                            requests.append({"tool": self._key_of(tc.name), "arguments": tc.arguments})
                            result, ok, status = _PENDING_RESULT, False, "pending"
                    else:
                        result, ok = await self._audited_dispatch(tc, offered, input, invocation_id, session_id, allowed_skills=set(allowed_skills))

                    new_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    yield _tool_event({
                        "type": "tool_result",
                        "id": tc.id,
                        "name": tc.name,
                        "ok": ok,
                        "output": self._public_tool_result(tc.name, result, ok),
                        **({"status": status} if status else {}),
                    })

        except Exception as exc:
            if auditor:
                await auditor.emit(
                    "invocation.error",
                    invocation_id=invocation_id,
                    session_id=session_id,
                    data={"error_type": type(exc).__name__, "message": str(exc)},
                    latency_ms=(time.monotonic() - invocation_start) * 1000,
                    status="error",
                )
            raise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _load_history(self, session_id: str) -> list[Message]:
        """The session's thread, or its recent end when the store can serve one.

        The store is asked for less only when it appends: a backend that saves
        rewrites the thread from what was loaded, so it must load all of it.
        """
        turns = self._max_history_turns
        if (
            turns is not None
            and isinstance(self._memory, AppendableMemory)
            and isinstance(self._memory, WindowedMemory)
        ):
            return await load_recent(self._memory, session_id, turns)
        return await self._memory.load(session_id)

    async def _persist_turn(
        self,
        session_id: str,
        input: AgentInput,
        turn_text: str,
        history: list[Message],
        new_messages: list[dict],
        invocation_id: str,
    ) -> None:
        """Write this turn to memory — keep the full assistant/tool structure so
        replayed histories remain valid provider payloads (tool messages must
        follow an assistant turn carrying the matching tool_calls ids).
        """
        if not self._memory or input.metadata.get(MEMORY_METADATA_KEY) == "off":
            return
        # The turn's text with its files described: memory never holds file bytes.
        turn = [Message(role="user", content=turn_text)]
        skill_call_ids = {
            tool_call["id"]
            for message in new_messages if message["role"] == "assistant"
            for tool_call in message.get("tool_calls", [])
            if tool_call["function"]["name"] == SKILL_TOOL_NAME
        }
        for m in new_messages:
            if m["role"] == "assistant":
                turn.append(Message(
                    role="assistant",
                    content=m.get("content", "") or "",
                    tool_calls=[
                        ToolCall(
                            id=t["id"],
                            name=t["function"]["name"],
                            arguments=json.loads(t["function"]["arguments"] or "{}"),
                            thought_signature=t.get("thought_signature"),
                        )
                        for t in m.get("tool_calls", [])
                    ],
                ))
            elif m["role"] == "tool":
                turn.append(Message(
                    role="tool",
                    content=SKILL_MEMORY_MARKER if m.get("tool_call_id") in skill_call_ids else m.get("content", "") or "",
                    tool_call_id=m.get("tool_call_id"),
                ))
        if isinstance(self._memory, AppendableMemory):
            # One turn, not the whole thread: the store already holds
            # what ``load`` returned. ``subject`` and the session's
            # metadata ride on the input — a channel ingress sets them;
            # a plain caller may leave them out.
            await self._memory.append(
                session_id,
                turn,
                subject=_optional_str(input.metadata.get("subject")),
                metadata=_session_metadata(input.metadata),
                turn_id=_turn_id(input.metadata, invocation_id),
            )
        else:
            await self._memory.save(session_id, list(history) + turn)

    async def _user_content(self, input: AgentInput) -> tuple[str, str | list[dict]]:
        """The turn as text (files described), and the user message for the model.

        Without files both are ``input.message``. Each file adds a line that
        names it (``describe``): kind, name, type and size, never its bytes.
        With ``vision`` in the model config, an image of a common type is
        also sent as image content (OpenAI ``image_url`` form, a data URL;
        each provider converts it), within the turn's limits: at most
        ``max_images_per_turn`` read attempts (sent or failed, so also at most
        that many resolver calls) and ``max_image_bytes_per_turn`` bytes in
        all, each at most ``MAX_MODEL_IMAGE_BYTES``. A repeated image (the
        same reference, or the same inline bytes) is read once, and its
        outcome, sent or failed, is reused. An
        image past the limits, one that can't be read, or one whose bytes are
        not its declared type is described, with the reason, instead.
        """
        media = media_of(input.parts)
        if not media:
            return input.message, input.message
        lines = [input.message] if input.message else []
        images: list[dict] = []
        # One outcome per distinct image (reference or bytes): None when it
        # was sent, else the reason it wasn't. A repeat is never read again.
        outcomes: dict[str, str | None] = {}
        attempts = 0
        budget = self._max_image_bytes_per_turn
        vision = bool(self._model_config.get("vision"))
        for part in media:
            # Only images a vision model takes are read; every other file
            # is described without touching the resolver.
            if not (vision and isinstance(part, ImagePart) and part.mime_type in MODEL_IMAGE_TYPES):
                lines.append(part.describe())
                continue
            key = part.uri or "sha256:" + hashlib.sha256(part.data or b"").hexdigest()
            if key in outcomes:
                reason = outcomes[key]
                lines.append(f"{part.describe()} " + ("(repeated; shown once)" if reason is None else f"(unavailable: {reason})"))
                continue
            # Every attempt counts, failed or not: at most this many reads
            # (resolver calls) per turn, whatever the request carries.
            if attempts >= self._max_images_per_turn or budget <= 0:
                lines.append(f"{part.describe()} (not shown: image limit for this turn reached)")
                continue
            attempts += 1
            reason = None
            try:
                data = await read_part(
                    part,
                    max_bytes=min(MAX_MODEL_IMAGE_BYTES, budget),
                    metadata=input.metadata,
                    resolver=self._files,
                )
            except FileAccessError as exc:
                reason = exc.reason
            else:
                if not looks_like(part.mime_type, data):
                    reason = "content does not match its type"
            outcomes[key] = reason
            if reason is not None:
                log.info("image not sent to the model: %s", reason)
                lines.append(f"{part.describe()} (unavailable: {reason})")
                continue
            budget -= len(data)
            lines.append(part.describe())
            url = f"data:{part.mime_type};base64,{base64.b64encode(data).decode()}"
            images.append({"type": "image_url", "image_url": {"url": url}})
        text = "\n".join(lines)
        if not images:
            return text, text
        return text, [{"type": "text", "text": text}, *images]

    def _offered_tools(self, caller: Caller | None, overrides: dict[str, frozenset[Audience]]) -> set[str]:
        """Names of the tools offered on this caller's turn, by ``effective_audiences``."""
        names = set()
        for n, w in self._local_tools.items():
            marking = _marking_of(w)
            if may_offer(caller, effective_audiences(n, marking.slug, marking.audiences, overrides)):
                names.add(n)
        names |= {
            n for n, spec in self._mcp_tools.items()
            # MCP: `customer_safe_tools` is the config's (legacy) marking; the
            # platform's setting names an MCP tool by name only.
            if may_offer(
                caller, effective_audiences(n, None, marked(n in spec["config"].get("customer_safe_tools", [])), overrides)
            )
        }
        return names

    def _key_of(self, name: str) -> str:
        """The platform's key for a tool: a manifest tool's slug key, else its name."""
        tool = self._local_tools.get(name)
        slug = _marking_of(tool).slug if tool is not None else None
        return slug_key(slug) if slug else name

    def _tool_by_key(self, key: str) -> str | None:
        """The tool a platform key names, by the same rule as audiences (§5)."""
        for name in [*self._local_tools, *self._mcp_tools]:
            if self._key_of(name) == key:
                return name
        return None

    def _sensitive_tools(self, flags: dict[str, bool]) -> set[str]:
        """Names of the sensitive tools: the author's marking, plus the platform's ``true`` (add-only)."""
        names = {
            n for n, w in self._local_tools.items()
            if _is_sensitive(self._key_of(n), _marking_of(w).sensitive, flags)
        }
        names |= {
            n for n, spec in self._mcp_tools.items()
            if _is_sensitive(n, n in spec["config"].get("sensitive_tools", []), flags)
        }
        return names

    async def _audited_dispatch(
        self,
        tc: ToolCall,
        offered: set[str],
        input: AgentInput,
        invocation_id: str,
        session_id: str,
        *,
        contain_tool_errors: bool = False,
        allowed_skills: set[str] | None = None,
    ) -> tuple[str, bool]:
        """One tool call, inside the turn, with its ``tool.call`` / ``tool.result`` audit events.

        With ``contain_tool_errors`` (an approved call), a tool that raises
        becomes a failed result the model reports, instead of ending the turn.
        """
        auditor = self._auditor
        if auditor:
            await auditor.emit(
                "tool.call",
                invocation_id=invocation_id,
                session_id=session_id,
                data={"tool_name": tc.name, "arguments": auditor.truncate(str(tc.arguments))},
            )
        tool_start = time.monotonic()
        try:
            result, ok = await _turn.call_within(input.metadata, self._dispatch_tool(tc, offered, allowed_skills))
        except Exception as exc:
            if not contain_tool_errors:
                raise
            log.exception("tool %s failed", tc.name)
            result, ok = f"Error: the approved action failed ({type(exc).__name__}).", False
        if auditor:
            await auditor.emit(
                "tool.result",
                invocation_id=invocation_id,
                session_id=session_id,
                data={"tool_name": tc.name, "result": auditor.truncate(self._public_tool_result(tc.name, result, ok))},
                latency_ms=(time.monotonic() - tool_start) * 1000,
            )
        return result, ok

    @staticmethod
    def _public_tool_result(name: str, result: str, ok: bool) -> str:
        return SKILL_MEMORY_MARKER if name == SKILL_TOOL_NAME and ok else result

    def _read_skill(self, tc: ToolCall, allowed_skills: set[str] | None) -> tuple[str, bool]:
        name = tc.arguments.get("name")
        skill = self._skills.get(name) if isinstance(name, str) else None
        if skill is None or (allowed_skills is not None and name not in allowed_skills):
            return "Error: skill not found", False
        path = tc.arguments.get("path")
        if path is not None:
            content = skill.references.get(path) if isinstance(path, str) else None
            return (content, True) if content is not None else ("Error: skill reference not found", False)
        references = "\nReferences: " + ", ".join(skill.references) if skill.references else ""
        return f"Skill: {skill.name}\nVersion: {skill.version}{references}\n\n{skill.instructions}", True

    async def _dispatch_tool(self, tc: ToolCall, offered: set[str] | None = None, allowed_skills: set[str] | None = None) -> tuple[str, bool]:
        """The tool's result for the model, and whether the tool ran.

        ``ok`` is false when the tool is unknown, not offered on this turn, or
        an MCP call failed; the result is then the error text the model sees.
        A local tool that raises still ends the run, as it always has.
        """
        if offered is not None and tc.name not in offered:
            # Same answer as an unknown tool: a customer's turn learns nothing
            # about the operator-only tools it cannot see.
            return f"Error: tool '{tc.name}' not found", False
        if tc.name == SKILL_TOOL_NAME and self._skills:
            return self._read_skill(tc, allowed_skills)
        if tc.name in self._local_tools:
            return await self._local_tools[tc.name].call(tc), True
        if tc.name in self._mcp_tools:
            return await self._call_mcp_tool(tc)
        return f"Error: tool '{tc.name}' not found", False

    async def _init_mcp(self) -> None:
        for config in self._mcp_configs:
            try:
                tools = await _fetch_mcp_tools(config)
                # An unwrapped ToolProtocol object (a peer) carries no marking,
                # so it gives way to an MCP tool of the same name, which keeps
                # its config's `sensitive_tools` and `customer_safe_tools`.
                for name in [n for n in tools if n in self._local_tools]:
                    if not isinstance(self._local_tools[name], ToolWrapper):
                        log.warning("tool %r skipped: the name is taken by an MCP tool", name)
                        del self._local_tools[name]
                # An MCP server names its own tools. One that reuses a local
                # tool's name is dropped: dispatch goes by name, so it could
                # otherwise lend its customer-safe mark to an operator-only tool.
                for name in [n for n in tools if n in self._local_tools or n in self._mcp_tools or (self._skills and n == SKILL_TOOL_NAME)]:
                    log.warning("MCP tool %r skipped: the name is already taken by another tool", name)
                    del tools[name]
                # A name starting with `@` is a slug key's form:
                # an MCP tool can't take one and borrow a manifest tool's settings.
                for name in [n for n in tools if n.startswith(("@", "skill:"))]:
                    log.warning("MCP tool %r skipped: a tool name can't start with '@'", name)
                    del tools[name]
                self._mcp_tools.update(tools)
            except Exception:
                log.exception("Failed to initialise MCP server: %s", config)
        self._mcp_ready = True

    async def _call_mcp_tool(self, tc: ToolCall) -> tuple[str, bool]:
        spec = self._mcp_tools[tc.name]
        try:
            return await _invoke_mcp_tool(spec["config"], tc.name, tc.arguments)
        except Exception as exc:
            return f"Error calling MCP tool '{tc.name}': {exc}", False

    @staticmethod
    def _message_to_dict(msg: Message) -> dict:
        d: dict = {"role": msg.role, "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [_tool_call_to_dict(tc) for tc in msg.tool_calls]
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        return d


# ------------------------------------------------------------------
# MCP helpers (lazy-import mcp package)
# ------------------------------------------------------------------

async def _fetch_mcp_tools(config: MCPConfig) -> dict[str, dict]:
    """Connect to an MCP server and return its tool schemas."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
    except ImportError:
        raise ImportError(
            "MCP support requires the mcp extra: pip install 'genfleet-sdk[mcp]'"
        )

    tools: dict[str, dict] = {}

    if config["type"] == "stdio":
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                for tool in result.tools:
                    tools[tool.name] = {
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        "config": config,
                    }

    elif config["type"] == "sse":
        async with sse_client(config["url"]) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                for tool in result.tools:
                    tools[tool.name] = {
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        "config": config,
                    }

    return tools


async def _invoke_mcp_tool(config: MCPConfig, name: str, arguments: dict) -> tuple[str, bool]:
    """The MCP tool's result text, and whether the call succeeded.

    A tool that fails on the server's side does not raise: per the MCP spec
    ``call_tool`` returns a result with ``isError`` set, and that is a failure.
    """
    if config["type"] not in ("stdio", "sse"):
        return "Error: unsupported MCP transport", False

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client

    if config["type"] == "stdio":
        transport = stdio_client(StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
        ))
    else:
        transport = sse_client(config["url"])
    async with transport as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(name, arguments)
        return str(result.content), not result.isError


def _optional_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


#: Input-metadata keys that describe the *session* rather than the turn, and
#: are worth keeping with it. Everything else in ``metadata`` is per request.
#: Deliberately one key: ``channel`` says which ingress a thread arrived on
#: and identifies nobody. The party is already named by
#: ``subject``, an opaque id; ``contact`` — a phone number or an address — is
#: customer PII and is not copied into the store until a spec asks for it.
_SESSION_METADATA_KEYS = ("channel",)


def _session_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    kept = {k: metadata[k] for k in _SESSION_METADATA_KEYS if metadata.get(k) is not None}
    return kept or None


def _turn_id(metadata: dict[str, Any], invocation_id: str) -> str:
    """The idempotency key for this turn.

    A caller's own id is what makes a retry recognisable — the invocation id
    is fresh on every attempt, so the fallback only dedupes a redelivery of
    the same in-flight call.
    """
    return _turn.turn_id_of(metadata) or invocation_id
