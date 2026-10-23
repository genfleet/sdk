from __future__ import annotations

from typing import Any, Literal, NotRequired, Required, TypedDict

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .parts import Part, TextPart, text_of


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_calls: list["ToolCall"] = Field(default_factory=list)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Gemini 3.x returns an opaque signature alongside each function call and
    # rejects replayed histories that omit it. Stored base64-encoded (the wire
    # value is bytes) so history stays JSON-serializable for memory backends.
    # Always None for every other provider.
    thought_signature: str | None = None


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class AgentInput(BaseModel):
    """One turn for an agent.

    ``message`` is the turn's text. ``parts`` is the whole turn, text and
    files (``genfleet.sdk.parts``). Either may be given: an empty ``message``
    is the text parts joined, and a ``message`` with no text part in
    ``parts`` is added to them as the first part. A text-only agent reads
    ``message`` and never looks at ``parts``.
    """

    message: str = ""
    parts: list[Part] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)
    tool_schemas: list[ToolSchema] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    #: The file types ``serve`` admitted this turn under (its effective
    #: ``accepted_input_types``), set by ``serve`` only. Private: never
    #: serialised, never read from a request. ``Agent.run`` checks against it
    #: instead of its own types, so an app's override needs no change to the
    #: shared agent.
    _served_input_types: tuple[str, ...] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _text_and_parts(self) -> "AgentInput":
        texts = [p for p in self.parts if isinstance(p, TextPart)]
        if not self.message and texts:
            self.message = text_of(texts)
        elif self.message and not texts:
            self.parts = [TextPart(text=self.message), *self.parts]
        return self


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AgentOutput(BaseModel):
    """One chunk of an agent's reply.

    ``content`` is text. ``parts`` carries typed results — files, images —
    alongside it; ``serve`` sends them as A2A file parts. When ``content`` is
    left out it is the text parts joined; when it is set, it is the text and
    the text parts are not sent again.
    """

    content: str | None = None
    parts: list[Part] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage | None = None

    @model_validator(mode="after")
    def _content_from_parts(self) -> "AgentOutput":
        if self.content is None and any(isinstance(p, TextPart) for p in self.parts):
            self.content = text_of(self.parts)
        return self


#: ``AgentOutput.metadata`` key of a tool event: an output that reports a
#: tool the agent ran itself, rather than text or a call for the caller to
#: dispatch. ``Agent.run`` yields one per call and one per result, each as
#: ``AgentOutput(content="", done=False, metadata={TOOL_EVENT_KEY: {...}})``:
#:
#: * ``{"type": "tool_call", "id", "name", "arguments"}`` once the call's
#:   input is complete, before the tool runs;
#: * ``{"type": "tool_result", "id", "name", "ok", "output"}`` when it
#:   returns, always after the matching call.
#:
#: The same shape the platform's runner emits for tools it dispatches. Values
#: are raw (arguments as the model produced them, the tool's full output):
#: whoever shows them to a user redacts and truncates. A consumer that only
#: reads ``content`` can ignore these outputs; their content is empty.
TOOL_EVENT_KEY = "tool_event"


# ---------------------------------------------------------------------------
# Configuration TypedDicts — passed as plain dicts by developers
# ---------------------------------------------------------------------------

class ModelConfig(TypedDict, total=False):
    model: Required[str]      # e.g. "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-6", "gemini/gemini-1.5-pro", "openrouter/google/gemini-2.5-flash"
    api_key: str              # a direct provider's key; a model client may need none
    base_url: str             # optional — for OpenAI-compatible endpoints
    default_headers: dict[str, str]  # optional — sent on every request (OpenAI-compatible providers)
    options: dict[str, Any]   # optional — passed to the model client unchanged; the SDK never reads it
    vision: bool              # optional — the model takes images; `Agent` then sends image parts as images


class MemoryConfig(TypedDict, total=False):
    type: Required[Literal["redis", "platform"]]
    connection: str  # redis only, e.g. "redis://localhost:6379/0"
    user: str
    password: str


class MCPStdioConfig(TypedDict):
    type: Literal["stdio"]
    command: str
    args: list[str]
    #: Names of this server's tools offered on a customer's turn.
    #: Every other tool of the server is operator-only.
    customer_safe_tools: NotRequired[list[str]]
    #: Names of this server's tools that wait for an owner's approval.
    sensitive_tools: NotRequired[list[str]]


class MCPSseConfig(TypedDict):
    type: Literal["sse"]
    url: str
    customer_safe_tools: NotRequired[list[str]]
    #: Names of this server's tools that wait for an owner's approval.
    sensitive_tools: NotRequired[list[str]]


MCPConfig = MCPStdioConfig | MCPSseConfig


