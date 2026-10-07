from __future__ import annotations

from typing import Any, Literal, Required, TypedDict

from pydantic import BaseModel, Field


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
    message: str
    history: list[Message] = Field(default_factory=list)
    tool_schemas: list[ToolSchema] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AgentOutput(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage | None = None


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


class MemoryConfig(TypedDict, total=False):
    type: Required[Literal["redis", "platform"]]
    connection: str  # redis only, e.g. "redis://localhost:6379/0"
    user: str
    password: str


class MCPStdioConfig(TypedDict):
    type: Literal["stdio"]
    command: str
    args: list[str]


class MCPSseConfig(TypedDict):
    type: Literal["sse"]
    url: str


MCPConfig = MCPStdioConfig | MCPSseConfig


