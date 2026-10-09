from .agent import Agent
from .audit.schemas import AuditConfig, AuditEvent
from .caller import (
    CALLER_METADATA_KEY,
    TOOL_AUDIENCES_METADATA_KEY,
    Audience,
    Caller,
    caller_of,
    may_offer,
    may_use,
    tool_audiences_of,
)
from .channels import ChannelSendError, send_message, send_message_tool
from .manifest import (
    AgentManifest,
    ConfigField,
    ManifestError,
    SecretField,
    ToolManifest,
    ToolRef,
    load_agent_manifest,
    load_tool_manifest,
)
from .protocol import AgentFactory, AgentProtocol, ToolProtocol
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
from .tool import ToolWrapper, tool
from .tools_loader import (
    LocalToolResolver,
    ToolResolutionError,
    ToolResolver,
    load_tools,
)

__all__ = [
    # Core
    "Agent",
    "AgentProtocol",
    "ToolProtocol",
    "AgentFactory",
    # Schemas
    "AgentInput",
    "AgentOutput",
    "Message",
    "ToolCall",
    "ToolSchema",
    "TokenUsage",
    "TOOL_EVENT_KEY",
    # Caller roles (ADR-0028)
    "CALLER_METADATA_KEY",
    "Caller",
    "caller_of",
    "may_use",
    "may_offer",
    "tool_audiences_of",
    "TOOL_AUDIENCES_METADATA_KEY",
    "Audience",
    # Config
    "ModelConfig",
    "MemoryConfig",
    "MCPConfig",
    "AuditConfig",
    # Audit
    "AuditEvent",
    # Manifests
    "AgentManifest",
    "ToolManifest",
    "ToolRef",
    "ConfigField",
    "SecretField",
    "ManifestError",
    "load_agent_manifest",
    "load_tool_manifest",
    # Tools
    "load_tools",
    "ToolResolver",
    "LocalToolResolver",
    "ToolResolutionError",
    "ToolWrapper",
    "tool",
    # Channels (ADR-0019)
    "send_message",
    "send_message_tool",
    "ChannelSendError",
]
