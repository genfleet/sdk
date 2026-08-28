"""
Agent with MCP (Model Context Protocol) server connections.

MCP tool schemas are fetched from the server on first run() and merged
with any local tools. Both stdio and SSE transports are supported.

Requires:
    pip install 'withfleet-sdk[openai,mcp,serve]'
    npx -y @modelcontextprotocol/server-filesystem /tmp   # example MCP server

Run:
    export OPENAI_API_KEY=sk-...
    python examples/mcp_agent.py

Test:
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{
        "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
        "params": {
          "id": "t1",
          "message": {"role": "user", "parts": [{"type": "text", "text": "List the files in /tmp"}]}
        }
      }' | python -m json.tool
"""

import os

from fleet.sdk import Agent
from fleet.sdk.serve import serve


# ── stdio MCP server (spawns a subprocess) ────────────────────────────────
filesystem_mcp: dict = {
    "type":    "stdio",
    "command": "npx",
    "args":    ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
}

# ── SSE MCP server (connects to a running HTTP server) ────────────────────
# remote_mcp: dict = {
#     "type": "sse",
#     "url":  "http://localhost:3001/sse",
# }


def read_file(path: str) -> str:
    """Reads a local file (local tool alongside MCP tools)."""
    try:
        with open(path) as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


agent = Agent(
    role="You are a helpful file assistant. Use the available tools to read and list files.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    tools=[read_file],     # local tools — merged with MCP tools at runtime
    mcps=[filesystem_mcp], # MCP servers — tools discovered on first run()
)

if __name__ == "__main__":
    serve(agent, name="mcp-agent", description="Agent with filesystem MCP server", port=8000)
