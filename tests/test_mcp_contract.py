"""
The MCP symbols `Agent` imports lazily, asserted to still exist.

Not a test that MCP works — that is MCP's job. This pins the *contract this
SDK depends on*, which nothing else covers: the imports live inside methods, so
a breaking change in the `mcp` package surfaces only when a user with MCPs
configured actually runs an agent.

Found during a security upgrade that moved `mcp` from 1.x to 2.0. The suite was
green throughout, because no test reached this code at all.
"""

from __future__ import annotations

import importlib

import pytest

# module -> the names agentinc/sdk/agent.py imports from it
MCP_IMPORTS = {
    "mcp": ["ClientSession", "StdioServerParameters"],
    "mcp.client.stdio": ["stdio_client"],
    "mcp.client.sse": ["sse_client"],
}


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [(module, symbol) for module, symbols in MCP_IMPORTS.items() for symbol in symbols],
)
def test_mcp_symbol_the_agent_imports_still_exists(module_name: str, symbol: str):
    mcp_module = pytest.importorskip(
        module_name, reason="the `mcp` extra is not installed"
    )
    assert hasattr(mcp_module, symbol), (
        f"agent.py imports {symbol} from {module_name}, and it is gone. "
        "An MCP-configured agent would fail at run time, not here."
    )
