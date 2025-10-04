"""Simplified MCP client integration utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

from .config import MCPServerDefinition
from .mcp_metadata import get_server_functions


@dataclass
class MCPActionResult:
    """Represents the outcome of an executed MCP action."""

    server: str
    action: str
    parameters: Dict[str, str]
    output: str


class MCPClient:
    """Simple facade around registered MCP servers.

    This class does not spawn MCP server processes by itself. Instead, it stores the
    registration information and provides a centralized location to integrate with
    MCP SDKs or HTTP clients. For demonstration purposes the ``execute`` method
    returns a simulated response so the remainder of the agent can be exercised
    without live MCP servers.
    """

    def __init__(self, registry: Dict[str, MCPServerDefinition]) -> None:
        print(
            "[agent/mcp_client.py][MCPClient.__init__][Start] "
            f"registry_keys={list(registry.keys())}"
        )
        self._registry = registry
        self._log_available_functions()
        print(
            "[agent/mcp_client.py][MCPClient.__init__][End] registry_size="
            f"{len(self._registry)}"
        )

    def _log_available_functions(self) -> None:
        """Log the available functions for each registered server."""
        print(
            "[agent/mcp_client.py][MCPClient._log_available_functions][Start] "
            f"registry_size={len(self._registry)}"
        )
        for server_name in sorted(self._registry):
            functions = get_server_functions(server_name)
            print(
                "[agent/mcp_client.py][MCPClient._log_available_functions][Info] "
                f"server_name={server_name} functions={functions}"
            )
        print(
            "[agent/mcp_client.py][MCPClient._log_available_functions][End] "
            f"registry_size={len(self._registry)}"
        )

    def describe_servers(self) -> str:
        """Return a JSON string describing the available servers for prompting."""
        print(
            "[agent/mcp_client.py][MCPClient.describe_servers][Start] "
            f"registry_size={len(self._registry)}"
        )
        payload = {
            name: {
                "command": definition.command,
                "args": definition.args,
                "env": definition.env,
            }
            for name, definition in self._registry.items()
        }
        description = json.dumps(payload, ensure_ascii=False, indent=2)
        print(
            "[agent/mcp_client.py][MCPClient.describe_servers][End] "
            f"description_length={len(description)}"
        )
        return description

    def execute(self, server_name: str, action: str, parameters: Optional[Dict[str, str]] = None) -> MCPActionResult:
        """Execute an action using the specified MCP server.

        Replace the simulated return value with real MCP invocation logic as needed.
        """
        print(
            "[agent/mcp_client.py][MCPClient.execute][Start] "
            f"server_name={server_name} action={action} parameters={parameters}"
        )
        if server_name not in self._registry:
            raise ValueError(f"Unknown MCP server: {server_name}")
        params = parameters or {}
        output = (
            "Simulated MCP execution. Integrate with actual MCP server here.\n"
            f"Server: {server_name}\nAction: {action}\nParameters: {json.dumps(params, ensure_ascii=False)}"
        )
        result = MCPActionResult(server=server_name, action=action, parameters=params, output=output)
        print(
            "[agent/mcp_client.py][MCPClient.execute][End] "
            f"server_name={server_name} action={action}"
        )
        return result
