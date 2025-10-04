"""Configuration utilities for the MCP client agent."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .mcp_metadata import get_server_functions


@dataclass
class MCPServerDefinition:
    """Represents a single MCP server registration."""

    name: str
    command: str
    args: List[str]
    env: Dict[str, str]


class ConfigurationError(RuntimeError):
    """Raised when configuration files are missing or malformed."""


def load_system_prompt(path: Path) -> str:
    """Load the system prompt from the provided path."""
    print(
        "[agent/config.py][load_system_prompt][Start] "
        f"path={path}"
    )
    if not path.exists():
        raise ConfigurationError(f"System prompt file not found: {path}")
    prompt = path.read_text(encoding="utf-8").strip()
    print(
        "[agent/config.py][load_system_prompt][End] "
        f"length={len(prompt)}"
    )
    return prompt


def load_mcp_servers(path: Path) -> Dict[str, MCPServerDefinition]:
    """Load MCP server definitions from the given JSON file."""
    print(
        "[agent/config.py][load_mcp_servers][Start] "
        f"path={path}"
    )
    if not path.exists():
        raise ConfigurationError(f"MCP server configuration not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        raise ConfigurationError("Invalid MCP configuration: 'mcpServers' missing or not an object")

    registry: Dict[str, MCPServerDefinition] = {}
    for name, definition in servers.items():
        if not isinstance(definition, dict):
            raise ConfigurationError(f"Invalid server definition for '{name}'")
        command = definition.get("command")
        args = definition.get("args", [])
        env = definition.get("env", {})

        if not isinstance(command, str):
            raise ConfigurationError(f"Server '{name}' is missing a command")
        if not isinstance(args, list):
            raise ConfigurationError(f"Server '{name}' has invalid args")
        if not isinstance(env, dict):
            raise ConfigurationError(f"Server '{name}' has invalid env")

        registry[name] = MCPServerDefinition(
            name=name,
            command=command,
            args=[str(arg) for arg in args],
            env={str(key): str(value) for key, value in env.items()},
        )
        functions = get_server_functions(name)
        print(
            "[agent/config.py][load_mcp_servers][Info] "
            f"server_name={name} functions={functions}"
        )

    print(
        "[agent/config.py][load_mcp_servers][End] "
        f"server_count={len(registry)}"
    )
    return registry
