"""Metadata helpers for MCP server capabilities."""
from __future__ import annotations

from typing import Dict, List


_SIMULATED_SERVER_FUNCTIONS: Dict[str, List[str]] = {
    "filesystem": [
        "list_directory",
        "read_file",
        "write_file",
        "delete_path",
    ],
    "brave-search": [
        "search",
    ],
    "playwright": [
        "open_page",
        "click",
        "type",
        "screenshot",
    ],
    "excel": [
        "open_workbook",
        "list_sheets",
        "read_range",
        "write_range",
    ],
    "computer-use": [
        "move_mouse",
        "click",
        "type",
        "screenshot",
    ],
}


def get_server_functions(server_name: str) -> List[str]:
    """Return the known functions for the given server name."""
    print(
        "[agent/mcp_metadata.py][get_server_functions][Start] "
        f"server_name={server_name}"
    )
    functions = _SIMULATED_SERVER_FUNCTIONS.get(server_name, ["execute"])
    print(
        "[agent/mcp_metadata.py][get_server_functions][End] "
        f"server_name={server_name} function_count={len(functions)}"
    )
    return list(functions)
