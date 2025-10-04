"""Execution logic for running planned MCP actions."""
from __future__ import annotations

from typing import List, Tuple

from .mcp_client import MCPActionResult, MCPClient
from .planner import PlanStep


class Executor:
    """Executes plan steps using the configured MCP client."""

    def __init__(self, mcp_client: MCPClient) -> None:
        self.mcp_client = mcp_client

    def execute_plan(self, plan_steps: List[PlanStep]) -> List[Tuple[PlanStep, MCPActionResult]]:
        """Execute each step of the plan sequentially."""
        results: List[Tuple[PlanStep, MCPActionResult]] = []
        for step in plan_steps:
            action_result = self.mcp_client.execute(step.server, step.action, step.parameters)
            results.append((step, action_result))
        return results
