"""Execution logic for running planned MCP actions."""
from __future__ import annotations

from typing import List, Tuple

from .mcp_client import MCPActionResult, MCPClient
from .planner import PlanStep


class Executor:
    """Executes plan steps using the configured MCP client."""

    def __init__(self, mcp_client: MCPClient) -> None:
        print(
            "[agent/executor.py][Executor.__init__][Start] "
            f"mcp_client={mcp_client}"
        )
        self.mcp_client = mcp_client
        print(
            "[agent/executor.py][Executor.__init__][End] mcp_client_set=True"
        )

    def execute_plan(self, plan_steps: List[PlanStep]) -> List[Tuple[PlanStep, MCPActionResult]]:
        """Execute each step of the plan sequentially."""
        print(
            "[agent/executor.py][Executor.execute_plan][Start] "
            f"plan_steps_count={len(plan_steps)}"
        )
        results: List[Tuple[PlanStep, MCPActionResult]] = []
        for step in plan_steps:
            print(
                "[agent/executor.py][Executor.execute_plan][Info] "
                f"Executing MCP server action server={step.server} action={step.action} "
                f"parameters={step.parameters}"
            )
            action_result = self.mcp_client.execute(step.server, step.action, step.parameters)
            results.append((step, action_result))
        print(
            "[agent/executor.py][Executor.execute_plan][End] "
            f"executed_steps={len(results)}"
        )
        return results
