"""Execution history tracking for the agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class StepRecord:
    """Represents a single execution step in the agent loop."""

    iteration: int
    plan_summary: str
    server: str
    action: str
    parameters: str
    result: str


@dataclass
class ExecutionHistory:
    """Keeps track of the executed steps."""

    steps: List[StepRecord] = field(default_factory=list)

    def add_step(
        self,
        iteration: int,
        plan_summary: str,
        server: str,
        action: str,
        parameters: str,
        result: str,
    ) -> None:
        self.steps.append(
            StepRecord(
                iteration=iteration,
                plan_summary=plan_summary,
                server=server,
                action=action,
                parameters=parameters,
                result=result,
            )
        )

    def to_prompt(self) -> str:
        """Return a human-readable summary for LLM prompting."""
        if not self.steps:
            return "(No previous steps executed.)"

        lines: List[str] = []
        for step in self.steps:
            lines.append(
                "\n".join(
                    [
                        f"Iteration: {step.iteration}",
                        f"Plan summary: {step.plan_summary}",
                        f"Server: {step.server}",
                        f"Action: {step.action}",
                        f"Parameters: {step.parameters}",
                        f"Result: {step.result}",
                    ]
                )
            )
        return "\n\n".join(lines)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)
