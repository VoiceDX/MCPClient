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
        print(
            "[agent/history.py][ExecutionHistory.add_step][Start] "
            f"iteration={iteration} plan_summary={plan_summary} server={server} "
            f"action={action}"
        )
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
        print(
            "[agent/history.py][ExecutionHistory.add_step][End] "
            f"total_steps={len(self.steps)}"
        )

    def to_prompt(self) -> str:
        """Return a human-readable summary for LLM prompting."""
        print(
            "[agent/history.py][ExecutionHistory.to_prompt][Start] "
            f"steps_count={len(self.steps)}"
        )
        if not self.steps:
            print(
                "[agent/history.py][ExecutionHistory.to_prompt][End] result=(No previous steps executed.)"
            )
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
        result_text = "\n\n".join(lines)
        print(
            "[agent/history.py][ExecutionHistory.to_prompt][End] "
            f"result_length={len(result_text)}"
        )
        return result_text

    def __len__(self) -> int:
        print(
            "[agent/history.py][ExecutionHistory.__len__][Start] "
            f"steps_count={len(self.steps)}"
        )
        length = len(self.steps)
        print(
            "[agent/history.py][ExecutionHistory.__len__][End] "
            f"length={length}"
        )
        return length

    def __iter__(self):
        print(
            "[agent/history.py][ExecutionHistory.__iter__][Start] "
            f"steps_count={len(self.steps)}"
        )
        iterator = iter(self.steps)
        print(
            "[agent/history.py][ExecutionHistory.__iter__][End] iterator_created=True"
        )
        return iterator
