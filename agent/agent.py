"""High-level orchestration logic for the MCP agent."""
from __future__ import annotations

import logging
from typing import Optional

from .evaluator import Evaluator
from .executor import Executor
from .history import ExecutionHistory
from .planner import Planner

logger = logging.getLogger(__name__)


class Agent:
    """Combines planning, execution, and evaluation into a ReAct-style loop."""

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        evaluator: Evaluator,
        history: Optional[ExecutionHistory] = None,
        max_iterations: int = 10,
    ) -> None:
        print(
            "[agent/agent.py][Agent.__init__][Start] "
            f"planner={planner} executor={executor} evaluator={evaluator} "
            f"history={history} max_iterations={max_iterations}"
        )
        self.planner = planner
        self.executor = executor
        self.evaluator = evaluator
        self.history = history or ExecutionHistory()
        self.max_iterations = max_iterations
        print(
            "[agent/agent.py][Agent.__init__][End] "
            f"history={self.history} max_iterations={self.max_iterations}"
        )

    def run(self, goal: str) -> None:
        """Run the agent loop for the provided goal."""
        print(
            "[agent/agent.py][Agent.run][Start] "
            f"goal={goal} max_iterations={self.max_iterations}"
        )
        for iteration in range(1, self.max_iterations + 1):
            logger.info("Planning iteration %s", iteration)
            plan_steps = self.planner.create_plan(goal, self.history, self.executor.mcp_client)
            execution_pairs = self.executor.execute_plan(plan_steps)

            latest_results = []
            for step, result in execution_pairs:
                self.history.add_step(
                    iteration=iteration,
                    plan_summary=step.summary,
                    server=step.server,
                    action=step.action,
                    parameters=str(step.parameters),
                    result=result.output,
                )
                latest_results.append(result)

            evaluation = self.evaluator.assess_goal(goal, self.history, latest_results)
            logger.info("Evaluation: %s", evaluation.reason)
            if evaluation.achieved:
                print("✅ 目的を達成しました。")
                print(f"理由: {evaluation.reason}")
                print(
                    "[agent/agent.py][Agent.run][End] "
                    f"status=achieved iteration={iteration} reason={evaluation.reason}"
                )
                return

        print("⚠️ 目的を達成できませんでした。追加の指示が必要かもしれません。")
        print("最終的な実行履歴:")
        print(self.history.to_prompt())
        print(
            "[agent/agent.py][Agent.run][End] status=not_achieved "
            f"iterations={self.max_iterations}"
        )
