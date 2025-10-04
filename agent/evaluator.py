"""Goal completion evaluation using the LLM."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from .history import ExecutionHistory
from .llm import CompletionMessage, LLMClient
from .mcp_client import MCPActionResult


@dataclass
class EvaluationResult:
    achieved: bool
    reason: str


class Evaluator:
    """Determines whether the user's goal has been achieved."""

    def __init__(self, llm: LLMClient, system_prompt: str) -> None:
        print(
            "[agent/evaluator.py][Evaluator.__init__][Start] "
            f"llm={llm} system_prompt_length={len(system_prompt)}"
        )
        self.llm = llm
        self.system_prompt = system_prompt
        print(
            "[agent/evaluator.py][Evaluator.__init__][End] "
            f"system_prompt_length={len(self.system_prompt)}"
        )

    def assess_goal(
        self,
        goal: str,
        history: ExecutionHistory,
        latest_results: List[MCPActionResult],
    ) -> EvaluationResult:
        """Ask the LLM whether the goal has been satisfied."""
        print(
            "[agent/evaluator.py][Evaluator.assess_goal][Start] "
            f"goal={goal} latest_results_count={len(latest_results)}"
        )
        latest_summary = "\n\n".join(
            [
                "\n".join(
                    [
                        f"Server: {result.server}",
                        f"Action: {result.action}",
                        f"Parameters: {json.dumps(result.parameters, ensure_ascii=False)}",
                        f"Output: {result.output}",
                    ]
                )
                for result in latest_results
            ]
        ) or "(No actions executed in this iteration.)"

        user_prompt = f"""
目的: {goal}

実行履歴の要約:
{history.to_prompt()}

最新の実行結果:
{latest_summary}

上記の情報をもとに目的が達成されたか評価してください。
以下のJSONのみで回答してください。
{{
  "goalAchieved": true または false,
  "reason": "判断の根拠"
}}
"""
        response = self.llm.complete(
            [
                CompletionMessage(role="system", content=self.system_prompt),
                CompletionMessage(role="user", content=user_prompt.strip()),
            ],
            temperature=0.0,
        )
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("Evaluation response was not valid JSON") from exc

        achieved = bool(payload.get("goalAchieved"))
        reason = str(payload.get("reason", ""))
        result = EvaluationResult(achieved=achieved, reason=reason)
        print(
            "[agent/evaluator.py][Evaluator.assess_goal][End] "
            f"achieved={result.achieved} reason={result.reason}"
        )
        return result
