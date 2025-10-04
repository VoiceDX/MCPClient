"""Planning utilities for the MCP agent."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from .history import ExecutionHistory
from .llm import CompletionMessage, LLMClient
from .mcp_client import MCPClient


@dataclass
class PlanStep:
    """Represents a single step proposed by the planner."""

    summary: str
    server: str
    action: str
    parameters: Dict[str, str]


class Planner:
    """Uses an LLM to create execution plans based on user goals."""

    def __init__(self, llm: LLMClient, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def create_plan(self, goal: str, history: ExecutionHistory, mcp_client: MCPClient) -> List[PlanStep]:
        """Generate a plan for the next iteration."""
        history_prompt = history.to_prompt()
        servers_description = mcp_client.describe_servers()
        user_prompt = f"""
目的: {goal}

これまでの実行履歴:
{history_prompt}

利用可能なMCPサーバの一覧 (JSON):
{servers_description}

上記の目的を達成するために、利用可能なMCPサーバのみを用いた実行計画を策定してください。
出力は必ず以下のJSONスキーマに従ってください。
{{
  "steps": [
    {{
      "summary": "ステップの説明",
      "server": "使用するサーバ名",
      "action": "サーバで呼び出すアクションやコマンド",
      "parameters": {{"key": "value"}}
    }}
  ]
}}

各ステップはMCPサーバを具体的に使用する行動にしてください。
"""
        response = self.llm.complete(
            [
                CompletionMessage(role="system", content=self.system_prompt),
                CompletionMessage(role="user", content=user_prompt.strip()),
            ],
            temperature=0.2,
        )
        return self._parse_plan(response)

    def _parse_plan(self, response_text: str) -> List[PlanStep]:
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Planner response was not valid JSON") from exc

        steps_data = data.get("steps")
        if not isinstance(steps_data, list) or not steps_data:
            raise ValueError("Planner response must include a non-empty 'steps' array")

        steps: List[PlanStep] = []
        for entry in steps_data:
            if not isinstance(entry, dict):
                raise ValueError("Each plan step must be an object")
            summary = str(entry.get("summary", ""))
            server = str(entry.get("server", ""))
            action = str(entry.get("action", ""))
            parameters_field = entry.get("parameters", {})
            if not summary or not server or not action:
                raise ValueError("Plan steps must include summary, server, and action")
            if not isinstance(parameters_field, dict):
                raise ValueError("Plan step 'parameters' must be an object")

            parameters = {str(key): str(value) for key, value in parameters_field.items()}
            steps.append(
                PlanStep(
                    summary=summary,
                    server=server,
                    action=action,
                    parameters=parameters,
                )
            )
        return steps
