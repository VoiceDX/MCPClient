"""Minimal AutoGen-based ReAct agent implementation.

This module exposes a CLI that accepts a user goal and iteratively plans,
executes, and evaluates actions using an OpenAI model accessed via AutoGen.
The agent only executes predefined Python scripts, satisfying the constraint
that all tools are shell-invoked Python programs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from autogen import AssistantAgent

DEFAULT_MODEL = "gpt-4.1-mini"
MAX_ITERATIONS = 6


@dataclass
class ToolDefinition:
    """Description of a callable tool."""

    name: str
    description: str
    script_path: Path


@dataclass
class PlanStep:
    """Single step within a plan."""

    summary: str
    tool: str
    parameters: Dict[str, str]


@dataclass
class ToolResult:
    """Represents the outcome of executing a tool."""

    tool: str
    parameters: Dict[str, str]
    output: str
    returncode: int


class ToolRegistry:
    """Keeps track of available Python-script tools."""

    def __init__(self, tools: Iterable[ToolDefinition]):
        self._tools = {tool.name: tool for tool in tools}

    def describe(self) -> str:
        """Return a human-readable summary for prompts."""

        lines = []
        for tool in self._tools.values():
            lines.append(
                f"- {tool.name}: {tool.description} (python {tool.script_path})"
            )
        return "\n".join(lines)

    def execute(self, tool_name: str, parameters: Dict[str, str]) -> ToolResult:
        """Execute a tool and capture its output."""

        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool requested: {tool_name}")

        tool = self._tools[tool_name]
        payload = json.dumps(parameters, ensure_ascii=False)
        command = ["python", str(tool.script_path), payload]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        combined_output = stdout if not stderr else f"{stdout}\n[stderr]\n{stderr}".strip()
        return ToolResult(
            tool=tool_name,
            parameters=parameters,
            output=combined_output or "<no output>",
            returncode=completed.returncode,
        )


class AutoGenReActAgent:
    """AutoGen-powered agent that follows a ReAct-style loop."""

    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        if "OPENAI_API_KEY" not in os.environ:
            raise EnvironmentError("OPENAI_API_KEY environment variable is required")

        llm_config = {
            "config_list": [
                {
                    "model": model,
                    "api_key": os.environ["OPENAI_API_KEY"],
                }
            ],
            "temperature": 0,
        }
        self.planner = AssistantAgent(
            name="planner",
            system_message="You are a meticulous planner that outputs JSON only.",
            llm_config=llm_config,
        )
        self.evaluator = AssistantAgent(
            name="evaluator",
            system_message="You judge if the goal has been achieved. Respond in JSON only.",
            llm_config=llm_config,
        )
        self.tools = tools
        self.max_iterations = max_iterations

    def run(self, goal: str) -> None:
        """Execute the planning/execution/evaluation loop."""

        history: List[str] = []
        for iteration in range(1, self.max_iterations + 1):
            logging.info("Planning iteration %s", iteration)
            plan = self._request_plan(goal, history)
            logging.debug("Plan: %s", plan)

            results: List[ToolResult] = []
            for step in plan:
                result = self.tools.execute(step.tool, step.parameters)
                logging.info(
                    "Executed %s (returncode=%s)", step.tool, result.returncode
                )
                results.append(result)
                history.append(
                    textwrap.dedent(
                        f"Iteration {iteration}: {step.summary}\n"
                        f"Tool: {step.tool}\n"
                        f"Parameters: {json.dumps(step.parameters, ensure_ascii=False)}\n"
                        f"Output: {result.output}"
                    ).strip()
                )

            evaluation = self._evaluate(goal, history, results)
            logging.info("Evaluation: %s", evaluation.get("reason", "(no reason)"))
            if evaluation.get("achieved") is True:
                print("✅ 目的を達成しました。")
                print(f"理由: {evaluation.get('reason', '理由は提供されませんでした。')}")
                return

            if iteration == self.max_iterations:
                break

        print("⚠️ 目的を達成できませんでした。追加の指示が必要かもしれません。")

    def _request_plan(self, goal: str, history: List[str]) -> List[PlanStep]:
        """Ask the planner LLM for the next execution plan."""

        history_text = "\n\n".join(history) if history else "(まだ実行履歴はありません)"
        prompt = textwrap.dedent(
            f"""
            あなたは目的達成のための計画を作成するエージェントです。
            目的: {goal}

            これまでの実行履歴:
            {history_text}

            利用可能なツール:
            {self.tools.describe()}

            上記の目的を達成するための次の行動計画をJSONで出力してください。
            フォーマット:
            {{"steps": [{{"summary": "説明", "tool": "ツール名", "parameters": {{"key": "value"}} }}]}}
            必ず存在するツール名のみを使用し、parametersは文字列値のJSONオブジェクトにしてください。
            """
        ).strip()
        response = self.planner.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = self._extract_text(response)
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Planner response was not valid JSON: {response_text}"
            ) from exc

        steps_field = data.get("steps")
        if not isinstance(steps_field, list) or not steps_field:
            raise ValueError("Planner must return a non-empty 'steps' list")

        plan_steps: List[PlanStep] = []
        for entry in steps_field:
            if not isinstance(entry, dict):
                raise ValueError("Each plan step must be an object")
            summary = str(entry.get("summary", "")).strip()
            tool = str(entry.get("tool", "")).strip()
            parameters_field = entry.get("parameters", {})
            if not summary or not tool:
                raise ValueError("Plan steps must include 'summary' and 'tool'")
            if not isinstance(parameters_field, dict):
                raise ValueError("Plan step parameters must be an object")
            parameters = {str(k): str(v) for k, v in parameters_field.items()}
            plan_steps.append(PlanStep(summary=summary, tool=tool, parameters=parameters))

        return plan_steps

    def _evaluate(
        self,
        goal: str,
        history: List[str],
        results: List[ToolResult],
    ) -> Dict[str, object]:
        """Ask the evaluator LLM whether the goal has been achieved."""

        history_text = "\n\n".join(history) if history else "(まだ実行履歴はありません)"
        latest_results = "\n\n".join(
            textwrap.dedent(
                f"Tool: {result.tool}\n"
                f"Parameters: {json.dumps(result.parameters, ensure_ascii=False)}\n"
                f"Return code: {result.returncode}\n"
                f"Output: {result.output}"
            ).strip()
            for result in results
        ) or "(今回の実行結果はありません)"
        prompt = textwrap.dedent(
            f"""
            あなたは目的達成度を評価する審査員です。
            目的: {goal}

            累積履歴:
            {history_text}

            最新の実行結果:
            {latest_results}

            JSON形式で評価を返してください。例:
            {{"achieved": true, "reason": "..."}}
            """
        ).strip()
        response = self.evaluator.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = self._extract_text(response)
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Evaluator response was not valid JSON: {response_text}"
            ) from exc

    @staticmethod
    def _extract_text(reply: object) -> str:
        """Normalize AutoGen replies to plain text."""

        if isinstance(reply, str):
            return reply
        if isinstance(reply, dict) and "content" in reply:
            return str(reply["content"])
        raise TypeError(f"Unexpected reply type: {type(reply)!r}")


def build_agent(model: str | None = None) -> AutoGenReActAgent:
    """Factory function to create the agent with default tools."""

    base_path = Path(__file__).parent / "tools"
    tools = ToolRegistry(
        [
            ToolDefinition(
                name="python_runner",
                description="任意のPythonコードを実行し、result変数を出力します。",
                script_path=base_path / "python_runner.py",
            )
        ]
    )
    return AutoGenReActAgent(model=model or DEFAULT_MODEL, tools=tools)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Minimal AutoGen ReAct agent")
    parser.add_argument("goal", nargs="?", help="Goal prompt provided by the user")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name used for planning and evaluation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Command-line entry point."""

    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    goal = args.goal or input("目的を入力してください: ").strip()
    if not goal:
        print("目標が入力されませんでした。終了します。")
        return 1

    agent = build_agent(model=args.model)
    try:
        agent.run(goal)
    except Exception as exc:  # pragma: no cover - CLI safety
        logging.exception("エージェントの実行中にエラーが発生しました")
        print(f"❌ エージェントの実行に失敗しました: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
