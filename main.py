"""Entry point for the MCP ReAct agent."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from agent.agent import Agent
from agent.config import ConfigurationError, load_mcp_servers, load_system_prompt
from agent.evaluator import Evaluator
from agent.executor import Executor
from agent.history import ExecutionHistory
from agent.llm import LLMClient
from agent.mcp_client import MCPClient
from agent.planner import Planner

DEFAULT_SYSTEM_PROMPT_PATH = Path("config/system_prompt.txt")
DEFAULT_MCP_SERVERS_PATH = Path("mcp_servers.json")


def parse_args(argv: list[str]) -> argparse.Namespace:
    print(
        "[main.py][parse_args][Start] "
        f"argv={argv}"
    )
    parser = argparse.ArgumentParser(description="MCP ReAct agent")
    parser.add_argument("goal", nargs="?", help="Goal prompt provided by the user")
    parser.add_argument("--system-prompt", dest="system_prompt", default=str(DEFAULT_SYSTEM_PROMPT_PATH))
    parser.add_argument("--mcp-config", dest="mcp_config", default=str(DEFAULT_MCP_SERVERS_PATH))
    parser.add_argument("--model", dest="model", default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--log-level", dest="log_level", default="INFO", help="Logging level")
    parsed = parser.parse_args(argv)
    print(
        "[main.py][parse_args][End] "
        f"goal={parsed.goal} system_prompt={parsed.system_prompt} mcp_config={parsed.mcp_config} "
        f"model={parsed.model} log_level={parsed.log_level}"
    )
    return parsed


def main(argv: list[str] | None = None) -> int:
    print(
        "[main.py][main][Start] "
        f"argv={argv}"
    )
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    goal = args.goal or input("目的を入力してください: ").strip()
    if not goal:
        print("目標が入力されませんでした。終了します。")
        print("[main.py][main][End] status=no_goal")
        return 1

    try:
        system_prompt = load_system_prompt(Path(args.system_prompt))
        mcp_registry = load_mcp_servers(Path(args.mcp_config))
    except ConfigurationError as exc:
        logging.error("設定ファイルの読み込みに失敗しました: %s", exc)
        print(
            "[main.py][main][End] status=configuration_error "
            f"error={exc}"
        )
        return 1

    llm = LLMClient(model=args.model)
    mcp_client = MCPClient(mcp_registry)
    planner = Planner(llm, system_prompt)
    executor = Executor(mcp_client)
    evaluator = Evaluator(llm, system_prompt)
    history = ExecutionHistory()

    agent = Agent(planner=planner, executor=executor, evaluator=evaluator, history=history)
    agent.run(goal)
    print(
        "[main.py][main][End] status=success goal={goal}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
