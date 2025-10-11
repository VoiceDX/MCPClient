"""Utility script executed by the AutoGen agent tools.

This script expects a single JSON payload via the command line. The payload must
contain a ``code`` field with Python statements to execute. When the code
defines a variable named ``result``, its representation is printed. Otherwise,
the script prints a generic completion message. Any stderr output is bubbled
back to the caller to support debugging from the agent loop.
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any, Dict


def execute_user_code(code: str) -> str:
    """Execute user-provided code in an isolated namespace."""

    print(f"[python_runner.py][execute_user_code] start code={code}")
    namespace: Dict[str, Any] = {}
    exec(code, {}, namespace)  # noqa: S102 - intentional execution of trusted code

    if "result" in namespace:
        output = repr(namespace["result"])
        print(f"[python_runner.py][execute_user_code] end output={output}")
        return output
    default_message = "コードの実行が完了しました (result 変数は定義されていません)。"
    print(f"[python_runner.py][execute_user_code] end output={default_message}")
    return default_message


def parse_args(argv: list[str] | None = None) -> SimpleNamespace:
    """Parse command line arguments."""

    print(f"[python_runner.py][parse_args] start argv={argv}")
    parser = argparse.ArgumentParser(description="Execute Python code for the AutoGen tool")
    parser.add_argument(
        "payload",
        help="JSON string containing a 'code' field with Python statements",
    )
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
        raise SystemExit(f"Invalid JSON payload: {exc}") from exc

    if not isinstance(payload, dict) or "code" not in payload:
        raise SystemExit("Payload must be a JSON object containing a 'code' field")

    args = SimpleNamespace(code=str(payload["code"]))
    print(f"[python_runner.py][parse_args] end args={args}")
    return args


def main(argv: list[str] | None = None) -> int:
    """Script entry point."""

    print(f"[python_runner.py][main] start argv={argv}")
    args = parse_args(argv)
    output = execute_user_code(args.code)
    print(output)
    print("[python_runner.py][main] end return=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
