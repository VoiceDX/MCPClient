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

    namespace: Dict[str, Any] = {}
    exec(code, {}, namespace)  # noqa: S102 - intentional execution of trusted code

    if "result" in namespace:
        return repr(namespace["result"])
    return "コードの実行が完了しました (result 変数は定義されていません)。"


def parse_args(argv: list[str] | None = None) -> SimpleNamespace:
    """Parse command line arguments."""

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

    return SimpleNamespace(code=str(payload["code"]))


def main(argv: list[str] | None = None) -> int:
    """Script entry point."""

    args = parse_args(argv)
    output = execute_user_code(args.code)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
