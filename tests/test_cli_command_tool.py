from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

from mobile_agent.tools.external import create_external_tools


class FakeCommandRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], float]] = []

    async def run(
        self,
        command: str,
        args: Sequence[str],
        timeout: float,
    ) -> dict[str, Any]:
        self.calls.append((command, list(args), timeout))
        return {
            "command": command,
            "args": list(args),
            "returncode": 0,
            "stdout": {"ok": True},
            "stderr": "",
            "ok": True,
        }


def _tools(runner: FakeCommandRunner) -> dict[str, Any]:
    return {tool.name: tool for tool in create_external_tools(command_runner=runner)}


def test_run_cli_command_executes_lark_cli_and_returns_output() -> None:
    runner = FakeCommandRunner()
    tool = _tools(runner)["run_cli_command"]

    result = json.loads(
        asyncio.run(tool.ainvoke({"command": "lark-cli calendar list --limit 5"}))
    )

    assert runner.calls == [("lark-cli", ["calendar", "list", "--limit", "5"], 30.0)]
    assert result["stdout"] == {"ok": True}
    assert result["ok"] is True


def test_run_cli_command_accepts_windows_cmd_executable_names() -> None:
    runner = FakeCommandRunner()
    tool = _tools(runner)["run_cli_command"]

    result = json.loads(
        asyncio.run(tool.ainvoke({"command": "wecom-cli.cmd message list"}))
    )

    assert runner.calls == [("wecom-cli.cmd", ["message", "list"], 30.0)]
    assert result["command"] == "wecom-cli.cmd"


def test_run_cli_command_preserves_quoted_argument_groups() -> None:
    runner = FakeCommandRunner()
    tool = _tools(runner)["run_cli_command"]

    result = json.loads(
        asyncio.run(tool.ainvoke({"command": 'lark-cli docs search "weekly report"'}))
    )

    assert runner.calls == [("lark-cli", ["docs", "search", "weekly report"], 30.0)]
    assert result["args"] == ["docs", "search", "weekly report"]


def test_run_cli_command_rejects_non_lark_or_wecom_command_before_running() -> None:
    runner = FakeCommandRunner()
    tool = _tools(runner)["run_cli_command"]

    result = json.loads(asyncio.run(tool.ainvoke({"command": "python -c pass"})))

    assert runner.calls == []
    assert result["error"] == "disallowed_command"
    assert result["allowed_commands"] == [
        "lark-cli",
        "lark-cli.cmd",
        "wecom-cli",
        "wecom-cli.cmd",
    ]
