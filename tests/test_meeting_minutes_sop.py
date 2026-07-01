from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from mobile_agent import progress
from mobile_agent.agent.meeting_minutes_sop import (
    FIXED_MEETING_MINUTES_UTTERANCE,
    MeetingMinutesSopMiddleware,
    MeetingMinutesSopRunner,
    is_meeting_minutes_sop_request,
)
from mobile_agent.tools.external import CommandRunner


class _FakeCommandRunner(CommandRunner):
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
            "stdout": {"ok": True, "message_id": "demo-message"},
            "stderr": "",
            "ok": True,
        }


def _progress_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("phase") == "meeting_minutes_sop"]


def test_fixed_utterance_is_recognized() -> None:
    assert is_meeting_minutes_sop_request(FIXED_MEETING_MINUTES_UTTERANCE)
    assert is_meeting_minutes_sop_request(
        "帮我把今天的会议记录整理成纪要，提取待办事项，并发送到项目群"
    )
    assert not is_meeting_minutes_sop_request("帮我总结一下这段会议记录")


def test_meeting_minutes_sop_runs_fixed_closed_loop(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    meeting_file = tmp_path / "2026-07-02-项目会议记录.md"
    meeting_file.write_text(
        "\n".join(
            [
                "项目例会",
                "张三：完成 Echo 悬浮窗五步状态联调，今天下班前给出验收包。",
                "李四：明天补齐企业微信群发送配置。",
                "王五：本周五前整理演示脚本和风险清单。",
            ]
        ),
        encoding="utf-8",
    )
    emitted: list[dict[str, Any]] = []
    sender = _FakeCommandRunner()
    runner = MeetingMinutesSopRunner(
        search_roots=(tmp_path,),
        now=lambda: "2026-07-02",
        command_runner=sender,
    )
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    result = asyncio.run(runner.run())

    assert result["task_type"] == "meeting_minutes_send"
    assert result["selected_file"] == str(meeting_file)
    assert result["sent"] is True
    assert sender.calls
    command, args, _timeout = sender.calls[0]
    assert command == "wecom-cli"
    assert args[:4] == ["message", "send", "--to", "项目群"]
    assert "会议纪要" in args[-1]
    assert "张三" in args[-1]

    steps = _progress_steps(emitted)
    assert [
        (step["currentStep"], step["totalSteps"], step["message"], step["toolName"])
        for step in steps
    ] == [
        (1, 5, "第 1/5 步：正在查找会议记录", "search_files"),
        (2, 5, "第 2/5 步：正在读取会议内容", "read_text_file"),
        (3, 5, "第 3/5 步：正在生成会议纪要", "llm_summary"),
        (4, 5, "第 4/5 步：等待确认是否发送到项目群", "needs_confirmation"),
        (5, 5, "第 5/5 步：正在发送到项目群", "wecom_cli"),
    ]
    assert result["final_message"] == "会议纪要已整理完成，并已发送到项目群。"


def test_meeting_minutes_middleware_short_circuits_fixed_demo_request(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    (tmp_path / "2026-07-02-会议记录.txt").write_text(
        "赵六：今天完成 SOP 接入。\n",
        encoding="utf-8",
    )
    emitted: list[dict[str, Any]] = []
    runner = MeetingMinutesSopRunner(
        search_roots=(tmp_path,),
        now=lambda: "2026-07-02",
        command_runner=_FakeCommandRunner(),
    )
    middleware = MeetingMinutesSopMiddleware(runner)
    request = type(
        "Request",
        (),
        {"messages": [HumanMessage(content=FIXED_MEETING_MINUTES_UTTERANCE)]},
    )()
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    async def fail_handler(_request: Any) -> ModelResponse[Any]:
        raise AssertionError("fixed SOP request should not reach the model")

    response = asyncio.run(middleware.awrap_model_call(request, fail_handler))

    assert isinstance(response, ModelResponse)
    assert isinstance(response.result[0], AIMessage)
    assert response.result[0].content == "会议纪要已整理完成，并已发送到项目群。"
    assert json.loads(response.result[0].additional_kwargs["meeting_minutes_sop"])[
        "sent"
    ] is True
    assert [event["toolName"] for event in _progress_steps(emitted)] == [
        "search_files",
        "read_text_file",
        "llm_summary",
        "needs_confirmation",
        "wecom_cli",
    ]
