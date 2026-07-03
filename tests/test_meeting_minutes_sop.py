from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from starlette.testclient import TestClient

from mobile_agent import progress
from mobile_agent.agent.meeting_minutes_sop import (
    FIXED_MEETING_MINUTES_UTTERANCE,
    MeetingMinutesSopMiddleware,
    MeetingMinutesSopRunner,
    is_meeting_minutes_sop_request,
)
from mobile_agent.confirmations import confirmation_store
from mobile_agent.http_app import app
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


class _MissingCommandRunner(CommandRunner):
    async def run(
        self,
        command: str,
        args: Sequence[str],
        timeout: float,
    ) -> dict[str, Any]:
        return {
            "error": "command_not_found",
            "command": command,
            "message": f"Command {command!r} was not found in PATH.",
        }


def _progress_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("phase") == "meeting_minutes_sop"]


def _write_minimal_docx(path: Path, paragraphs: Sequence[str]) -> None:
    escaped_paragraphs = "\n".join(
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>" for paragraph in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{escaped_paragraphs}</w:body></w:document>"
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as docx:
        docx.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>"
            ),
        )
        docx.writestr("word/document.xml", document_xml)


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
        auto_confirm=True,
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


def test_meeting_minutes_sop_reads_today_docx_record(
    tmp_path: Path,
) -> None:
    meeting_file = tmp_path / "2026-07-03-项目会议记录.docx"
    _write_minimal_docx(
        meeting_file,
        [
            "项目例会",
            "张三：今天完成会议纪要端到端验证。",
            "李四：明天确认企业微信发送 CLI。",
        ],
    )
    runner = MeetingMinutesSopRunner(
        search_roots=(tmp_path,),
        now=lambda: "2026-07-03",
        command_runner=_FakeCommandRunner(),
        auto_confirm=True,
    )

    result = asyncio.run(runner.run())

    assert result["selected_file"] == str(meeting_file)
    assert "张三" in result["minutes"]
    assert "李四" in result["minutes"]
    assert "今天完成会议纪要端到端验证" in result["minutes"]


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
        auto_confirm=True,
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


def test_meeting_minutes_default_waits_for_confirmation_without_sending(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    (tmp_path / "2026-07-02-会议记录.txt").write_text(
        "赵六：今天完成 SOP 接入。\n",
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

    assert result["sent"] is False
    assert sender.calls == []
    assert result["confirmation"]["status"] == "needs_confirmation"
    confirmation_id = result["confirmation"]["confirmationId"]
    assert confirmation_id
    needs_confirmation_events = [
        event for event in emitted if event.get("type") == "needs_confirmation"
    ]
    assert needs_confirmation_events[-1]["confirmationId"] == confirmation_id
    assert needs_confirmation_events[-1]["toolName"] == "wecom_cli"
    assert needs_confirmation_events[-1]["dryRun"] is False
    waiting_events = [
        event
        for event in _progress_steps(emitted)
        if event.get("status") == "waiting_confirmation"
    ]
    assert waiting_events[-1]["confirmationId"] == confirmation_id
    assert waiting_events[-1]["canCancel"] is True
    assert waiting_events[-1]["canTakeOver"] is True
    assert waiting_events[-1]["dryRun"] is False
    assert [event["toolName"] for event in _progress_steps(emitted)] == [
        "search_files",
        "read_text_file",
        "llm_summary",
        "needs_confirmation",
    ]


def test_meeting_minutes_confirmation_confirm_sends_minutes_to_project_group(
    tmp_path: Path,
) -> None:
    confirmation_store.clear()
    (tmp_path / "2026-07-02-会议记录.txt").write_text(
        "赵六：今天完成 SOP 接入。\n",
        encoding="utf-8",
    )
    sender = _FakeCommandRunner()
    runner = MeetingMinutesSopRunner(
        search_roots=(tmp_path,),
        now=lambda: "2026-07-02",
        command_runner=sender,
    )

    result = asyncio.run(runner.run())
    confirmation_id = result["confirmation"]["confirmationId"]

    assert sender.calls == []
    response = TestClient(app).post(f"/mobile/confirmations/{confirmation_id}/confirm")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "confirmed"
    assert payload["dryRun"] is False
    assert sender.calls
    command, args, _timeout = sender.calls[0]
    assert command == "wecom-cli"
    assert args[:4] == ["message", "send", "--to", "项目群"]
    assert "会议纪要" in args[-1]
    assert [event["status"] for event in payload["events"]] == ["running", "completed"]
    assert all(event["dryRun"] is False for event in payload["events"])
    assert all(event["currentStep"] == 5 for event in payload["events"])
    assert all(event["totalSteps"] == 5 for event in payload["events"])
    assert all(event["phase"] == "meeting_minutes_sop" for event in payload["events"])
    assert payload["events"][0]["toolName"] == "wecom_cli"
    assert payload["events"][0]["message"] == "第 5/5 步：正在发送到项目群"
    assert payload["events"][1]["message"] == "第 5/5 步：会议纪要已发送到项目群"


def test_meeting_minutes_confirmation_reports_missing_cli_configuration(
    tmp_path: Path,
) -> None:
    confirmation_store.clear()
    (tmp_path / "2026-07-02-会议记录.txt").write_text(
        "赵六：今天完成 SOP 接入。\n",
        encoding="utf-8",
    )
    runner = MeetingMinutesSopRunner(
        search_roots=(tmp_path,),
        now=lambda: "2026-07-02",
        command_runner=_MissingCommandRunner(),
    )

    result = asyncio.run(runner.run())
    confirmation_id = result["confirmation"]["confirmationId"]
    response = TestClient(app).post(f"/mobile/confirmations/{confirmation_id}/confirm")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "confirmed"
    assert [event["status"] for event in payload["events"]] == ["running", "failed"]
    assert payload["events"][1]["toolName"] == "wecom_cli"
    assert payload["events"][1]["message"] == (
        "企业微信 CLI 未安装或未加入 PATH，请配置 WECOM_CLI_BIN 后重试。"
    )


def test_default_meeting_file_lookup_runs_off_event_loop_thread(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    search_threads: list[str] = []
    read_threads: list[str] = []
    meeting_file = tmp_path / "2026-07-03-会议记录.txt"
    meeting_file.write_text("钱七：今天完成阻塞调用修复。\n", encoding="utf-8")

    def fake_search_files(search_roots: Sequence[Path], today: str) -> list[Path]:
        search_threads.append(threading.current_thread().name)
        return [meeting_file]

    def fake_read_text_file(path: Path | None) -> str:
        read_threads.append(threading.current_thread().name)
        return "钱七：今天完成阻塞调用修复。"

    monkeypatch.setattr(
        "mobile_agent.agent.meeting_minutes_sop.search_files",
        fake_search_files,
    )
    monkeypatch.setattr(
        "mobile_agent.agent.meeting_minutes_sop.read_text_file",
        fake_read_text_file,
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    runner = MeetingMinutesSopRunner(
        now=lambda: "2026-07-03",
        command_runner=_FakeCommandRunner(),
        auto_confirm=True,
    )

    result = asyncio.run(runner.run())

    assert result["sent"] is True
    assert search_threads and search_threads[0] != threading.current_thread().name
    assert read_threads and read_threads[0] != threading.current_thread().name
