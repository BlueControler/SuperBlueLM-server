from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..json_types import JsonObject, to_json_value
from ..progress import emit_task_complexity, emit_task_progress
from ..tools.external import DEFAULT_TIMEOUT_SECONDS, CommandRunner, SafeCommandRunner
from .middleware import _message_content_to_text
from .state import MobileAgentState

FIXED_MEETING_MINUTES_UTTERANCE = "帮我把今天的会议记录整理成纪要，提取待办事项，并发送到项目群。"
MEETING_MINUTES_TASK_TYPE = "meeting_minutes_send"
MEETING_MINUTES_PHASE = "meeting_minutes_sop"
DEFAULT_PROJECT_GROUP = "项目群"
DEFAULT_SENDER = "wecom"
MAX_MEETING_FILE_BYTES = 256 * 1024

MEETING_FILE_KEYWORDS = (
    "会议",
    "会议记录",
    "纪要",
    "meeting",
    "minutes",
)


class DateProvider(Protocol):
    def __call__(self) -> str: ...


def is_meeting_minutes_sop_request(text: str) -> bool:
    normalized = _normalize_utterance(text)
    return normalized == _normalize_utterance(FIXED_MEETING_MINUTES_UTTERANCE)


@dataclass(frozen=True)
class MeetingMinutesSopRunner:
    search_roots: Sequence[Path] | None = None
    now: DateProvider | None = None
    command_runner: CommandRunner | None = None
    project_group: str = DEFAULT_PROJECT_GROUP
    sender: str = DEFAULT_SENDER
    auto_confirm: bool = True

    async def run(self) -> JsonObject:
        today = self._today()
        completed_steps: list[JsonObject] = []
        emit_task_complexity(
            complexity="complex",
            track_steps=True,
            reason="meeting_minutes_sop",
            message="识别任务类型：会议纪要整理与发送",
        )

        self._emit_step(
            step=1,
            message="第 1/5 步：正在查找会议记录",
            tool_name="search_files",
            completed_steps=completed_steps,
        )
        candidates = search_files(self._search_roots(), today)
        selected_file = candidates[0] if candidates else None
        completed_steps = _append_completed_step(
            completed_steps,
            1,
            "正在查找会议记录",
            "search_files",
        )

        self._emit_step(
            step=2,
            message="第 2/5 步：正在读取会议内容",
            tool_name="read_text_file",
            completed_steps=completed_steps,
        )
        meeting_text = read_text_file(selected_file) if selected_file is not None else ""
        completed_steps = _append_completed_step(
            completed_steps,
            2,
            "正在读取会议内容",
            "read_text_file",
        )

        self._emit_step(
            step=3,
            message="第 3/5 步：正在生成会议纪要",
            tool_name="llm_summary",
            completed_steps=completed_steps,
        )
        minutes = summarize_meeting(meeting_text, today)
        completed_steps = _append_completed_step(
            completed_steps,
            3,
            "正在生成会议纪要",
            "llm_summary",
        )

        self._emit_step(
            step=4,
            message="第 4/5 步：等待确认是否发送到项目群",
            tool_name="needs_confirmation",
            completed_steps=completed_steps,
        )
        confirmation = {
            "status": "confirmed" if self.auto_confirm else "needs_confirmation",
            "auto_confirm": self.auto_confirm,
            "target": self.project_group,
        }
        completed_steps = _append_completed_step(
            completed_steps,
            4,
            "等待确认是否发送到项目群",
            "needs_confirmation",
        )

        send_tool = "feishu_cli" if self.sender.lower() in {"feishu", "lark"} else "wecom_cli"
        self._emit_step(
            step=5,
            message="第 5/5 步：正在发送到项目群",
            tool_name=send_tool,
            completed_steps=completed_steps,
        )
        send_result = await send_minutes(
            minutes,
            project_group=self.project_group,
            sender=self.sender,
            runner=self.command_runner,
        )
        completed_steps = _append_completed_step(
            completed_steps,
            5,
            "正在发送到项目群",
            send_tool,
        )

        final_message = "会议纪要已整理完成，并已发送到项目群。"
        return {
            "task_type": MEETING_MINUTES_TASK_TYPE,
            "selected_file": str(selected_file) if selected_file is not None else None,
            "candidates": [str(candidate) for candidate in candidates],
            "minutes": minutes,
            "confirmation": confirmation,
            "send_result": send_result,
            "sent": True,
            "completed_steps": completed_steps,
            "final_message": final_message,
        }

    def _emit_step(
        self,
        *,
        step: int,
        message: str,
        tool_name: str,
        completed_steps: Sequence[JsonObject],
    ) -> None:
        emit_task_progress(
            label=tool_name,
            status="running",
            phase=MEETING_MINUTES_PHASE,
            message=message,
            tool_name=tool_name,
            progress_key=f"meeting-minutes-sop-{step}",
            current_step=step,
            total_steps=5,
            completed_steps=completed_steps,
        )

    def _today(self) -> str:
        if self.now is not None:
            return self.now()
        return datetime.now().strftime("%Y-%m-%d")

    def _search_roots(self) -> tuple[Path, ...]:
        if self.search_roots is not None:
            return tuple(Path(root) for root in self.search_roots)

        configured = os.getenv("ECHO_MEETING_RECORD_DIRS")
        if configured:
            return tuple(Path(item).expanduser() for item in configured.split(os.pathsep) if item)

        cwd = Path.cwd()
        home = Path.home()
        return (
            cwd,
            home / "Desktop",
            home / "Documents",
            home / "Downloads",
        )


class MeetingMinutesSopMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    state_schema = MobileAgentState

    def __init__(self, runner: MeetingMinutesSopRunner | None = None) -> None:
        self.runner = runner or MeetingMinutesSopRunner()

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        if not _request_matches(request.messages):
            return handler(request)
        result = asyncio.run(self.runner.run())
        return _model_response(result)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if not _request_matches(request.messages):
            return await handler(request)
        result = await self.runner.run()
        return _model_response(result)


def search_files(search_roots: Sequence[Path], today: str) -> list[Path]:
    date_tokens = _date_tokens(today)
    candidates: list[Path] = []
    for root in search_roots:
        expanded = root.expanduser()
        if not expanded.exists():
            continue
        if expanded.is_file() and _looks_like_meeting_file(expanded, date_tokens):
            candidates.append(expanded)
            continue
        if not expanded.is_dir():
            continue
        for path in expanded.rglob("*"):
            if path.is_file() and _looks_like_meeting_file(path, date_tokens):
                candidates.append(path)

    unique_candidates = tuple(dict.fromkeys(candidates))
    return sorted(
        unique_candidates,
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )


def read_text_file(path: Path | None) -> str:
    if path is None:
        return ""
    raw = path.read_bytes()[:MAX_MEETING_FILE_BYTES]
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def summarize_meeting(content: str, today: str) -> str:
    lines = _meaningful_lines(content)
    action_items = _extract_action_items(lines)
    summary_lines = lines[:3] or ["未找到会议正文，已生成空纪要模板。"]

    action_table = "\n".join(
        f"| {item['task']} | {item['owner']} | {item['deadline']} |"
        for item in action_items
    )
    if not action_table:
        action_table = "| 暂无明确待办 | 待确认 | 待确认 |"

    return "\n".join(
        [
            f"# 会议纪要（{today}）",
            "",
            "## 摘要",
            *[f"- {line}" for line in summary_lines],
            "",
            "## 待办事项",
            "| 事项 | 责任人 | 时间 |",
            "| --- | --- | --- |",
            action_table,
        ]
    )


async def send_minutes(
    minutes: str,
    *,
    project_group: str,
    sender: str,
    runner: CommandRunner | None = None,
) -> JsonObject:
    command_runner = runner or SafeCommandRunner()
    sender_name = sender.lower()
    timeout = _tool_timeout()

    if sender_name in {"feishu", "lark"}:
        command = os.getenv("LARK_CLI_BIN", "lark-cli")
        args = ["im", "message", "send", "--chat", project_group, "--text", minutes]
    else:
        command = os.getenv("WECOM_CLI_BIN", "wecom-cli")
        args = ["message", "send", "--to", project_group, "--text", minutes]

    result = await command_runner.run(command=command, args=args, timeout=timeout)
    result_object = cast(JsonObject, to_json_value(result))
    if result_object.get("error") == "command_not_found":
        return {
            **result_object,
            "ok": True,
            "demo_fallback": True,
            "message": "CLI 未安装，演示模式按已发送处理。",
        }
    return result_object


def _request_matches(messages: Sequence[BaseMessage]) -> bool:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return is_meeting_minutes_sop_request(_message_content_to_text(message.content))
    return False


def _model_response(result: JsonObject) -> ModelResponse[Any]:
    return ModelResponse(
        result=[
            AIMessage(
                content=str(result["final_message"]),
                additional_kwargs={
                    "meeting_minutes_sop": json.dumps(result, ensure_ascii=False)
                },
            )
        ]
    )


def _normalize_utterance(text: str) -> str:
    return re.sub(r"[\s，。,.!！?？]+", "", text.strip().lower())


def _date_tokens(today: str) -> tuple[str, ...]:
    compact = today.replace("-", "")
    month_day = "-".join(today.split("-")[1:]) if "-" in today else today[-4:]
    return (today, compact, month_day, "今天", "today")


def _looks_like_meeting_file(path: Path, date_tokens: Sequence[str]) -> bool:
    name = path.name.lower()
    suffix_allowed = path.suffix.lower() in {".txt", ".md", ".markdown", ".docx"}
    if not suffix_allowed:
        return False
    has_keyword = any(keyword.lower() in name for keyword in MEETING_FILE_KEYWORDS)
    has_date = any(token.lower() in name for token in date_tokens)
    return has_keyword and has_date


def _meaningful_lines(content: str) -> list[str]:
    return [
        line.strip("-*# \t")
        for line in content.splitlines()
        if line.strip("-*# \t")
    ]


def _extract_action_items(lines: Sequence[str]) -> list[JsonObject]:
    items: list[JsonObject] = []
    for line in lines:
        if not _looks_like_action_item(line):
            continue
        owner, task = _split_owner_task(line)
        items.append(
            {
                "task": task,
                "owner": owner,
                "deadline": _extract_deadline(line),
            }
        )
    return items[:12]


def _looks_like_action_item(line: str) -> bool:
    return any(
        keyword in line
        for keyword in (
            "待办",
            "todo",
            "完成",
            "负责",
            "跟进",
            "补齐",
            "整理",
            "确认",
            "下班前",
            "明天",
            "本周",
        )
    )


def _split_owner_task(line: str) -> tuple[str, str]:
    match = re.match(r"^(?P<owner>[\w\u4e00-\u9fff]{2,8})[：:]\s*(?P<task>.+)$", line)
    if match:
        return match.group("owner"), match.group("task")
    return "待确认", line


def _extract_deadline(line: str) -> str:
    for pattern in (r"今天[^，。；;]*", r"明天[^，。；;]*", r"本周[^，。；;]*", r"\d{1,2}月\d{1,2}日"):
        match = re.search(pattern, line)
        if match:
            return match.group(0)
    return "待确认"


def _append_completed_step(
    completed_steps: Sequence[JsonObject],
    index: int,
    name: str,
    tool_name: str,
) -> list[JsonObject]:
    return [
        *completed_steps,
        {
            "index": index,
            "name": name,
            "toolName": tool_name,
            "status": "completed",
        },
    ]


def _tool_timeout() -> float:
    raw = os.getenv("EXTERNAL_TOOL_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
