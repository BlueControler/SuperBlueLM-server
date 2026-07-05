from __future__ import annotations

import asyncio
import json
import os
import re
import zipfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast
from xml.etree import ElementTree

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from ..confirmations import ConfirmationTransaction, create_confirmation
from ..json_types import JsonObject, to_json_value
from ..progress import emit_needs_confirmation, emit_task_complexity, emit_task_progress
from ..tools.external import DEFAULT_TIMEOUT_SECONDS, CommandRunner, SafeCommandRunner
from .middleware import _message_content_to_text
from .state import MobileAgentState, device_id_from_mapping

FIXED_MEETING_MINUTES_UTTERANCE = "帮我把今天的会议记录整理成纪要，提取待办事项，并发送到项目群。"
MEETING_MINUTES_TASK_TYPE = "meeting_minutes_send"
MEETING_MINUTES_PHASE = "meeting_minutes_sop"
DEFAULT_PROJECT_GROUP = "项目群"
DEFAULT_SENDER = "wecom"
MAX_MEETING_FILE_BYTES = 256 * 1024
WAITING_CONFIRMATION_MESSAGE = "会议纪要已整理完成，等待你确认是否发送到项目群。"

MEETING_FILE_KEYWORDS = (
    "会议",
    "会议记录",
    "纪要",
    "meeting",
    "minutes",
)


class DateProvider(Protocol):
    def __call__(self) -> str: ...


class AsyncToolCall(Protocol):
    def __call__(self, arguments: JsonObject) -> Awaitable[Any]: ...


def is_meeting_minutes_sop_request(text: str) -> bool:
    normalized = _normalize_utterance(text)
    if normalized == _normalize_utterance(FIXED_MEETING_MINUTES_UTTERANCE):
        return True
    has_meeting_target = any(
        marker in normalized
        for marker in ("会议纪要", "会议总结", "会议记录", "会议")
    )
    has_summary_action = any(
        marker in normalized
        for marker in ("整理", "总结", "提取待办", "待办事项", "行动项", "发送到项目群")
    )
    has_demo_scope = any(marker in normalized for marker in ("今天", "会议纪要", "会议总结"))
    return has_meeting_target and has_summary_action and has_demo_scope


@dataclass(frozen=True)
class MeetingMaterialCandidate:
    title: str
    path: str | None
    uri: str | None
    source: str
    modified_time: str
    snippet: str
    confidence: float
    content_length: int = 0

    def ref(self) -> str:
        return self.path or self.uri or self.title

    def to_json(self) -> JsonObject:
        payload: JsonObject = {
            "title": self.title,
            "source": self.source,
            "modifiedTime": self.modified_time,
            "snippet": self.snippet,
            "confidence": round(self.confidence, 3),
            "contentLength": self.content_length,
        }
        if self.path:
            payload["path"] = self.path
        if self.uri:
            payload["uri"] = self.uri
        return payload


@dataclass(frozen=True)
class MeetingMinutesSopRunner:
    search_roots: Sequence[Path] | None = None
    now: DateProvider | None = None
    command_runner: CommandRunner | None = None
    project_group: str = DEFAULT_PROJECT_GROUP
    sender: str = DEFAULT_SENDER
    auto_confirm: bool = False
    file_search: AsyncToolCall | None = None
    file_reader: AsyncToolCall | None = None

    async def run(self, device_id: str | None = None) -> JsonObject:
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
        candidates = await self._search_candidates(today, device_id)
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
        meeting_text = await self._read_meeting_text(selected_file, device_id)
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

        send_tool = "feishu_cli" if self.sender.lower() in {"feishu", "lark"} else "wecom_cli"
        if not self.auto_confirm:
            confirmation_transaction = _create_meeting_minutes_confirmation(
                minutes=minutes,
                project_group=self.project_group,
                sender=self.sender,
                runner=self.command_runner,
                tool_name=send_tool,
            )
            _emit_needs_confirmation_event(confirmation_transaction)
            _emit_waiting_confirmation_progress(
                confirmation_transaction,
                completed_steps=completed_steps,
            )
            return {
                "task_type": MEETING_MINUTES_TASK_TYPE,
                "selected_file": _display_path(selected_file),
                "candidates": [_display_path(candidate) for candidate in candidates],
                "candidate_details": [candidate.to_json() for candidate in candidates],
                "minutes": minutes,
                "confirmation": {
                    "status": "needs_confirmation",
                    "auto_confirm": False,
                    "target": self.project_group,
                    "confirmationId": confirmation_transaction.confirmation_id,
                    "needsConfirmation": confirmation_transaction.needs_confirmation_event(),
                },
                "send_result": {
                    "skipped": True,
                    "reason": "needs_confirmation",
                    "confirmationId": confirmation_transaction.confirmation_id,
                    "dryRun": confirmation_transaction.dry_run,
                },
                "sent": False,
                "completed_steps": completed_steps,
                "final_message": _meeting_minutes_waiting_message(
                    minutes=minutes,
                    selected_file=selected_file,
                    candidates=candidates,
                    project_group=self.project_group,
                ),
            }

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
        sent = _send_result_ok(send_result)
        completed_steps = _append_completed_step(
            completed_steps,
            5,
            "正在发送到项目群",
            send_tool,
        )

        final_message = _meeting_minutes_final_message(
            minutes=minutes,
            selected_file=selected_file,
            candidates=candidates,
            send_result=send_result,
            sender=self.sender,
            project_group=self.project_group,
        )
        return {
            "task_type": MEETING_MINUTES_TASK_TYPE,
            "selected_file": _display_path(selected_file),
            "candidates": [_display_path(candidate) for candidate in candidates],
            "candidate_details": [candidate.to_json() for candidate in candidates],
            "minutes": minutes,
            "confirmation": confirmation,
            "send_result": send_result,
            "sent": sent,
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

    def _search_files_for_today(self, today: str) -> list[Path]:
        return search_files(self._search_roots(), today)

    async def _search_candidates(self, today: str, device_id: str | None) -> list[MeetingMaterialCandidate]:
        if self.file_search is not None:
            arguments: JsonObject = {
                "keywords": ["会议", "纪要", "记录", today],
                "limit": 20,
                "includeMetadata": True,
            }
            roots = [str(root) for root in self._search_roots()] if self.search_roots else []
            if roots:
                arguments["roots"] = roots
            if device_id:
                arguments["device_id"] = device_id
            result = await _call_tool(self.file_search, arguments)
            candidates = _file_candidates_from_tool_result(result)
            if candidates:
                return _rank_candidates(candidates, today)
        paths = await asyncio.to_thread(self._search_files_for_today, today)
        local_candidates = [
            _candidate_from_path(path, today)
            for path in paths
        ]
        return _rank_candidates(local_candidates, today)

    async def _read_meeting_text(self, selected_file: MeetingMaterialCandidate | None, device_id: str | None) -> str:
        if selected_file is None:
            return ""
        if self.file_reader is not None:
            arguments: JsonObject = {
                "path": selected_file.ref(),
                "uri": selected_file.uri or selected_file.ref(),
                "source": selected_file.source,
                "max_bytes": MAX_MEETING_FILE_BYTES,
            }
            if device_id:
                arguments["device_id"] = device_id
            result = await _call_tool(self.file_reader, arguments)
            text = _text_from_tool_result(result)
            if text:
                return text
        path = Path(selected_file.ref())
        return await asyncio.to_thread(read_text_file, path)

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
        result = asyncio.run(self.runner.run(_request_device_id(request)))
        return _model_response(result)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if not _request_matches(request.messages):
            return await handler(request)
        result = await self.runner.run(_request_device_id(request))
        return _model_response(result)


def build_meeting_minutes_sop_runner(
    *,
    scenario_system_tools: Sequence[BaseTool],
) -> MeetingMinutesSopRunner:
    tools = _tools_by_name(scenario_system_tools)
    return MeetingMinutesSopRunner(
        file_search=_base_tool_adapter(tools.get("search_files")),
        file_reader=_base_tool_adapter(tools.get("read_text_file")),
    )


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
    if path.suffix.lower() == ".docx":
        return _read_docx_text(path)
    raw = path.read_bytes()[:MAX_MEETING_FILE_BYTES]
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


async def _call_tool(tool_call: AsyncToolCall, arguments: JsonObject) -> Any:
    return await tool_call(arguments)


def _base_tool_adapter(tool: BaseTool | None) -> AsyncToolCall | None:
    if tool is None:
        return None

    async def call(arguments: JsonObject) -> Any:
        return await tool.ainvoke(arguments)

    return call


def _tools_by_name(tools: Sequence[BaseTool]) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in tools}


def _request_device_id(request: object) -> str | None:
    state_device_id = device_id_from_mapping(getattr(request, "state", None))
    if state_device_id is not None:
        return state_device_id
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None)
    if isinstance(config, dict):
        return device_id_from_mapping(config.get("configurable")) or device_id_from_mapping(
            config.get("metadata")
        )
    return None


def _file_candidates_from_tool_result(result: Any) -> list[MeetingMaterialCandidate]:
    payload = _json_payload(result)
    candidates: list[MeetingMaterialCandidate] = []
    for item in _candidate_items(payload):
        if isinstance(item, str) and item.strip():
            ref = item.strip()
            candidates.append(
                MeetingMaterialCandidate(
                    title=Path(ref).name or ref,
                    path=ref,
                    uri=None,
                    source="search_files",
                    modified_time="",
                    snippet="",
                    confidence=0.55,
                )
            )
        elif isinstance(item, dict):
            path = item.get("path") or item.get("absolutePath") or item.get("uri")
            if isinstance(path, str) and path.strip():
                ref = path.strip()
                title = item.get("title") or item.get("name") or Path(ref).name or ref
                confidence = item.get("confidence")
                candidates.append(
                    MeetingMaterialCandidate(
                        title=str(title),
                        path=str(item.get("path") or item.get("absolutePath") or "") or None,
                        uri=str(item.get("uri") or "") or None,
                        source=str(item.get("source") or "search_files"),
                        modified_time=str(item.get("modifiedTime") or item.get("modified_time") or ""),
                        snippet=str(item.get("snippet") or item.get("preview") or ""),
                        confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.6,
                        content_length=int(item.get("contentLength") or item.get("size") or 0),
                    )
                )
    deduped: dict[str, MeetingMaterialCandidate] = {}
    for candidate in candidates:
        deduped.setdefault(candidate.ref(), candidate)
    return list(deduped.values())


def _candidate_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("files", "matches", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if isinstance(payload.get("path"), str):
            return [payload]
    return []


def _text_from_tool_result(result: Any) -> str:
    payload = _json_payload(result)
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("text", "content", "result", "data"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _json_payload(result: Any) -> Any:
    if isinstance(result, str):
        try:
            return json.loads(result)
        except ValueError:
            return result
    return result


def _display_path(path: MeetingMaterialCandidate | Path | str | None) -> str | None:
    if path is None:
        return None
    if isinstance(path, MeetingMaterialCandidate):
        return path.ref()
    return str(path)


def _candidate_from_path(path: Path, today: str) -> MeetingMaterialCandidate:
    stat = path.stat()
    snippet = _snippet_from_path(path)
    modified_time = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    return MeetingMaterialCandidate(
        title=path.name,
        path=str(path),
        uri=path.as_uri() if path.is_absolute() else None,
        source="local_file",
        modified_time=modified_time,
        snippet=snippet,
        confidence=_candidate_confidence(
            title=path.name,
            modified_time=modified_time,
            snippet=snippet,
            content_length=stat.st_size,
            today=today,
        ),
        content_length=stat.st_size,
    )


def _snippet_from_path(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        text = _read_docx_text(path)
    else:
        raw = path.read_bytes()[:4096]
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
    return " ".join(_meaningful_lines(text)[:4])[:240]


def _rank_candidates(
    candidates: Sequence[MeetingMaterialCandidate],
    today: str,
) -> list[MeetingMaterialCandidate]:
    rescored = [
        MeetingMaterialCandidate(
            title=candidate.title,
            path=candidate.path,
            uri=candidate.uri,
            source=candidate.source,
            modified_time=candidate.modified_time,
            snippet=candidate.snippet,
            confidence=max(
                candidate.confidence,
                _candidate_confidence(
                    title=candidate.title,
                    modified_time=candidate.modified_time,
                    snippet=candidate.snippet,
                    content_length=candidate.content_length,
                    today=today,
                ),
            ),
            content_length=candidate.content_length,
        )
        for candidate in candidates
    ]
    return sorted(
        rescored,
        key=lambda candidate: (
            candidate.confidence,
            _modified_sort_key(candidate.modified_time),
            candidate.content_length,
            candidate.title,
        ),
        reverse=True,
    )


def _candidate_confidence(
    *,
    title: str,
    modified_time: str,
    snippet: str,
    content_length: int,
    today: str,
) -> float:
    score = 0.2
    lowered_title = title.lower()
    if any(keyword.lower() in lowered_title for keyword in MEETING_FILE_KEYWORDS):
        score += 0.25
    if any(token.lower() in lowered_title for token in _date_tokens(today)):
        score += 0.2
    if today in modified_time:
        score += 0.15
    if 80 <= content_length <= MAX_MEETING_FILE_BYTES:
        score += 0.1
    if any(marker in snippet for marker in ("待办", "结论", "讨论", "风险", "负责")):
        score += 0.1
    return min(score, 0.99)


def _modified_sort_key(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _read_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as docx:
            document_xml = docx.read("word/document.xml")
    except (KeyError, OSError, zipfile.BadZipFile):
        return ""

    try:
        root = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError:
        return ""

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        text_parts = [
            text_node.text or ""
            for text_node in paragraph.findall(".//w:t", namespace)
        ]
        paragraph_text = "".join(text_parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)
    return "\n".join(paragraphs)


def summarize_meeting(content: str, today: str) -> str:
    lines = _meaningful_lines(content)
    topic = _meeting_topic(lines, today)
    conclusions = _section_lines(lines, ("结论", "决定", "共识", "目标"), fallback_count=2)
    discussion_points = _discussion_points(lines)
    action_items = _extract_action_items(lines)
    risks = _section_lines(lines, ("风险", "待确认", "阻塞", "问题"), fallback_count=0)

    action_table = "\n".join(
        f"| {item['task']} | {item['owner']} | {item['deadline']} | {item['evidence']} |"
        for item in action_items
    )
    if not action_table:
        action_table = "| 暂无明确待办 | 待确认 | 待确认 | 未在会议资料中识别到明确行动项 |"

    risk_lines = risks or ["暂无明确风险；发送前仍建议确认项目群、负责人和截止时间是否准确。"]
    message_body = _group_message_body(topic, conclusions, action_items, risk_lines)

    return "\n".join(
        [
            f"# 会议纪要（{today}）",
            "",
            "## 会议主题",
            topic,
            "",
            "## 核心结论",
            *[f"- {line}" for line in (conclusions or ["会议资料未给出明确结论，需会后补充确认。"])],
            "",
            "## 讨论要点",
            *[f"- {line}" for line in (discussion_points or ["会议正文较短，未识别到更多讨论要点。"])],
            "",
            "## 待办事项",
            "| 事项 | 负责人 | 截止时间 | 来源证据 |",
            "| --- | --- | --- | --- |",
            action_table,
            "",
            "## 风险或待确认事项",
            *[f"- {line}" for line in risk_lines],
            "",
            "## 可发送消息正文",
            message_body,
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
    result_object["target"] = project_group
    result_object["toolName"] = "feishu_cli" if sender_name in {"feishu", "lark"} else "wecom_cli"
    message_id = _message_id_from_send_result(result_object)
    if message_id is not None:
        result_object["messageId"] = message_id
    return result_object


def _send_result_ok(send_result: JsonObject) -> bool:
    return bool(send_result.get("ok")) and not send_result.get("error")


def _send_failure_message(send_result: JsonObject, *, sender: str) -> str:
    if send_result.get("error") == "command_not_found":
        env_var = "LARK_CLI_BIN" if sender.lower() in {"feishu", "lark"} else "WECOM_CLI_BIN"
        app_name = "飞书" if sender.lower() in {"feishu", "lark"} else "企业微信"
        return f"{app_name} CLI 未安装或未加入 PATH，请配置 {env_var} 后重试。"
    return "会议纪要已整理完成，但发送到项目群失败，请检查 CLI 输出后重试。"


def _message_id_from_send_result(send_result: Mapping[str, Any]) -> str | None:
    for key in ("messageId", "message_id", "id"):
        value = send_result.get(key)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    stdout = send_result.get("stdout")
    if isinstance(stdout, Mapping):
        return _message_id_from_send_result(stdout)
    return None


def _meeting_minutes_waiting_message(
    *,
    minutes: str,
    selected_file: MeetingMaterialCandidate | None,
    candidates: Sequence[MeetingMaterialCandidate],
    project_group: str,
) -> str:
    material_line = _material_line(selected_file, candidates)
    return "\n".join(
        [
            "已整理会议后处理草稿，发送到项目群前需要你确认。",
            "",
            material_line,
            "",
            minutes,
            "",
            f"待确认动作：发送到 {project_group}。",
        ]
    )


def _meeting_minutes_final_message(
    *,
    minutes: str,
    selected_file: MeetingMaterialCandidate | None,
    candidates: Sequence[MeetingMaterialCandidate],
    send_result: JsonObject,
    sender: str,
    project_group: str,
) -> str:
    sender_label = "飞书" if sender.lower() in {"feishu", "lark"} else "企业微信"
    status = "成功" if _send_result_ok(send_result) else f"失败：{_send_failure_message(send_result, sender=sender)}"
    message_id = send_result.get("messageId") or send_result.get("stdout")
    message_line = f"消息编号/输出：{message_id}" if message_id else ""
    return "\n".join(
        part
        for part in [
            "已完成会议后处理。",
            "",
            _material_line(selected_file, candidates),
            "",
            minutes,
            "",
            f"已发送到：{sender_label} / {project_group}",
            f"发送状态：{status}",
            message_line,
        ]
        if part
    )


def _material_line(
    selected_file: MeetingMaterialCandidate | None,
    candidates: Sequence[MeetingMaterialCandidate],
) -> str:
    if selected_file is None:
        return "会议资料：未找到匹配资料，已基于空内容生成待补充纪要。"
    if len(candidates) > 1:
        return f"会议资料：已找到 {len(candidates)} 份会议资料，采用 {selected_file.title}。"
    return f"会议资料：{selected_file.title}"


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
                    "meeting_minutes": _meeting_minutes_payload(result),
                    "meeting_minutes_sop": json.dumps(result, ensure_ascii=False),
                },
            )
        ]
    )


def _meeting_minutes_payload(result: JsonObject) -> JsonObject:
    return {
        "taskType": result.get("task_type"),
        "selectedFile": result.get("selected_file"),
        "candidates": result.get("candidates", []),
        "candidateDetails": result.get("candidate_details", []),
        "minutes": result.get("minutes"),
        "confirmation": result.get("confirmation", {}),
        "sendResult": result.get("send_result", {}),
        "sent": result.get("sent", False),
        "completedSteps": result.get("completed_steps", []),
        "finalMessage": result.get("final_message"),
    }


def _normalize_utterance(text: str) -> str:
    return re.sub(r"[\s，。,.!！?？]+", "", text.strip().lower())


def _create_meeting_minutes_confirmation(
    *,
    minutes: str,
    project_group: str,
    sender: str,
    runner: CommandRunner | None,
    tool_name: str,
) -> ConfirmationTransaction:
    target_app = "飞书" if sender.lower() in {"feishu", "lark"} else "企业微信"
    preview = "\n".join(
        [
            f"发送对象：{target_app} / {project_group}",
            f"待办数量：{_action_item_count(minutes)}",
            f"风险提示：请确认项目群、负责人和截止时间准确，确认后将真实发送。",
            "正文预览：",
            minutes[:900],
        ]
    )
    return create_confirmation(
        task_title="会议纪要发送",
        operation="发送会议纪要到项目群",
        target_app=target_app,
        tool_name=tool_name,
        risk_level="high",
        payload_preview=preview,
        confirm_text="确认发送",
        cancel_text="取消",
        dry_run=False,
        confirm_handler=_meeting_minutes_confirm_handler(
            minutes=minutes,
            project_group=project_group,
            sender=sender,
            runner=runner,
            tool_name=tool_name,
        ),
    )


def _action_item_count(minutes: str) -> int:
    count = 0
    in_table = False
    for line in minutes.splitlines():
        if line.startswith("| 事项 |") or line.startswith("| --- |"):
            in_table = True
            continue
        if in_table and line.startswith("|"):
            if "暂无明确待办" not in line:
                count += 1
        elif in_table and not line.strip():
            break
    return count


def _meeting_minutes_confirm_handler(
    *,
    minutes: str,
    project_group: str,
    sender: str,
    runner: CommandRunner | None,
    tool_name: str,
) -> Callable[[ConfirmationTransaction], Awaitable[list[JsonObject]]]:
    async def handle(transaction: ConfirmationTransaction) -> list[JsonObject]:
        running = _meeting_minutes_confirmation_task_progress_event(
            transaction,
            status="running",
            step_title="正在发送到项目群",
            message="第 5/5 步：正在发送到项目群",
            tool_name=tool_name,
            confirmation_id=transaction.confirmation_id,
        )
        send_result = await send_minutes(
            minutes,
            project_group=project_group,
            sender=sender,
            runner=runner,
        )
        if _send_result_ok(send_result):
            terminal = _meeting_minutes_confirmation_task_progress_event(
                transaction,
                status="completed",
                step_title="会议纪要已发送到项目群",
                message="第 5/5 步：会议纪要已发送到项目群",
                tool_name=tool_name,
                confirmation_id=None,
                send_result=send_result,
            )
            result_event = _meeting_minutes_task_result_event(
                status="completed",
                minutes=minutes,
                project_group=project_group,
                sender=sender,
                send_result=send_result,
            )
        else:
            terminal = _meeting_minutes_confirmation_task_progress_event(
                transaction,
                status="failed",
                step_title="发送到项目群失败",
                message=_send_failure_message(send_result, sender=sender),
                tool_name=tool_name,
                confirmation_id=None,
                send_result=send_result,
            )
            result_event = _meeting_minutes_task_result_event(
                status="failed",
                minutes=minutes,
                project_group=project_group,
                sender=sender,
                send_result=send_result,
            )
        return [running, terminal, result_event]

    return handle


def _meeting_minutes_task_result_event(
    *,
    status: str,
    minutes: str,
    project_group: str,
    sender: str,
    send_result: JsonObject,
) -> JsonObject:
    return {
        "type": "task_result",
        "taskType": MEETING_MINUTES_TASK_TYPE,
        "status": status,
        "target": project_group,
        "toolName": send_result.get("toolName")
        or ("feishu_cli" if sender.lower() in {"feishu", "lark"} else "wecom_cli"),
        "messageId": send_result.get("messageId"),
        "result": send_result,
        "finalMessage": _meeting_minutes_final_message(
            minutes=minutes,
            selected_file=None,
            candidates=[],
            send_result=send_result,
            sender=sender,
            project_group=project_group,
        ),
    }


def _meeting_minutes_confirmation_task_progress_event(
    transaction: ConfirmationTransaction,
    *,
    status: str,
    step_title: str,
    message: str,
    tool_name: str,
    confirmation_id: str | None,
    send_result: JsonObject | None = None,
) -> JsonObject:
    event: JsonObject = {
        "type": "task_progress",
        "label": tool_name,
        "taskTitle": transaction.task_title,
        "status": status,
        "phase": MEETING_MINUTES_PHASE,
        "currentStep": 5,
        "totalSteps": 5,
        "stepTitle": step_title,
        "message": message,
        "toolName": tool_name,
        "requiresConfirmation": False,
        "canCancel": False,
        "canTakeOver": False,
        "progressKey": f"meeting-minutes-sop-confirm-{transaction.confirmation_id}",
        "dryRun": transaction.dry_run,
    }
    if confirmation_id:
        event["confirmationId"] = confirmation_id
    if transaction.run_id:
        event["runId"] = transaction.run_id
    if transaction.thread_id:
        event["threadId"] = transaction.thread_id
    if send_result is not None:
        event["result"] = send_result
    return event


def _emit_needs_confirmation_event(transaction: ConfirmationTransaction) -> None:
    emit_needs_confirmation(
        confirmation_id=transaction.confirmation_id,
        run_id=transaction.run_id,
        thread_id=transaction.thread_id,
        task_title=transaction.task_title,
        operation=transaction.operation,
        target_app=transaction.target_app,
        tool_name=transaction.tool_name,
        risk_level=transaction.risk_level,
        payload_preview=transaction.payload_preview,
        confirm_text=transaction.confirm_text,
        cancel_text=transaction.cancel_text,
        dry_run=transaction.dry_run,
    )


def _emit_waiting_confirmation_progress(
    transaction: ConfirmationTransaction,
    *,
    completed_steps: Sequence[JsonObject],
) -> None:
    emit_task_progress(
        label="needs_confirmation",
        status="waiting_confirmation",
        phase=MEETING_MINUTES_PHASE,
        task_title=transaction.task_title,
        step_title="等待确认是否发送到项目群",
        message="会议纪要已整理完成，等待确认后才会发送。",
        tool_name="needs_confirmation",
        progress_key=f"meeting-minutes-sop-confirmation-{transaction.confirmation_id}",
        current_step=4,
        total_steps=5,
        completed_steps=completed_steps,
        requires_confirmation=True,
        confirmation_id=transaction.confirmation_id,
        can_cancel=True,
        can_take_over=True,
        dry_run=transaction.dry_run,
    )


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
                "evidence": line,
            }
        )
    return items[:12]


def _looks_like_action_item(line: str) -> bool:
    if re.match(r"^(主题|结论|决定|共识|讨论|风险|问题|待确认)[：:]", line):
        return False
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


def _meeting_topic(lines: Sequence[str], today: str) -> str:
    for line in lines:
        match = re.match(r"^(?:主题|会议主题|标题)[：:]\s*(?P<topic>.+)$", line)
        if match:
            return match.group("topic").strip()
    for line in lines:
        if any(keyword in line for keyword in ("会议", "例会", "复盘", "评审")) and len(line) <= 48:
            return line
    return f"{today} 项目会议"


def _section_lines(
    lines: Sequence[str],
    prefixes: Sequence[str],
    *,
    fallback_count: int,
) -> list[str]:
    matched: list[str] = []
    for line in lines:
        for prefix in prefixes:
            match = re.match(rf"^{re.escape(prefix)}[：:]\s*(?P<value>.+)$", line)
            if match:
                matched.append(match.group("value").strip())
                break
    if matched:
        return matched[:6]
    if fallback_count <= 0:
        return []
    return [
        line
        for line in lines
        if not _looks_like_action_item(line)
        and not re.match(r"^(主题|风险|问题|待确认)[：:]", line)
    ][:fallback_count]


def _discussion_points(lines: Sequence[str]) -> list[str]:
    explicit = _section_lines(lines, ("讨论", "议题", "要点"), fallback_count=0)
    if explicit:
        return explicit[:8]
    points = []
    for line in lines:
        if _looks_like_action_item(line):
            continue
        if re.match(r"^(主题|结论|决定|共识|风险|问题|待确认)[：:]", line):
            continue
        points.append(line)
    return points[:8]


def _group_message_body(
    topic: str,
    conclusions: Sequence[str],
    action_items: Sequence[JsonObject],
    risks: Sequence[str],
) -> str:
    actions = "\n".join(
        f"{index}. {item['owner']}：{item['task']}（截止：{item['deadline']}）"
        for index, item in enumerate(action_items, start=1)
    ) or "暂无明确待办，需会后确认。"
    conclusion_text = "；".join(conclusions) if conclusions else "会议结论待补充确认。"
    risk_text = "；".join(risks) if risks else "暂无明确风险。"
    return "\n".join(
        [
            f"【会议纪要】{topic}",
            f"核心结论：{conclusion_text}",
            "待办事项：",
            actions,
            f"风险/待确认：{risk_text}",
        ]
    )


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
