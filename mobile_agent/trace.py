"""Safe, user-visible execution trace events for a single agent request."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import os
import re
from threading import RLock
from time import monotonic
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.config import get_stream_writer

TraceStepStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "waiting_for_user",
]
TraceRunStatus = Literal["succeeded", "failed", "cancelled", "waiting_for_user"]
TraceDetailKind = Literal[
    "reasoning_summary",
    "plan",
    "decision",
    "tool_call",
    "tool_args_summary",
    "tool_result",
    "observation",
    "retry",
    "warning",
    "error",
]

TRACE_EVENT_TYPE = "trace.v1"
TRACE_VERSION = 1
MAX_TRACE_TITLE_CHARS = 48
MAX_TRACE_SUMMARY_CHARS = 240
MAX_TRACE_DETAIL_TEXT_CHARS = 500
MAX_TRACE_EVENT_BYTES = 4096
_FALLBACK_SUMMARY = "处理详情已省略。"
_MAX_IDENTIFIER_CHARS = 128
TRACE_DETAIL_KINDS: set[str] = {
    "reasoning_summary",
    "plan",
    "decision",
    "tool_call",
    "tool_args_summary",
    "tool_result",
    "observation",
    "retry",
    "warning",
    "error",
}
_SENSITIVE_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*[^\s,;，。]+"
)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_IMAGE_DATA = re.compile(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE)
_UI_NODE = re.compile(r"<(?:node|hierarchy)\b[^>]*(?:/>|>.*?</(?:node|hierarchy)>)", re.IGNORECASE | re.DOTALL)
_VERIFICATION_CODE = re.compile(r"验证码\s*[:=：]?\s*[A-Za-z0-9]{0,16}")


class ToolDisplaySpec(TypedDict):
    title: str
    kind: str
    risk: Literal["low", "high"]
    safe_args_summary: Callable[[dict[str, object]], str]
    safe_result_summary: Callable[[object], str]


def _fixed_summary(value: str) -> Callable[[dict[str, object]], str]:
    return lambda _: value


def _fixed_result(value: str) -> Callable[[object], str]:
    return lambda _: value


def _spec(
    title: str,
    kind: str,
    args_summary: str,
    result_summary: str,
    *,
    risk: Literal["low", "high"] = "low",
) -> ToolDisplaySpec:
    return {
        "title": title,
        "kind": kind,
        "risk": risk,
        "safe_args_summary": _fixed_summary(args_summary),
        "safe_result_summary": _fixed_result(result_summary),
    }


TOOL_DISPLAY_REGISTRY: dict[str, ToolDisplaySpec] = {
    "observe": _spec("观察屏幕", "vision", "读取当前屏幕的安全概要，不展示截图原图。", "已获取当前屏幕概要。"),
    "launch": _spec("打开应用", "phone_action", "尝试打开目标应用。", "已发起打开应用操作。"),
    "tap": _spec("点击屏幕", "phone_action", "点击目标位置。", "点击操作已执行。"),
    "type": _spec("输入内容", "phone_action", "在目标输入框输入内容。", "输入操作已执行。"),
    "swipe": _spec("滑动屏幕", "phone_action", "按目标方向移动页面。", "页面移动操作已执行。"),
    "scroll": _spec("滑动屏幕", "phone_action", "按目标方向移动页面。", "页面移动操作已执行。"),
    "list_apps": _spec("读取应用", "system", "正在读取应用列表。", "已完成应用列表读取。"),
    "list_events": _spec("读取日程", "system", "正在查询日程。", "已完成日程查询。"),
    "get_location": _spec("获取位置", "system", "正在获取当前位置。", "已完成位置获取。"),
    "create_event": _spec("创建日程", "approval", "该操作涉及日程变更，需要用户确认。", "", risk="high"),
    "update_event": _spec("修改日程", "approval", "该操作涉及日程变更，需要用户确认。", "", risk="high"),
    "update_reminders": _spec("修改提醒", "approval", "该操作涉及日程变更，需要用户确认。", "", risk="high"),
    "execute_phone_todo": _spec("执行手机操作", "phone_action", "根据用户目标执行手机自动化任务。", "手机自动化任务已返回安全结果。"),
    "interact": _spec("等待你处理", "approval", "该操作需要你确认。我不会继续自动执行。", "", risk="high"),
    "take_over": _spec("等待你接管", "approval", "该操作需要你确认。我不会继续自动执行。", "", risk="high"),
}

_FALLBACK_TOOL_SPEC = _spec(
    "执行受控操作",
    "generic",
    "正在执行受控操作。",
    "已完成受控操作。",
)
_HIGH_RISK_PHONE_TODO_MARKERS = (
    "发送",
    "删除",
    "支付",
    "下单",
    "转账",
    "拨打",
    "授权",
    "登录",
    "密码",
    "验证码",
    "提交",
)


@dataclass
class TraceRequestContext:
    run_id: str
    thread_id: str | None


@dataclass
class TraceRunState:
    thread_id: str | None
    next_seq: int = 1
    terminal_status: TraceRunStatus | None = None
    last_accessed_at: float = 0.0
    step_ids: set[str] = field(default_factory=set)


class TraceRunRegistry:
    """Keeps one trace sequence alive while LangGraph moves between nodes."""

    _STALE_AFTER_SECONDS = 300.0

    def __init__(self) -> None:
        self._lock = RLock()
        self._runs: dict[str, TraceRunState] = {}

    def start(self, run_id: str, thread_id: str | None) -> None:
        with self._lock:
            self._prune_stale_locked()
            self._runs[run_id] = TraceRunState(
                thread_id=thread_id,
                last_accessed_at=monotonic(),
            )

    def resume(self, run_id: str) -> bool:
        with self._lock:
            self._prune_stale_locked()
            state = self._runs.get(run_id)
            if state is None:
                return False
            state.last_accessed_at = monotonic()
            return state.terminal_status is None

    def emit(
        self,
        run_id: str,
        event_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        with self._lock:
            state = self._runs.get(run_id)
            if state is None or state.terminal_status is not None:
                return None
            if event_name == "step.detail.append":
                step_id = payload.get("stepId")
                if not isinstance(step_id, str) or step_id not in state.step_ids:
                    return None
            event: dict[str, Any] = {
                "type": TRACE_EVENT_TYPE,
                "version": TRACE_VERSION,
                "runId": run_id,
                "threadId": state.thread_id,
                "eventId": f"evt_{uuid4().hex}",
                "seq": state.next_seq,
                "event": event_name,
                **payload,
            }
            state.next_seq += 1
            state.last_accessed_at = monotonic()
            if event_name == "step.upsert":
                step = payload.get("step")
                if isinstance(step, dict) and isinstance(step.get("stepId"), str):
                    state.step_ids.add(step["stepId"])
            if event_name == "run.terminal":
                state.terminal_status = payload["status"]
            return event

    def close(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)

    def _prune_stale_locked(self) -> None:
        cutoff = monotonic() - self._STALE_AFTER_SECONDS
        stale_run_ids = [
            run_id
            for run_id, state in self._runs.items()
            if state.last_accessed_at < cutoff
        ]
        for run_id in stale_run_ids:
            self._runs.pop(run_id, None)


_trace_context: ContextVar[TraceRequestContext | None] = ContextVar(
    "mobile_agent_trace_context",
    default=None,
)
_trace_runs = TraceRunRegistry()


@contextmanager
def request_context(
    *,
    thread_id: str | None,
    run_id: str | None = None,
) -> Generator[TraceRequestContext, None, None]:
    context = activate_request_context(thread_id=thread_id, run_id=run_id)
    try:
        yield context
    finally:
        clear_request_context()


def activate_request_context(
    *,
    thread_id: str | None,
    run_id: str | None = None,
    resume: bool = False,
) -> TraceRequestContext | None:
    normalized_run_id = _limit_text(run_id or f"run_{uuid4().hex}", _MAX_IDENTIFIER_CHARS)
    normalized_thread_id = _limit_text(thread_id, _MAX_IDENTIFIER_CHARS) if thread_id else None
    if resume:
        if not _trace_runs.resume(normalized_run_id):
            return None
    else:
        _trace_runs.start(normalized_run_id, normalized_thread_id)
    context = TraceRequestContext(
        run_id=normalized_run_id,
        thread_id=normalized_thread_id,
    )
    _trace_context.set(context)
    return context


def clear_request_context() -> None:
    _trace_context.set(None)


def current_context() -> TraceRequestContext | None:
    return _trace_context.get()


def display_spec_for(tool_name: str) -> ToolDisplaySpec:
    return TOOL_DISPLAY_REGISTRY.get(tool_name, _FALLBACK_TOOL_SPEC)


def tool_trace_step_id(tool_call_id: str | None) -> str | None:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    return f"tool_{_limit_text(tool_call_id, _MAX_IDENTIFIER_CHARS - len('tool_'))}"


def is_high_risk_phone_todo(todo: str) -> bool:
    return any(marker in todo for marker in _HIGH_RISK_PHONE_TODO_MARKERS)


def is_high_risk_tool(tool_name: str, args: dict[str, object]) -> bool:
    spec = display_spec_for(tool_name)
    if spec["risk"] == "high":
        return True
    if tool_name != "execute_phone_todo":
        return False
    todo = args.get("todo")
    return isinstance(todo, str) and is_high_risk_phone_todo(todo)


class TraceEmitter:
    """Emits bounded user-visible events for the active request context."""

    def __init__(
        self,
        writer_provider: Callable[[], Callable[[dict[str, Any]], None]] | None = None,
        *,
        enabled: bool | None = None,
    ) -> None:
        self._writer_provider = writer_provider or get_stream_writer
        if enabled is not None:
            self._enabled = enabled
        elif writer_provider is not None:
            self._enabled = True
        else:
            self._enabled = os.getenv("TRACE_V1_EMIT_ENABLED", "false").strip().lower() == "true"

    def run_started(self, summary: str = "正在分析请求。") -> dict[str, Any] | None:
        return self._emit("run.started", {"summary": _limit_text(summary, MAX_TRACE_SUMMARY_CHARS)})

    def step_upsert(
        self,
        *,
        step_id: str,
        kind: str,
        title: str,
        summary: str,
        status: TraceStepStatus,
        parent_id: str | None = None,
        visible_to_user: bool = True,
    ) -> dict[str, Any] | None:
        step: dict[str, Any] = {
            "stepId": _limit_text(step_id, _MAX_IDENTIFIER_CHARS),
            "kind": _limit_text(kind, _MAX_IDENTIFIER_CHARS),
            "title": _limit_text(title, MAX_TRACE_TITLE_CHARS),
            "summary": _limit_text(summary, MAX_TRACE_SUMMARY_CHARS),
            "status": status,
            "visibleToUser": visible_to_user,
        }
        if parent_id:
            step["parentId"] = _limit_text(parent_id, _MAX_IDENTIFIER_CHARS)
        return self._emit("step.upsert", {"step": step})

    def step_detail_append(
        self,
        *,
        step_id: str,
        kind: TraceDetailKind,
        title: str,
        text: str,
        detail_id: str | None = None,
        visible_to_user: bool = True,
    ) -> dict[str, Any] | None:
        if kind not in TRACE_DETAIL_KINDS:
            return None
        safe_text = _safe_detail_text(text)
        if not safe_text:
            return None
        detail: dict[str, Any] = {
            "detailId": _limit_text(detail_id or f"detail_{uuid4().hex}", _MAX_IDENTIFIER_CHARS),
            "kind": kind,
            "title": _safe_detail_title(title),
            "text": safe_text,
            "visibleToUser": visible_to_user,
        }
        return self._emit(
            "step.detail.append",
            {
                "stepId": _limit_text(step_id, _MAX_IDENTIFIER_CHARS),
                "detail": detail,
            },
        )

    def run_terminal(self, status: TraceRunStatus) -> dict[str, Any] | None:
        context = current_context()
        if context is None:
            return None
        event = self._emit("run.terminal", {"status": status})
        if event is not None:
            _trace_runs.close(context.run_id)
        return event

    def _emit(self, event_name: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        context = current_context()
        if not self._enabled or context is None:
            return None
        event = _trace_runs.emit(context.run_id, event_name, payload)
        if event is None:
            return None
        event = _fit_event(event)
        try:
            writer = self._writer_provider()
        except Exception:
            return event
        try:
            writer(event)
        except Exception:
            # 轨迹属于可观测性能力，不能因为流写入异常中断真实 Agent 执行。
            return event
        return event


def _fit_event(event: dict[str, Any]) -> dict[str, Any]:
    if _event_size(event) <= MAX_TRACE_EVENT_BYTES:
        return event
    detail = event.get("detail")
    if isinstance(detail, dict):
        detail["text"] = _limit_text(str(detail.get("text", "")), 256)
        detail["title"] = _limit_text(str(detail.get("title", "")), 32)
        if _event_size(event) <= MAX_TRACE_EVENT_BYTES:
            return event
        detail["text"] = _FALLBACK_SUMMARY
        detail["title"] = "详情"
        if _event_size(event) <= MAX_TRACE_EVENT_BYTES:
            return event
    step = event.get("step")
    if isinstance(step, dict):
        step["summary"] = _limit_text(str(step.get("summary", "")), 64)
    event.pop("summary", None)
    if _event_size(event) <= MAX_TRACE_EVENT_BYTES:
        return event
    if isinstance(step, dict):
        step["title"] = "执行过程更新"
        step["summary"] = _FALLBACK_SUMMARY
        step["kind"] = "generic"
    return event


def _event_size(event: dict[str, Any]) -> int:
    return len(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _limit_text(value: str, limit: int) -> str:
    normalized = value.strip()
    if len(normalized) <= limit:
        return normalized
    if limit <= 1:
        return normalized[:limit]
    return f"{normalized[: limit - 1]}…"


def _safe_detail_title(value: str) -> str:
    return _limit_text(_safe_trace_text(value), MAX_TRACE_TITLE_CHARS) or "详情"


def _safe_detail_text(value: str) -> str:
    return _limit_text(_safe_trace_text(value), MAX_TRACE_DETAIL_TEXT_CHARS)


def _safe_trace_text(value: str) -> str:
    text = _THINK_BLOCK.sub("", value)
    text = _IMAGE_DATA.sub("[已隐藏图片数据]", text)
    text = _UI_NODE.sub("[已隐藏界面树]", text)
    text = _SENSITIVE_VALUE.sub(lambda match: f"{match.group(1)}=***", text)
    text = _VERIFICATION_CODE.sub("[已隐藏敏感内容]", text)
    return text.strip()


__all__ = [
    "MAX_TRACE_EVENT_BYTES",
    "MAX_TRACE_DETAIL_TEXT_CHARS",
    "MAX_TRACE_SUMMARY_CHARS",
    "MAX_TRACE_TITLE_CHARS",
    "TRACE_DETAIL_KINDS",
    "TOOL_DISPLAY_REGISTRY",
    "TraceDetailKind",
    "TraceEmitter",
    "TraceRequestContext",
    "TraceRunStatus",
    "TraceStepStatus",
    "activate_request_context",
    "clear_request_context",
    "current_context",
    "display_spec_for",
    "is_high_risk_phone_todo",
    "is_high_risk_tool",
    "request_context",
    "tool_trace_step_id",
]
