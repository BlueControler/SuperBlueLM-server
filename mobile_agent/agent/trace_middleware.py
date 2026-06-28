"""LangChain middleware that turns agent activity into safe trace events."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from loguru import logger

from ..action_control import phone_action_registry, phone_action_scope
from ..trace import (
    TraceEmitter,
    activate_request_context,
    clear_request_context,
    current_context,
    display_spec_for,
    tool_trace_step_id,
)
from .state import MobileAgentState

ModelHandler: TypeAlias = Callable[[ModelRequest[Any]], ModelResponse[Any]]
AsyncModelHandler: TypeAlias = Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]]
ToolResult: TypeAlias = ToolMessage | Command[Any]
ToolHandler: TypeAlias = Callable[[ToolCallRequest], ToolResult]
AsyncToolHandler: TypeAlias = Callable[[ToolCallRequest], Awaitable[ToolResult]]

WAITING_FOR_USER_GUIDANCE = "该操作需要你确认。我不会继续自动执行。请发送新消息说明继续、取消或换一种方式。"


class TraceMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    """Emits only registry-derived, user-safe descriptions of real activity."""

    state_schema = MobileAgentState

    def __init__(self, emitter: TraceEmitter | None = None) -> None:
        self.emitter = emitter or TraceEmitter()

    def before_agent(
        self,
        state: MobileAgentState,
        runtime: Runtime[None],
    ) -> dict[str, object]:
        context = activate_request_context(
            thread_id=_thread_id(runtime, state),
            run_id=_mobile_run_id(runtime, state),
        )
        logger.info(
            "trace_before_agent run_id={} thread_id={} emitter_enabled={}",
            context.run_id, context.thread_id, self.emitter._enabled,
        )
        phone_action_registry.start_run(context.run_id, context.thread_id)
        analysis_step_id = f"phase_{context.run_id[:24]}"
        self.emitter.run_started()
        self.emitter.step_upsert(
            step_id=analysis_step_id,
            kind="summary",
            title="分析请求",
            summary="正在分析请求并确定下一步。",
            status="running",
        )
        self.emitter.step_detail_append(
            step_id=analysis_step_id,
            kind="plan",
            title="处理思路",
            text="我会先判断请求类型，并在需要时调用安全工具完成任务。",
        )
        return {
            "trace_run_id": context.run_id,
            "trace_analysis_step_id": analysis_step_id,
        }

    async def abefore_agent(
        self,
        state: MobileAgentState,
        runtime: Runtime[None],
    ) -> dict[str, object]:
        return self.before_agent(state, runtime)

    def after_agent(
        self,
        state: MobileAgentState,
        runtime: Runtime[None],
    ) -> dict[str, object] | None:
        if not self._resume_trace_context(state, runtime):
            logger.warning("trace_after_agent_skipped reason=context_resume_failed")
            return None
        trace_run_id = state.get("trace_run_id")
        if isinstance(state.get("run_failure_reason"), str) and state["run_failure_reason"]:
            status = "failed"
            reason = state["run_failure_reason"]
            self._finish_analysis(state, "failed")
            self.emitter.run_terminal("failed")
            self._mark_action_run_terminal(state)
        elif state.get("awaiting_user_action") is True:
            status = "waiting_for_user"
            reason = "awaiting_user_action"
            self._finish_analysis(state, "waiting_for_user")
            self.emitter.run_terminal("waiting_for_user")
            self._mark_action_run_terminal(state)
        else:
            status = "succeeded"
            reason = "normal_completion"
            self._finish_analysis(state, "succeeded")
            self.emitter.run_terminal("succeeded")
            self._mark_action_run_terminal(state)
        logger.info(
            "trace_after_agent run_id={} status={} reason={} emitter_enabled={}",
            trace_run_id, status, reason, self.emitter._enabled,
        )
        clear_request_context()
        return None

    async def aafter_agent(
        self,
        state: MobileAgentState,
        runtime: Runtime[None],
    ) -> dict[str, object] | None:
        return self.after_agent(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: ModelHandler,
    ) -> ModelResponse[Any]:
        try:
            return handler(request)
        except Exception:
            self._resume_trace_context(request.state, request.runtime)
            self.emitter.run_terminal("failed")
            self._mark_action_run_terminal(request.state)
            clear_request_context()
            raise

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: AsyncModelHandler,
    ) -> ModelResponse[Any]:
        try:
            return await handler(request)
        except Exception:
            self._resume_trace_context(request.state, request.runtime)
            self.emitter.run_terminal("failed")
            self._mark_action_run_terminal(request.state)
            clear_request_context()
            raise

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolResult:
        self._ensure_context(request)
        step_id = self._start_tool_step(request)
        try:
            with phone_action_scope(request.tool_call["id"]):
                result = handler(request)
        except Exception:
            self._complete_tool_step(
                request,
                step_id,
                "failed",
                detail_kind="error",
                detail_title="错误",
                detail_text="工具执行失败，已停止该步骤。",
            )
            self.emitter.run_terminal("failed")
            self._mark_action_run_terminal(request.state)
            clear_request_context()
            raise
        self._complete_for_result(request, step_id, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
    ) -> ToolResult:
        self._ensure_context(request)
        step_id = self._start_tool_step(request)
        try:
            with phone_action_scope(request.tool_call["id"]):
                result = await handler(request)
        except Exception:
            self._complete_tool_step(
                request,
                step_id,
                "failed",
                detail_kind="error",
                detail_title="错误",
                detail_text="工具执行失败，已停止该步骤。",
            )
            self.emitter.run_terminal("failed")
            self._mark_action_run_terminal(request.state)
            clear_request_context()
            raise
        self._complete_for_result(request, step_id, result)
        return result

    def _ensure_context(self, request: ToolCallRequest) -> None:
        if current_context() is not None:
            return
        state = request.state if isinstance(request.state, Mapping) else {}
        self._resume_trace_context(state, getattr(request, "runtime", None))

    def _resume_trace_context(
        self,
        state: Mapping[str, Any],
        runtime: object,
    ) -> bool:
        run_id = state.get("trace_run_id")
        if not isinstance(run_id, str) or not run_id:
            return False
        context = current_context()
        if context is not None and context.run_id == run_id:
            return True
        return activate_request_context(
            thread_id=_thread_id(runtime, state),
            run_id=run_id,
            resume=True,
        ) is not None

    def _start_tool_step(self, request: ToolCallRequest) -> str:
        tool_name = request.tool_call["name"]
        spec = display_spec_for(tool_name)
        step_id = tool_trace_step_id(request.tool_call["id"]) or "tool_unknown"
        self.emitter.step_upsert(
            step_id=step_id,
            kind=spec["kind"],
            title=spec["title"],
            summary=spec["safe_args_summary"](_tool_args(request)),
            status="running",
        )
        self.emitter.step_detail_append(
            step_id=step_id,
            kind="tool_call",
            title="工具调用",
            text=_tool_call_summary(tool_name, spec),
        )
        self.emitter.step_detail_append(
            step_id=step_id,
            kind="tool_args_summary",
            title="安全参数",
            text=spec["safe_args_summary"](_tool_args(request)),
        )
        return step_id

    def _complete_for_result(
        self,
        request: ToolCallRequest,
        step_id: str,
        result: ToolResult,
    ) -> None:
        if _is_waiting_command(result):
            self._complete_tool_step(
                request,
                step_id,
                "waiting_for_user",
                detail_kind="warning",
                detail_title="需要确认",
                detail_text=WAITING_FOR_USER_GUIDANCE,
            )
            return
        if isinstance(result, ToolMessage) and result.status == "error":
            self._complete_tool_step(
                request,
                step_id,
                "failed",
                result,
                detail_kind="error",
                detail_title="执行结果",
                detail_text="工具返回了可恢复错误，未展示原始结果。",
            )
            return
        spec = display_spec_for(request.tool_call["name"])
        self._complete_tool_step(
            request,
            step_id,
            "succeeded",
            result,
            detail_kind="observation" if request.tool_call["name"] == "observe" else "tool_result",
            detail_title="观察结果" if request.tool_call["name"] == "observe" else "工具结果",
            detail_text=spec["safe_result_summary"](result),
        )

    def _complete_tool_step(
        self,
        request: ToolCallRequest,
        step_id: str,
        status: str,
        result: object | None = None,
        *,
        detail_kind: str | None = None,
        detail_title: str | None = None,
        detail_text: str | None = None,
    ) -> None:
        spec = display_spec_for(request.tool_call["name"])
        summary = (
            WAITING_FOR_USER_GUIDANCE
            if status == "waiting_for_user"
            else "操作未完成，可稍后重试。"
            if status == "failed"
            else spec["safe_result_summary"](result)
        )
        if detail_kind and detail_title and detail_text:
            self.emitter.step_detail_append(
                step_id=step_id,
                kind=detail_kind,  # type: ignore[arg-type]
                title=detail_title,
                text=detail_text,
            )
        self.emitter.step_upsert(
            step_id=step_id,
            kind=spec["kind"],
            title=spec["title"],
            summary=summary,
            status=status,  # type: ignore[arg-type]
        )

    def _finish_analysis(self, state: MobileAgentState, status: str) -> None:
        step_id = state.get("trace_analysis_step_id")
        if not isinstance(step_id, str):
            return
        self.emitter.step_upsert(
            step_id=step_id,
            kind="summary",
            title="分析请求",
            summary=(
                "需要你处理后再继续。"
                if status == "waiting_for_user"
                else "请求处理未完成。"
                if status == "failed"
                else "已完成请求处理。"
            ),
            status=status,  # type: ignore[arg-type]
        )

    def _mark_action_run_terminal(self, state: Mapping[str, Any]) -> None:
        run_id = state.get("trace_run_id")
        if isinstance(run_id, str) and run_id:
            phone_action_registry.mark_terminal(run_id)


def _tool_args(request: ToolCallRequest) -> dict[str, object]:
    args = request.tool_call.get("args", {})
    return args if isinstance(args, dict) else {}


def _is_waiting_command(result: ToolResult) -> bool:
    if not isinstance(result, Command) or not isinstance(result.update, Mapping):
        return False
    return result.update.get("awaiting_user_action") is True


def _tool_call_summary(tool_name: str, spec: Mapping[str, Any]) -> str:
    if tool_name == "observe":
        return "观察当前屏幕。"
    if tool_name == "launch":
        return "打开应用。"
    if tool_name in {"tap"}:
        return "点击屏幕。"
    if tool_name in {"type"}:
        return "输入文本。"
    if tool_name in {"swipe", "scroll"}:
        return "滚动页面。"
    if tool_name in {"create_event", "update_event", "update_reminders"}:
        return "日程相关操作。"
    title = spec.get("title")
    return str(title) if isinstance(title, str) and title else "执行受控操作。"


def _thread_id(runtime: object, state: Mapping[str, Any]) -> str | None:
    config = getattr(runtime, "config", {}) if runtime is not None else {}
    if not isinstance(config, Mapping):
        config = {}
    configurable = config.get("configurable", {})
    if isinstance(configurable, Mapping):
        value = configurable.get("thread_id")
        if isinstance(value, str) and value:
            return value
    metadata = config.get("metadata", {})
    if isinstance(metadata, Mapping):
        value = metadata.get("thread_id") or metadata.get("threadId")
        if isinstance(value, str) and value:
            return value
    value = state.get("thread_id") or state.get("threadId")
    return value if isinstance(value, str) and value else None


def _mobile_run_id(runtime: object, state: Mapping[str, Any]) -> str | None:
    config = getattr(runtime, "config", {}) if runtime is not None else {}
    if isinstance(config, Mapping):
        configurable = config.get("configurable", {})
        if isinstance(configurable, Mapping):
            value = configurable.get("mobile_run_id")
            if isinstance(value, str) and value:
                return value
        metadata = config.get("metadata", {})
        if isinstance(metadata, Mapping):
            value = metadata.get("mobile_run_id") or metadata.get("mobileRunId")
            if isinstance(value, str) and value:
                return value
    value = state.get("mobile_run_id")
    return value if isinstance(value, str) and value else None


__all__ = ["TraceMiddleware"]
