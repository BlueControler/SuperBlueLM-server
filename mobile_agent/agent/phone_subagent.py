from __future__ import annotations

import json
import os
import re
import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, Protocol, cast
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
)
from langchain.agents.structured_output import ToolStrategy
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.runtime import Runtime
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from ..action_control import PhoneActionCancelledError, PhoneActionControlError, PhoneActionLimitError, dispatch_phone_command
from ..gateways.phone import ConnectedDeviceSession, DeviceGateway, DeviceNotConnectedError
from ..prompt_assets import PHONE_SUBAGENT_SYSTEM_PROMPT
from ..trace import TraceDetailKind, TraceEmitter, TraceStepStatus, current_context
from ..tools.phone import create_phone_tools
from .middleware import SyncPhoneStateMiddleware

PhoneTodoStatus = Literal[
    "completed",
    "failed",
    "rejected",
    "cancelled",
    "timeout",
    "stopped",
    "needs_user_action",
    "budget_exhausted",
]

_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "rejected", "cancelled", "timeout", "stopped"}
)
_NON_RETRYABLE_STATUSES: frozenset[str] = frozenset(
    {"rejected", "cancelled", "timeout", "stopped", "completed"}
)

BASE64_IMAGE_PATTERN = re.compile(
    r"data:image/[^,\s]+;base64,[A-Za-z0-9+/=\r\n]+",
    re.IGNORECASE,
)
UI_TAG_PATTERN = re.compile(r"<[^>]+>")
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    (["']?(?:
        token|secret|key|password|passwd|authorization|credential|
        cookie|session(?:_id)?|access_token|api_key
    )["']?\s*[:=]\s*)
    (?:
        ["'][^"']*["']|
        [^,\s;]+
    )
    """
)
BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[a-z0-9._~+/=-]+")
AUTHORIZATION_HEADER_PATTERN = re.compile(
    r"(?i)(authorization\s*:\s*)(?:basic|bearer)\s+[^,\s;]+"
)
COOKIE_HEADER_PATTERN = re.compile(r"(?i)(cookie\s*:\s*)[^\r\n]+")
_FAST_LAUNCH_PACKAGES = {
    "微信": "com.tencent.mm",
    "wechat": "com.tencent.mm",
}


class PhoneStateSummary(BaseModel):
    current_package: str | None = Field(alias="currentPackage")
    activity: str | None
    has_screenshot: bool = Field(alias="hasScreenshot")
    has_ui: bool = Field(alias="hasUi")

    model_config = {"populate_by_name": True}


class PhoneSubagentDecision(BaseModel):
    status: Literal["completed", "failed", "needs_user_action"]
    summary: str
    needs_main_agent_plan: bool = Field(alias="needsMainAgentPlan")
    error: str | None = None

    model_config = {"populate_by_name": True}


class PhoneTodoExecution(BaseModel):
    status: PhoneTodoStatus
    todo: str
    summary: str
    phone_state: PhoneStateSummary = Field(alias="phoneState")
    tool_call_count: int = Field(alias="toolCallCount")
    needs_main_agent_plan: bool = Field(alias="needsMainAgentPlan")
    error: str | None = None
    terminal: bool = False
    retryable: bool = True

    model_config = {"populate_by_name": True}

    @classmethod
    def completed(cls, todo: str, summary: str, phone_state: PhoneStateSummary, *, tool_call_count: int = 0, needs_main_agent_plan: bool = False) -> "PhoneTodoExecution":
        return cls(status="completed", todo=todo, summary=summary, phoneState=phone_state, toolCallCount=tool_call_count, needsMainAgentPlan=needs_main_agent_plan, terminal=True, retryable=False)

    @classmethod
    def rejected(cls, todo: str, reason: str, phone_state: PhoneStateSummary, *, error: str | None = None) -> "PhoneTodoExecution":
        return cls(status="rejected", todo=todo, summary=reason, phoneState=phone_state, toolCallCount=0, needsMainAgentPlan=False, error=error, terminal=True, retryable=False)

    @classmethod
    def cancelled(cls, todo: str, reason: str, phone_state: PhoneStateSummary) -> "PhoneTodoExecution":
        return cls(status="cancelled", todo=todo, summary=reason, phoneState=phone_state, toolCallCount=0, needsMainAgentPlan=False, terminal=True, retryable=False)

    @classmethod
    def timeout(cls, todo: str, phone_state: PhoneStateSummary) -> "PhoneTodoExecution":
        return cls(status="timeout", todo=todo, summary="手机操作超时。", phoneState=phone_state, toolCallCount=0, needsMainAgentPlan=False, terminal=True, retryable=False)

    @classmethod
    def stopped(cls, todo: str, reason: str, phone_state: PhoneStateSummary) -> "PhoneTodoExecution":
        return cls(status="stopped", todo=todo, summary=reason, phoneState=phone_state, toolCallCount=0, needsMainAgentPlan=False, terminal=True, retryable=False)

    @classmethod
    def failed(cls, todo: str, error: str, phone_state: PhoneStateSummary, *, retryable: bool = False) -> "PhoneTodoExecution":
        return cls(status="failed", todo=todo, summary="手机操作执行失败。", phoneState=phone_state, toolCallCount=0, needsMainAgentPlan=True, error=error, terminal=not retryable, retryable=retryable)


class PhoneToolBudgetState(AgentState[PhoneSubagentDecision], total=False):
    phone_tool_call_count: NotRequired[Annotated[int, PrivateStateAttr]]
    phone_last_action_signature: NotRequired[Annotated[str, PrivateStateAttr]]
    phone_identical_action_count: NotRequired[Annotated[int, PrivateStateAttr]]


class PhoneToolBudgetExceededError(RuntimeError):
    def __init__(self, *, executed_count: int, limit: int) -> None:
        super().__init__(f"Phone tool budget exhausted after {executed_count}/{limit} calls.")
        self.executed_count = executed_count
        self.limit = limit


class PhoneToolSequenceError(RuntimeError):
    pass


class PhoneToolRepeatedActionError(RuntimeError):
    pass


class PhoneToolBudgetMiddleware(
    AgentMiddleware[PhoneToolBudgetState, None, PhoneSubagentDecision]
):
    state_schema = PhoneToolBudgetState

    def __init__(
        self,
        phone_tool_names: set[str],
        limit: int,
        identical_action_limit: int = 3,
    ) -> None:
        self.phone_tool_names = frozenset(phone_tool_names)
        self.limit = max(limit, 1)
        self.identical_action_limit = max(identical_action_limit, 1)

    def after_model(
        self,
        state: PhoneToolBudgetState,
        runtime: Runtime[None],
    ) -> dict[str, int | str] | None:
        phone_calls = self._latest_phone_calls(state)
        if not phone_calls:
            return None
        if len(phone_calls) > 1:
            raise PhoneToolSequenceError(
                "Phone tools must be called sequentially so each action uses fresh UI state."
            )

        executed_count = state.get("phone_tool_call_count", 0)
        if executed_count >= self.limit:
            raise PhoneToolBudgetExceededError(
                executed_count=executed_count,
                limit=self.limit,
            )

        signature = _phone_tool_call_signature(phone_calls[0])
        identical_count = (
            state.get("phone_identical_action_count", 0) + 1
            if state.get("phone_last_action_signature") == signature
            else 1
        )
        if identical_count > self.identical_action_limit:
            raise PhoneToolRepeatedActionError(
                f"Phone tool repeated more than {self.identical_action_limit} times."
            )
        return {
            "phone_tool_call_count": executed_count + 1,
            "phone_last_action_signature": signature,
            "phone_identical_action_count": identical_count,
        }

    async def aafter_model(
        self,
        state: PhoneToolBudgetState,
        runtime: Runtime[None],
    ) -> dict[str, int | str] | None:
        return self.after_model(state, runtime)

    def _latest_phone_calls(self, state: PhoneToolBudgetState) -> list[ToolCall]:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage):
                return [
                    call
                    for call in message.tool_calls
                    if call["name"] in self.phone_tool_names
                ]
        return []


class PhoneSubagentTrace:
    """Small no-op-safe bridge from phone subagent actions to the active trace."""

    def __init__(
        self,
        parent_id: str | None,
        *,
        emitter: TraceEmitter | None = None,
    ) -> None:
        self.parent_id = parent_id
        self.emitter = emitter or TraceEmitter()
        self._next_index = 1

    def start_step(
        self,
        *,
        kind: str,
        title: str,
        summary: str,
    ) -> str | None:
        if not self._active:
            return None
        step_id = f"phone_{uuid4().hex}_{self._next_index}"
        self._next_index += 1
        self.emitter.step_upsert(
            step_id=step_id,
            parent_id=self.parent_id,
            kind=kind,
            title=title,
            summary=summary,
            status="running",
            visible_to_user=True,
        )
        return step_id

    def complete_step(
        self,
        *,
        step_id: str | None,
        status: TraceStepStatus,
        kind: str,
        title: str,
        summary: str,
    ) -> None:
        if not self._active or step_id is None:
            return
        self.emitter.step_upsert(
            step_id=step_id,
            parent_id=self.parent_id,
            kind=kind,
            title=title,
            summary=summary,
            status=status,
            visible_to_user=True,
        )

    def detail(
        self,
        *,
        step_id: str | None,
        kind: TraceDetailKind,
        title: str,
        text: str,
    ) -> None:
        if not self._active or step_id is None:
            return
        self.emitter.step_detail_append(
            step_id=step_id,
            kind=kind,
            title=title,
            text=text,
        )

    @property
    def _active(self) -> bool:
        return bool(self.parent_id) and current_context() is not None


class PhoneSubagentTraceMiddleware(
    AgentMiddleware[PhoneToolBudgetState, None, PhoneSubagentDecision]
):
    """Emits safe child trace steps for real phone-tool calls."""

    state_schema = PhoneToolBudgetState

    def __init__(
        self,
        parent_id: str | None,
        *,
        emitter: TraceEmitter | None = None,
    ) -> None:
        self.trace = PhoneSubagentTrace(parent_id, emitter=emitter)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        spec = _phone_trace_spec(request.tool_call["name"])
        step_id = self.trace.start_step(
            kind=spec["kind"],
            title=spec["title"],
            summary=spec["running_summary"],
        )
        self.trace.detail(
            step_id=step_id,
            kind=spec["detail_kind"],
            title=spec["detail_title"],
            text=spec["detail_text"],
        )
        try:
            result = handler(request)
        except Exception:
            self._fail_step(step_id, spec)
            raise
        self._complete_for_result(step_id, spec, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        spec = _phone_trace_spec(request.tool_call["name"])
        step_id = self.trace.start_step(
            kind=spec["kind"],
            title=spec["title"],
            summary=spec["running_summary"],
        )
        self.trace.detail(
            step_id=step_id,
            kind=spec["detail_kind"],
            title=spec["detail_title"],
            text=spec["detail_text"],
        )
        try:
            result = await handler(request)
        except Exception:
            self._fail_step(step_id, spec)
            raise
        self._complete_for_result(step_id, spec, result)
        return result

    def _complete_for_result(
        self,
        step_id: str | None,
        spec: Mapping[str, Any],
        result: Any,
    ) -> None:
        if _tool_result_failed(result):
            self.trace.detail(
                step_id=step_id,
                kind="error",
                title="执行结果",
                text="手机工具返回错误，未展示原始结果。",
            )
            self.trace.complete_step(
                step_id=step_id,
                kind=spec["kind"],
                title=spec["title"],
                summary="操作未完成，可稍后重试。",
                status="failed",
            )
            return
        result_kind: TraceDetailKind = (
            "observation" if spec["title"] == "观察屏幕" else "tool_result"
        )
        result_title = "观察结果" if result_kind == "observation" else "工具结果"
        result_text = (
            "已读取当前页面的安全概要，不展示截图原图。"
            if result_kind == "observation"
            else spec["success_summary"]
        )
        self.trace.detail(
            step_id=step_id,
            kind=result_kind,
            title=result_title,
            text=result_text,
        )
        self.trace.complete_step(
            step_id=step_id,
            kind=spec["kind"],
            title=spec["title"],
            summary=spec["success_summary"],
            status="succeeded",
        )

    def _fail_step(self, step_id: str | None, spec: Mapping[str, Any]) -> None:
        self.trace.detail(
            step_id=step_id,
            kind="error",
            title="错误",
            text="手机工具执行失败，未展示内部错误。",
        )
        self.trace.complete_step(
            step_id=step_id,
            kind=spec["kind"],
            title=spec["title"],
            summary="操作未完成，可稍后重试。",
            status="failed",
        )


def _phone_trace_spec(tool_name: str) -> dict[str, Any]:
    if tool_name == "observe" or tool_name == "wait":
        return {
            "kind": "observation",
            "title": "观察屏幕",
            "running_summary": "正在获取当前屏幕概要。",
            "success_summary": "已获取当前屏幕概要。",
            "detail_kind": "observation",
            "detail_title": "观察结果",
            "detail_text": "已读取当前页面的安全概要，不展示截图原图。",
        }
    if tool_name == "launch":
        return {
            "kind": "phone_action",
            "title": "打开应用",
            "running_summary": "正在打开目标应用。",
            "success_summary": "已发起打开应用操作。",
            "detail_kind": "tool_args_summary",
            "detail_title": "安全参数",
            "detail_text": "打开目标应用，不展示内部包名。",
        }
    if tool_name in {"tap", "long_press", "double_tap"}:
        return {
            "kind": "phone_action",
            "title": "点击屏幕",
            "running_summary": "正在点击目标位置。",
            "success_summary": "点击操作已执行。",
            "detail_kind": "tool_args_summary",
            "detail_title": "安全参数",
            "detail_text": "点击目标位置，不展示精确坐标。",
        }
    if tool_name == "type":
        return {
            "kind": "phone_action",
            "title": "输入文本",
            "running_summary": "正在输入内容。",
            "success_summary": "输入操作已执行。",
            "detail_kind": "tool_args_summary",
            "detail_title": "安全参数",
            "detail_text": "在目标输入框输入内容，不展示完整敏感文本。",
        }
    if tool_name in {"swipe", "scroll"}:
        return {
            "kind": "phone_action",
            "title": "滑动屏幕",
            "running_summary": "正在移动页面。",
            "success_summary": "页面移动操作已执行。",
            "detail_kind": "tool_args_summary",
            "detail_title": "安全参数",
            "detail_text": "按目标方向移动页面，不展示精确坐标轨迹。",
        }
    if tool_name in {"back", "home", "keyevent"}:
        return {
            "kind": "phone_action",
            "title": "导航操作",
            "running_summary": "正在执行导航操作。",
            "success_summary": "导航操作已执行。",
            "detail_kind": "tool_args_summary",
            "detail_title": "安全参数",
            "detail_text": "执行系统导航操作，不展示内部按键码。",
        }
    return {
        "kind": "phone_action",
        "title": "执行手机操作",
        "running_summary": "正在执行受控手机操作。",
        "success_summary": "手机操作已执行。",
        "detail_kind": "tool_args_summary",
        "detail_title": "安全参数",
        "detail_text": "执行受控手机操作，不展示原始参数。",
    }


def _tool_result_failed(result: Any) -> bool:
    if isinstance(result, ToolMessage):
        if getattr(result, "status", None) == "error":
            return True
        content = result.content
    else:
        content = result
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except ValueError:
            return False
        return isinstance(payload, Mapping) and "error" in payload
    return False


def _decision_trace_status(status: str) -> TraceStepStatus:
    if status == "completed":
        return "succeeded"
    if status == "needs_user_action":
        return "waiting_for_user"
    return "failed"


class InvokableAgent(Protocol):
    async def ainvoke(self, state: dict[str, Any]) -> Mapping[str, Any]: ...


AgentFactory = Callable[..., InvokableAgent]


class PhoneSubagentRunner:
    def __init__(
        self,
        phone_gateway: DeviceGateway,
        model: BaseChatModel | str,
        *,
        agent_factory: AgentFactory | None = None,
        trace_emitter: TraceEmitter | None = None,
    ) -> None:
        self.phone_gateway = phone_gateway
        self.model = model
        self.agent_factory = agent_factory or cast(AgentFactory, create_agent)
        self.trace_emitter = trace_emitter

    async def execute(
        self,
        todo: str,
        *,
        allow_short_chain: bool,
        device_id: str | None = None,
        trace_parent_id: str | None = None,
    ) -> PhoneTodoExecution:
        trace = PhoneSubagentTrace(trace_parent_id, emitter=self.trace_emitter)
        if phone_text_contains_sensitive_data(todo):
            logger.warning("Phone TODO contains sensitive data and requires user takeover.")
            return self._needs_user_action(
                todo,
                "phone_todo_contains_sensitive_data",
                device_id,
            )

        # 快速路径：已知应用的启动任务不需要子模型推理。即使主 agent 声明了
        # allow_short_chain，直接启动仍是更小、更可靠的一步。
        launch_result = await self._try_fast_launch(todo, device_id, trace=trace)
        if launch_result is not None:
            return launch_result

        session: ConnectedDeviceSession
        try:
            session = (
                self.phone_gateway.get_session(device_id)
                if device_id is not None
                else self.phone_gateway.get_session()
            )
        except DeviceNotConnectedError:
            logger.warning("Phone subagent cannot start: device not connected.")
            return PhoneTodoExecution.failed(
                todo=redact_phone_text(todo),
                error="device_not_connected",
                phone_state=self._phone_state_summary(device_id),
                retryable=False,
            )
        budget = _tool_budget(allow_short_chain)
        tools = create_phone_tools(
            self.phone_gateway,
            default_device_id=device_id,
            expected_session=session,
        )
        phone_tool_names = {tool.name for tool in tools}
        agent = self.agent_factory(
            model=self.model,
            tools=tools,
            system_prompt=PHONE_SUBAGENT_SYSTEM_PROMPT,
            middleware=[
                SyncPhoneStateMiddleware(self.phone_gateway, device_id=device_id),
                PhoneSubagentTraceMiddleware(
                    trace_parent_id,
                    emitter=self.trace_emitter,
                ),
                PhoneToolBudgetMiddleware(
                    phone_tool_names,
                    budget,
                    identical_action_limit=_identical_action_limit(),
                ),
            ],
            response_format=ToolStrategy(PhoneSubagentDecision),
        )
        try:
            async with asyncio.timeout(_execution_timeout_seconds()):
                state = await agent.ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=(
                                    f"当前 TODO: {todo}\n"
                                    f"allow_short_chain={str(allow_short_chain).lower()}\n"
                                    f"最多调用 {budget} 次手机工具。"
                                )
                            )
                        ]
                    }
                )
        except TimeoutError:
            logger.warning("Phone subagent execution timed out before completion.")
            self._trace_stop(
                trace,
                kind="error",
                title="停止自动操作",
                text="手机操作超时，已停止自动执行。",
            )
            return PhoneTodoExecution.timeout(todo, self._phone_state_summary(device_id))
        except PhoneToolBudgetExceededError as exc:
            logger.warning("Phone subagent stopped because its tool budget was exhausted.")
            self._trace_stop(
                trace,
                kind="warning",
                title="停止自动操作",
                text="自动操作已停止，避免继续重复执行。",
            )
            return self._budget_exhausted(
                todo,
                exc.executed_count,
                exc.limit,
                device_id,
            )
        except PhoneToolSequenceError:
            logger.warning("Phone subagent attempted parallel UI tool calls.")
            self._trace_stop(
                trace,
                kind="warning",
                title="停止自动操作",
                text="并行手机操作已被拒绝，已停止自动执行。",
            )
            return PhoneTodoExecution.stopped(todo, "并行调用手机工具已被拒绝。", self._phone_state_summary(device_id))
        except PhoneToolRepeatedActionError:
            logger.warning("Phone subagent stopped after repeating the same phone action.")
            self._trace_stop(
                trace,
                kind="retry",
                title="重试操作",
                text="上一步未达到目标且重复执行，已停止自动操作。",
            )
            return PhoneTodoExecution.stopped(todo, "重复执行同一手机操作，已停止。", self._phone_state_summary(device_id))
        except Exception as exc:
            logger.bind(
                error_type=type(exc).__name__,
                model=_model_label(self.model),
                request_id=getattr(exc, "request_id", None),
                status_code=getattr(exc, "status_code", None),
            ).exception("Phone subagent execution failed.")
            self._trace_stop(
                trace,
                kind="error",
                title="错误",
                text="手机子任务执行失败，未展示内部异常。",
            )
            return self._failed(todo, _execution_error_code(exc), device_id)

        structured_response = state.get("structured_response")
        if not structured_response:
            messages = state.get("messages", ())
            successful_tool_calls = _successful_phone_tool_call_count(
                messages,
                phone_tool_names,
            )
            if successful_tool_calls:
                logger.warning(
                    "Phone subagent response missing structured_response after {} successful phone tool call(s). "
                    "Returning a recoverable completion for main-agent verification.",
                    successful_tool_calls,
                )
                self._trace_terminal_decision(
                    trace,
                    status="succeeded",
                    detail_kind="tool_result",
                    text="已执行手机操作，正在确认结果。",
                )
                return PhoneTodoExecution.completed(
                    todo=redact_phone_text(todo),
                    summary="已执行手机操作，正在确认结果。",
                    phone_state=self._phone_state_summary(device_id),
                    tool_call_count=successful_tool_calls,
                    needs_main_agent_plan=True,
                )
            logger.warning(
                "Phone subagent response missing structured_response. "
                "state_keys={} last_message_type={} tool_message_count={} model={}",
                sorted(str(key) for key in state.keys()),
                _last_message_type(messages),
                _count_phone_tool_calls(messages, phone_tool_names),
                _model_label(self.model),
            )
            self._trace_terminal_decision(
                trace,
                status="failed",
                detail_kind="error",
                text="手机子任务未返回明确结果，已停止自动操作。",
            )
            return PhoneTodoExecution.failed(
                redact_phone_text(todo),
                "missing_structured_response",
                self._phone_state_summary(device_id),
            )
        try:
            decision = PhoneSubagentDecision.model_validate(structured_response)
        except Exception:
            logger.bind(
                state_keys=sorted(str(key) for key in state.keys()),
                model=_model_label(self.model),
            ).exception("Phone subagent returned an invalid structured response.")
            self._trace_terminal_decision(
                trace,
                status="failed",
                detail_kind="error",
                text="手机子任务返回了无效结果，未展示原始内容。",
            )
            return PhoneTodoExecution.failed(
                redact_phone_text(todo),
                "invalid_structured_response",
                self._phone_state_summary(device_id),
            )
        execution = PhoneTodoExecution(
            status=decision.status,
            todo=redact_phone_text(todo),
            summary=redact_phone_text(decision.summary),
            phoneState=self._phone_state_summary(device_id),
            toolCallCount=_count_phone_tool_calls(
                state.get("messages", ()),
                phone_tool_names,
            ),
            needsMainAgentPlan=decision.needs_main_agent_plan,
            error=redact_phone_text(decision.error) if decision.error else None,
            terminal=decision.status in _TERMINAL_STATUSES,
            retryable=decision.status not in _NON_RETRYABLE_STATUSES,
        )
        self._trace_terminal_decision(
            trace,
            status=_decision_trace_status(decision.status),
            detail_kind="decision" if decision.status == "completed" else "warning",
            text=execution.summary,
        )
        return execution

    def _trace_stop(
        self,
        trace: PhoneSubagentTrace,
        *,
        kind: TraceDetailKind,
        title: str,
        text: str,
    ) -> None:
        step_id = trace.start_step(
            kind="phone_action",
            title=title,
            summary="自动操作已停止。",
        )
        trace.detail(
            step_id=step_id,
            kind=kind,
            title=title,
            text=text,
        )
        trace.complete_step(
            step_id=step_id,
            kind="phone_action",
            title=title,
            summary="自动操作已停止。",
            status="failed",
        )

    def _trace_terminal_decision(
        self,
        trace: PhoneSubagentTrace,
        *,
        status: TraceStepStatus,
        detail_kind: TraceDetailKind,
        text: str,
    ) -> None:
        step_id = trace.start_step(
            kind="phone_action",
            title="判断结果",
            summary="正在判断手机操作结果。",
        )
        trace.detail(
            step_id=step_id,
            kind=detail_kind,
            title="判断结果",
            text=text,
        )
        trace.complete_step(
            step_id=step_id,
            kind="phone_action",
            title="判断结果",
            summary=text,
            status=status,
        )

    def _phone_state_summary(self, device_id: str | None = None) -> PhoneStateSummary:
        try:
            session = (
                self.phone_gateway.get_session(device_id)
                if device_id is not None
                else self.phone_gateway.get_session()
            )
        except Exception:
            return PhoneStateSummary(
                currentPackage=None,
                activity=None,
                hasScreenshot=False,
                hasUi=False,
            )

        info = session.device_info
        return PhoneStateSummary(
            currentPackage=info.current_package if info else None,
            activity=info.activity if info else None,
            hasScreenshot=bool(info and info.screenshot),
            hasUi=bool(info and info.ui),
        )

    def _budget_exhausted(
        self,
        todo: str,
        executed_count: int,
        limit: int,
        device_id: str | None = None,
    ) -> PhoneTodoExecution:
        return PhoneTodoExecution(
            status="budget_exhausted",
            todo=redact_phone_text(todo),
            summary=f"Phone TODO stopped after reaching the {limit}-tool budget.",
            phoneState=self._phone_state_summary(device_id),
            toolCallCount=executed_count,
            needsMainAgentPlan=True,
            error="phone_tool_budget_exhausted",
            terminal=True,
            retryable=False,
        )

    def _failed(
        self,
        todo: str,
        error: str,
        device_id: str | None = None,
    ) -> PhoneTodoExecution:
        return PhoneTodoExecution(
            status="failed",
            todo=redact_phone_text(todo),
            summary="手机操作执行失败。",
            phoneState=self._phone_state_summary(device_id),
            toolCallCount=0,
            needsMainAgentPlan=True,
            error=error,
            terminal=False,
            retryable=False,
        )


    def _needs_user_action(
        self,
        todo: str,
        error: str,
        device_id: str | None = None,
    ) -> PhoneTodoExecution:
        return PhoneTodoExecution(
            status="needs_user_action",
            todo=redact_phone_text(todo),
            summary="Phone TODO requires user takeover because it contains sensitive data.",
            phoneState=self._phone_state_summary(device_id),
            toolCallCount=0,
            needsMainAgentPlan=True,
            error=error,
            terminal=True,
            retryable=False,
        )
    async def _try_fast_launch(
        self,
        todo: str,
        device_id: str | None = None,
        *,
        trace: PhoneSubagentTrace,
    ) -> PhoneTodoExecution | None:
        """纯启动应用任务跳过 LLM 推理，直接通过 monkey 启动。"""
        import re
        # 主 agent 有时会生成“找到并打开微信 App”的导航说明，而不是简洁的
        # “打开微信”。只要目标是已知应用/明确包名，仍可直接启动。
        match = re.search(
            r"(?i)(?:打开|启动|launch|open)",
            todo,
        )
        if not match:
            return None
        # 排除已经要求应用内观察或操作的多步任务；仅“找到”或“左右滑动”
        # 是模型对启动方式的冗余描述，不应阻止已知包名的直接启动。
        if _contains_any(todo, ("截图", "查看", "检查", "确认", "搜索")):
            return None
        target = todo[match.end() :].strip()
        package = _FAST_LAUNCH_PACKAGES.get(target.lower())
        if package is None:
            # 主 agent 常把应用名和包名一起写入 TODO，例如
            # “打开微信应用（包名 com.tencent.mm）”。这仍是单步启动，
            # 不能因为附注不匹配而退回不稳定的子模型推理。
            package_match = re.search(
                r"(?<![\w.])([a-zA-Z][\w]*(?:\.[\w]+)+)(?![\w.])",
                target,
            )
            if package_match:
                package = package_match.group(1)
        if package is None:
            for app_name, known_package in _FAST_LAUNCH_PACKAGES.items():
                if re.search(
                    rf"{re.escape(app_name)}(?:应用|app)?(?=$|[\s，。,.、:：;；!！?？）)】'\"”’])",
                    todo,
                    re.IGNORECASE,
                ):
                    package = known_package
                    break
        if package is None and re.fullmatch(r"[a-zA-Z][\w.]*", target):
            package = target
        if package is None:
            return None
        step_id = trace.start_step(
            kind="phone_action",
            title="打开应用",
            summary="正在打开目标应用。",
        )
        trace.detail(
            step_id=step_id,
            kind="tool_args_summary",
            title="安全参数",
            text="打开目标应用，不展示内部包名。",
        )
        try:
            session = (
                self.phone_gateway.get_session(device_id)
                if device_id is not None
                else self.phone_gateway.get_session()
            )
            await dispatch_phone_command(
                session,
                "launch",
                {"package": package},
                device_id=device_id,
            )
            logger.info(f"Fast launch succeeded: package={package}")
            trace.detail(
                step_id=step_id,
                kind="tool_result",
                title="工具结果",
                text="已发起打开应用操作。",
            )
            trace.complete_step(
                step_id=step_id,
                kind="phone_action",
                title="打开应用",
                summary="已发起打开应用操作。",
                status="succeeded",
            )
            return PhoneTodoExecution.completed(
                todo=redact_phone_text(todo),
                summary="已发起打开应用操作。",
                phone_state=self._phone_state_summary(device_id),
                tool_call_count=1,
            )
        except PhoneActionCancelledError as exc:
            logger.warning(f"Fast launch cancelled for {package}: {exc}")
            trace.complete_step(
                step_id=step_id,
                kind="phone_action",
                title="打开应用",
                summary="任务已被取消。",
                status="cancelled",
            )
            return PhoneTodoExecution.cancelled(todo, "任务已被取消。", self._phone_state_summary(device_id))
        except PhoneActionLimitError as exc:
            # 预算耗尽或重复操作——表示前一次启动已成功，通知主 Agent 任务完成
            logger.info(f"Fast launch already done for {package}: {exc}")
            trace.detail(
                step_id=step_id,
                kind="tool_result",
                title="工具结果",
                text="系统已确认打开应用操作。",
            )
            trace.complete_step(
                step_id=step_id,
                kind="phone_action",
                title="打开应用",
                summary="系统已确认打开应用操作。",
                status="succeeded",
            )
            return PhoneTodoExecution.completed(
                todo=redact_phone_text(todo),
                summary="系统已确认打开应用操作。",
                phone_state=self._phone_state_summary(device_id),
            )
        except PhoneActionControlError as exc:
            logger.warning(f"Fast launch rejected for {package}: {exc}")
            trace.detail(
                step_id=step_id,
                kind="error",
                title="执行结果",
                text="手机操作已被系统安全策略拒绝。",
            )
            trace.complete_step(
                step_id=step_id,
                kind="phone_action",
                title="打开应用",
                summary="手机操作已被系统安全策略拒绝。",
                status="failed",
            )
            return PhoneTodoExecution.rejected(todo, "手机操作已被系统安全策略拒绝。", self._phone_state_summary(device_id))
        except DeviceNotConnectedError:
            logger.warning(f"Fast launch skipped for {package}: device not connected")
            trace.detail(
                step_id=step_id,
                kind="error",
                title="设备断开",
                text="手机连接已断开，无法执行操作。",
            )
            trace.complete_step(
                step_id=step_id,
                kind="phone_action",
                title="打开应用",
                summary="手机连接已断开，操作未执行。",
                status="failed",
            )
            return PhoneTodoExecution.failed(
                todo=redact_phone_text(todo),
                error="device_not_connected",
                phone_state=self._phone_state_summary(device_id),
                retryable=False,
            )
        except Exception as exc:
            logger.warning(f"Fast launch failed for {package}: {exc}")
            trace.detail(
                step_id=step_id,
                kind="error",
                title="错误",
                text="打开应用操作失败，未展示内部异常。",
            )
            trace.complete_step(
                step_id=step_id,
                kind="phone_action",
                title="打开应用",
                summary="打开应用操作未完成。",
                status="failed",
            )
            return None  # fallback 到 LLM 正常流程


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text for kw in keywords)


def _execution_error_code(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    if status_code == 500:
        return "llm_http_500"
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return "llm_http_5xx"
    return "phone_subagent_execution_failed"


def _model_label(model: BaseChatModel | str) -> str:
    value = getattr(model, "model_name", None) or getattr(model, "model", None) or model
    return value if isinstance(value, str) else type(model).__name__


def _last_message_type(messages: object) -> str | None:
    if not isinstance(messages, Sequence) or not messages:
        return None
    return type(messages[-1]).__name__

def _tool_budget(allow_short_chain: bool) -> int:
    env_name = (
        "PHONE_SUBAGENT_SHORT_CHAIN_MAX_TOOL_CALLS"
        if allow_short_chain
        else "PHONE_SUBAGENT_MAX_TOOL_CALLS"
    )
    default = "12"
    try:
        return max(int(os.getenv(env_name, default)), 1)
    except ValueError:
        return int(default)


def _execution_timeout_seconds() -> int:
    try:
        return max(int(os.getenv("PHONE_SUBAGENT_MAX_EXECUTION_SECONDS", "90")), 1)
    except ValueError:
        return 90


def _identical_action_limit() -> int:
    try:
        return max(int(os.getenv("PHONE_SUBAGENT_MAX_IDENTICAL_ACTIONS", "3")), 1)
    except ValueError:
        return 3


def _phone_tool_call_signature(call: ToolCall) -> str:
    return json.dumps(
        {"name": call["name"], "args": call["args"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _count_phone_tool_calls(
    messages: Sequence[Any],
    phone_tool_names: set[str],
) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, ToolMessage) and message.name in phone_tool_names
    )


def _successful_phone_tool_call_count(
    messages: object,
    phone_tool_names: set[str],
) -> int:
    if not isinstance(messages, Sequence):
        return 0
    phone_results = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.name in phone_tool_names
    ]
    if not phone_results or any(message.status == "error" for message in phone_results):
        return 0
    return len(phone_results)


def redact_phone_text(value: str) -> str:
    without_images = BASE64_IMAGE_PATTERN.sub("data:image/***;base64,***", value)
    without_ui_tags = UI_TAG_PATTERN.sub("<ui-redacted>", without_images)
    without_authorization = AUTHORIZATION_HEADER_PATTERN.sub(
        r"\1***",
        without_ui_tags,
    )
    without_cookies = COOKIE_HEADER_PATTERN.sub(r"\1***", without_authorization)
    without_bearer = BEARER_PATTERN.sub(r"\1***", without_cookies)
    return SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1***", without_bearer)


def phone_text_contains_sensitive_data(value: str) -> bool:
    return redact_phone_text(value) != value


__all__ = [
    "PhoneStateSummary",
    "PhoneSubagentRunner",
    "PhoneSubagentTrace",
    "PhoneSubagentTraceMiddleware",
    "PhoneTodoExecution",
    "PhoneTodoStatus",
    "phone_text_contains_sensitive_data",
    "redact_phone_text",
]
