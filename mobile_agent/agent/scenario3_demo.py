from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..confirmations import (
    ConfirmationResolveResult,
    ConfirmationTransaction,
    confirm_confirmation,
    create_confirmation,
    reject_confirmation,
    take_over_confirmation,
)
from ..json_types import JsonObject
from ..progress import emit_needs_confirmation, emit_task_complexity, emit_task_progress
from ..trace import current_context
from .middleware import _message_content_to_text
from .state import MobileAgentState

SCENARIO3_TASK_TITLE = "为会议通知创建提醒"
SCENARIO3_PROGRESS_KEY = "scenario3-demo"
SCENARIO3_TOTAL_STEPS = 3
SCENARIO3_DRY_RUN = True


@dataclass(frozen=True)
class Scenario3DemoTask:
    confirmation_id: str
    run_id: str | None
    thread_id: str | None


_TASKS: dict[str, Scenario3DemoTask] = {}


def is_scenario3_demo_request(text: str) -> bool:
    normalized = _normalize_utterance(text)
    return (
        "会议通知" in normalized
        and ("模拟" in normalized or "收到" in normalized or "检测" in normalized)
    )


def start_scenario3_demo(
    *,
    thread_id: str | None = None,
    run_id: str | None = None,
    emit: bool = False,
) -> JsonObject:
    confirmation_id = f"scenario3-{uuid4().hex[:16]}"
    _TASKS[confirmation_id] = Scenario3DemoTask(
        confirmation_id=confirmation_id,
        run_id=run_id,
        thread_id=thread_id,
    )
    confirmation = create_confirmation(
        confirmation_id=confirmation_id,
        run_id=run_id,
        thread_id=thread_id,
        task_title=SCENARIO3_TASK_TITLE,
        operation="创建会议提醒",
        target_app="系统日历",
        tool_name="create_event",
        risk_level="medium",
        payload_preview="检测到会议通知，是否创建会议提醒？",
        confirm_text="确认创建",
        cancel_text="取消",
        dry_run=SCENARIO3_DRY_RUN,
        confirm_handler=_confirm_scenario3_transaction,
        reject_handler=_reject_scenario3_transaction,
        take_over_handler=_take_over_scenario3_transaction,
    )
    events = [
        _progress_event(
            run_id=run_id,
            thread_id=thread_id,
            status="running",
            current_step=1,
            step_title="检测到会议通知",
            phase="phone_tool",
            tool_name="list_notifications",
            message="检测到一条会议通知",
            requires_confirmation=False,
            confirmation_id=None,
            can_cancel=True,
            can_take_over=True,
        ),
        _progress_event(
            run_id=run_id,
            thread_id=thread_id,
            status="waiting_confirmation",
            current_step=2,
            step_title="等待确认是否创建会议提醒",
            phase="confirmation",
            tool_name="needs_confirmation",
            message="检测到会议通知，是否创建提醒？",
            requires_confirmation=True,
            confirmation_id=confirmation_id,
            can_cancel=True,
            can_take_over=True,
        ),
    ]
    if emit:
        emit_task_complexity(
            complexity="complex",
            track_steps=True,
            reason="scenario3_notification_demo",
            message="识别任务类型：通知感知与主动介入",
        )
        _emit_needs_confirmation_event(confirmation)
        for event in events:
            _emit_progress_event(event)
    return {
        "dryRun": SCENARIO3_DRY_RUN,
        "confirmationId": confirmation_id,
        "needsConfirmation": confirmation.needs_confirmation_event(),
        "events": events,
    }


def confirm_scenario3_demo(confirmation_id: str) -> JsonObject | None:
    result = confirm_confirmation(confirmation_id)
    return _scenario3_resolution_payload(result)


def _confirm_scenario3_transaction(
    transaction: ConfirmationTransaction,
) -> list[JsonObject]:
    task = _TASKS.pop(transaction.confirmation_id, None)
    if task is None:
        return []
    return [
        _progress_event(
            run_id=task.run_id,
            thread_id=task.thread_id,
            status="running",
            current_step=3,
            step_title="正在创建会议提醒",
            phase="system_tool",
            tool_name="create_event",
            message="正在创建会议提醒",
            requires_confirmation=False,
            confirmation_id=transaction.confirmation_id,
            can_cancel=True,
            can_take_over=True,
        ),
        _progress_event(
            run_id=task.run_id,
            thread_id=task.thread_id,
            status="completed",
            current_step=3,
            step_title="会议提醒已创建",
            phase="finalizing",
            tool_name="create_event",
            message="会议提醒已创建",
            requires_confirmation=False,
            confirmation_id=None,
            can_cancel=False,
            can_take_over=False,
        ),
    ]


def reject_scenario3_demo(confirmation_id: str) -> JsonObject | None:
    result = reject_confirmation(confirmation_id)
    return _scenario3_resolution_payload(result)


def _reject_scenario3_transaction(
    transaction: ConfirmationTransaction,
) -> list[JsonObject]:
    task = _TASKS.pop(transaction.confirmation_id, None)
    if task is None:
        return []
    return [
        _progress_event(
            run_id=task.run_id,
            thread_id=task.thread_id,
            status="cancelled",
            current_step=2,
            step_title="已取消创建会议提醒",
            phase="finalizing",
            tool_name="needs_confirmation",
            message="已取消创建会议提醒",
            requires_confirmation=False,
            confirmation_id=None,
            can_cancel=False,
            can_take_over=False,
        )
    ]


def take_over_scenario3_demo(confirmation_id: str) -> JsonObject | None:
    result = take_over_confirmation(confirmation_id)
    return _scenario3_resolution_payload(result)


def _take_over_scenario3_transaction(
    transaction: ConfirmationTransaction,
) -> list[JsonObject]:
    task = _TASKS.pop(transaction.confirmation_id, None)
    if task is None:
        return []
    return [
        _progress_event(
            run_id=task.run_id,
            thread_id=task.thread_id,
            status="taken_over",
            current_step=2,
            step_title="已接管会议提醒处理",
            phase="finalizing",
            tool_name="take_over",
            message="已接管，自动执行已停止",
            requires_confirmation=False,
            confirmation_id=None,
            can_cancel=False,
            can_take_over=False,
        )
    ]


def _scenario3_resolution_payload(
    result: ConfirmationResolveResult,
) -> JsonObject | None:
    if result.status_code == 404:
        return None
    return result.payload


class Scenario3DemoMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    state_schema = MobileAgentState

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        if not _request_matches(request.messages):
            return handler(request)
        result = self._start_from_context()
        return _model_response(result)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if not _request_matches(request.messages):
            return await handler(request)
        result = self._start_from_context()
        return _model_response(result)

    @staticmethod
    def _start_from_context() -> JsonObject:
        context = current_context()
        return start_scenario3_demo(
            thread_id=context.thread_id if context is not None else None,
            run_id=context.run_id if context is not None else None,
            emit=True,
        )


def _request_matches(messages: Sequence[BaseMessage]) -> bool:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return is_scenario3_demo_request(_message_content_to_text(message.content))
    return False


def _model_response(result: Mapping[str, object]) -> ModelResponse[Any]:
    return ModelResponse(
        result=[
            AIMessage(
                content="检测到会议通知，是否创建提醒？",
                additional_kwargs={"scenario3_demo": result},
            )
        ]
    )


def _progress_event(
    *,
    run_id: str | None,
    thread_id: str | None,
    status: str,
    current_step: int,
    step_title: str,
    phase: str,
    tool_name: str,
    message: str,
    requires_confirmation: bool,
    confirmation_id: str | None,
    can_cancel: bool,
    can_take_over: bool,
) -> JsonObject:
    event: JsonObject = {
        "type": "task_progress",
        "label": "会议通知",
        "taskTitle": SCENARIO3_TASK_TITLE,
        "status": status,
        "currentStep": current_step,
        "totalSteps": SCENARIO3_TOTAL_STEPS,
        "stepTitle": step_title,
        "phase": phase,
        "toolName": tool_name,
        "requiresConfirmation": requires_confirmation,
        "canCancel": can_cancel,
        "canTakeOver": can_take_over,
        "message": message,
        "progressKey": SCENARIO3_PROGRESS_KEY,
        "dryRun": SCENARIO3_DRY_RUN,
    }
    if run_id:
        event["runId"] = run_id
    if thread_id:
        event["threadId"] = thread_id
    if confirmation_id is not None:
        event["confirmationId"] = confirmation_id
    return event


def _emit_progress_event(event: Mapping[str, object]) -> None:
    emit_task_progress(
        label=str(event["label"]),
        run_id=_optional_str(event, "runId"),
        thread_id=_optional_str(event, "threadId"),
        task_title=str(event["taskTitle"]),
        status=str(event["status"]),  # type: ignore[arg-type]
        current_step=int(event["currentStep"]),
        total_steps=int(event["totalSteps"]),
        step_title=str(event["stepTitle"]),
        phase=str(event["phase"]),
        tool_name=str(event["toolName"]),
        requires_confirmation=bool(event["requiresConfirmation"]),
        confirmation_id=_optional_str(event, "confirmationId"),
        can_cancel=bool(event["canCancel"]),
        can_take_over=bool(event["canTakeOver"]),
        message=str(event["message"]),
        progress_key=str(event["progressKey"]),
        dry_run=bool(event["dryRun"]),
    )


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


def _optional_str(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) and value else None


def _normalize_utterance(text: str) -> str:
    return re.sub(r"[\s，。,.!！?？]+", "", text.strip().lower())


__all__ = [
    "Scenario3DemoMiddleware",
    "confirm_scenario3_demo",
    "is_scenario3_demo_request",
    "reject_scenario3_demo",
    "start_scenario3_demo",
    "take_over_scenario3_demo",
]
