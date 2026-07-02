from __future__ import annotations

import inspect
from collections.abc import Awaitable
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from time import time
from typing import Literal
from uuid import uuid4

from .json_types import JsonObject, JsonValue, to_json_value

ConfirmationDecision = Literal["confirm", "reject", "take_over"]
ConfirmationStatus = Literal["pending", "confirmed", "rejected", "taken_over"]
ConfirmationEvents = list[JsonObject] | Awaitable[list[JsonObject]]
ConfirmationHandler = Callable[["ConfirmationTransaction"], ConfirmationEvents]


@dataclass
class ConfirmationTransaction:
    confirmation_id: str
    task_title: str
    operation: str
    target_app: str
    tool_name: str
    risk_level: str
    payload_preview: str
    confirm_text: str
    cancel_text: str
    dry_run: bool
    run_id: str | None = None
    thread_id: str | None = None
    status: ConfirmationStatus = "pending"
    created_at_ms: int = 0
    confirm_handler: ConfirmationHandler | None = None
    reject_handler: ConfirmationHandler | None = None
    take_over_handler: ConfirmationHandler | None = None

    def needs_confirmation_event(self) -> JsonObject:
        event: JsonObject = {
            "type": "needs_confirmation",
            "confirmationId": self.confirmation_id,
            "taskTitle": self.task_title,
            "operation": self.operation,
            "targetApp": self.target_app,
            "toolName": self.tool_name,
            "riskLevel": self.risk_level,
            "payloadPreview": self.payload_preview,
            "confirmText": self.confirm_text,
            "cancelText": self.cancel_text,
            "dryRun": self.dry_run,
        }
        if self.run_id:
            event["runId"] = self.run_id
        if self.thread_id:
            event["threadId"] = self.thread_id
        return event


@dataclass(frozen=True)
class ConfirmationResolveResult:
    status_code: int
    payload: JsonObject


class ConfirmationStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._transactions: dict[str, ConfirmationTransaction] = {}

    def register(
        self,
        *,
        task_title: str,
        operation: str,
        target_app: str,
        tool_name: str,
        risk_level: str,
        payload_preview: str,
        confirm_text: str,
        cancel_text: str,
        dry_run: bool,
        run_id: str | None = None,
        thread_id: str | None = None,
        confirmation_id: str | None = None,
        confirm_handler: ConfirmationHandler | None = None,
        reject_handler: ConfirmationHandler | None = None,
        take_over_handler: ConfirmationHandler | None = None,
    ) -> ConfirmationTransaction:
        normalized_id = confirmation_id or f"confirm-{uuid4().hex[:16]}"
        transaction = ConfirmationTransaction(
            confirmation_id=normalized_id,
            task_title=task_title,
            operation=operation,
            target_app=target_app,
            tool_name=tool_name,
            risk_level=risk_level,
            payload_preview=payload_preview,
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            dry_run=dry_run,
            run_id=run_id,
            thread_id=thread_id,
            status="pending",
            created_at_ms=int(time() * 1000),
            confirm_handler=confirm_handler,
            reject_handler=reject_handler,
            take_over_handler=take_over_handler,
        )
        with self._lock:
            self._transactions[normalized_id] = transaction
        return transaction

    def get(self, confirmation_id: str) -> ConfirmationTransaction | None:
        with self._lock:
            return self._transactions.get(confirmation_id)

    def resolve(
        self,
        confirmation_id: str,
        decision: ConfirmationDecision,
    ) -> ConfirmationResolveResult:
        with self._lock:
            transaction = self._transactions.get(confirmation_id)
            if transaction is None:
                return ConfirmationResolveResult(
                    404,
                    {"error": "confirmation_not_found", "confirmationId": confirmation_id},
                )
            if transaction.status != "pending":
                return ConfirmationResolveResult(
                    409,
                    {
                        "error": "confirmation_already_resolved",
                        "confirmationId": confirmation_id,
                        "status": transaction.status,
                    },
                )
            transaction.status = _status_for_decision(decision)
            handler = _handler_for_decision(transaction, decision)

        events = handler(transaction) if handler is not None else _default_events(transaction, decision)
        return ConfirmationResolveResult(
            200,
            {
                "confirmationId": confirmation_id,
                "status": transaction.status,
                "dryRun": transaction.dry_run,
                "events": events,
            },
        )

    async def resolve_async(
        self,
        confirmation_id: str,
        decision: ConfirmationDecision,
    ) -> ConfirmationResolveResult:
        with self._lock:
            transaction = self._transactions.get(confirmation_id)
            if transaction is None:
                return ConfirmationResolveResult(
                    404,
                    {"error": "confirmation_not_found", "confirmationId": confirmation_id},
                )
            if transaction.status != "pending":
                return ConfirmationResolveResult(
                    409,
                    {
                        "error": "confirmation_already_resolved",
                        "confirmationId": confirmation_id,
                        "status": transaction.status,
                    },
                )
            transaction.status = _status_for_decision(decision)
            handler = _handler_for_decision(transaction, decision)

        events_result = handler(transaction) if handler is not None else _default_events(transaction, decision)
        events = await events_result if inspect.isawaitable(events_result) else events_result
        return ConfirmationResolveResult(
            200,
            {
                "confirmationId": confirmation_id,
                "status": transaction.status,
                "dryRun": transaction.dry_run,
                "events": events,
            },
        )

    def clear(self) -> None:
        with self._lock:
            self._transactions.clear()


confirmation_store = ConfirmationStore()


def create_confirmation(
    **kwargs: object,
) -> ConfirmationTransaction:
    return confirmation_store.register(**kwargs)  # type: ignore[arg-type]


def get_confirmation(confirmation_id: str) -> ConfirmationTransaction | None:
    return confirmation_store.get(confirmation_id)


def confirm_confirmation(confirmation_id: str) -> ConfirmationResolveResult:
    return confirmation_store.resolve(confirmation_id, "confirm")


async def confirm_confirmation_async(confirmation_id: str) -> ConfirmationResolveResult:
    return await confirmation_store.resolve_async(confirmation_id, "confirm")


def reject_confirmation(confirmation_id: str) -> ConfirmationResolveResult:
    return confirmation_store.resolve(confirmation_id, "reject")


async def reject_confirmation_async(confirmation_id: str) -> ConfirmationResolveResult:
    return await confirmation_store.resolve_async(confirmation_id, "reject")


def take_over_confirmation(confirmation_id: str) -> ConfirmationResolveResult:
    return confirmation_store.resolve(confirmation_id, "take_over")


async def take_over_confirmation_async(confirmation_id: str) -> ConfirmationResolveResult:
    return await confirmation_store.resolve_async(confirmation_id, "take_over")


def create_high_risk_confirmation(
    *,
    tool_name: str,
    args: Mapping[str, object],
    run_id: str | None,
    thread_id: str | None,
) -> ConfirmationTransaction:
    spec = _high_risk_spec(tool_name, args)
    return create_confirmation(
        run_id=run_id,
        thread_id=thread_id,
        task_title=spec["taskTitle"],
        operation=spec["operation"],
        target_app=spec["targetApp"],
        tool_name=tool_name,
        risk_level=spec["riskLevel"],
        payload_preview=spec["payloadPreview"],
        confirm_text=spec["confirmText"],
        cancel_text=spec["cancelText"],
        dry_run=True,
    )


def _handler_for_decision(
    transaction: ConfirmationTransaction,
    decision: ConfirmationDecision,
) -> ConfirmationHandler | None:
    if decision == "confirm":
        return transaction.confirm_handler
    if decision == "reject":
        return transaction.reject_handler
    return transaction.take_over_handler


def _status_for_decision(decision: ConfirmationDecision) -> ConfirmationStatus:
    if decision == "confirm":
        return "confirmed"
    if decision == "reject":
        return "rejected"
    return "taken_over"


def _default_events(
    transaction: ConfirmationTransaction,
    decision: ConfirmationDecision,
) -> list[JsonObject]:
    if decision == "confirm":
        return [
            _task_progress_event(
                transaction,
                status="running",
                phase="confirmation",
                tool_name=transaction.tool_name,
                step_title=f"已确认：{transaction.operation}",
                message=f"已确认，当前为 dry-run，未执行真实{transaction.operation}。",
                can_cancel=False,
                can_take_over=False,
            ),
            _task_progress_event(
                transaction,
                status="completed",
                phase="finalizing",
                tool_name=transaction.tool_name,
                step_title=f"{transaction.operation}已完成 dry-run",
                message=f"dry-run 已完成，未执行真实{transaction.operation}。",
                can_cancel=False,
                can_take_over=False,
                confirmation_id=None,
            ),
        ]
    if decision == "take_over":
        return [
            _task_progress_event(
                transaction,
                status="taken_over",
                phase="finalizing",
                tool_name="take_over",
                step_title="已停止自动执行，请手动接管",
                message="已停止自动执行，请手动接管。",
                can_cancel=False,
                can_take_over=False,
                confirmation_id=None,
            )
        ]
    return [
        _task_progress_event(
            transaction,
            status="cancelled",
            phase="finalizing",
            tool_name="needs_confirmation",
            step_title=f"已取消：{transaction.operation}",
            message=f"已取消{transaction.operation}，未执行真实操作。",
            can_cancel=False,
            can_take_over=False,
            confirmation_id=None,
        )
    ]


def _task_progress_event(
    transaction: ConfirmationTransaction,
    *,
    status: str,
    phase: str,
    tool_name: str,
    step_title: str,
    message: str,
    can_cancel: bool,
    can_take_over: bool,
    confirmation_id: str | None | object = ...,
) -> JsonObject:
    event: JsonObject = {
        "type": "task_progress",
        "label": transaction.operation,
        "taskTitle": transaction.task_title,
        "status": status,
        "currentStep": 1,
        "totalSteps": 1,
        "stepTitle": step_title,
        "phase": phase,
        "toolName": tool_name,
        "requiresConfirmation": False,
        "canCancel": can_cancel,
        "canTakeOver": can_take_over,
        "message": message,
        "progressKey": f"confirmation-{transaction.confirmation_id}",
        "dryRun": transaction.dry_run,
    }
    if transaction.run_id:
        event["runId"] = transaction.run_id
    if transaction.thread_id:
        event["threadId"] = transaction.thread_id
    resolved_confirmation_id = (
        transaction.confirmation_id if confirmation_id is ... else confirmation_id
    )
    if isinstance(resolved_confirmation_id, str) and resolved_confirmation_id:
        event["confirmationId"] = resolved_confirmation_id
    return event


def _high_risk_spec(tool_name: str, args: Mapping[str, object]) -> JsonObject:
    if tool_name == "create_event":
        title = _event_title(args)
        return _spec(
            task_title="高风险日程操作确认",
            operation="创建日程",
            target_app="系统日历",
            payload_preview=f"是否创建日程：{title}？",
            confirm_text="确认创建",
        )
    if tool_name == "update_event":
        title = _event_title(args)
        return _spec(
            task_title="高风险日程操作确认",
            operation="修改日程",
            target_app="系统日历",
            payload_preview=f"是否修改日程：{title}？",
            confirm_text="确认修改",
        )
    if tool_name == "update_reminders":
        event_id = args.get("event_id") or args.get("eventId") or "未知日程"
        reminders = args.get("reminders")
        count = len(reminders) if isinstance(reminders, list) else 0
        return _spec(
            task_title="高风险提醒操作确认",
            operation="修改提醒",
            target_app="系统日历",
            payload_preview=f"是否为日程 {event_id} 替换 {count} 条提醒？",
            confirm_text="确认修改",
        )
    if tool_name == "archive_file":
        source = _short_text(args.get("source"), "待归档文件")
        target = _short_text(args.get("target_dir") or args.get("targetDir"), "目标目录")
        mode = _short_text(args.get("mode"), "copy")
        return _spec(
            task_title="高风险文件操作确认",
            operation="归档文件",
            target_app="本地文件",
            payload_preview=f"是否以 {mode} 模式将 {source} 归档到 {target}？",
            confirm_text="确认归档",
        )
    if tool_name == "feishu_cli":
        return _spec(
            task_title="高风险飞书操作确认",
            operation="执行飞书写入操作",
            target_app="飞书",
            payload_preview="即将调用飞书 CLI 写入能力，是否继续？",
            confirm_text="确认执行",
        )
    if tool_name == "wecom_cli":
        return _spec(
            task_title="高风险企业微信操作确认",
            operation="执行企业微信写入操作",
            target_app="企业微信",
            payload_preview="即将调用企业微信 CLI 写入能力，是否继续？",
            confirm_text="确认执行",
        )
    if tool_name == "run_cli_command":
        return _spec(
            task_title="高风险外部命令确认",
            operation="执行外部命令",
            target_app="飞书/企业微信 CLI",
            payload_preview="即将执行可能包含写入动作的外部命令，是否继续？",
            confirm_text="确认执行",
        )
    return _spec(
        task_title="高风险操作确认",
        operation="执行高风险操作",
        target_app="受控工具",
        payload_preview=f"是否允许执行 {tool_name}？",
        confirm_text="确认执行",
    )


def _spec(
    *,
    task_title: str,
    operation: str,
    target_app: str,
    payload_preview: str,
    confirm_text: str,
) -> JsonObject:
    return {
        "taskTitle": task_title,
        "operation": operation,
        "targetApp": target_app,
        "riskLevel": "medium",
        "payloadPreview": payload_preview,
        "confirmText": confirm_text,
        "cancelText": "取消",
    }


def _event_title(args: Mapping[str, object]) -> str:
    event = args.get("event")
    if isinstance(event, Mapping):
        for key in ("title", "description", "eventLocation"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return _short_text(value, "未命名日程")
    return "未命名日程"


def _short_text(value: object, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    text = value.strip()
    return text if len(text) <= 48 else f"{text[:47]}…"


def to_jsonable(value: object) -> JsonValue:
    return to_json_value(value)


__all__ = [
    "ConfirmationResolveResult",
    "ConfirmationTransaction",
    "confirm_confirmation",
    "confirm_confirmation_async",
    "confirmation_store",
    "create_confirmation",
    "create_high_risk_confirmation",
    "get_confirmation",
    "reject_confirmation",
    "reject_confirmation_async",
    "take_over_confirmation",
    "take_over_confirmation_async",
]
