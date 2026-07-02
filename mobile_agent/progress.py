from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langgraph.config import get_stream_writer

from .json_types import JsonObject, JsonValue, to_json_value

ProgressStatus = Literal[
    "started",
    "pending",
    "running",
    "waiting_confirmation",
    "completed",
    "failed",
    "cancelled",
    "taken_over",
]
TaskComplexity = Literal["simple", "complex"]

TASK_PROGRESS_TYPE = "task_progress"
TASK_COMPLEXITY_TYPE = "task_complexity"
NEEDS_CONFIRMATION_TYPE = "needs_confirmation"


def emit_needs_confirmation(
    *,
    confirmation_id: str,
    task_title: str,
    operation: str,
    target_app: str,
    tool_name: str,
    risk_level: str,
    payload_preview: str,
    confirm_text: str,
    cancel_text: str,
    run_id: str | None = None,
    thread_id: str | None = None,
    dry_run: bool | None = None,
) -> None:
    payload: JsonObject = {
        "type": NEEDS_CONFIRMATION_TYPE,
        "confirmationId": confirmation_id,
        "taskTitle": task_title,
        "operation": operation,
        "targetApp": target_app,
        "toolName": tool_name,
        "riskLevel": risk_level,
        "payloadPreview": payload_preview,
        "confirmText": confirm_text,
        "cancelText": cancel_text,
    }
    _put_if_present(payload, "runId", run_id)
    _put_if_present(payload, "threadId", thread_id)
    _put_if_present(payload, "dryRun", dry_run)

    try:
        writer = get_stream_writer()
    except Exception:
        return
    writer(payload)


def emit_task_progress(
    *,
    label: str,
    status: ProgressStatus,
    phase: str,
    run_id: str | None = None,
    thread_id: str | None = None,
    task_title: str | None = None,
    step_title: str | None = None,
    message: str | None = None,
    tool_name: str | None = None,
    progress_key: str | None = None,
    current_step: int | None = None,
    total_steps: int | None = None,
    completed_steps: Sequence[JsonObject] | None = None,
    requires_confirmation: bool | None = None,
    confirmation_id: str | None = None,
    can_cancel: bool | None = None,
    can_take_over: bool | None = None,
    dry_run: bool | None = None,
    error: str | None = None,
) -> None:
    payload: JsonObject = {
        "type": TASK_PROGRESS_TYPE,
        "label": label,
        "status": status,
        "phase": phase,
    }
    _put_if_present(payload, "runId", run_id)
    _put_if_present(payload, "threadId", thread_id)
    _put_if_present(payload, "taskTitle", task_title)
    _put_if_present(payload, "stepTitle", step_title)
    _put_if_present(payload, "message", message)
    _put_if_present(payload, "toolName", tool_name)
    _put_if_present(payload, "progressKey", progress_key)
    _put_if_present(payload, "currentStep", current_step)
    _put_if_present(payload, "totalSteps", total_steps)
    _put_if_present(
        payload,
        "completedSteps",
        list(completed_steps) if completed_steps is not None else None,
    )
    _put_if_present(payload, "requiresConfirmation", requires_confirmation)
    _put_if_present(payload, "confirmationId", confirmation_id)
    _put_if_present(payload, "canCancel", can_cancel)
    _put_if_present(payload, "canTakeOver", can_take_over)
    _put_if_present(payload, "dryRun", dry_run)
    _put_if_present(payload, "error", error)

    try:
        writer = get_stream_writer()
    except Exception:
        return
    writer(payload)


def emit_task_complexity(
    *,
    complexity: TaskComplexity,
    track_steps: bool,
    reason: str,
    message: str | None = None,
) -> None:
    payload: JsonObject = {
        "type": TASK_COMPLEXITY_TYPE,
        "complexity": complexity,
        "trackSteps": track_steps,
        "reason": reason,
    }
    _put_if_present(payload, "message", message)

    try:
        writer = get_stream_writer()
    except Exception:
        return
    writer(payload)


def _put_if_present(payload: JsonObject, key: str, value: JsonValue | None) -> None:
    if value is not None:
        payload[key] = to_json_value(value)
