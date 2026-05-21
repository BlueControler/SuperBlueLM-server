from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langgraph.config import get_stream_writer

from .json_types import JsonObject, JsonValue, to_json_value

ProgressStatus = Literal["started", "running", "completed", "failed"]
TaskComplexity = Literal["simple", "complex"]

TASK_PROGRESS_TYPE = "task_progress"
TASK_COMPLEXITY_TYPE = "task_complexity"


def emit_task_progress(
    *,
    label: str,
    status: ProgressStatus,
    phase: str,
    message: str | None = None,
    tool_name: str | None = None,
    current_step: int | None = None,
    total_steps: int | None = None,
    completed_steps: Sequence[JsonObject] | None = None,
    error: str | None = None,
) -> None:
    payload: JsonObject = {
        "type": TASK_PROGRESS_TYPE,
        "label": label,
        "status": status,
        "phase": phase,
    }
    _put_if_present(payload, "message", message)
    _put_if_present(payload, "toolName", tool_name)
    _put_if_present(payload, "currentStep", current_step)
    _put_if_present(payload, "totalSteps", total_steps)
    _put_if_present(
        payload,
        "completedSteps",
        list(completed_steps) if completed_steps is not None else None,
    )
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
