from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langgraph.config import get_stream_writer

from .json_types import JsonObject, JsonValue, to_json_value

ProgressStatus = Literal["started", "running", "completed", "failed"]

TASK_PROGRESS_TYPE = "task_progress"


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


def _put_if_present(payload: JsonObject, key: str, value: JsonValue | None) -> None:
    if value is not None:
        payload[key] = to_json_value(value)
