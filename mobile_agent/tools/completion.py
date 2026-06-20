from __future__ import annotations

from typing import Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from ..json_types import JsonObject
from ..progress import emit_task_progress

SubtaskStatus = Literal["completed", "failed", "skipped"]


class FinishSubtask(BaseModel):
    name: str = Field(description="Short name of the subtask, for example 消息发送 or 日程创建.")
    status: SubtaskStatus = Field(
        description="Final state of the subtask: completed, failed, or skipped."
    )
    detail: str | None = Field(
        default=None,
        description="Optional one-line result detail shown to the user.",
    )


class FinishArgs(BaseModel):
    summary: str = Field(
        min_length=1,
        max_length=2000,
        description="User-facing wrap-up of the whole task and its final outcome.",
    )
    subtasks: list[FinishSubtask] = Field(
        default_factory=list,
        description=(
            "Per-subtask completion states for the floating-window summary. "
            "Use this when the task was a multi-step plan."
        ),
    )


def create_completion_tools() -> list[BaseTool]:
    @tool(
        "finish",
        args_schema=FinishArgs,
        description=(
            "End the current task and report the final result. Call this once at the "
            "very end of a task to drive the floating-window completion feedback. "
            "Pass a user-facing summary and, for multi-step tasks, the per-subtask "
            "completion states."
        ),
        return_direct=True,
    )
    async def finish(summary: str, subtasks: list[FinishSubtask] | None = None) -> str:
        subtask_list = subtasks or []
        completed_steps: list[JsonObject] = [
            {
                "index": index,
                "name": subtask.name,
                "status": subtask.status,
                **({"detail": subtask.detail} if subtask.detail else {}),
            }
            for index, subtask in enumerate(subtask_list, start=1)
        ]
        emit_task_progress(
            label="finish",
            status="completed",
            phase="agent",
            message=summary,
            tool_name="finish",
            progress_key="finish",
            completed_steps=completed_steps or None,
        )
        return summary

    return [finish]


__all__ = ["create_completion_tools", "FinishArgs", "FinishSubtask"]
