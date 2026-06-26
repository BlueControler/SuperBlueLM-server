from __future__ import annotations

import inspect
import json
from typing import Annotated, Any, Literal, Protocol, cast

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from ..json_types import JsonObject
from ..gateways.phone import (
    DEVICE_NOT_CONNECTED_MESSAGE,
    DeviceGatewayError,
)
from ..progress import emit_task_progress
from ..trace import tool_trace_step_id
from .phone_subagent import PhoneTodoExecution, redact_phone_text
from .state import MobileAgentState, PhoneTodoStep, device_id_from_mapping


class PhoneTodoRunner(Protocol):
    async def execute(
        self,
        todo: str,
        *,
        allow_short_chain: bool,
        device_id: str | None = None,
        trace_parent_id: str | None = None,
    ) -> PhoneTodoExecution: ...


class ExecutePhoneTodoArgs(BaseModel):
    todo: str = Field(
        min_length=1,
        max_length=1000,
        description="One clear and verifiable phone UI TODO.",
    )
    allow_short_chain: bool = Field(
        default=False,
        description="Allow a bounded deterministic short chain of phone actions.",
    )
    runtime: Annotated[Any, SkipJsonSchema()] = None


class ResetPhoneTodoMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    state_schema = MobileAgentState

    def before_agent(
        self,
        state: MobileAgentState,
        runtime: Runtime[None],
    ) -> dict[str, object]:
        return {
            "phone_todo_steps": (),
            "task_complexity_emitted": False,
            "awaiting_user_action": False,
            "awaiting_user_reason": "",
            "run_failure_reason": "",
        }

    async def abefore_agent(
        self,
        state: MobileAgentState,
        runtime: Runtime[None],
    ) -> dict[str, object]:
        return self.before_agent(state, runtime)


async def execute_tracked_phone_todo(
    steps: tuple[PhoneTodoStep, ...],
    runner: PhoneTodoRunner,
    todo: str,
    *,
    allow_short_chain: bool,
    device_id: str | None = None,
    trace_parent_id: str | None = None,
) -> tuple[PhoneTodoExecution, tuple[PhoneTodoStep, ...]]:
    index = len(steps) + 1
    progress_key = f"phone-todo-{index}"
    redacted_todo = redact_phone_text(todo)
    emit_task_progress(
        label=redacted_todo,
        status="running",
        phase="agent",
        message=f"Executing phone TODO: {redacted_todo}",
        tool_name="execute_phone_todo",
        progress_key=progress_key,
        current_step=index,
        total_steps=index,
        completed_steps=_completed_step_payloads(steps),
    )

    try:
        trace_kwargs = _trace_parent_kwargs(runner, trace_parent_id)
        if device_id is None:
            result = await runner.execute(
                todo,
                allow_short_chain=allow_short_chain,
                **trace_kwargs,
            )
        else:
            result = await runner.execute(
                todo,
                allow_short_chain=allow_short_chain,
                device_id=device_id,
                **trace_kwargs,
            )
    except DeviceGatewayError:
        result = _device_not_connected_execution(todo)
    redacted_summary = redact_phone_text(result.summary)
    redacted_error = redact_phone_text(result.error) if result.error else None
    progress_status: Literal["completed", "failed"] = (
        "completed" if result.status == "completed" else "failed"
    )
    next_steps = (
        *steps,
        PhoneTodoStep(
            index=index,
            progressKey=progress_key,
            name=redacted_todo,
            status=progress_status,
            summary=redacted_summary,
        ),
    )
    emit_task_progress(
        label=redacted_todo,
        status=progress_status,
        phase="agent",
        message=redacted_summary,
        tool_name="execute_phone_todo",
        progress_key=progress_key,
        current_step=index,
        total_steps=len(next_steps),
        completed_steps=_completed_step_payloads(next_steps),
        error=redacted_error,
    )
    return result, next_steps


def _device_not_connected_execution(todo: str) -> PhoneTodoExecution:
    return PhoneTodoExecution(
        status="failed",
        todo=redact_phone_text(todo),
        summary=DEVICE_NOT_CONNECTED_MESSAGE,
        phoneState={
            "currentPackage": None,
            "activity": None,
            "hasScreenshot": False,
            "hasUi": False,
        },
        toolCallCount=0,
        needsMainAgentPlan=True,
        error="device_not_connected",
    )


def create_phone_delegation_tool(runner: PhoneTodoRunner) -> BaseTool:
    @tool(
        "execute_phone_todo",
        args_schema=ExecutePhoneTodoArgs,
        description=(
            "Delegate one clear Android UI TODO to the restricted phone subagent. "
            "Use allow_short_chain only for a small deterministic local sequence."
        ),
    )
    async def execute_phone_todo(
        *,
        todo: str,
        runtime: ToolRuntime,
        allow_short_chain: bool = False,
    ) -> Command:
        steps = cast(
            tuple[PhoneTodoStep, ...],
            runtime.state.get("phone_todo_steps", ()),
        )
        # 同一 run 内只允许执行一次手机 TODO——防止主 Agent 的 LLM 循环重试。
        # 重复调用直接返回之前已完成的步骤结果，不再启动子 Agent。
        if steps:
            last_step = steps[-1]
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=json.dumps({
                                "status": "completed",
                                "todo": redact_phone_text(todo),
                                "summary": f"已执行手机操作（共 {len(steps)} 步）。",
                                "terminal": True,
                                "retryable": False,
                                "already_done": True,
                            }),
                            tool_call_id=runtime.tool_call_id or "execute_phone_todo",
                            name="execute_phone_todo",
                            status="success",
                        ),
                    ],
                },
                goto=END,
            )

        result, next_steps = await execute_tracked_phone_todo(
            steps,
            runner,
            todo,
            allow_short_chain=allow_short_chain,
            device_id=_runtime_device_id(runtime),
            trace_parent_id=tool_trace_step_id(runtime.tool_call_id),
        )
        messages = [
            ToolMessage(
                content=result.model_dump_json(by_alias=True),
                tool_call_id=runtime.tool_call_id or "execute_phone_todo",
                name="execute_phone_todo",
                status="error" if result.status in {"failed", "rejected", "cancelled", "timeout", "stopped"} else "success",
            )
        ]
        update: dict[str, object] = {
            "phone_todo_steps": next_steps,
            "messages": messages,
        }
        # 不可重试终态 → 立即结束 run
        if result.status in {"failed", "rejected", "cancelled", "timeout", "stopped", "budget_exhausted"}:
            update["run_failure_reason"] = result.error or result.status
            return Command(update=update, goto=END)
        # 正常完成且不需要主 agent 再规划：结束工具节点，让直接意图中间件生成最终回复。
        if result.status == "completed" and not result.needs_main_agent_plan:
            return Command(update=update, goto=END)
        return Command(update=update)

    return execute_phone_todo


def _runtime_device_id(runtime: ToolRuntime) -> str | None:
    return (
        device_id_from_mapping(runtime.state)
        or device_id_from_mapping(runtime.config.get("metadata"))
        or device_id_from_mapping(runtime.config.get("configurable"))
    )


def _trace_parent_kwargs(
    runner: PhoneTodoRunner,
    trace_parent_id: str | None,
) -> dict[str, str]:
    if trace_parent_id is None:
        return {}
    parameters = inspect.signature(runner.execute).parameters
    return {"trace_parent_id": trace_parent_id} if "trace_parent_id" in parameters else {}


def _completed_step_payloads(steps: tuple[PhoneTodoStep, ...]) -> list[JsonObject]:
    return [
        {
            "index": step["index"],
            "name": step["name"],
            "status": step["status"],
        }
        for step in steps
        if step["status"] == "completed"
    ]


__all__ = [
    "ResetPhoneTodoMiddleware",
    "create_phone_delegation_tool",
    "execute_tracked_phone_todo",
]
