from __future__ import annotations

from typing import Any, Literal, Protocol, cast

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..json_types import JsonObject
from ..progress import emit_task_progress
from .phone_subagent import PhoneTodoExecution
from .state import MobileAgentState, PhoneTodoStep


class PhoneTodoRunner(Protocol):
    async def execute(
        self,
        todo: str,
        *,
        allow_short_chain: bool,
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
) -> tuple[PhoneTodoExecution, tuple[PhoneTodoStep, ...]]:
    index = len(steps) + 1
    progress_key = f"phone-todo-{index}"
    emit_task_progress(
        label=todo,
        status="running",
        phase="agent",
        message=f"Executing phone TODO: {todo}",
        tool_name="execute_phone_todo",
        progress_key=progress_key,
        current_step=index,
        total_steps=index,
        completed_steps=_completed_step_payloads(steps),
    )

    result = await runner.execute(todo, allow_short_chain=allow_short_chain)
    progress_status: Literal["completed", "failed"] = (
        "completed" if result.status == "completed" else "failed"
    )
    next_steps = (
        *steps,
        PhoneTodoStep(
            index=index,
            progressKey=progress_key,
            name=todo,
            status=progress_status,
            summary=result.summary,
        ),
    )
    emit_task_progress(
        label=todo,
        status=progress_status,
        phase="agent",
        message=result.summary,
        tool_name="execute_phone_todo",
        progress_key=progress_key,
        current_step=index,
        total_steps=len(next_steps),
        completed_steps=_completed_step_payloads(next_steps),
        error=result.error,
    )
    return result, next_steps


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
        todo: str,
        runtime: ToolRuntime,
        allow_short_chain: bool = False,
    ) -> Command:
        steps = cast(
            tuple[PhoneTodoStep, ...],
            runtime.state.get("phone_todo_steps", ()),
        )
        result, next_steps = await execute_tracked_phone_todo(
            steps,
            runner,
            todo,
            allow_short_chain=allow_short_chain,
        )
        return Command(
            update={
                "phone_todo_steps": next_steps,
                "messages": [
                    ToolMessage(
                        content=result.model_dump_json(by_alias=True),
                        tool_call_id=runtime.tool_call_id or "execute_phone_todo",
                        name="execute_phone_todo",
                    )
                ],
            }
        )

    return execute_phone_todo


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
