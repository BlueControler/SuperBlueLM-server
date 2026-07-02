"""Blocks high-risk actions until the user starts a new explicit request."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ..confirmations import create_high_risk_confirmation
from ..progress import emit_needs_confirmation, emit_task_progress
from ..trace import is_high_risk_tool
from ..trace import current_context
from .state import MobileAgentState

ToolResult: TypeAlias = ToolMessage | Command[Any]
ToolHandler: TypeAlias = Callable[[ToolCallRequest], ToolResult]
AsyncToolHandler: TypeAlias = Callable[[ToolCallRequest], Awaitable[ToolResult]]

_WAITING_MESSAGE = "该操作需要你确认后再继续处理。"
_RUN_BLOCKED_MESSAGE = "当前任务正在等待用户处理，不能继续自动执行工具。"


class HighRiskActionGateMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    """Converts risky calls into a run-level waiting state before side effects."""

    state_schema = MobileAgentState

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolResult:
        blocked = self._block_result(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
    ) -> ToolResult:
        blocked = self._block_result(request)
        return blocked if blocked is not None else await handler(request)

    def _block_result(self, request: ToolCallRequest) -> ToolResult | None:
        state = request.state if isinstance(request.state, Mapping) else {}
        if state.get("awaiting_user_action") is True:
            return self._blocked_message(request, _RUN_BLOCKED_MESSAGE)

        tool_call = request.tool_call
        tool_name = tool_call["name"]
        args = tool_call.get("args", {})
        if not isinstance(args, dict) or not is_high_risk_tool(tool_name, args):
            return None
        context = current_context()
        transaction = create_high_risk_confirmation(
            tool_name=tool_name,
            args=args,
            run_id=context.run_id if context is not None else None,
            thread_id=context.thread_id if context is not None else None,
        )
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
        emit_task_progress(
            label=transaction.operation,
            run_id=transaction.run_id,
            thread_id=transaction.thread_id,
            task_title=transaction.task_title,
            status="waiting_confirmation",
            phase="confirmation",
            current_step=1,
            total_steps=1,
            step_title=f"等待确认：{transaction.operation}",
            tool_name="needs_confirmation",
            requires_confirmation=True,
            confirmation_id=transaction.confirmation_id,
            can_cancel=True,
            can_take_over=True,
            message=transaction.payload_preview,
            progress_key=f"confirmation-{transaction.confirmation_id}",
            dry_run=transaction.dry_run,
        )
        return Command(
            update={
                "awaiting_user_action": True,
                "awaiting_user_reason": "high_risk_action",
                "awaiting_confirmation_id": transaction.confirmation_id,
                "messages": [self._blocked_message(request, _WAITING_MESSAGE)],
            }
        )

    @staticmethod
    def _blocked_message(request: ToolCallRequest, message: str) -> ToolMessage:
        return ToolMessage(
            content=message,
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )


__all__ = ["HighRiskActionGateMiddleware"]
