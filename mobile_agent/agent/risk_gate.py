"""Blocks high-risk actions until the user starts a new explicit request."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ..trace import is_high_risk_tool
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
        return Command(
            update={
                "awaiting_user_action": True,
                "awaiting_user_reason": "high_risk_action",
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
