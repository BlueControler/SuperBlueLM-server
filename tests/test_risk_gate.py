from __future__ import annotations

import asyncio
from typing import Any

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from mobile_agent.agent.risk_gate import HighRiskActionGateMiddleware


def _request(
    name: str,
    *,
    args: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args or {}, "id": "call-1", "type": "tool_call"},
        tool=None,
        state={"messages": [], **(state or {})},
        runtime=None,
    )


def test_read_only_system_tool_is_allowed() -> None:
    middleware = HighRiskActionGateMiddleware()
    called = False

    async def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return ToolMessage(content="{}", tool_call_id="call-1", name="list_events")

    result = asyncio.run(middleware.awrap_tool_call(_request("list_events"), handler))

    assert called is True
    assert isinstance(result, ToolMessage)


def test_calendar_mutation_is_blocked_without_calling_handler() -> None:
    middleware = HighRiskActionGateMiddleware()
    called = False

    async def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return ToolMessage(content="unexpected", tool_call_id="call-1", name="create_event")

    result = asyncio.run(middleware.awrap_tool_call(_request("create_event"), handler))

    assert called is False
    assert isinstance(result, Command)
    assert result.update["awaiting_user_action"] is True
    assert "需要你确认" in result.update["messages"][0].content


def test_run_waiting_for_user_blocks_every_followup_tool() -> None:
    middleware = HighRiskActionGateMiddleware()
    called = False

    async def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return ToolMessage(content="unexpected", tool_call_id="call-1", name="list_events")

    result = asyncio.run(
        middleware.awrap_tool_call(
            _request("list_events", state={"awaiting_user_action": True}),
            handler,
        )
    )

    assert called is False
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "等待用户处理" in result.content
