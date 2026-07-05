from __future__ import annotations

import asyncio
from typing import Any

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from mobile_agent.agent.factory import build_middleware_stack
from mobile_agent.agent.risk_gate import HighRiskActionGateMiddleware
from mobile_agent.agent.trace_middleware import TraceMiddleware
from mobile_agent.trace import TraceEmitter


def test_trace_stack_order_is_locked() -> None:
    stack = build_middleware_stack(
        phone_gateway=object(),
        system_gateway=object(),
        phone_tool_names={"tap"},
        device_scoped_tool_names={"tap"},
    )

    assert [type(item).__name__ for item in stack] == [
        "ResetAgentRunStateMiddleware",
        "TraceMiddleware",
        "AppInventoryQueryMiddleware",
        "MeetingMinutesSopMiddleware",
        "Scenario3DemoMiddleware",
        "MedicalTravelSopMiddleware",
        "HighRiskActionGateMiddleware",
        "ModeToolAccessMiddleware",
        "TaskComplexityMiddleware",
        "RouteModelMiddleware",
        "RoutedSystemPromptMiddleware",
        "SyncPhoneStateMiddleware",
    ]


def test_tool_wrapping_order_and_high_risk_short_circuit_are_locked() -> None:
    trace_events: list[dict[str, Any]] = []
    trace = TraceMiddleware(TraceEmitter(lambda: trace_events.append))
    risk = HighRiskActionGateMiddleware()
    state: dict[str, Any] = {"messages": []}
    state.update(trace.before_agent(state, type("Runtime", (), {"config": {}})()) or {})
    order: list[str] = []

    async def real_handler(_: ToolCallRequest) -> ToolMessage:
        order.append("real")
        return ToolMessage(content="ok", tool_call_id="call-1", name="observe")

    async def mode_gate(request: ToolCallRequest) -> ToolMessage:
        order.append("mode")
        return await real_handler(request)

    async def high_gate(request: ToolCallRequest) -> ToolMessage:
        order.append("high_risk")
        return await risk.awrap_tool_call(request, mode_gate)

    request = ToolCallRequest(
        tool_call={"name": "observe", "args": {}, "id": "call-1", "type": "tool_call"},
        tool=None,
        state=state,
        runtime=None,
    )
    asyncio.run(trace.awrap_tool_call(request, high_gate))

    assert order == ["high_risk", "mode", "real"]
    assert [event["event"] for event in trace_events[:2]] == ["run.started", "step.upsert"]
    assert trace_events[-1]["step"]["status"] == "succeeded"

    order.clear()
    high_risk_request = ToolCallRequest(
        tool_call={"name": "create_event", "args": {}, "id": "call-2", "type": "tool_call"},
        tool=None,
        state=state,
        runtime=None,
    )
    result = asyncio.run(trace.awrap_tool_call(high_risk_request, high_gate))

    assert order == ["high_risk"]
    assert result.update["awaiting_user_action"] is True
    assert trace_events[-1]["step"]["status"] == "waiting_for_user"
