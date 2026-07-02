from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from starlette.testclient import TestClient

from mobile_agent import progress
from mobile_agent.confirmations import confirmation_store
from mobile_agent.http_app import app


class _FakeMedicalTravelTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def weather_query(self, arguments: dict[str, Any]) -> str:
        self.calls.append(("weather_query", arguments))
        return "明天多云，18-24℃，上午体感舒适。"

    async def amap_mcp_tool(self, arguments: dict[str, Any]) -> str:
        self.calls.append(("amap_mcp_tool", arguments))
        return "高德地图路线：地铁 2 号线转 5 号线，预计 42 分钟。"

    async def needs_confirmation(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("needs_confirmation", arguments))
        return {"status": "confirmed", "confirmed": True}

    async def write_reminder(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("create_event / update_reminders", arguments))
        return {"ok": True, "eventId": "demo-event-1", "reminderId": "demo-reminder-1"}


def _medical_module() -> Any:
    try:
        return importlib.import_module("mobile_agent.agent.medical_travel_sop")
    except ModuleNotFoundError as exc:
        raise AssertionError("medical travel SOP module is missing") from exc


def _progress_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("phase") == "medical_travel_sop"]


def test_fixed_medical_travel_utterance_is_recognized() -> None:
    module = _medical_module()

    assert module.is_medical_travel_sop_request(module.FIXED_MEDICAL_TRAVEL_UTTERANCE)
    assert module.is_medical_travel_sop_request(
        "明天上午我要去医院复诊，帮我查天气和路线，并设置提醒"
    )
    assert not module.is_medical_travel_sop_request("明天帮我查一下天气")


def test_medical_travel_sop_runs_fixed_confirmed_demo_flow(monkeypatch: Any) -> None:
    module = _medical_module()
    emitted: list[dict[str, Any]] = []
    tools = _FakeMedicalTravelTools()
    runner = module.MedicalTravelSopRunner(
        weather_query=tools.weather_query,
        route_query=tools.amap_mcp_tool,
        confirmation_request=tools.needs_confirmation,
        reminder_writer=tools.write_reminder,
        auto_confirm=True,
        now=lambda: "2026-07-02",
    )
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    result = asyncio.run(runner.run())

    assert result["task_type"] == "medical_travel_reminder"
    assert result["weather_result"] == "明天多云，18-24℃，上午体感舒适。"
    assert result["route_result"] == "高德地图路线：地铁 2 号线转 5 号线，预计 42 分钟。"
    assert "建议提前" in result["travel_advice"]
    assert result["confirmation"]["status"] == "confirmed"
    assert result["reminder_created"] is True
    assert result["reminder_result"]["eventId"] == "demo-event-1"
    assert result["final_message"] == "已整理明日出行信息，并创建复诊提醒。"
    assert [name for name, _args in tools.calls] == [
        "weather_query",
        "amap_mcp_tool",
        "needs_confirmation",
        "create_event / update_reminders",
    ]
    assert tools.calls[0][1]["city"] == "南京"
    assert tools.calls[1][1]["destination"] == "医院"
    assert "复诊" in tools.calls[2][1]["message"]
    assert tools.calls[3][1]["title"] == "医院复诊出行提醒"

    steps = _progress_steps(emitted)
    assert [
        (step["currentStep"], step["totalSteps"], step["message"], step["toolName"])
        for step in steps
    ] == [
        (1, 5, "第 1/5 步：正在识别出行需求", "task_complexity"),
        (2, 5, "第 2/5 步：正在查询天气", "weather_query"),
        (3, 5, "第 3/5 步：正在规划路线", "amap_mcp_tool"),
        (4, 5, "第 4/5 步：等待确认是否创建提醒", "needs_confirmation"),
        (5, 5, "第 5/5 步：正在写入日程提醒", "create_event / update_reminders"),
    ]


def test_medical_travel_middleware_short_circuits_fixed_demo_request(
    monkeypatch: Any,
) -> None:
    module = _medical_module()
    emitted: list[dict[str, Any]] = []
    runner = module.MedicalTravelSopRunner()
    middleware = module.MedicalTravelSopMiddleware(runner)
    request = type(
        "Request",
        (),
        {"messages": [HumanMessage(content=module.FIXED_MEDICAL_TRAVEL_UTTERANCE)]},
    )()
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    async def fail_handler(_request: Any) -> ModelResponse[Any]:
        raise AssertionError("fixed SOP request should not reach the model")

    response = asyncio.run(middleware.awrap_model_call(request, fail_handler))

    assert isinstance(response, ModelResponse)
    assert isinstance(response.result[0], AIMessage)
    assert response.result[0].content == "已整理明日出行信息，等待你确认是否创建复诊提醒。"
    payload = json.loads(response.result[0].additional_kwargs["medical_travel_sop"])
    assert payload["reminder_created"] is False
    assert payload["confirmation"]["status"] == "needs_confirmation"
    confirmation_id = payload["confirmation"]["confirmationId"]
    assert confirmation_id
    needs_confirmation_events = [
        event for event in emitted if event.get("type") == "needs_confirmation"
    ]
    assert needs_confirmation_events[-1]["confirmationId"] == confirmation_id
    assert needs_confirmation_events[-1]["toolName"] == "create_event"
    assert needs_confirmation_events[-1]["dryRun"] is True
    waiting_events = [
        event
        for event in _progress_steps(emitted)
        if event.get("status") == "waiting_confirmation"
    ]
    assert waiting_events[-1]["confirmationId"] == confirmation_id
    assert waiting_events[-1]["canCancel"] is True
    assert waiting_events[-1]["canTakeOver"] is True
    assert [event["toolName"] for event in _progress_steps(emitted)] == [
        "task_complexity",
        "weather_query",
        "amap_mcp_tool",
        "needs_confirmation",
    ]


def test_medical_travel_runner_adapters_call_real_tool_interfaces() -> None:
    module = _medical_module()
    calls: list[tuple[str, dict[str, Any]]] = []

    @tool("weather_query")
    async def weather_query(city: str | None = None) -> str:
        """Query weather."""
        calls.append(("weather_query", {"city": city}))
        return "真实天气结果"

    @tool("amap_mcp_tool")
    async def amap_mcp_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call AMap MCP."""
        calls.append(("amap_mcp_tool", {"tool_name": tool_name, "arguments": arguments or {}}))
        return "真实路线结果"

    @tool("create_event")
    async def create_event(event: dict[str, Any], device_id: str | None = None) -> dict[str, Any]:
        """Create event."""
        calls.append(("create_event", {"event": event, "device_id": device_id}))
        return {"ok": True, "eventId": 123}

    @tool("update_reminders")
    async def update_reminders(
        event_id: int,
        reminders: list[dict[str, Any]],
        device_id: str | None = None,
    ) -> dict[str, Any]:
        """Update reminders."""
        calls.append(
            (
                "update_reminders",
                {"event_id": event_id, "reminders": reminders, "device_id": device_id},
            )
        )
        return {"ok": True}

    runner = module.build_medical_travel_sop_runner(
        external_tools=[weather_query, amap_mcp_tool],
        system_tools=[create_event, update_reminders],
    )

    result = asyncio.run(runner.run(device_id="device-1"))

    assert result["weather_result"] == "真实天气结果"
    assert result["route_result"] == "真实路线结果"
    assert result["reminder_result"]["skipped"] is True
    assert result["confirmation"]["status"] == "needs_confirmation"
    assert result["reminder_created"] is False
    assert [name for name, _args in calls] == [
        "weather_query",
        "amap_mcp_tool",
    ]
    assert calls[0][1] == {"city": "南京"}
    assert calls[1][1]["tool_name"] == "maps_direction_transit_integrated"


def test_medical_travel_confirmation_confirm_completes_dry_run_without_writing() -> None:
    module = _medical_module()
    confirmation_store.clear()
    calls: list[tuple[str, dict[str, Any]]] = []
    tools = _FakeMedicalTravelTools()

    async def write_reminder(arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append(("create_event / update_reminders", arguments))
        return {"ok": True, "eventId": "event-1", "reminderId": "reminder-1"}

    runner = module.MedicalTravelSopRunner(
        weather_query=tools.weather_query,
        route_query=tools.amap_mcp_tool,
        reminder_writer=write_reminder,
    )

    result = asyncio.run(runner.run(device_id="device-1"))
    confirmation_id = result["confirmation"]["confirmationId"]

    assert calls == []
    response = TestClient(app).post(f"/mobile/confirmations/{confirmation_id}/confirm")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "confirmed"
    assert payload["dryRun"] is True
    assert calls == []
    assert [event["status"] for event in payload["events"]] == ["running", "completed"]
    assert all(event["dryRun"] is True for event in payload["events"])
    assert all(event["currentStep"] == 5 for event in payload["events"])
    assert all(event["totalSteps"] == 5 for event in payload["events"])
    assert payload["events"][0]["toolName"] == "create_event / update_reminders"
    assert payload["events"][0]["requiresConfirmation"] is False
    assert payload["events"][0]["message"] == "第 5/5 步：正在创建复诊提醒（演示模式）"
    assert payload["events"][1]["message"] == "第 5/5 步：创建提醒 dry-run 已完成"

    second = TestClient(app).post(f"/mobile/confirmations/{confirmation_id}/confirm")
    assert second.status_code == 409
    assert calls == []
