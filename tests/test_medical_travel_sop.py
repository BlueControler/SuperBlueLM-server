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
        return "高德地图路线：地铁 2 号线转 5 号线，预计 42 分钟，少步行。"

    async def write_reminder(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("create_event / update_reminders", arguments))
        return {"ok": True, "eventId": "demo-event-1", "reminderId": "demo-reminder-1"}

    async def get_location(self, arguments: dict[str, Any]) -> str:
        self.calls.append(("get_location", arguments))
        return json.dumps({"address": "天津市南开区学生公寓"}, ensure_ascii=False)


def _medical_module() -> Any:
    return importlib.import_module("mobile_agent.agent.medical_travel_sop")


def _progress_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("phase") == "medical_travel"]


def test_medical_travel_request_recognizes_travel_but_not_medical_advice() -> None:
    module = _medical_module()

    assert module.is_medical_travel_request("明天上午我要去医院复诊，帮我查天气和路线，并设置提醒。")
    assert module.is_medical_travel_request("后天上午 9 点去医院检查，帮我看看天气和出行时间。")
    assert module.is_medical_travel_request("明天陪老人去医院复诊，帮我整理出行注意事项，并设置提醒。")
    assert not module.is_medical_travel_request("我头疼怎么办？")
    assert not module.is_medical_travel_request("这个药怎么吃？")


def test_parse_medical_travel_intent_extracts_time_destination_and_elderly() -> None:
    module = _medical_module()

    intent = module.parse_medical_travel_intent("后天上午 9 点陪老人去天津医科大学总医院检查，提前 60 分钟提醒。")

    assert intent.date_text == "后天"
    assert intent.time_text == "上午9点"
    assert intent.city == "天津"
    assert intent.destination == "天津医科大学总医院"
    assert intent.purpose == "检查"
    assert intent.reminder_offset_minutes == 60
    assert intent.for_elderly is True


def test_medical_travel_reads_memory_and_returns_decision_payload(monkeypatch: Any) -> None:
    module = _medical_module()
    emitted: list[dict[str, Any]] = []
    tools = _FakeMedicalTravelTools()
    memory = {
        "origin": "南开大学宿舍",
        "preferred_hospital": "天津医科大学总医院",
        "city": "天津",
        "travel_preference": "优先地铁，少步行",
        "medical_checklist": "身份证、医保卡、既往检查资料",
        "reminder_offset_minutes": 30,
    }
    runner = module.MedicalTravelSopRunner(
        weather_query=tools.weather_query,
        route_query=tools.amap_mcp_tool,
        reminder_writer=tools.write_reminder,
        memory_reader=lambda _keys, _context=None: memory,
    )
    intent = module.parse_medical_travel_intent("明天上午我要去医院复诊，帮我查天气和路线，并设置提醒。")
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    result = asyncio.run(runner.run(intent=intent, device_id="device-1"))

    assert result["medical_travel"]["intent"]["origin"] == "南开大学宿舍"
    assert result["medical_travel"]["intent"]["destination"] == "天津医科大学总医院"
    assert result["medical_travel"]["memory"]["memoryUsed"] is True
    assert result["medical_travel"]["memory"]["memorySource"] == "user_memory"
    assert result["medical_travel"]["memory"]["items"] == [
        {"key": "origin", "value": "南开大学宿舍"},
        {"key": "preferred_hospital", "value": "天津医科大学总医院"},
        {"key": "city", "value": "天津"},
        {"key": "travel_preference", "value": "优先地铁，少步行"},
        {"key": "medical_checklist", "value": "身份证、医保卡、既往检查资料"},
        {"key": "reminder_offset_minutes", "value": 30},
    ]
    assert result["medical_travel"]["decision"]["origin"] == "南开大学宿舍"
    assert "优先地铁" in result["medical_travel"]["decision"]["routeChoiceReason"]
    assert result["medical_travel"]["decision"]["checklist"] == [
        "身份证",
        "医保卡",
        "既往检查资料",
    ]
    assert result["medical_travel"]["confirmation"]["status"] == "needs_confirmation"
    assert result["medical_travel"]["reminder"]["created"] is False
    assert [name for name, _args in tools.calls] == ["weather_query", "amap_mcp_tool"]

    step_events = _progress_steps(emitted)
    titles = [event["stepTitle"] for event in step_events if event["status"] in {"running", "waiting_confirmation"}]
    assert titles == [
        "识别就医需求",
        "读取过往记忆",
        "查询天气",
        "规划出行路线",
        "形成出行决策",
        "等待确认",
    ]
    assert all(event["totalSteps"] == 7 for event in step_events)
    assert step_events[1]["message"] == "已识别为：明天上午前往天津医科大学总医院复诊。"
    assert any(event["toolName"] == "read_user_memory" for event in step_events)
    assert any("已参考过往偏好" in event["message"] for event in step_events)


def test_medical_travel_without_memory_uses_defaults() -> None:
    module = _medical_module()
    tools = _FakeMedicalTravelTools()
    runner = module.MedicalTravelSopRunner(
        weather_query=tools.weather_query,
        route_query=tools.amap_mcp_tool,
        reminder_writer=tools.write_reminder,
        memory_reader=lambda _keys, _context=None: {},
    )

    result = asyncio.run(runner.run(intent=module.parse_medical_travel_intent("明天上午我要去医院复诊，帮我查天气和路线，并设置提醒。")))

    payload = result["medical_travel"]
    assert payload["memory"]["memoryUsed"] is False
    assert payload["memory"]["memorySource"] == "none"
    assert payload["intent"]["origin"] == "当前位置"
    assert payload["intent"]["destination"]


def test_medical_travel_uses_location_tool_when_memory_has_no_origin() -> None:
    module = _medical_module()
    tools = _FakeMedicalTravelTools()
    runner = module.MedicalTravelSopRunner(
        weather_query=tools.weather_query,
        route_query=tools.amap_mcp_tool,
        reminder_writer=tools.write_reminder,
        memory_reader=lambda _keys, _context=None: {},
        location_query=tools.get_location,
    )

    result = asyncio.run(
        runner.run(
            intent=module.parse_medical_travel_intent("明天上午我要去医院复诊，帮我查天气和路线，并设置提醒。"),
            device_id="device-1",
        )
    )

    assert [name for name, _args in tools.calls] == [
        "get_location",
        "weather_query",
        "amap_mcp_tool",
    ]
    assert tools.calls[0][1] == {"device_id": "device-1"}
    assert tools.calls[2][1]["origin"] == "天津市南开区学生公寓"
    assert result["medical_travel"]["intent"]["origin"] == "天津市南开区学生公寓"
    assert result["medical_travel"]["decision"]["origin"] == "天津市南开区学生公寓"


def test_medical_travel_middleware_returns_structured_payload(monkeypatch: Any) -> None:
    module = _medical_module()
    emitted: list[dict[str, Any]] = []
    tools = _FakeMedicalTravelTools()
    runner = module.MedicalTravelSopRunner(
        weather_query=tools.weather_query,
        route_query=tools.amap_mcp_tool,
        reminder_writer=tools.write_reminder,
        memory_reader=lambda _keys, _context=None: {"travel_preference": "少换乘"},
    )
    middleware = module.MedicalTravelSopMiddleware(runner)
    request = type(
        "Request",
        (),
        {
            "messages": [HumanMessage(content="明天陪老人去医院复诊，帮我查天气和路线，并设置提醒。")],
            "runtime": type("Runtime", (), {"config": {"configurable": {"device_id": "device-1"}}})(),
            "state": {},
        },
    )()
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    async def fail_handler(_request: Any) -> ModelResponse[Any]:
        raise AssertionError("medical travel skill should not reach the model")

    response = asyncio.run(middleware.awrap_model_call(request, fail_handler))

    assert isinstance(response.result[0], AIMessage)
    payload = response.result[0].additional_kwargs["medical_travel"]
    assert payload["intent"]["forElderly"] is True
    assert payload["memory"]["memoryUsed"] is True
    assert payload["decision"]["reminderOffsetMinutes"] == 30
    assert "身份证" in payload["decision"]["checklist"]
    assert json.loads(response.result[0].additional_kwargs["medical_travel_sop"])["medical_travel"]["intent"]["forElderly"] is True


def test_medical_travel_confirmation_confirm_creates_calendar_with_device_id() -> None:
    module = _medical_module()
    confirmation_store.clear()
    calls: list[tuple[str, dict[str, Any]]] = []

    @tool("weather_query")
    async def weather_query(city: str | None = None) -> str:
        """Query weather."""
        calls.append(("weather_query", {"city": city}))
        return "明天多云，18-24℃。"

    @tool("amap_mcp_tool")
    async def amap_mcp_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call AMap."""
        calls.append(("amap_mcp_tool", {"tool_name": tool_name, "arguments": arguments or {}}))
        return "预计 42 分钟，地铁优先。"

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
        calls.append(("update_reminders", {"event_id": event_id, "reminders": reminders, "device_id": device_id}))
        return {"ok": True}

    runner = module.build_medical_travel_sop_runner(
        external_tools=[weather_query, amap_mcp_tool],
        system_tools=[create_event, update_reminders],
    )
    result = asyncio.run(
        runner.run(
            intent=module.parse_medical_travel_intent("明天上午我要去医院复诊，帮我查天气和路线，并设置提醒。"),
            device_id="device-1",
        )
    )
    confirmation_id = result["medical_travel"]["confirmation"]["confirmationId"]

    assert [name for name, _args in calls] == ["weather_query", "amap_mcp_tool"]
    response = TestClient(app).post(f"/mobile/confirmations/{confirmation_id}/confirm")

    assert response.status_code == 200
    assert [name for name, _args in calls] == [
        "weather_query",
        "amap_mcp_tool",
        "create_event",
        "update_reminders",
    ]
    assert calls[2][1]["device_id"] == "device-1"
    assert calls[3][1]["device_id"] == "device-1"
    assert calls[3][1]["reminders"] == [{"minutes": 30, "method": "alert"}]
    events = response.json()["events"]
    assert [event["status"] for event in events] == ["running", "completed"]
    assert all(event["currentStep"] == 7 for event in events)
    assert all(event["totalSteps"] == 7 for event in events)
    assert events[-1]["message"] == "已创建就医出行提醒。"
