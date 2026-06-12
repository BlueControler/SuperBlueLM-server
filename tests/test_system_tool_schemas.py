from __future__ import annotations

import asyncio
from typing import Any

from mobile_agent.gateways.system import SystemToolGateway
from mobile_agent.tools.system import create_system_tools


class _FakeSystemClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def send_request(self, message: str, data: Any) -> dict[str, bool]:
        self.calls.append((message, data))
        return {"ok": True}


class _FakeSystemGateway:
    def __init__(self) -> None:
        self.client = _FakeSystemClient()
        self.device_ids: list[str | None] = []

    def get_default_client(self, device_id: str | None = None) -> _FakeSystemClient:
        self.device_ids.append(device_id)
        return self.client


def _tool_by_name() -> dict[str, Any]:
    return {tool.name: tool for tool in create_system_tools(SystemToolGateway())}


def _fake_tools() -> tuple[dict[str, Any], _FakeSystemGateway]:
    gateway = _FakeSystemGateway()
    return {tool.name: tool for tool in create_system_tools(gateway)}, gateway


def _properties(tool: Any) -> dict[str, dict[str, Any]]:
    schema = tool.args_schema.model_json_schema()
    return schema["properties"]


def test_list_apps_parameter_description() -> None:
    props = _properties(_tool_by_name()["list_apps"])
    assert props["app_type"]["description"] == (
        "Application filter from the system API protocol. Use 'all' for every app, "
        "'third' for third-party apps, and 'system' for system apps."
    )


def test_calendar_event_schema_describes_protocol_fields() -> None:
    props = _properties(_tool_by_name()["create_event"])
    event_schema = props["event"]
    assert (
        "Calendar event payload following Android CalendarContract.EventsColumns."
        in event_schema["description"]
    )

    definitions = _tool_by_name()["create_event"].args_schema.model_json_schema()["$defs"]
    calendar_event = definitions["CalendarEventArgs"]["properties"]
    assert (
        calendar_event["dtstart"]["description"]
        == "Event start time as Unix timestamp in milliseconds."
    )
    assert (
        calendar_event["eventTimezone"]["description"]
        == "IANA timezone ID, for example Asia/Shanghai."
    )
    assert (
        calendar_event["status"]["description"]
        == "Event confirmation status. Use cancelled to delete an event."
    )


def test_reminder_schema_describes_protocol_fields() -> None:
    definitions = _tool_by_name()["update_reminders"].args_schema.model_json_schema()[
        "$defs"
    ]
    reminder = definitions["CalendarReminderArgs"]["properties"]
    assert (
        reminder["minutes"]["description"]
        == "Reminder offset before event start, in minutes."
    )
    assert (
        reminder["method"]["description"]
        == "Reminder method supported by the Android client: alert or alarm."
    )


def test_get_location_uses_device_scoped_explicit_schema() -> None:
    tool = _tool_by_name()["get_location"]
    assert tool.args_schema.__name__ == "GetLocationArgs"
    assert tool.args_schema.model_json_schema()["properties"].keys() == {"device_id"}


def test_create_event_serializes_aliases_through_tool_schema() -> None:
    tools, gateway = _fake_tools()

    asyncio.run(
        tools["create_event"].ainvoke({"event": {"_id": 123, "title": "Meeting"}})
    )

    assert gateway.client.calls == [
        ("createEvent", {"event": {"_id": 123, "title": "Meeting"}})
    ]


def test_update_event_serializes_aliases_through_tool_schema() -> None:
    tools, gateway = _fake_tools()

    asyncio.run(
        tools["update_event"].ainvoke({"event": {"_id": 123, "title": "Meeting"}})
    )

    assert gateway.client.calls == [
        ("updateEvent", {"event": {"_id": 123, "title": "Meeting"}})
    ]


def test_update_reminders_serializes_protocol_payload() -> None:
    tools, gateway = _fake_tools()

    asyncio.run(
        tools["update_reminders"].ainvoke(
            {
                "event_id": 123,
                "reminders": [{"minutes": 10, "method": "alert"}],
            }
        )
    )

    assert gateway.client.calls == [
        (
            "updateReminders",
            {
                "eventId": 123,
                "reminders": [{"minutes": 10, "method": "alert"}],
            },
        )
    ]


def test_list_apps_uses_default_app_type_through_tool_schema() -> None:
    tools, gateway = _fake_tools()

    asyncio.run(tools["list_apps"].ainvoke({}))

    assert gateway.client.calls == [("listApps", {"type": "all"})]


def test_system_tool_routes_to_configured_device_id() -> None:
    tools, gateway = _fake_tools()

    asyncio.run(tools["list_apps"].ainvoke({"app_type": "all", "device_id": "device-1"}))

    assert gateway.device_ids == ["device-1"]


def test_all_system_tool_arguments_have_descriptions() -> None:
    for tool in _tool_by_name().values():
        schema = tool.args_schema.model_json_schema()
        properties = schema.get("properties", {})
        for name, property_schema in properties.items():
            assert property_schema.get(
                "description"
            ), f"{tool.name}.{name} is missing description"
