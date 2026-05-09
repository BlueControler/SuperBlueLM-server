from __future__ import annotations

import json
from typing import Literal, TypedDict

from langchain_core.tools import BaseTool, tool

from ..gateways.system import SystemGatewayError, SystemToolGateway
from ..json_types import JsonObject, JsonValue, to_json_value


class CalendarEvent(TypedDict, total=False):
    _id: int
    title: str
    description: str
    eventLocation: str
    dtstart: int
    dtend: int
    allDay: bool
    eventTimezone: str
    duration: str
    rrule: str
    availability: Literal["busy", "free", "tentative"]
    status: Literal["confirmed", "tentative", "cancelled"]


class CalendarReminder(TypedDict):
    minutes: int
    method: Literal["alert", "alarm"]


def create_system_tools(gateway: SystemToolGateway) -> list[BaseTool]:
    async def send(message: str, data: JsonValue) -> str:
        try:
            return _dump(await gateway.get_default_client().send_request(message, data))
        except SystemGatewayError as exc:
            return _dump({"error": str(exc)})

    @tool(
        "list_apps",
        description=(
            "List installed Android apps through the /system tool client. "
            "app_type is all, third, or system. Returns {packageName: appLabel}."
        ),
    )
    async def list_apps(app_type: str = "all") -> str:
        return await send("listApps", {"type": app_type})

    @tool(
        "create_event",
        description="Create a calendar event through the /system tool client. The event follows CalendarContract fields.",
    )
    async def create_event(event: CalendarEvent) -> str:
        return await send("createEvent", {"event": _json_object(event)})

    @tool(
        "list_events",
        description="List calendar events whose start or end time falls within [start, end], timestamps in milliseconds.",
    )
    async def list_events(start: int, end: int) -> str:
        return await send("listEvents", {"start": start, "end": end})

    @tool(
        "update_event",
        description="Update an existing calendar event. To delete an event, set status to cancelled.",
    )
    async def update_event(event: CalendarEvent) -> str:
        return await send("updateEvent", {"event": _json_object(event)})

    @tool("list_reminders", description="List all reminders attached to a calendar event.")
    async def list_reminders(event_id: int) -> str:
        return await send("listReminders", {"eventId": event_id})

    @tool(
        "update_reminders",
        description="Replace all reminders on a calendar event. Passing an empty list removes all reminders.",
    )
    async def update_reminders(event_id: int, reminders: list[CalendarReminder]) -> str:
        return await send(
            "updateReminders",
            {
                "eventId": event_id,
                "reminders": [_json_object(reminder) for reminder in reminders],
            },
        )

    @tool(
        "get_location", description="Get current device location through the /system tool client."
    )
    async def get_location() -> str:
        return await send("getLocation", None)

    return [
        list_apps,
        create_event,
        list_events,
        update_event,
        list_reminders,
        update_reminders,
        get_location,
    ]


def _json_object(value: CalendarEvent | CalendarReminder) -> JsonObject:
    return {key: to_json_value(item) for key, item in value.items()}


def _dump(data: JsonValue) -> str:
    return json.dumps(data, ensure_ascii=False)
