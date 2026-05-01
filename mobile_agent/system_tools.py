from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from .system_gateway import SystemGatewayError, SystemToolGateway


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _error_payload(exc: Exception) -> str:
    return _dump({"error": str(exc)})


def create_system_tools(gateway: SystemToolGateway) -> list[Any]:
    async def send(message: str, data: Any) -> str:
        try:
            return _dump(await gateway.get_default_client().send_request(message, data))
        except SystemGatewayError as exc:
            return _error_payload(exc)

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
    async def create_event(event: dict[str, Any]) -> str:
        return await send("createEvent", {"event": event})

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
    async def update_event(event: dict[str, Any]) -> str:
        return await send("updateEvent", {"event": event})

    @tool(
        "list_reminders",
        description="List all reminders attached to a calendar event.",
    )
    async def list_reminders(event_id: int) -> str:
        return await send("listReminders", {"eventId": event_id})

    @tool(
        "update_reminders",
        description="Replace all reminders on a calendar event. Passing an empty list removes all reminders.",
    )
    async def update_reminders(event_id: int, reminders: list[dict[str, Any]]) -> str:
        return await send("updateReminders", {"eventId": event_id, "reminders": reminders})

    @tool(
        "get_location",
        description="Get current device location through the /system tool client.",
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
