from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from ..gateways.system import SystemGatewayError, SystemToolGateway
from ..json_types import JsonObject, JsonValue, to_json_value


class CalendarEventArgs(BaseModel):
    id: int | None = Field(
        default=None,
        alias="_id",
        description=(
            "System-assigned event ID. Omit when creating a new event; include when "
            "updating an existing event."
        ),
    )
    title: str | None = Field(default=None, description="Calendar event title.")
    description: str | None = Field(
        default=None, description="Calendar event description or notes."
    )
    eventLocation: str | None = Field(
        default=None, description="Human-readable event location."
    )
    dtstart: int | None = Field(
        default=None,
        description="Event start time as Unix timestamp in milliseconds.",
    )
    dtend: int | None = Field(
        default=None,
        description="Event end time as Unix timestamp in milliseconds.",
    )
    allDay: bool | None = Field(
        default=None, description="Whether the event lasts the whole day."
    )
    eventTimezone: str | None = Field(
        default=None,
        description="IANA timezone ID, for example Asia/Shanghai.",
    )
    duration: str | None = Field(
        default=None,
        description="RFC2445 duration, for example PT1H for one hour.",
    )
    rrule: str | None = Field(
        default=None,
        description="RFC2445 recurrence rule, for example FREQ=DAILY;INTERVAL=1.",
    )
    availability: Literal["busy", "free", "tentative"] | None = Field(
        default=None,
        description="Event availability: busy, free, or tentative.",
    )
    status: Literal["confirmed", "tentative", "cancelled"] | None = Field(
        default=None,
        description="Event confirmation status. Use cancelled to delete an event.",
    )

    model_config = {"populate_by_name": True}


class CalendarReminderArgs(BaseModel):
    minutes: int = Field(
        description="Reminder offset before event start, in minutes."
    )
    method: Literal["alert", "alarm"] = Field(
        description="Reminder method supported by the Android client: alert or alarm."
    )


class ListAppsArgs(BaseModel):
    app_type: Literal["all", "third", "system"] = Field(
        default="all",
        description=(
            "Application filter from the system API protocol. Use 'all' for every app, "
            "'third' for third-party apps, and 'system' for system apps."
        ),
    )


class CreateEventArgs(BaseModel):
    event: CalendarEventArgs = Field(
        description="Calendar event payload following Android CalendarContract.EventsColumns."
    )


class ListEventsArgs(BaseModel):
    start: int = Field(
        description="Inclusive query start time as Unix timestamp in milliseconds."
    )
    end: int = Field(
        description="Inclusive query end time as Unix timestamp in milliseconds."
    )


class UpdateEventArgs(BaseModel):
    event: CalendarEventArgs = Field(
        description="Calendar event payload with _id set for the event to update."
    )


class ListRemindersArgs(BaseModel):
    event_id: int = Field(
        description="Calendar event ID whose reminders should be listed."
    )


class UpdateRemindersArgs(BaseModel):
    event_id: int = Field(
        description="Calendar event ID whose reminders should be replaced."
    )
    reminders: list[CalendarReminderArgs] = Field(
        description="Complete reminder list. Passing an empty list removes all reminders."
    )


class GetLocationArgs(BaseModel):
    pass


def create_system_tools(gateway: SystemToolGateway) -> list[BaseTool]:
    async def send(message: str, data: JsonValue) -> str:
        try:
            return _dump(await gateway.get_default_client().send_request(message, data))
        except SystemGatewayError as exc:
            return _dump({"error": str(exc)})

    @tool(
        "list_apps",
        args_schema=ListAppsArgs,
        description=(
            "List installed Android apps through the /system tool client. "
            "app_type is all, third, or system. Returns {packageName: appLabel}."
        ),
    )
    async def list_apps(app_type: str = "all") -> str:
        return await send("listApps", {"type": app_type})

    @tool(
        "create_event",
        args_schema=CreateEventArgs,
        description="Create a calendar event through the /system tool client. The event follows CalendarContract fields.",
    )
    async def create_event(event: CalendarEventArgs) -> str:
        return await send("createEvent", {"event": _json_object(event)})

    @tool(
        "list_events",
        args_schema=ListEventsArgs,
        description="List calendar events whose start or end time falls within [start, end], timestamps in milliseconds.",
    )
    async def list_events(start: int, end: int) -> str:
        return await send("listEvents", {"start": start, "end": end})

    @tool(
        "update_event",
        args_schema=UpdateEventArgs,
        description="Update an existing calendar event. To delete an event, set status to cancelled.",
    )
    async def update_event(event: CalendarEventArgs) -> str:
        return await send("updateEvent", {"event": _json_object(event)})

    @tool(
        "list_reminders",
        args_schema=ListRemindersArgs,
        description="List all reminders attached to a calendar event.",
    )
    async def list_reminders(event_id: int) -> str:
        return await send("listReminders", {"eventId": event_id})

    @tool(
        "update_reminders",
        args_schema=UpdateRemindersArgs,
        description="Replace all reminders on a calendar event. Passing an empty list removes all reminders.",
    )
    async def update_reminders(
        event_id: int, reminders: list[CalendarReminderArgs]
    ) -> str:
        return await send(
            "updateReminders",
            {
                "eventId": event_id,
                "reminders": [_json_object(reminder) for reminder in reminders],
            },
        )

    @tool(
        "get_location",
        args_schema=GetLocationArgs,
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


def _json_object(value: BaseModel | JsonObject) -> JsonObject:
    if isinstance(value, BaseModel):
        data = value.model_dump(mode="json", by_alias=True, exclude_none=True)
    else:
        data = value
    return {key: to_json_value(item) for key, item in data.items()}


def _dump(data: JsonValue) -> str:
    return json.dumps(data, ensure_ascii=False)
