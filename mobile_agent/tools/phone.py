from __future__ import annotations

import asyncio
import json
from typing import TypedDict

from langchain_core.tools import BaseTool, tool

from ..gateways.phone import DeviceGateway
from ..json_types import JsonObject, JsonValue
from ..progress import emit_task_progress


class PhoneToolSummary(TypedDict):
    ok: bool
    currentPackage: JsonValue
    activity: JsonValue
    has_screenshot: bool
    has_ui: bool


def create_phone_tools(gateway: DeviceGateway) -> list[BaseTool]:
    async def send(tool_name: str, message: str, data: JsonValue) -> JsonObject:
        emit_task_progress(
            label=tool_name,
            status="running",
            phase="phone_tool",
            message=f"Running phone tool: {tool_name}",
            tool_name=tool_name,
        )
        try:
            result = await gateway.get_session().send_command(message, data)
        except Exception as exc:
            emit_task_progress(
                label=tool_name,
                status="failed",
                phase="phone_tool",
                message=f"Phone tool failed: {tool_name}",
                tool_name=tool_name,
                error=str(exc),
            )
            raise
        emit_task_progress(
            label=tool_name,
            status="completed",
            phase="phone_tool",
            message=f"Completed phone tool: {tool_name}",
            tool_name=tool_name,
        )
        return result

    @tool("observe", description="Get the latest screenshot and UI tree from the phone.")
    async def observe() -> str:
        return _dump_result(_summarize_result(await send("observe", "observe", None)))

    @tool("launch", description="Launch an Android app by package name.")
    async def launch(package: str) -> str:
        return _dump_result(
            _summarize_result(await send("launch", "launch", {"package": package}))
        )

    @tool("tap", description="Tap a screen coordinate in pixels.")
    async def tap(x: int, y: int) -> str:
        return _dump_result(_summarize_result(await send("tap", "tap", {"x": x, "y": y})))

    @tool("type", description="Type text into the currently focused input box.")
    async def type_text(text: str) -> str:
        return _dump_result(_summarize_result(await send("type", "type", {"text": text})))

    @tool("swipe", description="Swipe from one coordinate to another.")
    async def swipe(start_x: int, start_y: int, end_x: int, end_y: int) -> str:
        result = await send(
            "swipe",
            "swipe",
            {"startX": start_x, "startY": start_y, "endX": end_x, "endY": end_y},
        )
        return _dump_result(_summarize_result(result))

    @tool("long_press", description="Long press a screen coordinate.")
    async def long_press(x: int, y: int) -> str:
        return _dump_result(
            _summarize_result(await send("long_press", "longPress", {"x": x, "y": y}))
        )

    @tool("double_tap", description="Double tap a screen coordinate.")
    async def double_tap(x: int, y: int) -> str:
        return _dump_result(
            _summarize_result(await send("double_tap", "doubleTap", {"x": x, "y": y}))
        )

    @tool("back", description="Agent-level wrapper that sends Android keyevent 4 (BACK).")
    async def back() -> str:
        return _dump_result(
            _summarize_result(await send("back", "keyevent", {"keyevent": 4}))
        )

    @tool("home", description="Agent-level wrapper that sends Android keyevent 3 (HOME).")
    async def home() -> str:
        return _dump_result(
            _summarize_result(await send("home", "keyevent", {"keyevent": 3}))
        )

    @tool(
        "keyevent",
        description=(
            "Send an Android keyevent code. Common codes: 3 HOME, 4 BACK, "
            "66 ENTER, 67 DEL/BACKSPACE, 82 MENU, 187 APP_SWITCH/RECENTS, "
            "24 VOLUME_UP, 25 VOLUME_DOWN, 26 POWER. Prefer back() and home() "
            "for normal navigation."
        ),
    )
    async def keyevent(keyevent: int) -> str:
        return _dump_result(
            _summarize_result(await send("keyevent", "keyevent", {"keyevent": keyevent}))
        )

    @tool("wait", description="Wait for a number of seconds so the page can complete loading.")
    async def wait(duration: float) -> str:
        await asyncio.sleep(max(duration, 0))
        return _dump_result(_summarize_result(await send("wait", "observe", None)))

    @tool(
        "interact",
        description="Ask the user to choose one of several reasonable next actions.",
        return_direct=True,
    )
    async def interact(message: str) -> str:
        await send("interact", "interact", {"message": message})
        return message

    @tool(
        "take_over",
        description="Hand control back to the user when the user must operate the phone directly.",
        return_direct=True,
    )
    async def take_over(message: str) -> str:
        await send("take_over", "interact", {"message": message})
        return message

    return [
        observe,
        launch,
        tap,
        type_text,
        swipe,
        long_press,
        double_tap,
        back,
        home,
        keyevent,
        wait,
        interact,
        take_over,
    ]


def _summarize_result(result: JsonObject) -> PhoneToolSummary:
    return {
        "ok": True,
        "currentPackage": result.get("currentPackage"),
        "activity": result.get("activity"),
        "has_screenshot": bool(result.get("screenshot")),
        "has_ui": bool(result.get("ui")),
    }


def _dump_result(result: PhoneToolSummary) -> str:
    return json.dumps(result, ensure_ascii=False)
