from __future__ import annotations

import asyncio
import json
from typing import Literal

from langchain_core.tools import BaseTool, tool

from ..action_control import PhoneActionControlError, dispatch_phone_command
from ..gateways.phone import (
    DEVICE_NOT_CONNECTED_CODE,
    DEVICE_NOT_CONNECTED_MESSAGE,
    DeviceGateway,
    DeviceGatewayError,
    DeviceNotConnectedError,
)
from ..json_types import JsonObject, JsonValue
from ..progress import emit_task_progress


def create_phone_tools(
    gateway: DeviceGateway,
    default_device_id: str | None = None,
    expected_session: object | None = None,
) -> list[BaseTool]:
    async def send(
        tool_name: str,
        message: str,
        data: JsonValue,
        device_id: str | None,
    ) -> JsonObject:
        emit_task_progress(
            label=tool_name,
            status="running",
            phase="phone_tool",
            message=f"Running phone tool: {tool_name}",
            tool_name=tool_name,
            can_cancel=True,
            can_take_over=True,
        )
        try:
            selected_device_id = _select_device_id(device_id, default_device_id)
            session = await gateway.wait_for_session(selected_device_id)
            if expected_session is not None and session is not expected_session:
                raise DeviceNotConnectedError()
            result = await dispatch_phone_command(
                session,
                message,
                data,
                device_id=selected_device_id,
            )
        except DeviceNotConnectedError as exc:
            emit_task_progress(
                label=tool_name,
                status="failed",
                phase="phone_tool",
                message=f"Phone tool failed: {tool_name}",
                tool_name=tool_name,
                can_cancel=False,
                can_take_over=False,
                error=str(exc),
            )
            return _device_not_connected_result()
        except DeviceGatewayError as exc:
            emit_task_progress(
                label=tool_name,
                status="failed",
                phase="phone_tool",
                message=f"Phone tool failed: {tool_name}",
                tool_name=tool_name,
                can_cancel=False,
                can_take_over=False,
                error=str(exc),
            )
            if _is_device_not_connected_error(exc):
                return _device_not_connected_result()
            return {"error": "phone_tool_failed", "message": str(exc), "recoverable": False}
        except PhoneActionControlError as exc:
            emit_task_progress(
                label=tool_name,
                status="failed",
                phase="phone_tool",
                message=f"Phone tool rejected: {tool_name}",
                tool_name=tool_name,
                can_cancel=False,
                can_take_over=False,
                error=str(exc),
            )
            return {
                "error": "phone_action_rejected",
                "message": str(exc),
                "recoverable": False,
            }
        emit_task_progress(
            label=tool_name,
            status="completed",
            phase="phone_tool",
            message=f"Completed phone tool: {tool_name}",
            tool_name=tool_name,
            can_cancel=False,
            can_take_over=False,
        )
        return result

    @tool("observe", description="Get the latest screenshot and UI tree from the phone.")
    async def observe(device_id: str | None = None) -> str:
        return _dump_result(_summarize_result(await send("observe", "observe", None, device_id)))

    @tool("launch", description="Launch an Android app by package name.")
    async def launch(package: str, device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("launch", "launch", {"package": package}, device_id))
        )

    @tool("tap", description="Tap a screen coordinate in pixels.")
    async def tap(x: int, y: int, device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("tap", "tap", {"x": x, "y": y}, device_id))
        )

    @tool("type", description="Type text into the currently focused input box.")
    async def type_text(text: str, device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("type", "type", {"text": text}, device_id))
        )

    @tool("swipe", description="Swipe from one coordinate to another.")
    async def swipe(
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        device_id: str | None = None,
    ) -> str:
        result = await send(
            "swipe",
            "swipe",
            {"startX": start_x, "startY": start_y, "endX": end_x, "endY": end_y},
            device_id,
        )
        return _dump_result(_summarize_result(result))

    @tool(
        "scroll",
        description=(
            "Scroll the current page using stable safe-area coordinates. "
            "Prefer this over swipe for normal page scrolling."
        ),
    )
    async def scroll(
        direction: Literal["up", "down"],
        distance: Literal["short", "medium", "long"] = "medium",
        device_id: str | None = None,
    ) -> str:
        try:
            selected_device_id = _select_device_id(device_id, default_device_id)
            session = await gateway.wait_for_session(selected_device_id)
            if expected_session is not None and session is not expected_session:
                raise DeviceNotConnectedError()
        except DeviceNotConnectedError:
            return _dump_result(_device_not_connected_result())
        except DeviceGatewayError as exc:
            if _is_device_not_connected_error(exc):
                return _dump_result(_device_not_connected_result())
            return _dump_result(
                {"error": "phone_tool_failed", "message": str(exc), "recoverable": False}
            )
        info = getattr(session, "device_info", None)
        if info is None:
            return _dump_result(
                {
                    "error": "phone_tool_failed",
                    "message": "Device screen dimensions are unavailable.",
                    "recoverable": False,
                }
            )
        start_x, start_y, end_x, end_y = _scroll_coordinates(
            info.width,
            info.height,
            direction,
            distance,
        )
        result = await send(
            "scroll",
            "swipe",
            {
                "startX": start_x,
                "startY": start_y,
                "endX": end_x,
                "endY": end_y,
            },
            device_id,
        )
        return _dump_result(_summarize_result(result))

    @tool("long_press", description="Long press a screen coordinate.")
    async def long_press(x: int, y: int, device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(
                await send("long_press", "longPress", {"x": x, "y": y}, device_id)
            )
        )

    @tool("double_tap", description="Double tap a screen coordinate.")
    async def double_tap(x: int, y: int, device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(
                await send("double_tap", "doubleTap", {"x": x, "y": y}, device_id)
            )
        )

    @tool("back", description="Agent-level wrapper that sends Android keyevent 4 (BACK).")
    async def back(device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("back", "keyevent", {"keyevent": 4}, device_id))
        )

    @tool("home", description="Agent-level wrapper that sends Android keyevent 3 (HOME).")
    async def home(device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("home", "keyevent", {"keyevent": 3}, device_id))
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
    async def keyevent(keyevent: int, device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(
                await send("keyevent", "keyevent", {"keyevent": keyevent}, device_id)
            )
        )

    @tool("wait", description="Wait for a number of seconds so the page can complete loading.")
    async def wait(duration: float, device_id: str | None = None) -> str:
        await asyncio.sleep(max(duration, 0))
        return _dump_result(_summarize_result(await send("wait", "observe", None, device_id)))

    @tool(
        "interact",
        description="Ask the user to choose one of several reasonable next actions.",
        return_direct=True,
    )
    async def interact(message: str, device_id: str | None = None) -> str:
        result = await send("interact", "interact", {"message": message}, device_id)
        if "error" in result:
            return _dump_result(result)
        return message

    @tool(
        "take_over",
        description="Hand control back to the user when the user must operate the phone directly.",
        return_direct=True,
    )
    async def take_over(message: str, device_id: str | None = None) -> str:
        result = await send("take_over", "interact", {"message": message}, device_id)
        if "error" in result:
            return _dump_result(result)
        return message

    return [
        observe,
        launch,
        tap,
        type_text,
        swipe,
        scroll,
        long_press,
        double_tap,
        back,
        home,
        keyevent,
        wait,
        interact,
        take_over,
    ]


def _summarize_result(result: JsonObject) -> JsonObject:
    if "error" in result:
        return result
    return {
        "ok": True,
        "currentPackage": result.get("currentPackage"),
        "activity": result.get("activity"),
        "has_screenshot": bool(result.get("screenshot")),
        "has_ui": bool(result.get("ui")),
    }


def _select_device_id(
    requested_device_id: str | None,
    bound_device_id: str | None,
) -> str | None:
    if bound_device_id is None:
        return requested_device_id
    if requested_device_id is not None and requested_device_id != bound_device_id:
        raise DeviceGatewayError("Phone tool cannot override the bound device_id.")
    return bound_device_id


def _device_not_connected_result() -> JsonObject:
    return {
        "error": DEVICE_NOT_CONNECTED_CODE,
        "message": DEVICE_NOT_CONNECTED_MESSAGE,
        "recoverable": True,
    }


def _scroll_coordinates(
    width: int,
    height: int,
    direction: Literal["up", "down"],
    distance: Literal["short", "medium", "long"],
) -> tuple[int, int, int, int]:
    center_x = width // 2
    spans = {
        "short": (0.65, 0.35),
        "medium": (0.75, 0.25),
        "long": (0.82, 0.18),
    }
    lower, upper = spans[distance]
    start_y = int(height * lower)
    end_y = int(height * upper)
    if direction == "down":
        start_y, end_y = end_y, start_y
    return center_x, start_y, center_x, end_y


def _is_device_not_connected_error(error: DeviceGatewayError) -> bool:
    message = str(error).lower()
    return isinstance(error, DeviceNotConnectedError) or any(
        marker in message
        for marker in (
            DEVICE_NOT_CONNECTED_CODE,
            "no connected device",
            "device disconnected",
            "device is disconnected",
            "device connection",
        )
    )


def _dump_result(result: JsonObject) -> str:
    return json.dumps(result, ensure_ascii=False)
