from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import tool

from phone_gateway import ConnectedDeviceSession, DeviceGateway


def _dump_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "currentPackage": result.get("currentPackage"),
        "activity": result.get("activity"),
        "has_screenshot": bool(result.get("screenshot")),
        "has_ui": bool(result.get("ui")),
    }


def _require_session(gateway: DeviceGateway, device_id: str | None) -> ConnectedDeviceSession:
    if device_id:
        return gateway.get_device(device_id)
    return gateway.get_default_device()


def create_phone_tools(gateway: DeviceGateway) -> list[Any]:
    async def send(message: str, data: Any, device_id: str | None = None) -> dict[str, Any]:
        session = _require_session(gateway, device_id)
        return await session.send_command(message, data)

    @tool("observe", description="Get the latest screenshot and UI tree from the phone.")
    async def observe(device_id: str | None = None) -> str:
        return _dump_result(_summarize_result(await send("observe", None, device_id)))

    @tool("list_packages", description="List installed app packages on the phone.")
    async def list_packages(
        package_type: str = "all",
        device_id: str | None = None,
    ) -> str:
        return _dump_result(
            {
                "ok": True,
                "type": package_type,
                "packages": (await send("listPackages", {"type": package_type}, device_id)).get(
                    "packages"
                ),
            }
        )

    @tool("launch", description="Launch an Android app by package name.")
    async def launch(package: str, device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("launch", {"package": package}, device_id))
        )

    @tool("tap", description="Tap a screen coordinate in pixels.")
    async def tap(x: int, y: int, device_id: str | None = None) -> str:
        return _dump_result(_summarize_result(await send("tap", {"x": x, "y": y}, device_id)))

    @tool("type", description="Type text into the currently focused input box.")
    async def type_text(text: str, device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("type", {"text": text}, device_id))
        )

    @tool("swipe", description="Swipe from one coordinate to another.")
    async def swipe(
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        device_id: str | None = None,
    ) -> str:
        return _dump_result(
            _summarize_result(
                await send(
                    "swipe",
                    {"startX": start_x, "startY": start_y, "endX": end_x, "endY": end_y},
                    device_id,
                )
            )
        )

    @tool("long_press", description="Long press a screen coordinate.")
    async def long_press(x: int, y: int, device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("longPress", {"x": x, "y": y}, device_id))
        )

    @tool("double_tap", description="Double tap a screen coordinate.")
    async def double_tap(x: int, y: int, device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("doubleTap", {"x": x, "y": y}, device_id))
        )

    @tool("back", description="Go back once on the phone.")
    async def back(device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("keyevent", {"keyevent": 4}, device_id))
        )

    @tool("home", description="Return to the home screen.")
    async def home(device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("keyevent", {"keyevent": 3}, device_id))
        )

    @tool("keyevent", description="Send a raw Android keyevent code.")
    async def keyevent(keyevent: int, device_id: str | None = None) -> str:
        return _dump_result(
            _summarize_result(await send("keyevent", {"keyevent": keyevent}, device_id))
        )

    @tool("wait", description="Wait for a number of seconds so the page can finish loading.")
    async def wait(duration: float, device_id: str | None = None) -> str:
        await asyncio.sleep(max(duration, 0))
        return _dump_result(_summarize_result(await send("observe", None, device_id)))

    @tool(
        "interact",
        description="Ask the user to choose one of several reasonable next actions.",
        return_direct=True,
    )
    async def interact(message: str, device_id: str | None = None) -> str:
        await send("interact", {"message": message}, device_id)
        return message

    @tool(
        "take_over",
        description="Hand control back to the user when the user must operate the phone directly.",
        return_direct=True,
    )
    async def take_over(message: str, device_id: str | None = None) -> str:
        await send("interact", {"message": message}, device_id)
        return message

    @tool(
        "finish",
        description="End the current task after the phone operation is complete.",
        return_direct=True,
    )
    async def finish(message: str, device_id: str | None = None) -> str:
        return message

    return [
        observe,
        list_packages,
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
        finish,
    ]
