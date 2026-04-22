from __future__ import annotations

import asyncio
import json

from websockets.asyncio.server import serve

from custom_deep_agent import _remove_old_images_from_messages
from mock_portal_client import MockPortalClient
from phone_gateway import DeviceGateway
from phone_tools import create_phone_tools


def _tool_map(gateway: DeviceGateway) -> dict[str, object]:
    tools = create_phone_tools(gateway)
    return {tool.name: tool for tool in tools}


async def _call_tool(tool_map: dict[str, object], name: str, **kwargs) -> dict:
    tool = tool_map[name]
    result = await tool.ainvoke(kwargs)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"message": result}
    return result


def _demo_remove_old_images() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "旧页面"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,old-image"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "新页面"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,new-image"}},
            ],
        },
    ]
    filtered_messages = _remove_old_images_from_messages(messages)
    print("[fake-agent] before_model remove_old_images demo:")
    print(f"  input messages: {len(messages)}")
    if filtered_messages == messages:
        print("  no changes")
    else:
        print(f"  output messages: {len(filtered_messages)}")


async def run_fake_agent(task: str) -> None:
    host = "127.0.0.1"
    port = 8771
    gateway = DeviceGateway()

    async with serve(gateway.handler, host, port):
        client_task = asyncio.create_task(
            MockPortalClient(device_id="demo-device").run(url=f"ws://{host}:{port}/adb")
        )
        try:
            session = await gateway.wait_for_device("default", timeout=5.0)
            print(f"[fake-agent] connected device: {session.device_id}")
            _demo_remove_old_images()

            tools = _tool_map(gateway)

            print(f"[fake-agent] task: {task}")
            observe_result = await _call_tool(tools, "observe")
            print(f"[fake-agent] observe -> {observe_result}")

            lowered = task.lower()
            if "设置" in task or "setting" in lowered:
                launch_result = await _call_tool(
                    tools, "launch", package="com.android.settings"
                )
                print(f"[fake-agent] launch -> {launch_result}")

            if "输入" in task or "type" in lowered:
                type_result = await _call_tool(
                    tools, "type", text="hello from fake agent"
                )
                print(f"[fake-agent] type -> {type_result}")
            else:
                tap_result = await _call_tool(tools, "tap", x=320, y=640)
                print(f"[fake-agent] tap -> {tap_result}")

            finish_result = await _call_tool(
                tools, "finish", message="fake agent demo finished"
            )
            print(f"[fake-agent] finish -> {finish_result}")
            print("[fake-agent] success: no real model key was needed.")
        finally:
            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass


def main() -> None:
    asyncio.run(run_fake_agent("打开设置，然后点一个按钮"))


if __name__ == "__main__":
    main()
