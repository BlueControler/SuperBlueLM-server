from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from typing import Any

from websockets.asyncio.server import serve

from custom_deep_agent import build_agent, build_observation_message
from phone_gateway import DeviceGateway, DeviceGatewayError


async def agent_console_loop(gateway: DeviceGateway, stop_event: asyncio.Event) -> None:
    print("[system] 输入自然语言任务并回车，Agent 会通过 websocket tools 操作已连接手机")
    print("[system] 输入 /quit 关闭服务")

    conversation_messages: list[dict[str, Any]] = []
    agent = build_agent(gateway)

    while not stop_event.is_set():
        text = await asyncio.to_thread(input, "agent> ")
        text = text.strip()

        if not text:
            continue
        if text == "/quit":
            stop_event.set()
            break

        try:
            session = gateway.get_default_device()
        except DeviceGatewayError as exc:
            print(f"[system] {exc}")
            continue

        conversation_messages.append(build_observation_message(session, text))
        result = await agent.ainvoke({"messages": conversation_messages})
        conversation_messages, final_message = _consume_agent_result(conversation_messages, result)
        if final_message:
            print(f"[assistant] {final_message}")


def _consume_agent_result(
    conversation_messages: list[dict[str, Any]],
    result: object,
) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(result, dict) and "messages" in result:
        messages = list(result["messages"])
        return messages, _extract_last_text(messages)

    if isinstance(result, str):
        return conversation_messages, result

    return conversation_messages, str(result) if result is not None else None


def _extract_last_text(messages: list[object]) -> str | None:
    for message in reversed(messages):
        role = _get_field(message, "role")
        message_type = _get_field(message, "type")
        if role not in {"assistant", "tool"} and message_type not in {"ai", "tool"}:
            continue

        content = _get_field(message, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            text = "\n".join(part for part in text_parts if part)
            if text:
                return text
    return None


def _get_field(message: object, field: str) -> object:
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


async def run_server(host: str, port: int) -> None:
    gateway = DeviceGateway()
    stop_event = asyncio.Event()

    async with serve(gateway.handler, host, port):
        print(f"[system] websocket server listening at ws://{host}:{port}/ws/devices/{{deviceId}}")
        print("[system] waiting for mobile client connections...")
        input_task = asyncio.create_task(agent_console_loop(gateway, stop_event))

        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            print("\n[system] received Ctrl+C, shutting down...")
        finally:
            input_task.cancel()
            with suppress(asyncio.CancelledError):
                await input_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Agent websocket server for mobile control")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=8765, help="Listen port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\n[system] server exited")


if __name__ == "__main__":
    main()
