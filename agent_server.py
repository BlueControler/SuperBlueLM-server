from __future__ import annotations

import argparse
import asyncio
import ssl
from contextlib import suppress
from typing import Any

from websockets.asyncio.server import serve

from custom_deep_agent import build_agent, build_state_snapshot_message, build_user_message
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

        await _ensure_initial_observation(session)
        conversation_messages.append(build_user_message(text))
        conversation_messages.append(build_state_snapshot_message(session))
        try:
            result = await agent.ainvoke({"messages": conversation_messages})
        except Exception as exc:
            print(f"[assistant] 模型调用失败：{_format_model_error(exc)}")
            continue
        conversation_messages, final_message = _consume_agent_result(conversation_messages, result)
        if final_message:
            print(f"[assistant] {final_message}")


async def _ensure_initial_observation(session) -> None:
    device_info = session.device_info
    if device_info is None:
        return

    if device_info.screenshot:
        return

    try:
        await session.send_command("observe", None)
        print(f"[system] 首轮缺少截图，已为设备 {session.device_id} 自动补一次 observe")
    except Exception as exc:
        print(f"[system] 自动补 observe 失败：{exc}")


def _consume_agent_result(
    conversation_messages: list[dict[str, Any]],
    result: object,
) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(result, dict) and "messages" in result:
        messages = list(result["messages"])
        final_text = _extract_last_text(messages)
        if final_text:
            return messages, final_text
        # Some providers reject follow-up turns if the previous turn ended with an
        # incomplete tool-call exchange and no assistant-facing text. In that case,
        # keep the prior stable conversation state and only surface a fallback note.
        return conversation_messages, _fallback_result_text(messages)

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


def _fallback_result_text(messages: list[object]) -> str:
    if not messages:
        return "这一轮没有返回任何消息。"

    last_message = messages[-1]
    message_type = _get_field(last_message, "type")
    role = _get_field(last_message, "role")

    if message_type == "tool" or role == "tool":
        tool_name = _get_field(last_message, "name") or "unknown_tool"
        return f"这一轮执行到了工具 `{tool_name}`，但没有生成可显示的自然语言回复。"

    return "这一轮没有生成可显示的自然语言回复，可能是模型空回复，或返回内容格式暂未被控制台识别。"


def _format_model_error(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    return f"{type(exc).__name__}"


def _build_ssl_context(certfile: str | None, keyfile: str | None) -> ssl.SSLContext | None:
    if not certfile and not keyfile:
        return None
    if not certfile or not keyfile:
        raise ValueError("Both --certfile and --keyfile are required to enable TLS.")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return context


async def run_server(host: str, port: int, certfile: str | None, keyfile: str | None) -> None:
    gateway = DeviceGateway()
    stop_event = asyncio.Event()
    ssl_context = _build_ssl_context(certfile, keyfile)
    scheme = "wss" if ssl_context else "ws"

    async with serve(gateway.handler, host, port, ssl=ssl_context):
        print(f"[system] websocket server listening at {scheme}://{host}:{port}/ws/devices/{{deviceId}}")
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
    parser.add_argument("--certfile", default=None, help="TLS certificate path for wss")
    parser.add_argument("--keyfile", default=None, help="TLS private key path for wss")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_server(args.host, args.port, args.certfile, args.keyfile))
    except KeyboardInterrupt:
        print("\n[system] server exited")


if __name__ == "__main__":
    main()
