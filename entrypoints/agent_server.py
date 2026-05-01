from __future__ import annotations

import argparse
import asyncio
import ssl
from contextlib import suppress
from typing import Any, TypeAlias, cast

from langchain_core.messages import BaseMessage
from loguru import logger
from websockets.asyncio.server import serve

from mobile_agent.custom_deep_agent import (
    build_agent,
    build_user_message,
)
from mobile_agent.phone_gateway import ConnectedDeviceSession, DeviceGateway, DeviceGatewayError
from mobile_agent.system_gateway import SystemToolGateway

MessagePayload: TypeAlias = dict[str, Any]
MessageBlock: TypeAlias = MessagePayload | str
MessageContent: TypeAlias = str | list[MessageBlock] | None
AgentMessage: TypeAlias = BaseMessage | MessagePayload
AgentMessages: TypeAlias = list[AgentMessage]
InvokeResult: TypeAlias = dict[str, AgentMessages] | str | None


async def agent_console_loop(
    phone_gateway: DeviceGateway,
    system_gateway: SystemToolGateway,
    stop_event: asyncio.Event,
) -> None:
    logger.info("输入自然语言任务并回车，Agent 会通过 websocket tools 操作已连接手机")
    logger.info("输入 /quit 关闭服务")

    conversation_messages: AgentMessages = []
    agent = build_agent(phone_gateway, system_gateway)

    while not stop_event.is_set():
        text = await asyncio.to_thread(input, "agent> ")
        text = text.strip()

        if not text:
            continue
        if text == "/quit":
            stop_event.set()
            break

        try:
            session = phone_gateway.get_session()
        except DeviceGatewayError as exc:
            logger.warning(str(exc))
            continue

        await _ensure_initial_observation(session)
        conversation_messages.append(build_user_message(text))
        try:
            result = cast(
                InvokeResult, await agent.ainvoke(cast(Any, {"messages": conversation_messages}))
            )
        except Exception as exc:
            logger.error(f"模型调用失败：{_format_model_error(exc)}")
            continue
        conversation_messages, final_message = _consume_agent_result(conversation_messages, result)
        if final_message:
            logger.info(final_message)


async def _ensure_initial_observation(session: ConnectedDeviceSession) -> None:
    device_info = session.device_info
    if device_info is None:
        return

    if device_info.screenshot:
        return

    try:
        await session.send_command("observe", None)
        logger.info("首轮缺少截图，已自动补一次 observe")
    except Exception as exc:
        logger.warning(f"自动补 observe 失败：{exc}")


def _consume_agent_result(
    conversation_messages: AgentMessages,
    result: InvokeResult,
) -> tuple[AgentMessages, str | None]:
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
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


def _extract_last_text(messages: AgentMessages) -> str | None:
    for message in reversed(messages):
        role, message_type, content, _ = _split_message_fields(message)
        if role not in {"assistant", "tool"} and message_type not in {"ai", "tool"}:
            continue

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


def _split_message_fields(
    message: AgentMessage,
) -> tuple[str | None, str | None, MessageContent, str | None]:
    if isinstance(message, dict):
        role = message.get("role")
        message_type = message.get("type")
        content = message.get("content")
        name = message.get("name")
        return (
            role if isinstance(role, str) else None,
            message_type if isinstance(message_type, str) else None,
            cast(MessageContent, content),
            name if isinstance(name, str) else None,
        )

    if isinstance(message, BaseMessage):
        dumped = message.model_dump()
        role = dumped.get("role")
        message_type = dumped.get("type")
        content = dumped.get("content")
        name = dumped.get("name")
        return (
            role if isinstance(role, str) else None,
            message_type if isinstance(message_type, str) else None,
            cast(MessageContent, content),
            name if isinstance(name, str) else None,
        )

    return None, None, None, None


def _fallback_result_text(messages: AgentMessages) -> str:
    if not messages:
        return "这一轮没有返回任何消息。"

    last_message = messages[-1]
    role, message_type, _, name = _split_message_fields(last_message)

    if message_type == "tool" or role == "tool":
        tool_name = name or "unknown_tool"
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
    phone_gateway = DeviceGateway()
    system_gateway = SystemToolGateway()
    stop_event = asyncio.Event()
    ssl_context = _build_ssl_context(certfile, keyfile)
    scheme = "wss" if ssl_context else "ws"

    async def websocket_handler(websocket) -> None:
        request = websocket.request
        path = request.path.split("?", 1)[0] if request is not None else ""
        if path == system_gateway.path:
            await system_gateway.handler(websocket)
            return
        await phone_gateway.handler(websocket)

    async with serve(websocket_handler, host, port, ssl=ssl_context):
        logger.info(f"websocket server listening at {scheme}://{host}:{port}/adb")
        logger.info(f"system tool websocket listening at {scheme}://{host}:{port}/system")
        logger.info("waiting for mobile client connections...")
        input_task = asyncio.create_task(
            agent_console_loop(phone_gateway, system_gateway, stop_event)
        )

        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            logger.warning("received Ctrl+C, shutting down...")
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
        logger.warning("server exited")


if __name__ == "__main__":
    main()
