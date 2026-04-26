from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TypeAlias, cast

from deepagents import create_deep_agent
from langchain.agents.middleware.types import AgentState, StateT, before_model
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from dotenv import load_dotenv
from pydantic import SecretStr

from .phone_gateway import ConnectedDeviceSession, DeviceGateway
from .phone_tools import create_phone_tools
from .prompt_assets import SYSTEM_PROMPT, SYSTEM_TOOL_PROMPT, TOOL_PROMPT
from .system_gateway import SystemToolGateway
from .system_tools import create_system_tools

STATE_MESSAGE_PREFIX = "[PHONE_STATE]"

MessagePayload: TypeAlias = dict[str, object]
MessageBlock: TypeAlias = MessagePayload | str
MessageContent: TypeAlias = str | list[MessageBlock] | None
AgentMessage: TypeAlias = BaseMessage | MessagePayload
AgentMessages: TypeAlias = list[AgentMessage]
MiddlewarePatch: TypeAlias = dict[str, list[RemoveMessage | AgentMessage]]


def _message_content(message: AgentMessage) -> MessageContent:
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return cast(list[MessageBlock], content)
        return None
    if isinstance(message, BaseMessage):
        return message.content
    return None


def _set_message_content(message: AgentMessage, content: MessageContent) -> AgentMessage:
    if isinstance(message, dict):
        new_message = dict(message)
        new_message["content"] = content
        return new_message

    if isinstance(message, BaseMessage):
        return message.model_copy(update={"content": content})

    raise TypeError(f"Unsupported message type: {type(message)!r}")


def _is_image_block(block: MessageBlock) -> bool:
    if not isinstance(block, dict):
        return False

    block_type = block.get("type")
    return (
        block_type
        in {
            "image",
            "image_url",
            "input_image",
            "image_base64",
        }
        or "image_url" in block
    )


def _find_latest_image_position(messages: Sequence[AgentMessage]) -> tuple[int, int] | None:
    for message_index in range(len(messages) - 1, -1, -1):
        content = _message_content(messages[message_index])
        if not isinstance(content, list):
            continue

        for block_index in range(len(content) - 1, -1, -1):
            if _is_image_block(content[block_index]):
                return message_index, block_index
    return None


def _remove_old_images_from_messages(messages: Sequence[AgentMessage]) -> AgentMessages:
    latest_image_position = _find_latest_image_position(messages)
    if latest_image_position is None:
        return list(messages)

    latest_message_index, latest_block_index = latest_image_position
    filtered_messages: AgentMessages = []

    for message_index, message in enumerate(messages):
        content = _message_content(message)
        if not isinstance(content, list):
            filtered_messages.append(message)
            continue

        new_content: list[MessageBlock] = []
        for block_index, block in enumerate(content):
            if not _is_image_block(block):
                new_content.append(block)
                continue

            if message_index == latest_message_index and block_index == latest_block_index:
                new_content.append(block)

        if new_content:
            filtered_messages.append(_set_message_content(message, new_content))
            continue

        # 历史消息如果只剩旧图片，直接删除，避免把空消息送给模型。
        if any(_is_image_block(block) for block in content):
            continue

        filtered_messages.append(message)

    return filtered_messages


def _is_phone_state_message(message: AgentMessage) -> bool:
    content = _message_content(message)
    if not isinstance(content, list) or not content:
        return False

    first = content[0]
    if not isinstance(first, dict):
        return False

    first_type = first.get("type")
    first_text = first.get("text")
    return (
        first_type == "text"
        and isinstance(first_text, str)
        and first_text.startswith(STATE_MESSAGE_PREFIX)
    )


def _replace_phone_state_message(
    messages: Sequence[AgentMessage], session: ConnectedDeviceSession
) -> AgentMessages:
    filtered_messages = [m for m in messages if not _is_phone_state_message(m)]
    filtered_messages.append(build_state_snapshot_message(session))
    return filtered_messages


@before_model
def remove_old_images(
    state: AgentState[StateT],
    runtime: Runtime,
) -> MiddlewarePatch | None:
    messages = list(state["messages"])
    filtered_messages = _remove_old_images_from_messages(messages)

    if filtered_messages == messages:
        return None

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *filtered_messages,
        ],
    }


def make_sync_phone_state_middleware(gateway: DeviceGateway):
    @before_model
    def sync_phone_state(
        state: AgentState[StateT],
        runtime: Runtime,
    ) -> MiddlewarePatch | None:
        try:
            session = gateway.get_session()
        except Exception:
            return None

        messages = list(state["messages"])
        updated_messages = _replace_phone_state_message(messages, session)
        if updated_messages == messages:
            return None

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *updated_messages,
            ],
        }

    return sync_phone_state


def build_agent(gateway: DeviceGateway, system_gateway: SystemToolGateway | None = None):
    model = _build_model()
    tools = list(create_phone_tools(gateway))
    prompts = [SYSTEM_PROMPT, TOOL_PROMPT]
    if system_gateway is not None:
        tools.extend(create_system_tools(system_gateway))
        prompts.append(SYSTEM_TOOL_PROMPT)

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt="\n\n".join(prompts),
        middleware=[remove_old_images, make_sync_phone_state_middleware(gateway)],
    )


def _build_model():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
    openai_base_url = os.getenv("OPENAI_BASE_URL") or None
    if openai_key:
        return ChatOpenAI(
            api_key=SecretStr(openai_key),
            base_url=openai_base_url,
            model=openai_model,
            max_tokens=openai_max_tokens,
        )

    return "openai:gpt-5.4"


def build_user_message(user_text: str) -> HumanMessage:
    return HumanMessage(content=user_text)


def build_state_snapshot_message(session: ConnectedDeviceSession) -> HumanMessage:
    if session.device_info is None:
        raise RuntimeError("Device session has no device_info yet.")

    content: list[str | dict[str, object]] = [
        {
            "type": "text",
            "text": (
                f"{STATE_MESSAGE_PREFIX}\n"
                "当前手机页面状态如下，请基于这些信息决定下一步：\n"
                f"screenWidth={session.device_info.width}\n"
                f"screenHeight={session.device_info.height}\n"
                f"currentPackage={session.device_info.current_package}\n"
                f"activity={session.device_info.activity}\n"
                f"ui={session.device_info.ui}"
            ),
        }
    ]

    screenshot = session.device_info.screenshot
    if screenshot:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot}"},
            }
        )

    return HumanMessage(content=content)
