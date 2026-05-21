from __future__ import annotations

from typing import Annotated

from langchain.agents.middleware.types import AgentState, PrivateStateAttr
from langchain_core.messages import HumanMessage
from typing_extensions import NotRequired, TypedDict

from ..gateways.phone import ConnectedDeviceSession

STATE_MESSAGE_PREFIX = "[PHONE_STATE]"


class PhoneSnapshot(TypedDict):
    width: int
    height: int
    screenshot: str | None
    ui: str | None
    current_package: str | None
    activity: str | None


class MobileAgentState(AgentState[object], total=False):
    phone_snapshot: NotRequired[Annotated[PhoneSnapshot | None, PrivateStateAttr]]
    task_complexity_emitted: NotRequired[Annotated[bool, PrivateStateAttr]]


def build_phone_snapshot(session: ConnectedDeviceSession) -> PhoneSnapshot:
    if session.device_info is None:
        raise RuntimeError("Device session has no device_info yet.")

    return {
        "width": session.device_info.width,
        "height": session.device_info.height,
        "screenshot": session.device_info.screenshot,
        "ui": session.device_info.ui,
        "current_package": session.device_info.current_package,
        "activity": session.device_info.activity,
    }


def build_phone_state_message(snapshot: PhoneSnapshot) -> HumanMessage:
    content: list[str | dict[str, object]] = [
        {
            "type": "text",
            "text": (
                f"{STATE_MESSAGE_PREFIX}\n"
                "当前手机页面状态如下，请基于这些信息决定下一步：\n"
                f"screenWidth={snapshot['width']}\n"
                f"screenHeight={snapshot['height']}\n"
                f"currentPackage={snapshot['current_package']}\n"
                f"activity={snapshot['activity']}\n"
                f"ui={snapshot['ui']}"
            ),
        }
    ]

    screenshot = snapshot["screenshot"]
    if screenshot:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot}"},
            }
        )

    return HumanMessage(content=content)
