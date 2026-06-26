from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from langchain.agents.middleware.types import AgentState, PrivateStateAttr
from langchain_core.messages import HumanMessage
from typing_extensions import NotRequired, TypedDict

from ..gateways.phone import ConnectedDeviceSession

STATE_MESSAGE_PREFIX = "[PHONE_STATE]"


class PhoneSnapshot(TypedDict):
    device_id: str | None
    width: int
    height: int
    screenshot: str | None
    screenshot_mime_type: str
    ui: str | None
    current_package: str | None
    activity: str | None


class PhoneTodoStep(TypedDict):
    index: int
    progressKey: str
    name: str
    status: Literal["completed", "failed"]
    summary: str


class MobileAgentState(AgentState[object], total=False):
    device_id: NotRequired[str]
    deviceId: NotRequired[str]
    phone_snapshot: NotRequired[Annotated[PhoneSnapshot | None, PrivateStateAttr]]
    task_complexity_emitted: NotRequired[Annotated[bool, PrivateStateAttr]]
    phone_todo_steps: NotRequired[
        Annotated[tuple[PhoneTodoStep, ...], PrivateStateAttr]
    ]
    awaiting_user_action: NotRequired[Annotated[bool, PrivateStateAttr]]
    awaiting_user_reason: NotRequired[Annotated[str, PrivateStateAttr]]
    run_failure_reason: NotRequired[Annotated[str, PrivateStateAttr]]
    trace_run_id: NotRequired[Annotated[str, PrivateStateAttr]]
    trace_analysis_step_id: NotRequired[Annotated[str, PrivateStateAttr]]


def device_id_from_mapping(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    device_id = value.get("device_id") or value.get("deviceId")
    return device_id if isinstance(device_id, str) and device_id else None


def build_phone_snapshot(
    session: ConnectedDeviceSession,
    device_id: str | None = None,
) -> PhoneSnapshot:
    if session.device_info is None:
        raise RuntimeError("Device session has no device_info yet.")

    return {
        "device_id": device_id,
        "width": session.device_info.width,
        "height": session.device_info.height,
        "screenshot": session.device_info.screenshot,
        "screenshot_mime_type": getattr(
            session.device_info, "screenshot_mime_type", "image/png"
        ),
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
                f"deviceId={snapshot['device_id']}\n"
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
                "image_url": {
                    "url": f"data:{snapshot['screenshot_mime_type']};base64,{screenshot}"
                },
            }
        )

    return HumanMessage(content=content)
