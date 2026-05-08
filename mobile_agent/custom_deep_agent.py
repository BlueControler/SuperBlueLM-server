from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, NotRequired, TypeAlias, cast

from deepagents import create_deep_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from .external_tools import create_external_tools
from .local_model_runtime import build_cloud_model, model_runtime
from .phone_gateway import ConnectedDeviceSession, DeviceGateway
from .phone_tools import create_phone_tools
from .prompt_assets import LOCAL_MODEL_SYSTEM_PROMPT, SYSTEM_PROMPT
from .system_gateway import SystemToolGateway
from .system_tools import create_system_tools

STATE_MESSAGE_PREFIX = "[PHONE_STATE]"

ModelHandler: TypeAlias = Callable[[ModelRequest[Any]], ModelResponse[Any]]
AsyncModelHandler: TypeAlias = Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]]


class PhoneSnapshot(TypedDict):
    width: int
    height: int
    screenshot: str | None
    ui: str | None
    current_package: str | None
    activity: str | None


class MobileAgentState(AgentState[Any], total=False):
    phone_snapshot: NotRequired[Annotated[PhoneSnapshot | None, PrivateStateAttr]]


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


class SyncPhoneStateMiddleware(AgentMiddleware[MobileAgentState, Any, Any]):
    state_schema = MobileAgentState

    def __init__(self, phone_gateway: DeviceGateway) -> None:
        self.phone_gateway = phone_gateway

    def before_model(
        self,
        state: MobileAgentState,
        runtime: Runtime[Any],
    ) -> dict[str, PhoneSnapshot | None] | None:
        snapshot = self._current_snapshot()
        if state.get("phone_snapshot") == snapshot:
            return None

        return {"phone_snapshot": snapshot}

    async def abefore_model(
        self,
        state: MobileAgentState,
        runtime: Runtime[Any],
    ) -> dict[str, PhoneSnapshot | None] | None:
        return self.before_model(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: ModelHandler,
    ) -> ModelResponse[Any]:
        return handler(self._with_phone_state_message(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: AsyncModelHandler,
    ) -> ModelResponse[Any]:
        return await handler(self._with_phone_state_message(request))

    def _with_phone_state_message(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        snapshot = cast(PhoneSnapshot | None, request.state.get("phone_snapshot"))
        if snapshot is None:
            snapshot = self._current_snapshot()
        if snapshot is None:
            return request

        return request.override(
            messages=[
                *request.messages,
                build_phone_state_message(snapshot),
            ],
        )

    def _current_snapshot(self) -> PhoneSnapshot | None:
        try:
            session = self.phone_gateway.get_session()
        except Exception:
            return None

        if session.device_info is None:
            return None

        return build_phone_snapshot(session)


class RouteModelMiddleware(AgentMiddleware[MobileAgentState, Any, Any]):
    state_schema = MobileAgentState

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: ModelHandler,
    ) -> ModelResponse[Any]:
        return handler(self._with_routed_model(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: AsyncModelHandler,
    ) -> ModelResponse[Any]:
        return await handler(self._with_routed_model(request))

    def _with_routed_model(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        local_model = model_runtime.get_model_override()
        if local_model is None:
            return request
        return request.override(model=local_model)


class RoutedSystemPromptMiddleware(AgentMiddleware[MobileAgentState, Any, Any]):
    state_schema = MobileAgentState

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: ModelHandler,
    ) -> ModelResponse[Any]:
        return handler(self._with_routed_system_prompt(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: AsyncModelHandler,
    ) -> ModelResponse[Any]:
        return await handler(self._with_routed_system_prompt(request))

    def _with_routed_system_prompt(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        prompt = (
            LOCAL_MODEL_SYSTEM_PROMPT
            if model_runtime.status().get("mode") == "local"
            else SYSTEM_PROMPT
        )

        return request.override(
            messages=[
                SystemMessage(content=prompt),
                *request.messages,
            ],
        )


def build_agent(phone_gateway: DeviceGateway, system_gateway: SystemToolGateway):
    model = _build_model()
    tools = [
        *create_phone_tools(phone_gateway),
        *create_system_tools(system_gateway),
        *create_external_tools(),
    ]

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt="",
        middleware=[
            RouteModelMiddleware(),
            RoutedSystemPromptMiddleware(),
            SyncPhoneStateMiddleware(phone_gateway),
        ],  # pyright: ignore[reportArgumentType]
    )


def _build_model():
    return build_cloud_model()


def build_user_message(user_text: str) -> HumanMessage:
    return HumanMessage(content=user_text)


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
