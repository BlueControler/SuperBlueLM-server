from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from ..gateways.phone import DeviceGateway
from ..local_model_runtime import model_runtime
from ..prompt_assets import LOCAL_MODEL_SYSTEM_PROMPT, SYSTEM_PROMPT
from .state import (
    MobileAgentState,
    PhoneSnapshot,
    build_phone_snapshot,
    build_phone_state_message,
)

ModelHandler: TypeAlias = Callable[[ModelRequest[Any]], ModelResponse[Any]]
AsyncModelHandler: TypeAlias = Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]]


class SyncPhoneStateMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    state_schema = MobileAgentState

    def __init__(self, phone_gateway: DeviceGateway) -> None:
        self.phone_gateway = phone_gateway

    def before_model(
        self,
        state: MobileAgentState,
        runtime: Runtime[None],
    ) -> dict[str, PhoneSnapshot | None] | None:
        snapshot = self._current_snapshot()
        if state.get("phone_snapshot") == snapshot:
            return None
        return {"phone_snapshot": snapshot}

    async def abefore_model(
        self,
        state: MobileAgentState,
        runtime: Runtime[None],
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

    def _with_phone_state_message(
        self,
        request: ModelRequest[Any],
    ) -> ModelRequest[Any]:
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


class RouteModelMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
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


class RoutedSystemPromptMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
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

    def _with_routed_system_prompt(
        self,
        request: ModelRequest[Any],
    ) -> ModelRequest[Any]:
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
