from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone, tzinfo
import os
from typing import Any, TypeAlias, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
NowProvider: TypeAlias = Callable[[], datetime]
CURRENT_TIME_MESSAGE_MARKER = "mobile_agent_current_time"
DEFAULT_TIMEZONE = "Asia/Shanghai"
FALLBACK_TIMEZONE = timezone(timedelta(hours=8), DEFAULT_TIMEZONE)


def _agent_timezone() -> tzinfo:
    timezone_name = os.getenv("AGENT_TIMEZONE", DEFAULT_TIMEZONE)
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        if timezone_name != DEFAULT_TIMEZONE:
            try:
                return ZoneInfo(DEFAULT_TIMEZONE)
            except ZoneInfoNotFoundError:
                pass
        return FALLBACK_TIMEZONE


def _default_now() -> datetime:
    return datetime.now(_agent_timezone())


def build_current_time_message(now: datetime) -> SystemMessage:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("build_current_time_message requires a timezone-aware datetime")
    timezone_name = getattr(now.tzinfo, "key", now.tzname()) or "unknown"
    unix_millis = int(now.timestamp() * 1000)
    return SystemMessage(
        content=(
            "当前时间: "
            f"{now.isoformat()}\n"
            f"UnixMillis: {unix_millis}\n"
            f"Timezone: {timezone_name}\n"
            "请使用此时间解析相对时间表达，例如今天、明天、昨天、稍后、本周。"
        ),
        additional_kwargs={CURRENT_TIME_MESSAGE_MARKER: True},
    )


def _is_routed_system_prompt(message: Any) -> bool:
    return (
        isinstance(message, SystemMessage)
        and message.content in {SYSTEM_PROMPT, LOCAL_MODEL_SYSTEM_PROMPT}
    )


def _is_current_time_message(message: Any) -> bool:
    return (
        isinstance(message, SystemMessage)
        and message.additional_kwargs.get(CURRENT_TIME_MESSAGE_MARKER) is True
    )


def _without_leading_routed_messages(messages: list[Any]) -> list[Any]:
    remaining = list(messages)
    while remaining and (
        _is_routed_system_prompt(remaining[0]) or _is_current_time_message(remaining[0])
    ):
        remaining = remaining[1:]
    return remaining


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

    def __init__(self, now_provider: NowProvider = _default_now) -> None:
        self.now_provider = now_provider

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
                build_current_time_message(self.now_provider()),
                *_without_leading_routed_messages(request.messages),
            ],
        )
