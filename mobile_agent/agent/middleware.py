from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone, tzinfo
import os
from typing import Any, TypeAlias, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime
from langgraph.types import Command

from ..gateways.phone import DeviceGateway
from ..local_model_runtime import model_runtime
from ..prompt_assets import LOCAL_MODEL_SYSTEM_PROMPT, SYSTEM_PROMPT
from ..progress import emit_task_complexity
from .state import (
    MobileAgentState,
    PhoneSnapshot,
    build_phone_snapshot,
    build_phone_state_message,
)
from .task_complexity import classify_task_complexity

ModelHandler: TypeAlias = Callable[[ModelRequest[Any]], ModelResponse[Any]]
AsyncModelHandler: TypeAlias = Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]]
ToolResult: TypeAlias = ToolMessage | Command[Any]
ToolHandler: TypeAlias = Callable[[ToolCallRequest], ToolResult]
AsyncToolHandler: TypeAlias = Callable[[ToolCallRequest], Awaitable[ToolResult]]
NowProvider: TypeAlias = Callable[[], datetime]
CURRENT_TIME_MESSAGE_MARKER = "mobile_agent_current_time"
DEFAULT_TIMEZONE = "Asia/Shanghai"
FALLBACK_TIMEZONE = timezone(timedelta(hours=8), DEFAULT_TIMEZONE)
PHONE_DELEGATION_TOOL = "execute_phone_todo"
ALWAYS_BLOCKED_MAIN_TOOLS = frozenset(
    {
        "task",
        "execute",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
    }
)
LOCAL_ALLOWED_PHONE_TOOLS = frozenset(
    {
        "observe",
        "tap",
        "back",
        "home",
        "wait",
        "interact",
        "take_over",
    }
)


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


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_content_to_text(message.content)
    return ""


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


class TaskComplexityMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    state_schema = MobileAgentState

    def before_model(
        self,
        state: MobileAgentState,
        runtime: Runtime[None],
    ) -> dict[str, bool] | None:
        if state.get("task_complexity_emitted") is True:
            return None

        text = _latest_human_text(cast(list[BaseMessage], state.get("messages", [])))
        result = classify_task_complexity(text)
        emit_task_complexity(
            complexity=result.complexity,
            track_steps=result.track_steps,
            reason=result.reason,
        )
        return {"task_complexity_emitted": True}

    async def abefore_model(
        self,
        state: MobileAgentState,
        runtime: Runtime[None],
    ) -> dict[str, bool] | None:
        return self.before_model(state, runtime)


class SyncPhoneStateMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    state_schema = MobileAgentState

    def __init__(self, phone_gateway: DeviceGateway, device_id: str | None = None) -> None:
        self.phone_gateway = phone_gateway
        self.device_id = device_id

    def before_model(
        self,
        state: MobileAgentState,
        runtime: Runtime[None],
    ) -> dict[str, PhoneSnapshot | None] | None:
        snapshot = self._current_snapshot(state)
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
            snapshot = self._current_snapshot(request.state)
        if snapshot is None:
            return request

        return request.override(
            messages=[
                *request.messages,
                build_phone_state_message(snapshot),
            ],
        )

    def _current_snapshot(self, state: Mapping[str, Any] | None = None) -> PhoneSnapshot | None:
        device_id = self.device_id
        if device_id is None and state is not None:
            value = state.get("device_id") or state.get("deviceId")
            device_id = value if isinstance(value, str) and value else None
        try:
            session = (
                self.phone_gateway.get_session(device_id)
                if device_id is not None
                else self.phone_gateway.get_session()
            )
        except Exception:
            return None

        if session.device_info is None:
            return None
        return build_phone_snapshot(session, device_id=device_id)


class ModeToolAccessMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    state_schema = MobileAgentState

    def __init__(
        self,
        phone_tool_names: set[str],
        device_scoped_tool_names: set[str] | None = None,
    ) -> None:
        self.phone_tool_names = frozenset(phone_tool_names)
        self.device_scoped_tool_names = frozenset(
            device_scoped_tool_names or phone_tool_names
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: ModelHandler,
    ) -> ModelResponse[Any]:
        return handler(self._with_visible_tools(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: AsyncModelHandler,
    ) -> ModelResponse[Any]:
        return await handler(self._with_visible_tools(request))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolResult:
        request = self._with_bound_device_id(request)
        if self._is_blocked(request.tool_call["name"]) or self._exceeds_local_phone_limit(
            request
        ):
            return self._blocked_message(request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
    ) -> ToolResult:
        request = self._with_bound_device_id(request)
        if self._is_blocked(request.tool_call["name"]) or self._exceeds_local_phone_limit(
            request
        ):
            return self._blocked_message(request)
        return await handler(request)

    def _with_bound_device_id(self, request: ToolCallRequest) -> ToolCallRequest:
        if request.tool_call["name"] not in self.device_scoped_tool_names:
            return request
        state = request.state if isinstance(request.state, Mapping) else {}
        value = state.get("device_id") or state.get("deviceId")
        if not isinstance(value, str) or not value:
            return request
        tool_call = {
            **request.tool_call,
            "args": {**request.tool_call["args"], "device_id": value},
        }
        return request.override(tool_call=tool_call)

    def _with_visible_tools(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        return request.override(
            tools=[
                tool
                for tool in request.tools
                if self._is_visible(tool)
            ]
        )

    def _is_visible(self, tool: Any) -> bool:
        if isinstance(tool, dict):
            tool_name = tool.get("name")
        else:
            tool_name = tool.name
        return not isinstance(tool_name, str) or not self._is_blocked(tool_name)

    def _is_blocked(self, tool_name: str) -> bool:
        if tool_name in ALWAYS_BLOCKED_MAIN_TOOLS:
            return True
        if model_runtime.status().get("mode") == "local":
            return tool_name == PHONE_DELEGATION_TOOL or (
                tool_name in self.phone_tool_names
                and tool_name not in LOCAL_ALLOWED_PHONE_TOOLS
            )
        return tool_name in self.phone_tool_names

    def _exceeds_local_phone_limit(self, request: ToolCallRequest) -> bool:
        tool_name = request.tool_call["name"]
        if (
            model_runtime.status().get("mode") != "local"
            or tool_name not in self.phone_tool_names
        ):
            return False

        messages = request.state.get("messages", ())
        if any(
            isinstance(message, ToolMessage) and message.name in self.phone_tool_names
            for message in messages
        ):
            return True

        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return (
                    sum(
                        call["name"] in self.phone_tool_names
                        for call in message.tool_calls
                    )
                    > 1
                )
        return False

    def _blocked_message(self, request: ToolCallRequest) -> ToolMessage:
        tool_name = request.tool_call["name"]
        mode = model_runtime.status().get("mode", "unknown")
        return ToolMessage(
            content=f"Tool {tool_name!r} is unavailable in {mode} mode.",
            tool_call_id=request.tool_call["id"],
            name=tool_name,
            status="error",
        )


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
