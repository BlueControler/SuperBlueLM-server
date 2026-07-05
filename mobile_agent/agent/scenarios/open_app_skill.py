from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ...gateways.phone import DeviceGateway, DeviceGatewayError, DeviceNotConnectedError
from ...gateways.system import SystemGatewayError, SystemToolGateway
from ...json_types import JsonObject, to_json_value
from ...progress import emit_task_complexity, emit_task_progress
from ..middleware import _message_content_to_text
from ..state import MobileAgentState, device_id_from_mapping

OPEN_APP_PHASE = "open_app_skill"
OPEN_APP_TASK_TITLE = "打开应用"
TOTAL_STEPS = 4

OPEN_APP_VERBS = ("帮我打开", "打开", "启动", "进入", "launch", "open")
BUILTIN_APPS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "微信": ("微信", "com.tencent.mm", ("微信", "wechat")),
    "设置": ("设置", "com.android.settings", ("设置", "系统设置", "settings")),
    "高德地图": ("高德地图", "com.autonavi.minimap", ("高德地图", "高德", "amap")),
    "Chrome": ("Chrome", "com.android.chrome", ("chrome", "谷歌浏览器")),
}


@dataclass(frozen=True)
class OpenAppIntent:
    should_handle: bool
    target_app: str


@dataclass(frozen=True)
class ResolvedApp:
    app_label: str
    package_name: str
    source: str


class OpenAppSkillMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    state_schema = MobileAgentState

    def __init__(
        self,
        phone_gateway: DeviceGateway,
        system_gateway: SystemToolGateway,
    ) -> None:
        self.runner = OpenAppRunner(phone_gateway, system_gateway)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        intent = _intent_from_request(request.messages)
        if not intent.should_handle:
            return handler(request)
        result = asyncio.run(self.runner.run(intent, _request_device_id(request)))
        return _model_response(result)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        intent = _intent_from_request(request.messages)
        if not intent.should_handle:
            return await handler(request)
        result = await self.runner.run(intent, _request_device_id(request))
        return _model_response(result)


@dataclass(frozen=True)
class OpenAppRunner:
    phone_gateway: DeviceGateway
    system_gateway: SystemToolGateway

    async def run(self, intent: OpenAppIntent, device_id: str | None = None) -> JsonObject:
        emit_task_complexity(
            complexity="simple",
            track_steps=True,
            reason=OPEN_APP_PHASE,
            message="识别任务类型：打开指定应用",
        )
        self._emit_progress(
            step=1,
            status="running",
            step_title="识别目标应用",
            message="正在识别需要打开的应用。",
            tool_name="open_app_intent",
            completed_count=0,
        )
        self._emit_progress(
            step=2,
            status="running",
            step_title="查找应用包名",
            message=f"正在查找 {intent.target_app} 对应的应用包名。",
            tool_name="list_apps",
            completed_count=1,
        )

        resolved = await self._resolve_app(intent, device_id)
        if isinstance(resolved, str):
            return self._failed_result(intent, resolved, step=2, tool_name="list_apps")

        self._emit_progress(
            step=3,
            status="running",
            step_title="打开应用",
            message=f"正在打开 {resolved.app_label}。",
            tool_name="launch",
            completed_count=2,
        )
        try:
            session = await self.phone_gateway.wait_for_session(device_id)
            launch_result = await session.send_command(
                "launch",
                {"package": resolved.package_name},
            )
        except (DeviceGatewayError, DeviceNotConnectedError, Exception) as exc:
            return self._failed_result(intent, str(exc), step=3, tool_name="launch")

        final_message = f"已打开 {resolved.app_label}。"
        self._emit_progress(
            step=4,
            status="completed",
            step_title="完成",
            message=final_message,
            tool_name="finish",
            completed_count=4,
        )
        return cast(JsonObject, to_json_value({
            "success": True,
            "targetApp": intent.target_app,
            "appLabel": resolved.app_label,
            "packageName": resolved.package_name,
            "resolutionSource": resolved.source,
            "launchResult": launch_result,
            "finalMessage": final_message,
        }))

    async def _resolve_app(
        self,
        intent: OpenAppIntent,
        device_id: str | None,
    ) -> ResolvedApp | str:
        builtin = _builtin_app(intent.target_app)
        if builtin is not None:
            return builtin

        try:
            client = self.system_gateway.get_default_client(device_id)
            result = await client.send_request("listApps", {"type": "all"})
        except SystemGatewayError as exc:
            return _system_gateway_error_message(exc)
        except Exception as exc:
            return f"应用包名查找失败：{exc}"

        if not isinstance(result, Mapping):
            return "应用包名查找失败：系统工具返回的应用列表格式不正确。"
        match = _match_app_from_inventory(intent.target_app, result)
        if match is None:
            return f"未找到 {intent.target_app} 对应的应用包名。"
        return match

    def _failed_result(
        self,
        intent: OpenAppIntent,
        reason: str,
        *,
        step: int,
        tool_name: str,
    ) -> JsonObject:
        final_message = f"未能打开 {intent.target_app}：{reason}"
        self._emit_progress(
            step=step,
            status="failed",
            step_title="打开失败" if tool_name == "launch" else "查找应用包名",
            message=final_message,
            tool_name=tool_name,
            completed_count=max(step - 1, 0),
            error=final_message,
        )
        return {
            "success": False,
            "targetApp": intent.target_app,
            "appLabel": intent.target_app,
            "packageName": None,
            "finalMessage": final_message,
            "error": reason,
        }

    def _emit_progress(
        self,
        *,
        step: int,
        status: str,
        step_title: str,
        message: str,
        tool_name: str,
        completed_count: int,
        error: str | None = None,
    ) -> None:
        emit_task_progress(
            label=tool_name,
            status=cast(Any, status),
            phase=OPEN_APP_PHASE,
            task_title=OPEN_APP_TASK_TITLE,
            step_title=step_title,
            message=message,
            tool_name=tool_name,
            progress_key=f"open-app-{step}",
            current_step=step,
            total_steps=TOTAL_STEPS,
            completed_steps=_completed_steps(completed_count),
            error=error,
        )


def is_open_app_request(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    if not any(verb.lower() in normalized for verb in OPEN_APP_VERBS):
        return False
    return bool(_extract_target_app(text))


def parse_open_app_intent(text: str) -> OpenAppIntent:
    target_app = _extract_target_app(text)
    return OpenAppIntent(
        should_handle=bool(target_app) and is_open_app_request(text),
        target_app=target_app or "",
    )


def _extract_target_app(text: str) -> str:
    cleaned = re.sub(r"[。！？?!.，,]", " ", text).strip()
    normalized = _normalize(cleaned)
    for label, _package_name, aliases in BUILTIN_APPS.values():
        if any(alias.lower() in normalized for alias in aliases):
            return label

    pattern = r"(?:帮我打开|打开|启动|进入|launch|open)\s*(?P<target>[\w\u4e00-\u9fff. -]+)"
    match = re.search(pattern, cleaned, flags=re.IGNORECASE)
    if not match:
        return ""
    target = match.group("target").strip()
    target = re.sub(r"(应用|app)$", "", target, flags=re.IGNORECASE).strip()
    return target


def _builtin_app(target_app: str) -> ResolvedApp | None:
    normalized = _normalize(target_app)
    for label, package_name, aliases in BUILTIN_APPS.values():
        if normalized == _normalize(label) or any(alias.lower() in normalized for alias in aliases):
            return ResolvedApp(app_label=label, package_name=package_name, source="builtin")
    return None


def _match_app_from_inventory(
    target_app: str,
    apps: Mapping[object, object],
) -> ResolvedApp | None:
    normalized_target = _normalize(target_app)
    best: ResolvedApp | None = None
    best_score = 0
    for package_name, app_label in apps.items():
        if not isinstance(package_name, str) or not isinstance(app_label, str):
            continue
        label_lower = app_label.lower()
        package_lower = package_name.lower()
        score = 0
        if _normalize(app_label) == normalized_target:
            score = 100
        elif normalized_target in _normalize(app_label):
            score = 80
        elif normalized_target and normalized_target in package_lower:
            score = 60
        elif target_app.lower() in label_lower or target_app.lower() in package_lower:
            score = 60
        if score > best_score:
            best_score = score
            best = ResolvedApp(
                app_label=app_label,
                package_name=package_name,
                source="listApps",
            )
    return best if best_score >= 60 else None


def _intent_from_request(messages: Sequence[BaseMessage]) -> OpenAppIntent:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return parse_open_app_intent(_message_content_to_text(message.content))
    return OpenAppIntent(should_handle=False, target_app="")


def _request_device_id(request: ModelRequest[Any]) -> str | None:
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None)
    if isinstance(config, Mapping):
        device_id = (
            device_id_from_mapping(config.get("configurable"))
            or device_id_from_mapping(config.get("metadata"))
        )
        if device_id is not None:
            return device_id
    state = getattr(request, "state", None)
    return device_id_from_mapping(state)


def _model_response(result: JsonObject) -> ModelResponse[Any]:
    return ModelResponse(
        result=[
            AIMessage(
                content=str(result["finalMessage"]),
                additional_kwargs={"open_app": result},
            )
        ]
    )


def _system_gateway_error_message(exc: SystemGatewayError) -> str:
    message = str(exc)
    if "Multiple system tool clients are connected" in message:
        return "检测到多个设备连接，但当前请求未绑定设备 ID。"
    if "No connected system tool client is available" in message:
        return "系统工具客户端未连接。"
    return f"应用包名查找失败：{message}"


def _completed_steps(count: int) -> list[JsonObject]:
    definitions = (
        ("识别目标应用", "open_app_intent"),
        ("查找应用包名", "list_apps"),
        ("打开应用", "launch"),
        ("完成", "finish"),
    )
    return [
        {
            "index": index,
            "name": name,
            "toolName": tool_name,
            "status": "completed",
        }
        for index, (name, tool_name) in enumerate(definitions, start=1)
        if index <= count
    ]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


__all__ = [
    "OpenAppIntent",
    "OpenAppRunner",
    "OpenAppSkillMiddleware",
    "is_open_app_request",
    "parse_open_app_intent",
]
