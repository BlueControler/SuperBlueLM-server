from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import inspect
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from typing import Any, Protocol, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from ..confirmations import ConfirmationTransaction, create_confirmation
from ..json_types import JsonObject
from ..progress import emit_needs_confirmation, emit_task_complexity, emit_task_progress
from .middleware import _message_content_to_text
from .state import MobileAgentState, device_id_from_mapping

FIXED_MEDICAL_TRAVEL_UTTERANCE = "明天上午我要去医院复诊，帮我查天气和路线，并设置提醒。"
MEDICAL_TRAVEL_TASK_TYPE = "medical_travel_reminder"
MEDICAL_TRAVEL_PHASE = "medical_travel_sop"

DEFAULT_WEATHER_RESULT = "明天多云，18-24℃，上午体感舒适，建议携带薄外套。"
DEFAULT_ROUTE_RESULT = "高德地图路线：预计 42 分钟，建议优先选择地铁换乘路线，避开早高峰拥堵。"
DEFAULT_TRAVEL_ADVICE = "建议提前 20 分钟出发，带好医保卡、身份证和既往检查资料。"
DEFAULT_REMINDER_TIME = "明天上午出发前 30 分钟"
DEFAULT_CITY = "南京"
DEFAULT_ROUTE_DESTINATION = "医院"
FINAL_MESSAGE = "已整理明日出行信息，并创建复诊提醒。"
WAITING_CONFIRMATION_MESSAGE = "已整理明日出行信息，等待你确认是否创建复诊提醒。"


class DateProvider(Protocol):
    def __call__(self) -> str: ...


class AsyncToolCall(Protocol):
    def __call__(self, arguments: JsonObject) -> Awaitable[Any]: ...


def is_medical_travel_sop_request(text: str) -> bool:
    normalized = _normalize_utterance(text)
    return normalized == _normalize_utterance(FIXED_MEDICAL_TRAVEL_UTTERANCE)


@dataclass(frozen=True)
class MedicalTravelSopRunner:
    travel_advice: str = DEFAULT_TRAVEL_ADVICE
    reminder_time: str = DEFAULT_REMINDER_TIME
    now: DateProvider | None = None
    auto_confirm: bool = False
    weather_query: AsyncToolCall | None = None
    route_query: AsyncToolCall | None = None
    confirmation_request: AsyncToolCall | None = None
    reminder_writer: AsyncToolCall | None = None

    async def run(self, device_id: str | None = None) -> JsonObject:
        completed_steps: list[JsonObject] = []
        emit_task_complexity(
            complexity="complex",
            track_steps=True,
            reason="medical_travel_sop",
            message="识别任务类型：就医出行准备",
        )

        self._emit_step(
            step=1,
            message="第 1/5 步：正在识别出行需求",
            tool_name="task_complexity",
            completed_steps=completed_steps,
        )
        completed_steps = _append_completed_step(
            completed_steps,
            1,
            "正在识别出行需求",
            "task_complexity",
        )

        self._emit_step(
            step=2,
            message="第 2/5 步：正在查询天气",
            tool_name="weather_query",
            completed_steps=completed_steps,
        )
        weather_result = await _call_tool(
            self.weather_query or _demo_weather_query,
            {"city": DEFAULT_CITY, "date": "明天上午", "purpose": "医院复诊"},
        )
        completed_steps = _append_completed_step(
            completed_steps,
            2,
            "正在查询天气",
            "weather_query",
        )

        self._emit_step(
            step=3,
            message="第 3/5 步：正在规划路线",
            tool_name="amap_mcp_tool",
            completed_steps=completed_steps,
        )
        route_result = await _call_tool(
            self.route_query or _demo_route_query,
            {
                "origin": "当前位置",
                "destination": DEFAULT_ROUTE_DESTINATION,
                "departure_time": "明天上午",
                "purpose": "复诊",
            },
        )
        completed_steps = _append_completed_step(
            completed_steps,
            3,
            "正在规划路线",
            "amap_mcp_tool",
        )

        confirmation_transaction = _create_medical_travel_confirmation(
            weather_result=weather_result,
            route_result=route_result,
            travel_advice=self.travel_advice,
            reminder_time=self.reminder_time,
            device_id=device_id,
            reminder_writer=self.reminder_writer or _demo_reminder_writer,
        )
        if not self.auto_confirm:
            _emit_needs_confirmation_event(confirmation_transaction)
            _emit_waiting_confirmation_progress(
                confirmation_transaction,
                completed_steps=completed_steps,
            )
            return _waiting_confirmation_result(
                weather_result=weather_result,
                route_result=route_result,
                travel_advice=self.travel_advice,
                reminder_time=self.reminder_time,
                completed_steps=completed_steps,
                transaction=confirmation_transaction,
            )

        self._emit_step(
            step=4,
            message="第 4/5 步：等待确认是否创建提醒",
            tool_name="needs_confirmation",
            completed_steps=completed_steps,
        )
        confirmation = await _call_tool(
            self.confirmation_request or _demo_confirmation_request,
            {
                "message": "是否创建明天上午复诊出行提醒？",
                "action": "create_medical_travel_reminder",
                "auto_confirm": self.auto_confirm,
                "confirmation_id": confirmation_transaction.confirmation_id,
            },
        )
        completed_steps = _append_completed_step(
            completed_steps,
            4,
            "等待确认是否创建提醒",
            "needs_confirmation",
        )
        if not _confirmation_confirmed(confirmation):
            _emit_needs_confirmation_event(confirmation_transaction)
            _emit_waiting_confirmation_progress(
                confirmation_transaction,
                completed_steps=completed_steps,
            )
            return _waiting_confirmation_result(
                weather_result=weather_result,
                route_result=route_result,
                travel_advice=self.travel_advice,
                reminder_time=self.reminder_time,
                completed_steps=completed_steps,
                transaction=confirmation_transaction,
            )

        write_tool = "create_event / update_reminders"
        self._emit_step(
            step=5,
            message="第 5/5 步：正在写入日程提醒",
            tool_name=write_tool,
            completed_steps=completed_steps,
        )
        reminder_result = await _call_tool(
            self.reminder_writer or _demo_reminder_writer,
            {
                "title": "医院复诊出行提醒",
                "time": self.reminder_time,
                "weather": weather_result,
                "route": route_result,
                "advice": self.travel_advice,
                "tool": write_tool,
                "device_id": device_id,
            },
        )
        completed_steps = _append_completed_step(
            completed_steps,
            5,
            "正在写入日程提醒",
            write_tool,
        )

        return {
            "task_type": MEDICAL_TRAVEL_TASK_TYPE,
            "recognized_type": "就医出行准备",
            "weather_result": weather_result,
            "route_result": route_result,
            "travel_advice": self.travel_advice,
            "confirmation": confirmation,
            "reminder_time": self.reminder_time,
            "reminder_result": reminder_result,
            "reminder_created": _tool_result_ok(reminder_result),
            "completed_steps": completed_steps,
            "final_message": FINAL_MESSAGE,
        }

    def _emit_step(
        self,
        *,
        step: int,
        message: str,
        tool_name: str,
        completed_steps: Sequence[JsonObject],
    ) -> None:
        emit_task_progress(
            label=tool_name,
            status="running",
            phase=MEDICAL_TRAVEL_PHASE,
            message=message,
            tool_name=tool_name,
            progress_key=f"medical-travel-sop-{step}",
            current_step=step,
            total_steps=5,
            completed_steps=completed_steps,
        )


class MedicalTravelSopMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    state_schema = MobileAgentState

    def __init__(self, runner: MedicalTravelSopRunner | None = None) -> None:
        self.runner = runner or MedicalTravelSopRunner()

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        if not _request_matches(request.messages):
            return handler(request)
        result = asyncio.run(self.runner.run(_request_device_id(request)))
        return _model_response(result)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if not _request_matches(request.messages):
            return await handler(request)
        result = await self.runner.run(_request_device_id(request))
        return _model_response(result)


def _request_matches(messages: Sequence[BaseMessage]) -> bool:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return is_medical_travel_sop_request(_message_content_to_text(message.content))
    return False


def _request_device_id(request: object) -> str | None:
    return device_id_from_mapping(getattr(request, "state", None))


def build_medical_travel_sop_runner(
    *,
    external_tools: Sequence[BaseTool],
    system_tools: Sequence[BaseTool],
) -> MedicalTravelSopRunner:
    external = _tools_by_name(external_tools)
    system = _tools_by_name(system_tools)
    return MedicalTravelSopRunner(
        weather_query=_weather_tool_adapter(external.get("weather_query")),
        route_query=_route_tool_adapter(external.get("amap_mcp_tool")),
        confirmation_request=_demo_confirmation_request,
        reminder_writer=_reminder_tool_adapter(
            system.get("create_event"),
            system.get("update_reminders"),
        ),
    )


def _model_response(result: JsonObject) -> ModelResponse[Any]:
    return ModelResponse(
        result=[
            AIMessage(
                content=str(result["final_message"]),
                additional_kwargs={
                    "medical_travel_sop": json.dumps(result, ensure_ascii=False)
                },
            )
        ]
    )


def _normalize_utterance(text: str) -> str:
    return re.sub(r"[\s，。,.!！?？]+", "", text.strip().lower())


async def _call_tool(tool_call: AsyncToolCall, arguments: JsonObject) -> Any:
    result = tool_call(arguments)
    if inspect.isawaitable(result):
        result = await result
    return result


def _tools_by_name(tools: Sequence[BaseTool]) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in tools}


def _weather_tool_adapter(tool: BaseTool | None) -> AsyncToolCall:
    if tool is None:
        return _demo_weather_query

    async def call(arguments: JsonObject) -> Any:
        return await tool.ainvoke({"city": arguments.get("city") or DEFAULT_CITY})

    return call


def _route_tool_adapter(tool: BaseTool | None) -> AsyncToolCall:
    if tool is None:
        return _demo_route_query

    async def call(arguments: JsonObject) -> Any:
        origin = os.getenv("ECHO_MEDICAL_ROUTE_ORIGIN", "当前位置")
        destination = os.getenv(
            "ECHO_MEDICAL_ROUTE_DESTINATION",
            str(arguments.get("destination") or DEFAULT_ROUTE_DESTINATION),
        )
        city = os.getenv("DEFAULT_AMAP_CITY_NAME", DEFAULT_CITY)
        return await tool.ainvoke(
            {
                "tool_name": "maps_direction_transit_integrated",
                "arguments": {
                    "origin": origin,
                    "destination": destination,
                    "city": city,
                    "cityd": city,
                },
            }
        )

    return call


def _reminder_tool_adapter(
    create_event_tool: BaseTool | None,
    update_reminders_tool: BaseTool | None,
) -> AsyncToolCall:
    if create_event_tool is None or update_reminders_tool is None:
        return _demo_reminder_writer

    async def call(arguments: JsonObject) -> JsonObject:
        device_id = arguments.get("device_id")
        event = _calendar_event(arguments)
        create_args: JsonObject = {"event": event}
        if isinstance(device_id, str) and device_id:
            create_args["device_id"] = device_id
        create_result = await create_event_tool.ainvoke(create_args)
        create_payload = _json_payload(create_result)
        event_id = _event_id(create_payload)
        if event_id is None:
            return {
                "ok": bool(create_payload.get("ok", False)),
                "create_event": create_payload,
                "update_reminders": {
                    "skipped": True,
                    "reason": "create_event did not return an event id",
                },
            }

        reminder_args: JsonObject = {
            "event_id": event_id,
            "reminders": [{"minutes": 30, "method": "alert"}],
        }
        if isinstance(device_id, str) and device_id:
            reminder_args["device_id"] = device_id
        reminder_result = await update_reminders_tool.ainvoke(reminder_args)
        return {
            "ok": True,
            "eventId": event_id,
            "create_event": create_payload,
            "update_reminders": _json_payload(reminder_result),
        }

    return call


async def _demo_weather_query(_arguments: JsonObject) -> str:
    return DEFAULT_WEATHER_RESULT


async def _demo_route_query(_arguments: JsonObject) -> str:
    return DEFAULT_ROUTE_RESULT


async def _demo_confirmation_request(arguments: JsonObject) -> JsonObject:
    auto_confirm = bool(arguments.get("auto_confirm", False))
    return {
        "status": "confirmed" if auto_confirm else "needs_confirmation",
        "confirmed": auto_confirm,
        "confirmationId": str(arguments.get("confirmation_id") or ""),
        "message": str(arguments.get("message") or "是否继续？"),
    }


async def _demo_reminder_writer(arguments: JsonObject) -> JsonObject:
    return {
        "ok": True,
        "demo": True,
        "title": str(arguments.get("title") or "医院复诊出行提醒"),
        "time": str(arguments.get("time") or DEFAULT_REMINDER_TIME),
        "tool": str(arguments.get("tool") or "create_event / update_reminders"),
    }


def _tool_result_ok(result: Any) -> bool:
    if isinstance(result, Mapping):
        return bool(cast(Mapping[str, Any], result).get("ok", False))
    return False


def _confirmation_confirmed(result: Any) -> bool:
    if isinstance(result, Mapping):
        payload = cast(Mapping[str, Any], result)
        return payload.get("status") == "confirmed" or payload.get("confirmed") is True
    return False


def _create_medical_travel_confirmation(
    *,
    weather_result: Any,
    route_result: Any,
    travel_advice: str,
    reminder_time: str,
    device_id: str | None,
    reminder_writer: AsyncToolCall,
) -> ConfirmationTransaction:
    preview = "\n".join(
        [
            f"提醒：医院复诊出行提醒（{reminder_time}）",
            f"天气：{weather_result}",
            f"路线：{route_result}",
            f"建议：{travel_advice}",
        ]
    )
    if device_id:
        preview = f"{preview}\n设备：{device_id}"
    return create_confirmation(
        task_title="就医出行提醒",
        operation="创建复诊出行日程提醒",
        target_app="系统日历",
        tool_name="create_event",
        risk_level="high",
        payload_preview=preview,
        confirm_text="确认创建",
        cancel_text="取消",
        dry_run=False,
        confirm_handler=_medical_travel_confirm_handler(
            weather_result=weather_result,
            route_result=route_result,
            travel_advice=travel_advice,
            reminder_time=reminder_time,
            device_id=device_id,
            reminder_writer=reminder_writer,
        ),
    )


def _medical_travel_confirm_handler(
    *,
    weather_result: Any,
    route_result: Any,
    travel_advice: str,
    reminder_time: str,
    device_id: str | None,
    reminder_writer: AsyncToolCall,
) -> Callable[[ConfirmationTransaction], Awaitable[list[JsonObject]]]:
    async def handle(transaction: ConfirmationTransaction) -> list[JsonObject]:
        write_tool = "create_event / update_reminders"
        running = _confirmation_task_progress_event(
            transaction,
            status="running",
            step_title="正在写入日程提醒",
            message="已确认，正在创建复诊日程和提醒。",
            tool_name=write_tool,
            confirmation_id=transaction.confirmation_id,
        )
        reminder_result = await _call_tool(
            reminder_writer,
            {
                "title": "医院复诊出行提醒",
                "time": reminder_time,
                "weather": weather_result,
                "route": route_result,
                "advice": travel_advice,
                "tool": write_tool,
                "device_id": device_id,
            },
        )
        ok = _tool_result_ok(reminder_result)
        completed = _confirmation_task_progress_event(
            transaction,
            status="completed" if ok else "failed",
            step_title="复诊提醒已创建" if ok else "复诊提醒创建失败",
            message=FINAL_MESSAGE if ok else "复诊提醒创建失败，请稍后重试。",
            tool_name=write_tool,
            confirmation_id=None,
            error=None if ok else _safe_error_text(reminder_result),
        )
        return [running, completed]

    return handle


def _emit_needs_confirmation_event(transaction: ConfirmationTransaction) -> None:
    emit_needs_confirmation(
        confirmation_id=transaction.confirmation_id,
        run_id=transaction.run_id,
        thread_id=transaction.thread_id,
        task_title=transaction.task_title,
        operation=transaction.operation,
        target_app=transaction.target_app,
        tool_name=transaction.tool_name,
        risk_level=transaction.risk_level,
        payload_preview=transaction.payload_preview,
        confirm_text=transaction.confirm_text,
        cancel_text=transaction.cancel_text,
        dry_run=transaction.dry_run,
    )


def _emit_waiting_confirmation_progress(
    transaction: ConfirmationTransaction,
    *,
    completed_steps: Sequence[JsonObject],
) -> None:
    emit_task_progress(
        label="needs_confirmation",
        status="waiting_confirmation",
        phase=MEDICAL_TRAVEL_PHASE,
        task_title=transaction.task_title,
        step_title="等待确认是否创建提醒",
        message="已整理天气和路线信息，等待确认后才会创建日程提醒。",
        tool_name="needs_confirmation",
        progress_key=f"medical-travel-sop-confirmation-{transaction.confirmation_id}",
        current_step=4,
        total_steps=5,
        completed_steps=completed_steps,
        requires_confirmation=True,
        confirmation_id=transaction.confirmation_id,
        can_cancel=True,
        can_take_over=True,
        dry_run=transaction.dry_run,
    )


def _confirmation_task_progress_event(
    transaction: ConfirmationTransaction,
    *,
    status: str,
    step_title: str,
    message: str,
    tool_name: str,
    confirmation_id: str | None,
    error: str | None = None,
) -> JsonObject:
    event: JsonObject = {
        "type": "task_progress",
        "label": tool_name,
        "taskTitle": transaction.task_title,
        "status": status,
        "phase": MEDICAL_TRAVEL_PHASE,
        "currentStep": 5,
        "totalSteps": 5,
        "stepTitle": step_title,
        "message": message,
        "toolName": tool_name,
        "requiresConfirmation": False,
        "canCancel": False,
        "canTakeOver": False,
        "progressKey": f"medical-travel-sop-confirm-{transaction.confirmation_id}",
        "dryRun": False,
    }
    if confirmation_id:
        event["confirmationId"] = confirmation_id
    if transaction.run_id:
        event["runId"] = transaction.run_id
    if transaction.thread_id:
        event["threadId"] = transaction.thread_id
    if error:
        event["error"] = error
    return event


def _waiting_confirmation_result(
    *,
    weather_result: Any,
    route_result: Any,
    travel_advice: str,
    reminder_time: str,
    completed_steps: Sequence[JsonObject],
    transaction: ConfirmationTransaction,
) -> JsonObject:
    return {
        "task_type": MEDICAL_TRAVEL_TASK_TYPE,
        "recognized_type": "就医出行准备",
        "weather_result": weather_result,
        "route_result": route_result,
        "travel_advice": travel_advice,
        "confirmation": {
            "status": "needs_confirmation",
            "confirmed": False,
            "confirmationId": transaction.confirmation_id,
            "needsConfirmation": transaction.needs_confirmation_event(),
        },
        "reminder_time": reminder_time,
        "reminder_result": {
            "skipped": True,
            "reason": "needs_confirmation",
            "confirmationId": transaction.confirmation_id,
            "dryRun": transaction.dry_run,
        },
        "reminder_created": False,
        "completed_steps": list(completed_steps),
        "final_message": WAITING_CONFIRMATION_MESSAGE,
    }


def _calendar_event(arguments: JsonObject) -> JsonObject:
    start = _tomorrow_morning_ms()
    end = start + 60 * 60 * 1000
    description = "\n".join(
        str(arguments.get(key) or "")
        for key in ("weather", "route", "advice")
        if arguments.get(key)
    )
    return {
        "title": str(arguments.get("title") or "医院复诊出行提醒"),
        "description": description,
        "eventLocation": DEFAULT_ROUTE_DESTINATION,
        "dtstart": start,
        "dtend": end,
        "eventTimezone": "Asia/Shanghai",
        "availability": "busy",
        "status": "confirmed",
    }


def _tomorrow_morning_ms() -> int:
    timezone = ZoneInfo("Asia/Shanghai")
    tomorrow = datetime.now(timezone).date() + timedelta(days=1)
    start = datetime(
        tomorrow.year,
        tomorrow.month,
        tomorrow.day,
        9,
        0,
        tzinfo=timezone,
    )
    return int(start.timestamp() * 1000)


def _json_payload(result: Any) -> JsonObject:
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except ValueError:
            return {"ok": False, "raw": result}
        return parsed if isinstance(parsed, dict) else {"ok": False, "raw": parsed}
    if isinstance(result, Mapping):
        return {str(key): value for key, value in result.items()}
    return {"ok": False, "raw": result}


def _event_id(payload: Mapping[str, Any]) -> int | None:
    for key in ("eventId", "event_id", "id", "_id"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    event = payload.get("event")
    if isinstance(event, Mapping):
        return _event_id(event)
    return None


def _safe_error_text(result: Any) -> str:
    if isinstance(result, Mapping):
        message = result.get("message") or result.get("error") or result
        return str(message)[:200]
    return str(result)[:200]


def _append_completed_step(
    completed_steps: Sequence[JsonObject],
    index: int,
    name: str,
    tool_name: str,
) -> list[JsonObject]:
    return [
        *completed_steps,
        {
            "index": index,
            "name": name,
            "toolName": tool_name,
            "status": "completed",
        },
    ]


__all__ = [
    "FIXED_MEDICAL_TRAVEL_UTTERANCE",
    "MedicalTravelSopMiddleware",
    "MedicalTravelSopRunner",
    "build_medical_travel_sop_runner",
    "is_medical_travel_sop_request",
]
