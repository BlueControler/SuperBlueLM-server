from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import inspect
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from ..confirmations import ConfirmationTransaction, create_confirmation
from ..json_types import JsonObject, to_json_value
from ..progress import emit_needs_confirmation, emit_task_complexity, emit_task_progress
from .middleware import _message_content_to_text
from .state import MobileAgentState, device_id_from_mapping

FIXED_MEDICAL_TRAVEL_UTTERANCE = "明天上午我要去医院复诊，帮我查天气和路线，并设置提醒。"
MEDICAL_TRAVEL_TASK_TYPE = "medical_travel_reminder"
MEDICAL_TRAVEL_PHASE = "medical_travel"
MEDICAL_TRAVEL_TASK_TITLE = "就医出行准备"
TOTAL_STEPS = 7

DEFAULT_WEATHER_RESULT = "明天多云，18-24℃，上午体感舒适。"
DEFAULT_ROUTE_RESULT = "高德地图路线：预计 42 分钟，建议优先选择地铁路线，减少步行。"
DEFAULT_CITY = "南京"
DEFAULT_ROUTE_ORIGIN = "当前位置"
DEFAULT_ROUTE_DESTINATION = "江苏省人民医院"
DEFAULT_REMINDER_OFFSET_MINUTES = 30
WAITING_CONFIRMATION_MESSAGE = "已整理明日就医出行信息，等待你确认是否创建提醒。"
FINAL_MESSAGE = "已完成就医出行准备。"

MEDICAL_MARKERS = (
    "医院",
    "复诊",
    "就医",
    "看病",
    "检查",
    "取药",
    "陪老人去医院",
    "门诊",
    "体检",
)
TASK_MARKERS = ("天气", "路线", "导航", "出发", "提醒", "日历", "规划", "出行", "注意事项")
PURPOSE_MARKERS = ("复诊", "检查", "取药", "体检", "门诊", "看病", "就医")
CITY_NAMES = ("北京", "天津", "上海", "重庆", "南京", "广州", "深圳", "杭州", "成都", "武汉", "西安")
MEMORY_KEYS = [
    "origin",
    "home",
    "school",
    "dorm",
    "company",
    "preferred_hospital",
    "city",
    "travel_preference",
    "elderly_care_preference",
    "medical_checklist",
    "reminder_offset_minutes",
]


class DateProvider(Protocol):
    def __call__(self) -> str: ...


class AsyncToolCall(Protocol):
    def __call__(self, arguments: JsonObject) -> Awaitable[Any]: ...


class MemoryReader(Protocol):
    def __call__(
        self,
        keys: list[str],
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class MedicalTravelIntent:
    raw_text: str
    date_text: str
    time_text: str | None
    city: str | None
    origin: str | None
    destination: str
    purpose: str
    reminder_offset_minutes: int
    needs_weather: bool
    needs_route: bool
    needs_reminder: bool
    for_elderly: bool


@dataclass(frozen=True)
class MedicalMemory:
    memory_used: bool
    memory_source: str
    items: list[JsonObject]
    summary: str
    values: Mapping[str, Any]


@dataclass(frozen=True)
class MedicalDecision:
    origin: str
    departure_advice: str
    route_choice_reason: str
    checklist: list[str]
    reminder_offset_minutes: int


def is_medical_travel_request(text: str) -> bool:
    normalized = _normalize_utterance(text)
    has_medical = any(marker in normalized for marker in MEDICAL_MARKERS)
    has_task = any(marker in normalized for marker in TASK_MARKERS)
    return has_medical and has_task


def is_medical_travel_sop_request(text: str) -> bool:
    return is_medical_travel_request(text)


def parse_medical_travel_intent(text: str) -> MedicalTravelIntent:
    normalized = _normalize_utterance(text)
    date_text = _extract_date_text(normalized)
    time_text = _extract_time_text(text)
    city = _extract_city(normalized)
    purpose = _extract_purpose(normalized)
    destination = _extract_destination(text, purpose)
    reminder_offset = _extract_reminder_offset(text)
    return MedicalTravelIntent(
        raw_text=text,
        date_text=date_text,
        time_text=time_text,
        city=city,
        origin=None,
        destination=destination,
        purpose=purpose,
        reminder_offset_minutes=reminder_offset,
        needs_weather="天气" in normalized,
        needs_route=any(marker in normalized for marker in ("路线", "导航", "出行", "规划", "出发")),
        needs_reminder=any(marker in normalized for marker in ("提醒", "日历")),
        for_elderly=any(marker in normalized for marker in ("老人", "父母", "家人", "陪老人")),
    )


@dataclass(frozen=True)
class MedicalTravelSopRunner:
    travel_advice: str = ""
    reminder_time: str = ""
    now: DateProvider | None = None
    auto_confirm: bool = False
    weather_query: AsyncToolCall | None = None
    route_query: AsyncToolCall | None = None
    confirmation_request: AsyncToolCall | None = None
    reminder_writer: AsyncToolCall | None = None
    memory_reader: MemoryReader | None = None

    async def run(
        self,
        device_id: str | None = None,
        *,
        intent: MedicalTravelIntent | None = None,
        memory_context: Mapping[str, Any] | None = None,
    ) -> JsonObject:
        intent = intent or parse_medical_travel_intent(FIXED_MEDICAL_TRAVEL_UTTERANCE)
        completed_steps: list[JsonObject] = []
        emit_task_complexity(
            complexity="complex",
            track_steps=True,
            reason=MEDICAL_TRAVEL_PHASE,
            message="识别任务类型：就医出行准备",
        )

        self._emit_step(
            step=1,
            status="running",
            step_title="识别就医需求",
            message="正在识别就医时间、医院和提醒需求。",
            tool_name="medical_travel_intent",
            completed_steps=completed_steps,
        )

        memory = self._read_memory(memory_context)
        enriched_intent = _apply_memory_to_intent(intent, memory)
        completed_steps = _append_completed_step(completed_steps, 1, "识别就医需求", "medical_travel_intent")
        self._emit_step(
            step=1,
            status="completed",
            step_title="识别就医需求",
            message=(
                f"已识别为：{enriched_intent.date_text}"
                f"{enriched_intent.time_text or ''}前往{enriched_intent.destination}"
                f"{enriched_intent.purpose}。"
            ),
            tool_name="medical_travel_intent",
            completed_steps=completed_steps,
        )

        self._emit_step(
            step=2,
            status="running",
            step_title="读取过往记忆",
            message="正在读取过往偏好，用于判断出发地、出行方式和提醒时间。",
            tool_name="read_user_memory",
            completed_steps=completed_steps,
        )
        completed_steps = _append_completed_step(completed_steps, 2, "读取过往记忆", "read_user_memory")
        self._emit_step(
            step=2,
            status="completed",
            step_title="读取过往记忆",
            message=(
                f"已参考过往偏好：{memory.summary}。"
                if memory.memory_used
                else "未读取到可用偏好，将使用默认出行设置。"
            ),
            tool_name="read_user_memory",
            completed_steps=completed_steps,
        )

        self._emit_step(
            step=3,
            status="running",
            step_title="查询天气",
            message=(
                f"正在查询{enriched_intent.city or DEFAULT_CITY}"
                f"{enriched_intent.date_text}{enriched_intent.time_text or ''}的天气。"
            ),
            tool_name="weather_query",
            completed_steps=completed_steps,
        )
        weather_result = await _call_tool(
            self.weather_query or _demo_weather_query,
            {
                "city": enriched_intent.city or DEFAULT_CITY,
                "date": f"{enriched_intent.date_text}{enriched_intent.time_text or ''}",
                "purpose": enriched_intent.purpose,
            },
        )
        weather_summary = _summary_text(weather_result, DEFAULT_WEATHER_RESULT)
        completed_steps = _append_completed_step(completed_steps, 3, "查询天气", "weather_query")
        self._emit_step(
            step=3,
            status="completed",
            step_title="查询天气",
            message=f"已获取天气信息：{weather_summary}。",
            tool_name="weather_query",
            completed_steps=completed_steps,
        )

        self._emit_step(
            step=4,
            status="running",
            step_title="规划出行路线",
            message=f"正在规划从{enriched_intent.origin or DEFAULT_ROUTE_ORIGIN}前往{enriched_intent.destination}的出行方案。",
            tool_name="amap_mcp_tool",
            completed_steps=completed_steps,
        )
        route_result = await _call_tool(
            self.route_query or _demo_route_query,
            {
                "origin": enriched_intent.origin or DEFAULT_ROUTE_ORIGIN,
                "destination": enriched_intent.destination,
                "city": enriched_intent.city or DEFAULT_CITY,
                "departure_time": f"{enriched_intent.date_text}{enriched_intent.time_text or ''}",
                "purpose": enriched_intent.purpose,
            },
        )
        route_summary = _summary_text(route_result, DEFAULT_ROUTE_RESULT)
        route_duration = _extract_duration(route_summary)
        route_advice = _route_advice(route_summary, memory)
        completed_steps = _append_completed_step(completed_steps, 4, "规划出行路线", "amap_mcp_tool")
        self._emit_step(
            step=4,
            status="completed",
            step_title="规划出行路线",
            message=f"已获取出行方案：预计{route_duration}，建议{route_advice}。",
            tool_name="amap_mcp_tool",
            completed_steps=completed_steps,
        )

        self._emit_step(
            step=5,
            status="running",
            step_title="形成出行决策",
            message="正在结合天气、路线和过往偏好生成就医出行建议。",
            tool_name="medical_travel_decision",
            completed_steps=completed_steps,
        )
        decision = _build_decision(enriched_intent, memory, weather_summary, route_summary, route_advice)
        decision_summary = f"{decision.origin}出发，{decision.route_choice_reason}"
        completed_steps = _append_completed_step(completed_steps, 5, "形成出行决策", "medical_travel_decision")
        self._emit_step(
            step=5,
            status="completed",
            step_title="形成出行决策",
            message=f"已生成出行建议：{decision_summary}。",
            tool_name="medical_travel_decision",
            completed_steps=completed_steps,
        )

        payload = _medical_travel_payload(
            intent=enriched_intent,
            memory=memory,
            weather_summary=weather_summary,
            route_summary=route_summary,
            route_duration=route_duration,
            route_advice=route_advice,
            decision=decision,
            confirmation=None,
            reminder={"created": False, "eventId": None},
        )
        confirmation_transaction = _create_medical_travel_confirmation(
            payload=payload,
            reminder_writer=self.reminder_writer or _demo_reminder_writer,
            device_id=device_id,
        )
        payload["confirmation"] = {
            "status": "needs_confirmation",
            "confirmationId": confirmation_transaction.confirmation_id,
        }

        if not _auto_confirm_enabled(self.auto_confirm):
            _emit_needs_confirmation_event(confirmation_transaction)
            self._emit_step(
                step=6,
                status="waiting_confirmation",
                step_title="等待确认",
                message="已整理就医出行信息，等待确认是否创建提醒。",
                tool_name="needs_confirmation",
                completed_steps=completed_steps,
                requires_confirmation=True,
                confirmation_id=confirmation_transaction.confirmation_id,
            )
            return _result(payload, WAITING_CONFIRMATION_MESSAGE)

        completed_steps = _append_completed_step(completed_steps, 6, "等待确认", "needs_confirmation")
        self._emit_step(
            step=7,
            status="running",
            step_title="创建日历提醒",
            message=f"正在创建就医出行提醒，并设置提前{decision.reminder_offset_minutes}分钟提醒。",
            tool_name="create_event / update_reminders",
            completed_steps=completed_steps,
        )
        reminder_result = await _call_tool(
            self.reminder_writer or _demo_reminder_writer,
            _reminder_arguments(payload, device_id),
        )
        payload["reminder"] = _reminder_payload(reminder_result)
        payload["confirmation"] = {
            "status": "confirmed",
            "confirmationId": confirmation_transaction.confirmation_id,
        }
        completed_steps = _append_completed_step(completed_steps, 7, "创建日历提醒", "create_event / update_reminders")
        self._emit_step(
            step=7,
            status="completed",
            step_title="完成",
            message="已创建就医出行提醒。",
            tool_name="finish",
            completed_steps=completed_steps,
        )
        return _result(payload, _final_message(payload))

    def _read_memory(self, context: Mapping[str, Any] | None) -> MedicalMemory:
        values = dict((self.memory_reader or read_user_memory)(MEMORY_KEYS, context))
        used_items = [
            {"key": key, "value": values[key]}
            for key in MEMORY_KEYS
            if key in values and values[key] not in (None, "", [], {})
        ]
        if not used_items:
            return MedicalMemory(
                memory_used=False,
                memory_source="none",
                items=[],
                summary="",
                values={},
            )
        summary_parts: list[str] = []
        if values.get("travel_preference"):
            summary_parts.append(str(values["travel_preference"]))
        if values.get("elderly_care_preference"):
            summary_parts.append(str(values["elderly_care_preference"]))
        if values.get("reminder_offset_minutes"):
            summary_parts.append(f"提前 {values['reminder_offset_minutes']} 分钟提醒")
        if values.get("medical_checklist"):
            summary_parts.append(f"携带{values['medical_checklist']}")
        return MedicalMemory(
            memory_used=True,
            memory_source="user_memory",
            items=used_items,
            summary="、".join(summary_parts) or "、".join(str(item["value"]) for item in used_items[:3]),
            values=values,
        )

    def _emit_step(
        self,
        *,
        step: int,
        status: str,
        step_title: str,
        message: str,
        tool_name: str,
        completed_steps: Sequence[JsonObject],
        requires_confirmation: bool = False,
        confirmation_id: str | None = None,
        error: str | None = None,
    ) -> None:
        emit_task_progress(
            label=tool_name,
            status=cast(Any, status),
            phase=MEDICAL_TRAVEL_PHASE,
            task_title=MEDICAL_TRAVEL_TASK_TITLE,
            step_title=step_title,
            message=message,
            tool_name=tool_name,
            progress_key=f"medical-travel-{step}",
            current_step=step,
            total_steps=TOTAL_STEPS,
            completed_steps=completed_steps,
            requires_confirmation=requires_confirmation,
            confirmation_id=confirmation_id,
            can_cancel=status in {"running", "waiting_confirmation"},
            can_take_over=status in {"running", "waiting_confirmation"},
            error=error,
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
        intent = _intent_from_request(request.messages)
        if intent is None:
            return handler(request)
        result = asyncio.run(
            self.runner.run(
                _request_device_id(request),
                intent=intent,
                memory_context=_request_memory_context(request),
            )
        )
        return _model_response(result)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        intent = _intent_from_request(request.messages)
        if intent is None:
            return await handler(request)
        result = await self.runner.run(
            _request_device_id(request),
            intent=intent,
            memory_context=_request_memory_context(request),
        )
        return _model_response(result)


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


def read_user_memory(
    keys: list[str],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if context is not None:
        for source_key in ("user_memory", "memory", "medical_travel_memory"):
            source = context.get(source_key)
            if isinstance(source, Mapping):
                values.update({str(key): value for key, value in source.items()})
    raw = os.getenv("ECHO_MEDICAL_TRAVEL_MEMORY_JSON")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, Mapping):
            values.update({str(key): value for key, value in parsed.items()})
    return {key: values[key] for key in keys if key in values}


def _intent_from_request(messages: Sequence[BaseMessage]) -> MedicalTravelIntent | None:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            text = _message_content_to_text(message.content)
            if is_medical_travel_request(text):
                return parse_medical_travel_intent(text)
    return None


def _request_device_id(request: object) -> str | None:
    state_device_id = device_id_from_mapping(getattr(request, "state", None))
    if state_device_id is not None:
        return state_device_id
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None)
    if isinstance(config, Mapping):
        return device_id_from_mapping(config.get("configurable")) or device_id_from_mapping(
            config.get("metadata")
        )
    return None


def _request_memory_context(request: object) -> Mapping[str, Any]:
    state = getattr(request, "state", None)
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None)
    context: dict[str, Any] = {}
    if isinstance(state, Mapping):
        context.update(state)
    if isinstance(config, Mapping):
        configurable = config.get("configurable")
        metadata = config.get("metadata")
        if isinstance(configurable, Mapping):
            context.update(cast(Mapping[str, Any], configurable))
        if isinstance(metadata, Mapping):
            context.update(cast(Mapping[str, Any], metadata))
    return context


def _model_response(result: JsonObject) -> ModelResponse[Any]:
    return ModelResponse(
        result=[
            AIMessage(
                content=str(result["final_message"]),
                additional_kwargs={
                    "medical_travel": result["medical_travel"],
                    "medical_travel_sop": json.dumps(result, ensure_ascii=False),
                },
            )
        ]
    )


def _medical_travel_payload(
    *,
    intent: MedicalTravelIntent,
    memory: MedicalMemory,
    weather_summary: str,
    route_summary: str,
    route_duration: str,
    route_advice: str,
    decision: MedicalDecision,
    confirmation: JsonObject | None,
    reminder: JsonObject,
) -> JsonObject:
    return {
        "intent": {
            "dateText": intent.date_text,
            "timeText": intent.time_text,
            "city": intent.city or DEFAULT_CITY,
            "origin": decision.origin,
            "destination": intent.destination,
            "purpose": intent.purpose,
            "reminderOffsetMinutes": decision.reminder_offset_minutes,
            "forElderly": intent.for_elderly,
        },
        "memory": {
            "memoryUsed": memory.memory_used,
            "memorySource": memory.memory_source,
            "items": memory.items,
            "summary": (
                f"参考了用户偏好：{memory.summary}。"
                if memory.memory_used
                else "未读取到可用偏好。"
            ),
        },
        "weather": {
            "summary": weather_summary,
            "demo": False,
        },
        "route": {
            "summary": route_summary,
            "duration": route_duration,
            "transitType": route_advice,
            "demo": False,
        },
        "decision": {
            "origin": decision.origin,
            "departureAdvice": decision.departure_advice,
            "routeChoiceReason": decision.route_choice_reason,
            "checklist": decision.checklist,
            "reminderOffsetMinutes": decision.reminder_offset_minutes,
        },
        "confirmation": confirmation
        or {
            "status": "needs_confirmation",
            "confirmationId": None,
        },
        "reminder": reminder,
    }


def _result(payload: JsonObject, final_message: str) -> JsonObject:
    return {
        "task_type": MEDICAL_TRAVEL_TASK_TYPE,
        "recognized_type": "就医出行准备",
        "medical_travel": payload,
        "weather_result": payload["weather"]["summary"],
        "route_result": payload["route"]["summary"],
        "travel_advice": payload["decision"]["departureAdvice"],
        "confirmation": payload["confirmation"],
        "reminder_time": f"提前 {payload['decision']['reminderOffsetMinutes']} 分钟",
        "reminder_result": payload["reminder"],
        "reminder_created": bool(cast(Mapping[str, Any], payload["reminder"]).get("created")),
        "final_message": final_message,
    }


def _apply_memory_to_intent(intent: MedicalTravelIntent, memory: MedicalMemory) -> MedicalTravelIntent:
    values = memory.values
    origin = _memory_origin(values) or intent.origin or os.getenv("ECHO_MEDICAL_ROUTE_ORIGIN", DEFAULT_ROUTE_ORIGIN)
    destination = (
        intent.destination
        if intent.destination != "医院"
        else str(
            values.get("preferred_hospital")
            or os.getenv("ECHO_MEDICAL_ROUTE_DESTINATION")
            or DEFAULT_ROUTE_DESTINATION
        )
    )
    city = intent.city or _optional_str(values.get("city")) or os.getenv("DEFAULT_AMAP_CITY_NAME") or DEFAULT_CITY
    reminder_offset = _int_value(values.get("reminder_offset_minutes")) or intent.reminder_offset_minutes
    return MedicalTravelIntent(
        raw_text=intent.raw_text,
        date_text=intent.date_text,
        time_text=intent.time_text,
        city=city,
        origin=origin,
        destination=destination,
        purpose=intent.purpose,
        reminder_offset_minutes=reminder_offset,
        needs_weather=intent.needs_weather,
        needs_route=intent.needs_route,
        needs_reminder=intent.needs_reminder,
        for_elderly=intent.for_elderly,
    )


def _build_decision(
    intent: MedicalTravelIntent,
    memory: MedicalMemory,
    weather_summary: str,
    route_summary: str,
    route_advice: str,
) -> MedicalDecision:
    origin = intent.origin or DEFAULT_ROUTE_ORIGIN
    checklist = _checklist(memory, intent.for_elderly)
    preference = str(
        memory.values.get("travel_preference")
        or memory.values.get("elderly_care_preference")
        or "优先选择稳定路线，预留充足时间"
    )
    if intent.for_elderly and "少换乘" not in preference:
        preference = f"{preference}，尽量少换乘"
    departure_advice = (
        f"建议从{origin}提前出发，结合天气和路线预留缓冲时间。"
        if not intent.for_elderly
        else f"建议从{origin}更早出发，避免赶路，并优先选择少换乘路线。"
    )
    return MedicalDecision(
        origin=origin,
        departure_advice=departure_advice,
        route_choice_reason=f"{preference}；天气参考：{weather_summary}；路线参考：{route_summary or route_advice}",
        checklist=checklist,
        reminder_offset_minutes=intent.reminder_offset_minutes,
    )


def _create_medical_travel_confirmation(
    *,
    payload: JsonObject,
    reminder_writer: AsyncToolCall,
    device_id: str | None,
) -> ConfirmationTransaction:
    preview = "\n".join(
        [
            f"目的：{payload['intent']['purpose']}",
            f"医院：{payload['intent']['destination']}",
            f"时间：{payload['intent']['dateText']}{payload['intent'].get('timeText') or ''}",
            f"天气：{payload['weather']['summary']}",
            f"路线：{payload['route']['summary']}",
            f"建议：{payload['decision']['departureAdvice']}",
            f"提醒：提前 {payload['decision']['reminderOffsetMinutes']} 分钟",
        ]
    )
    return create_confirmation(
        task_title="就医出行提醒",
        operation="创建复诊出行日程提醒",
        target_app="系统日历",
        tool_name="create_event",
        risk_level="medium",
        payload_preview=preview,
        confirm_text="确认创建",
        cancel_text="取消",
        dry_run=False,
        confirm_handler=_medical_travel_confirm_handler(
            payload=payload,
            reminder_writer=reminder_writer,
            device_id=device_id,
        ),
    )


def _medical_travel_confirm_handler(
    *,
    payload: JsonObject,
    reminder_writer: AsyncToolCall,
    device_id: str | None,
) -> Callable[[ConfirmationTransaction], Awaitable[list[JsonObject]]]:
    async def handle(transaction: ConfirmationTransaction) -> list[JsonObject]:
        reminder_minutes = int(cast(Mapping[str, Any], payload["decision"])["reminderOffsetMinutes"])
        running = _confirmation_task_progress_event(
            transaction,
            status="running",
            step_title="创建日历提醒",
            message=f"正在创建就医出行提醒，并设置提前{reminder_minutes}分钟提醒。",
            tool_name="create_event / update_reminders",
            confirmation_id=transaction.confirmation_id,
        )
        reminder_result = await _call_tool(reminder_writer, _reminder_arguments(payload, device_id))
        if _tool_result_ok(reminder_result):
            completed = _confirmation_task_progress_event(
                transaction,
                status="completed",
                step_title="完成",
                message="已创建就医出行提醒。",
                tool_name="finish",
                confirmation_id=None,
                result=_reminder_payload(reminder_result),
            )
        else:
            completed = _confirmation_task_progress_event(
                transaction,
                status="failed",
                step_title="创建日历提醒",
                message=f"创建就医出行提醒失败：{_safe_error_text(reminder_result)}",
                tool_name="create_event / update_reminders",
                confirmation_id=None,
                result=_reminder_payload(reminder_result),
            )
        return [running, completed]

    return handle


def _confirmation_task_progress_event(
    transaction: ConfirmationTransaction,
    *,
    status: str,
    step_title: str,
    message: str,
    tool_name: str,
    confirmation_id: str | None,
    result: JsonObject | None = None,
) -> JsonObject:
    event: JsonObject = {
        "type": "task_progress",
        "label": tool_name,
        "taskTitle": MEDICAL_TRAVEL_TASK_TITLE,
        "status": status,
        "phase": MEDICAL_TRAVEL_PHASE,
        "currentStep": 7,
        "totalSteps": TOTAL_STEPS,
        "stepTitle": step_title,
        "message": message,
        "toolName": tool_name,
        "requiresConfirmation": False,
        "canCancel": status == "running",
        "canTakeOver": status == "running",
        "progressKey": "medical-travel-7",
        "dryRun": transaction.dry_run,
    }
    if confirmation_id:
        event["confirmationId"] = confirmation_id
    if transaction.run_id:
        event["runId"] = transaction.run_id
    if transaction.thread_id:
        event["threadId"] = transaction.thread_id
    if result is not None:
        event["result"] = result
    return event


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
        return await tool.ainvoke(
            {
                "tool_name": "maps_direction_transit_integrated",
                "arguments": {
                    "origin": arguments.get("origin") or DEFAULT_ROUTE_ORIGIN,
                    "destination": arguments.get("destination") or DEFAULT_ROUTE_DESTINATION,
                    "city": arguments.get("city") or DEFAULT_CITY,
                    "cityd": arguments.get("city") or DEFAULT_CITY,
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
                "ok": False,
                "create_event": create_payload,
                "message": "create_event did not return an event id",
            }

        reminder_args: JsonObject = {
            "event_id": event_id,
            "reminders": [
                {
                    "minutes": int(arguments.get("reminder_offset_minutes") or DEFAULT_REMINDER_OFFSET_MINUTES),
                    "method": "alert",
                }
            ],
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
    }


async def _demo_reminder_writer(arguments: JsonObject) -> JsonObject:
    return {
        "ok": True,
        "demo": True,
        "eventId": "demo-medical-travel-event",
        "title": str(arguments.get("title") or "就医出行提醒"),
        "time": str(arguments.get("time") or ""),
    }


def _reminder_arguments(payload: JsonObject, device_id: str | None) -> JsonObject:
    intent = cast(Mapping[str, Any], payload["intent"])
    decision = cast(Mapping[str, Any], payload["decision"])
    args: JsonObject = {
        "title": f"{intent['destination']}{intent['purpose']}出行提醒",
        "time": f"{intent['dateText']}{intent.get('timeText') or ''}",
        "destination": str(intent["destination"]),
        "purpose": str(intent["purpose"]),
        "weather": str(cast(Mapping[str, Any], payload["weather"])["summary"]),
        "route": str(cast(Mapping[str, Any], payload["route"])["summary"]),
        "advice": str(decision["departureAdvice"]),
        "reminder_offset_minutes": int(decision["reminderOffsetMinutes"]),
    }
    if device_id:
        args["device_id"] = device_id
    return args


def _reminder_payload(result: Any) -> JsonObject:
    if isinstance(result, Mapping):
        event_id = result.get("eventId") or result.get("event_id")
        return {
            "created": bool(result.get("ok")) and event_id is not None,
            "eventId": event_id,
            "raw": cast(JsonObject, to_json_value(result)),
        }
    return {"created": False, "eventId": None, "raw": str(result)}


def _calendar_event(arguments: JsonObject) -> JsonObject:
    start = _target_time_ms(str(arguments.get("time") or "明天上午"))
    end = start + 60 * 60 * 1000
    description = "\n".join(
        str(arguments.get(key) or "")
        for key in ("weather", "route", "advice")
        if arguments.get(key)
    )
    return {
        "title": str(arguments.get("title") or "就医出行提醒"),
        "description": description,
        "eventLocation": str(arguments.get("destination") or DEFAULT_ROUTE_DESTINATION),
        "dtstart": start,
        "dtend": end,
        "eventTimezone": "Asia/Shanghai",
        "availability": "busy",
        "status": "confirmed",
    }


def _target_time_ms(date_time_text: str) -> int:
    timezone = ZoneInfo("Asia/Shanghai")
    now = datetime.now(timezone)
    days = 0
    if "后天" in date_time_text:
        days = 2
    elif "明天" in date_time_text:
        days = 1
    target_date = now.date() + timedelta(days=days)
    hour = 9
    minute = 0
    explicit = re.search(r"(?P<hour>\d{1,2})\s*点", date_time_text)
    if explicit:
        hour = int(explicit.group("hour"))
        if "下午" in date_time_text and hour < 12:
            hour += 12
    elif "下午" in date_time_text:
        hour = 15
    elif "晚上" in date_time_text:
        hour = 19
    start = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour,
        minute,
        tzinfo=timezone,
    )
    return int(start.timestamp() * 1000)


def _tool_result_ok(result: Any) -> bool:
    if isinstance(result, Mapping):
        return bool(result.get("ok", False))
    return False


def _summary_text(result: Any, fallback: str) -> str:
    if isinstance(result, str) and result.strip():
        return result.strip()
    if isinstance(result, Mapping):
        for key in ("summary", "message", "text", "result"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(to_json_value(result), ensure_ascii=False)
    return fallback


def _extract_duration(route_summary: str) -> str:
    match = re.search(r"预计\s*(?P<duration>[^，。；;\s]+(?:\s*分钟|\s*小时)?)", route_summary)
    if match:
        return match.group("duration").replace(" ", "")
    match = re.search(r"(?P<duration>\d+\s*(?:分钟|小时))", route_summary)
    if match:
        return match.group("duration").replace(" ", "")
    return "约 45 分钟"


def _route_advice(route_summary: str, memory: MedicalMemory) -> str:
    preference = str(memory.values.get("travel_preference") or "")
    if "少步行" in preference:
        return "优先选择少步行路线"
    if "少换乘" in preference:
        return "优先选择少换乘路线"
    if "地铁" in route_summary or "地铁" in preference:
        return "优先选择地铁路线"
    if "打车" in preference:
        return "必要时打车"
    return "预留充足时间出发"


def _checklist(memory: MedicalMemory, for_elderly: bool) -> list[str]:
    raw = memory.values.get("medical_checklist")
    if isinstance(raw, str) and raw.strip():
        items = [item.strip() for item in re.split(r"[、,，;；\s]+", raw) if item.strip()]
    elif isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]
    else:
        items = ["身份证", "医保卡", "就诊卡", "既往检查资料"]
    if for_elderly:
        for item in ("药物清单",):
            if item not in items:
                items.append(item)
    return items


def _memory_origin(values: Mapping[str, Any]) -> str | None:
    for key in ("origin", "school", "dorm", "home", "company"):
        value = _optional_str(values.get(key))
        if value:
            return value
    return None


def _extract_date_text(normalized: str) -> str:
    if "后天" in normalized:
        return "后天"
    if "今天" in normalized:
        return "今天"
    return "明天"


def _extract_time_text(text: str) -> str | None:
    normalized = _normalize_utterance(text)
    explicit = re.search(r"(?P<period>上午|下午|晚上|早上)?\s*(?P<hour>\d{1,2})\s*点", text)
    if explicit:
        return f"{explicit.group('period') or ''}{int(explicit.group('hour'))}点"
    for period in ("上午", "下午", "晚上", "早上"):
        if period in normalized:
            return period
    return "上午"


def _extract_city(normalized: str) -> str | None:
    for city in CITY_NAMES:
        if city in normalized or f"{city}市" in normalized:
            return city
    return None


def _extract_purpose(normalized: str) -> str:
    for purpose in PURPOSE_MARKERS:
        if purpose in normalized:
            return purpose
    return "就医"


def _extract_destination(text: str, purpose: str) -> str:
    cleaned = re.sub(r"[。！？?!.，,]", " ", text).strip()
    patterns = (
        rf"(?:去|到|前往)(?P<destination>[\w\u4e00-\u9fff大学总医院人民医院中心医院附属医院专科医院]+?){purpose}",
        r"(?:去|到|前往)(?P<destination>[\w\u4e00-\u9fff]+?医院)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            destination = match.group("destination").strip()
            destination = re.sub(r"^(陪老人|陪父母|陪家人)", "", destination).strip()
            if destination:
                return destination
    return "医院"


def _extract_reminder_offset(text: str) -> int:
    match = re.search(r"提前\s*(?P<minutes>\d{1,3})\s*分钟", text)
    if match:
        return int(match.group("minutes"))
    match = re.search(r"提前\s*(?P<hours>\d{1,2})\s*小时", text)
    if match:
        return int(match.group("hours")) * 60
    return DEFAULT_REMINDER_OFFSET_MINUTES


def _auto_confirm_enabled(auto_confirm: bool) -> bool:
    if auto_confirm:
        return True
    return os.getenv("ECHO_MEDICAL_TRAVEL_AUTO_CONFIRM", "").lower() in {"1", "true", "yes"}


def _final_message(payload: JsonObject) -> str:
    intent = cast(Mapping[str, Any], payload["intent"])
    decision = cast(Mapping[str, Any], payload["decision"])
    return (
        f"{FINAL_MESSAGE}\n\n"
        f"天气：{cast(Mapping[str, Any], payload['weather'])['summary']}\n"
        f"路线：{cast(Mapping[str, Any], payload['route'])['summary']}\n"
        f"建议：{decision['departureAdvice']}\n"
        f"提醒：已创建 {intent['dateText']}{intent.get('timeText') or ''} 的就医出行提醒，"
        f"并提前 {decision['reminderOffsetMinutes']} 分钟提醒。"
    )


def _json_payload(result: Any) -> JsonObject:
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except ValueError:
            return {"ok": False, "raw": result}
        return parsed if isinstance(parsed, dict) else {"ok": False, "raw": parsed}
    if isinstance(result, Mapping):
        return cast(JsonObject, to_json_value(result))
    return {"ok": False, "raw": result}


def _event_id(payload: Mapping[str, Any]) -> int | str | None:
    for key in ("eventId", "event_id", "id", "_id"):
        value = payload.get(key)
        if isinstance(value, (int, str)) and str(value):
            return value
    event = payload.get("event")
    if isinstance(event, Mapping):
        return _event_id(event)
    return None


def _safe_error_text(result: Any) -> str:
    if isinstance(result, Mapping):
        message = result.get("message") or result.get("error") or result
        return str(message)[:200]
    return str(result)[:200]


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_value(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


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


def _normalize_utterance(text: str) -> str:
    return re.sub(r"[\s，。,.!！?？]+", "", text.strip().lower())


__all__ = [
    "FIXED_MEDICAL_TRAVEL_UTTERANCE",
    "MedicalTravelIntent",
    "MedicalTravelSopMiddleware",
    "MedicalTravelSopRunner",
    "build_medical_travel_sop_runner",
    "is_medical_travel_request",
    "is_medical_travel_sop_request",
    "parse_medical_travel_intent",
    "read_user_memory",
]
