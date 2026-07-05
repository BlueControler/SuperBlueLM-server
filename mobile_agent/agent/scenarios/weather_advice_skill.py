from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ...json_types import JsonObject, JsonValue, to_json_value
from ...progress import emit_task_complexity, emit_task_progress
from ...tools.external import (
    AmapClient,
    AmapHttpMcpClient,
    call_amap_mcp_tool,
    resolve_amap_city_adcode,
)
from ..middleware import _message_content_to_text
from ..state import MobileAgentState

WEATHER_PHASE = "weather_advice_skill"
WEATHER_TASK_TITLE = "天气建议"
TOTAL_STEPS = 4

CITY_NAMES = (
    "北京",
    "天津",
    "上海",
    "重庆",
    "广州",
    "深圳",
    "杭州",
    "南京",
    "成都",
    "武汉",
    "西安",
    "长沙",
    "郑州",
    "青岛",
    "苏州",
)
DATE_TEXTS = ("今天", "明天", "后天")
WEATHER_MARKERS = ("天气", "下雨", "带伞", "气温", "穿衣")
RAIN_MARKERS = ("雨", "阵雨", "雷阵雨", "降水", "雪", "冰雹")


@dataclass(frozen=True)
class WeatherAdviceIntent:
    should_handle: bool
    city: str
    date_text: str


class WeatherAdviceMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    state_schema = MobileAgentState

    def __init__(self, amap_client: AmapClient | None = None) -> None:
        self.runner = WeatherAdviceRunner(amap_client or AmapHttpMcpClient())

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        intent = _intent_from_request(request.messages)
        if not intent.should_handle:
            return handler(request)
        result = asyncio.run(self.runner.run(intent))
        return _model_response(result)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        intent = _intent_from_request(request.messages)
        if not intent.should_handle:
            return await handler(request)
        result = await self.runner.run(intent)
        return _model_response(result)


@dataclass(frozen=True)
class WeatherAdviceRunner:
    amap_client: AmapClient

    async def run(self, intent: WeatherAdviceIntent) -> JsonObject:
        emit_task_complexity(
            complexity="simple",
            track_steps=True,
            reason=WEATHER_PHASE,
            message="识别任务类型：天气查询与简短建议",
        )
        self._emit_progress(
            step=1,
            status="running",
            step_title="识别查询地点",
            message="正在识别天气查询的城市和日期。",
            tool_name="weather_intent",
            completed_count=0,
        )
        self._emit_progress(
            step=2,
            status="running",
            step_title="查询天气",
            message=f"正在查询 {intent.city}{intent.date_text} 的天气。",
            tool_name="weather_query",
            completed_count=1,
        )

        try:
            adcode = await resolve_amap_city_adcode(intent.city)
        except Exception as exc:
            return self._failed_result(intent, f"天气查询失败：{exc}", step=2)

        weather_result = await call_amap_mcp_tool(
            self.amap_client,
            "maps_weather",
            {"city": adcode},
        )
        if isinstance(weather_result, Mapping) and "error" in weather_result:
            reason = str(weather_result.get("message") or weather_result.get("error"))
            return self._failed_result(intent, f"天气查询失败：{reason}", step=2)

        report = _extract_weather_report(weather_result)
        if report is None:
            return self._failed_result(intent, "天气查询失败：天气服务返回格式不正确。", step=2)

        self._emit_progress(
            step=3,
            status="running",
            step_title="生成建议",
            message="正在根据天气判断是否需要带伞。",
            tool_name="weather_advice",
            completed_count=2,
        )
        weather_summary = _weather_summary(report)
        umbrella_advice = _umbrella_advice(report)
        final_message = f"{intent.city}{intent.date_text}：{weather_summary}。建议：{umbrella_advice}"
        self._emit_progress(
            step=4,
            status="completed",
            step_title="完成",
            message="已生成天气建议。",
            tool_name="finish",
            completed_count=4,
        )
        return cast(JsonObject, to_json_value({
            "success": True,
            "city": intent.city,
            "dateText": intent.date_text,
            "weatherSummary": weather_summary,
            "umbrellaAdvice": umbrella_advice,
            "rawWeather": weather_result,
            "finalMessage": final_message,
        }))

    def _failed_result(self, intent: WeatherAdviceIntent, message: str, *, step: int) -> JsonObject:
        self._emit_progress(
            step=step,
            status="failed",
            step_title="查询天气",
            message=message,
            tool_name="weather_query",
            completed_count=max(step - 1, 0),
            error=message,
        )
        return {
            "success": False,
            "city": intent.city,
            "dateText": intent.date_text,
            "finalMessage": message,
            "error": message,
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
            phase=WEATHER_PHASE,
            task_title=WEATHER_TASK_TITLE,
            step_title=step_title,
            message=message,
            tool_name=tool_name,
            progress_key=f"weather-advice-{step}",
            current_step=step,
            total_steps=TOTAL_STEPS,
            completed_steps=_completed_steps(completed_count),
            error=error,
        )


def is_weather_advice_request(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in WEATHER_MARKERS) and _extract_city(text) is not None


def parse_weather_advice_intent(text: str) -> WeatherAdviceIntent:
    city = _extract_city(text) or ""
    date_text = _extract_date_text(text)
    return WeatherAdviceIntent(
        should_handle=bool(city) and is_weather_advice_request(text),
        city=city,
        date_text=date_text,
    )


def _extract_city(text: str) -> str | None:
    normalized = _normalize(text)
    for city in CITY_NAMES:
        if city in normalized or f"{city}市" in normalized:
            return city
    return None


def _extract_date_text(text: str) -> str:
    normalized = _normalize(text)
    for date_text in DATE_TEXTS:
        if date_text in normalized:
            return date_text
    return "今天"


def _extract_weather_report(value: JsonValue) -> Mapping[str, object] | None:
    payload = _unwrap_weather_payload(value)
    if payload is None:
        return None
    forecasts = payload.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        return None
    first_forecast = forecasts[0]
    if not isinstance(first_forecast, Mapping):
        return None
    casts = first_forecast.get("casts")
    if isinstance(casts, list) and casts and isinstance(casts[0], Mapping):
        return casts[0]
    return first_forecast


def _unwrap_weather_payload(value: JsonValue) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    if "forecasts" in value:
        return value
    structured = value.get("structuredContent")
    if isinstance(structured, Mapping) and "forecasts" in structured:
        return structured
    content = value.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping) and "forecasts" in parsed:
            return parsed
    return None


def _weather_summary(report: Mapping[str, object]) -> str:
    day_weather = _weather_field(report, "dayweather") or _weather_field(report, "weather") or "未知"
    night_weather = _weather_field(report, "nightweather")
    day_temp = _weather_field(report, "daytemp") or _weather_field(report, "temperature")
    night_temp = _weather_field(report, "nighttemp")
    pieces = [f"白天{day_weather}"]
    if night_weather:
        pieces.append(f"夜间{night_weather}")
    temp_text = _temperature_text(day_temp, night_temp)
    if temp_text:
        pieces.append(f"气温 {temp_text}")
    return "，".join(pieces)


def _umbrella_advice(report: Mapping[str, object]) -> str:
    weather_text = " ".join(
        value
        for value in (
            _weather_field(report, "dayweather"),
            _weather_field(report, "nightweather"),
            _weather_field(report, "weather"),
        )
        if value
    )
    if any(marker in weather_text for marker in RAIN_MARKERS):
        return "有降水，建议带伞。"
    return "今天无明显降水，无需带伞。"


def _weather_field(report: Mapping[str, object], key: str) -> str | None:
    value = report.get(key)
    return str(value).strip() if value is not None and str(value).strip() else None


def _temperature_text(day_temp: str | None, night_temp: str | None) -> str | None:
    if day_temp and night_temp:
        try:
            values = sorted([int(float(day_temp)), int(float(night_temp))])
            return f"{values[0]}-{values[1]}℃"
        except ValueError:
            return f"{night_temp}-{day_temp}℃"
    if day_temp:
        return f"{day_temp}℃"
    return None


def _intent_from_request(messages: Sequence[BaseMessage]) -> WeatherAdviceIntent:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return parse_weather_advice_intent(_message_content_to_text(message.content))
    return WeatherAdviceIntent(should_handle=False, city="", date_text="今天")


def _model_response(result: JsonObject) -> ModelResponse[Any]:
    return ModelResponse(
        result=[
            AIMessage(
                content=str(result["finalMessage"]),
                additional_kwargs={"weather_advice": result},
            )
        ]
    )


def _completed_steps(count: int) -> list[JsonObject]:
    definitions = (
        ("识别查询地点", "weather_intent"),
        ("查询天气", "weather_query"),
        ("生成建议", "weather_advice"),
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
    "WeatherAdviceIntent",
    "WeatherAdviceMiddleware",
    "WeatherAdviceRunner",
    "is_weather_advice_request",
    "parse_weather_advice_intent",
]
