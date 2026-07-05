from __future__ import annotations

import asyncio
from typing import Any

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from mobile_agent import progress
from mobile_agent.agent.scenarios.weather_advice_skill import (
    WeatherAdviceMiddleware,
    WeatherAdviceRunner,
    is_weather_advice_request,
    parse_weather_advice_intent,
)


class _FakeAmapClient:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> object:
        self.calls.append((tool_name, arguments))
        return self.result


RAINY_WEATHER = {
    "status": "1",
    "forecasts": [
        {
            "city": "天津市",
            "casts": [
                {
                    "date": "2026-07-05",
                    "dayweather": "小雨",
                    "nightweather": "阴",
                    "daytemp": "31",
                    "nighttemp": "24",
                }
            ],
        }
    ],
}

SUNNY_WEATHER = {
    "content": [
        {
            "type": "text",
            "text": (
                '{"city":"北京市","forecasts":[{"casts":[{"dayweather":"晴",'
                '"nightweather":"多云","daytemp":"33","nighttemp":"25"}]}]}'
            ),
        }
    ]
}


def test_weather_advice_request_recognizes_supported_utterances() -> None:
    assert is_weather_advice_request("查询天津今天的天气，并告诉我需不需要带伞。")
    assert is_weather_advice_request("查一下北京今天会不会下雨。")
    assert is_weather_advice_request("今天去天津外面需要带伞吗？")
    assert not is_weather_advice_request("帮我打开天气。")


def test_parse_weather_advice_intent_extracts_city_and_date() -> None:
    intent = parse_weather_advice_intent("查询天津今天的天气，并告诉我需不需要带伞。")

    assert intent.should_handle is True
    assert intent.city == "天津"
    assert intent.date_text == "今天"


def test_runner_queries_weather_and_generates_umbrella_advice(monkeypatch: Any) -> None:
    emitted: list[dict[str, Any]] = []
    amap = _FakeAmapClient(RAINY_WEATHER)
    runner = WeatherAdviceRunner(amap)  # type: ignore[arg-type]
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    result = asyncio.run(runner.run(parse_weather_advice_intent("查询天津今天的天气，并告诉我需不需要带伞。")))

    assert amap.calls == [("maps_weather", {"city": "120000"})]
    assert result["success"] is True
    assert result["city"] == "天津"
    assert "天津今天：白天小雨，夜间阴，气温 24-31℃。" in result["finalMessage"]
    assert "建议带伞" in result["finalMessage"]
    messages = [event["message"] for event in emitted if event.get("type") == "task_progress"]
    assert messages == [
        "正在识别天气查询的城市和日期。",
        "正在查询 天津今天 的天气。",
        "正在根据天气判断是否需要带伞。",
        "已生成天气建议。",
    ]
    assert {event["status"] for event in emitted if event.get("type") == "task_progress"} >= {
        "running",
        "completed",
    }


def test_runner_handles_mcp_text_weather_without_rain() -> None:
    runner = WeatherAdviceRunner(_FakeAmapClient(SUNNY_WEATHER))  # type: ignore[arg-type]

    result = asyncio.run(runner.run(parse_weather_advice_intent("查一下北京今天会不会下雨。")))

    assert result["success"] is True
    assert "无需带伞" in result["finalMessage"]
    assert "北京今天：白天晴，夜间多云" in result["finalMessage"]


def test_runner_returns_chinese_error_when_weather_query_fails() -> None:
    runner = WeatherAdviceRunner(  # type: ignore[arg-type]
        _FakeAmapClient({"error": "weather_query_failed", "message": "INVALID_USER_KEY"})
    )

    result = asyncio.run(runner.run(parse_weather_advice_intent("查询天津今天的天气。")))

    assert result["success"] is False
    assert result["finalMessage"] == "天气查询失败：INVALID_USER_KEY"


def test_middleware_short_circuits_weather_request() -> None:
    middleware = WeatherAdviceMiddleware(_FakeAmapClient(RAINY_WEATHER))  # type: ignore[arg-type]
    request = type(
        "Request",
        (),
        {"messages": [HumanMessage(content="查询天津今天的天气，并告诉我需不需要带伞。")]},
    )()

    async def fail_handler(_request: Any) -> ModelResponse[Any]:
        raise AssertionError("weather advice skill should not reach the main model")

    response = asyncio.run(middleware.awrap_model_call(request, fail_handler))

    assert isinstance(response.result[0], AIMessage)
    assert "建议带伞" in str(response.result[0].content)
    assert response.result[0].additional_kwargs["weather_advice"]["city"] == "天津"


def test_middleware_ignores_non_weather_request() -> None:
    middleware = WeatherAdviceMiddleware(_FakeAmapClient(RAINY_WEATHER))  # type: ignore[arg-type]
    request = type("Request", (), {"messages": [HumanMessage(content="帮我打开微信")]})()

    async def handler(_request: Any) -> ModelResponse[Any]:
        return ModelResponse(result=[AIMessage(content="model called")])

    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert response.result[0].content == "model called"
