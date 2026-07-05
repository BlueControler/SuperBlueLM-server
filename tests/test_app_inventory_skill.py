from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from mobile_agent import progress
from mobile_agent.agent.app_inventory_skill import (
    AppInventoryIntent,
    AppInventoryQueryMiddleware,
    AppInventoryQueryRunner,
    filter_apps,
    format_app_inventory_result,
    is_app_inventory_query_request,
    parse_app_inventory_intent,
)


class _FakeClient:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def send_request(self, message: str, data: dict[str, str]) -> object:
        self.calls.append((message, data))
        return self.result


class _FakeSystemGateway:
    def __init__(self, result: object) -> None:
        self.client = _FakeClient(result)
        self.device_ids: list[str | None] = []

    def get_default_client(self, device_id: str | None = None) -> _FakeClient:
        self.device_ids.append(device_id)
        return self.client


APPS = {
    "com.android.chrome": "Chrome",
    "com.microsoft.emmx": "Edge",
    "com.vivo.browser": "系统浏览器",
    "com.tencent.mm": "微信",
    "com.autonavi.minimap": "高德地图",
    "com.tencent.meeting": "腾讯会议",
    "com.ss.android.lark": "飞书会议",
}


def test_app_inventory_request_recognizes_supported_utterances() -> None:
    assert is_app_inventory_query_request("读取当前手机已安装应用列表，并告诉我是否有浏览器类应用。")
    assert is_app_inventory_query_request("看看我手机里有没有高德地图。")
    assert is_app_inventory_query_request("我手机上有哪些浏览器")
    assert is_app_inventory_query_request("手机里有微信吗")
    assert not is_app_inventory_query_request("帮我打开微信")


def test_parse_intent_maps_builtin_categories_and_exact_apps() -> None:
    browser = parse_app_inventory_intent("读取当前手机已安装应用列表，并告诉我是否有浏览器类应用。")
    assert browser.should_handle is True
    assert browser.query_kind == "category"
    assert browser.category_key == "browser"
    assert browser.category_label == "浏览器类应用"

    exact = parse_app_inventory_intent("看看我手机里有没有高德地图。")
    assert exact.should_handle is True
    assert exact.query_kind == "exact_app"
    assert exact.query_text == "高德地图"

    map_navigation = parse_app_inventory_intent("读取应用列表，帮我找一下地图导航类应用。")
    assert map_navigation.category_key == "map_navigation"


def test_browser_filter_matches_label_and_package_without_wechat() -> None:
    intent = parse_app_inventory_intent("读取当前手机已安装应用列表，并告诉我是否有浏览器类应用。")

    matches = filter_apps(APPS, intent)

    assert [match.package_name for match in matches] == [
        "com.android.chrome",
        "com.microsoft.emmx",
        "com.vivo.browser",
    ]


def test_exact_app_filter_only_returns_gaode_map() -> None:
    intent = parse_app_inventory_intent("看看我手机里有没有高德地图。")

    matches = filter_apps(APPS, intent)

    assert [(match.app_label, match.package_name) for match in matches] == [
        ("高德地图", "com.autonavi.minimap")
    ]


def test_custom_query_finds_meeting_apps() -> None:
    intent = parse_app_inventory_intent("读取应用列表，找一下会议软件相关应用。")

    matches = filter_apps(APPS, intent)

    assert {match.app_label for match in matches} == {"腾讯会议", "飞书会议"}


def test_no_match_result_uses_clear_chinese_message() -> None:
    intent = AppInventoryIntent(
        should_handle=True,
        query_text="不存在应用",
        query_kind="custom",
        category_key="custom",
        category_label="不存在应用",
    )

    assert format_app_inventory_result(intent, []) == "已读取当前手机应用列表。未检测到明确的不存在应用。"


def test_runner_reads_list_apps_with_device_id_and_emits_progress(monkeypatch: Any) -> None:
    emitted: list[dict[str, Any]] = []
    gateway = _FakeSystemGateway(APPS)
    runner = AppInventoryQueryRunner(gateway)  # type: ignore[arg-type]
    intent = parse_app_inventory_intent("读取当前手机已安装应用列表，并告诉我是否有浏览器类应用。")
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    result = asyncio.run(runner.run(intent, "device-1"))

    assert gateway.device_ids == ["device-1"]
    assert gateway.client.calls == [("listApps", {"type": "all"})]
    assert result["matchedCount"] == 3
    assert result["totalApps"] == len(APPS)
    assert "共找到 3 个浏览器类应用" in result["finalMessage"]
    messages = [event["message"] for event in emitted if event.get("type") == "task_progress"]
    assert messages == [
        "正在分析需要检索的应用目标。",
        "已识别为检索：浏览器类应用。",
        "正在读取手机上的所有已安装应用。",
        f"已完成应用列表读取，共读取 {len(APPS)} 个应用。",
        "正在根据“浏览器类应用”过滤检索结果。",
        "已找到 3 个浏览器类应用。",
        result["finalMessage"],
    ]


def test_middleware_short_circuits_app_inventory_request() -> None:
    gateway = _FakeSystemGateway(APPS)
    middleware = AppInventoryQueryMiddleware(gateway)  # type: ignore[arg-type]
    request = type(
        "Request",
        (),
        {
            "messages": [HumanMessage(content="读取当前手机已安装应用列表，并告诉我是否有浏览器类应用。")],
            "runtime": type(
                "Runtime",
                (),
                {"config": {"configurable": {"device_id": "device-1"}}},
            )(),
            "state": {},
        },
    )()

    async def fail_handler(_request: Any) -> ModelResponse[Any]:
        raise AssertionError("app inventory skill should not reach the main model")

    response = asyncio.run(middleware.awrap_model_call(request, fail_handler))

    assert isinstance(response.result[0], AIMessage)
    assert "Chrome（com.android.chrome）" in str(response.result[0].content)
    payload = response.result[0].additional_kwargs["app_inventory_query"]
    assert payload["queryKind"] == "category"
    assert payload["matchedCount"] == 3
    assert gateway.device_ids == ["device-1"]


def test_middleware_ignores_non_inventory_request() -> None:
    gateway = _FakeSystemGateway(APPS)
    middleware = AppInventoryQueryMiddleware(gateway)  # type: ignore[arg-type]
    request = type(
        "Request",
        (),
        {"messages": [HumanMessage(content="帮我打开微信")], "runtime": None, "state": {}},
    )()

    async def handler(_request: Any) -> ModelResponse[Any]:
        return ModelResponse(result=[AIMessage(content="model called")])

    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert response.result[0].content == "model called"
    assert gateway.client.calls == []


def test_model_response_additional_kwargs_is_structured_json() -> None:
    gateway = _FakeSystemGateway(APPS)
    middleware = AppInventoryQueryMiddleware(gateway)  # type: ignore[arg-type]
    request = type(
        "Request",
        (),
        {"messages": [HumanMessage(content="看看我手机里有没有高德地图。")], "runtime": None, "state": {}},
    )()

    async def fail_handler(_request: Any) -> ModelResponse[Any]:
        raise AssertionError("app inventory skill should not reach the main model")

    response = asyncio.run(middleware.awrap_model_call(request, fail_handler))
    payload = response.result[0].additional_kwargs["app_inventory_query"]

    assert json.loads(json.dumps(payload, ensure_ascii=False))["matches"] == [
        {
            "appLabel": "高德地图",
            "packageName": "com.autonavi.minimap",
            "matchedKeyword": "高德地图",
            "score": 100,
        }
    ]
