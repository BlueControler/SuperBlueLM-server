from __future__ import annotations

import asyncio
from typing import Any

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from mobile_agent import progress
from mobile_agent.agent.scenarios.open_app_skill import (
    OpenAppSkillMiddleware,
    OpenAppRunner,
    is_open_app_request,
    parse_open_app_intent,
)


class _FakePhoneSession:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.result = result or {"currentPackage": "com.tencent.mm"}
        self.error = error
        self.commands: list[tuple[str, object]] = []

    async def send_command(self, message: str, data: object) -> dict[str, Any]:
        self.commands.append((message, data))
        if self.error is not None:
            raise self.error
        return self.result


class _FakePhoneGateway:
    def __init__(self, session: _FakePhoneSession) -> None:
        self.session = session
        self.device_ids: list[str | None] = []

    async def wait_for_session(self, device_id: str | None = None) -> _FakePhoneSession:
        self.device_ids.append(device_id)
        return self.session


class _FakeSystemClient:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def send_request(self, message: str, data: dict[str, str]) -> object:
        self.calls.append((message, data))
        return self.result


class _FakeSystemGateway:
    def __init__(self, result: object) -> None:
        self.client = _FakeSystemClient(result)
        self.device_ids: list[str | None] = []

    def get_default_client(self, device_id: str | None = None) -> _FakeSystemClient:
        self.device_ids.append(device_id)
        return self.client


def test_open_app_request_recognizes_supported_utterances() -> None:
    assert is_open_app_request("帮我打开微信。")
    assert is_open_app_request("打开高德地图。")
    assert is_open_app_request("启动 Chrome。")
    assert is_open_app_request("launch 设置")
    assert not is_open_app_request("看看我手机里有没有高德地图。")


def test_parse_open_app_intent_extracts_target_app() -> None:
    intent = parse_open_app_intent("帮我打开微信。")

    assert intent.should_handle is True
    assert intent.target_app == "微信"


def test_runner_launches_builtin_app_with_device_id_and_progress(monkeypatch: Any) -> None:
    emitted: list[dict[str, Any]] = []
    phone_session = _FakePhoneSession({"currentPackage": "com.tencent.mm"})
    phone_gateway = _FakePhoneGateway(phone_session)
    system_gateway = _FakeSystemGateway({})
    runner = OpenAppRunner(phone_gateway, system_gateway)  # type: ignore[arg-type]
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    result = asyncio.run(runner.run(parse_open_app_intent("帮我打开微信。"), "device-1"))

    assert phone_gateway.device_ids == ["device-1"]
    assert system_gateway.client.calls == []
    assert phone_session.commands == [("launch", {"package": "com.tencent.mm"})]
    assert result["success"] is True
    assert result["packageName"] == "com.tencent.mm"
    assert result["finalMessage"] == "已打开 微信。"
    messages = [event["message"] for event in emitted if event.get("type") == "task_progress"]
    assert messages == [
        "正在识别需要打开的应用。",
        "正在查找 微信 对应的应用包名。",
        "正在打开 微信。",
        "已打开 微信。",
    ]
    assert {event["status"] for event in emitted if event.get("type") == "task_progress"} >= {
        "running",
        "completed",
    }


def test_runner_falls_back_to_list_apps_for_unknown_app() -> None:
    phone_session = _FakePhoneSession({"currentPackage": "com.example.notes"})
    phone_gateway = _FakePhoneGateway(phone_session)
    system_gateway = _FakeSystemGateway({"com.example.notes": "示例笔记"})
    runner = OpenAppRunner(phone_gateway, system_gateway)  # type: ignore[arg-type]

    result = asyncio.run(runner.run(parse_open_app_intent("帮我打开示例笔记"), "device-1"))

    assert system_gateway.device_ids == ["device-1"]
    assert system_gateway.client.calls == [("listApps", {"type": "all"})]
    assert phone_session.commands == [("launch", {"package": "com.example.notes"})]
    assert result["appLabel"] == "示例笔记"
    assert result["finalMessage"] == "已打开 示例笔记。"


def test_runner_returns_chinese_failure_when_launch_fails() -> None:
    phone_session = _FakePhoneSession(error=RuntimeError("device offline"))
    phone_gateway = _FakePhoneGateway(phone_session)
    system_gateway = _FakeSystemGateway({})
    runner = OpenAppRunner(phone_gateway, system_gateway)  # type: ignore[arg-type]

    result = asyncio.run(runner.run(parse_open_app_intent("帮我打开设置。"), "device-1"))

    assert result["success"] is False
    assert result["finalMessage"] == "未能打开 设置：device offline"


def test_middleware_short_circuits_open_app_request() -> None:
    phone_session = _FakePhoneSession({"currentPackage": "com.android.settings"})
    middleware = OpenAppSkillMiddleware(
        _FakePhoneGateway(phone_session),  # type: ignore[arg-type]
        _FakeSystemGateway({}),  # type: ignore[arg-type]
    )
    request = type(
        "Request",
        (),
        {
            "messages": [HumanMessage(content="帮我打开设置。")],
            "runtime": type("Runtime", (), {"config": {"metadata": {"deviceId": "device-1"}}})(),
            "state": {},
        },
    )()

    async def fail_handler(_request: Any) -> ModelResponse[Any]:
        raise AssertionError("open app skill should not reach the main model")

    response = asyncio.run(middleware.awrap_model_call(request, fail_handler))

    assert isinstance(response.result[0], AIMessage)
    assert response.result[0].content == "已打开 设置。"
    assert response.result[0].additional_kwargs["open_app"]["packageName"] == "com.android.settings"


def test_middleware_ignores_non_open_app_request() -> None:
    middleware = OpenAppSkillMiddleware(
        _FakePhoneGateway(_FakePhoneSession()),  # type: ignore[arg-type]
        _FakeSystemGateway({}),  # type: ignore[arg-type]
    )
    request = type(
        "Request",
        (),
        {"messages": [HumanMessage(content="手机里有微信吗")], "runtime": None, "state": {}},
    )()

    async def handler(_request: Any) -> ModelResponse[Any]:
        return ModelResponse(result=[AIMessage(content="model called")])

    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert response.result[0].content == "model called"
