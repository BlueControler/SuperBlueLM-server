from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from mobile_agent import progress
from mobile_agent.gateways.phone import DeviceGatewayError
from mobile_agent.trace import clear_request_context
from mobile_agent.tools.external import create_external_tools
from mobile_agent.tools.phone import create_phone_tools
from mobile_agent.tools.system import create_system_tools


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.device_info = SimpleNamespace(width=1080, height=2400)

    async def send_command(self, message: str, data: Any) -> dict[str, Any]:
        self.calls.append((message, data))
        return {
            "currentPackage": "com.example",
            "activity": ".MainActivity",
            "screenshot": "base64",
            "ui": "<node />",
        }


class _FakePhoneGateway:
    def __init__(self) -> None:
        self.session = _FakeSession()
        self.device_ids: list[str | None] = []

    def get_session(self, device_id: str | None = None) -> _FakeSession:
        self.device_ids.append(device_id)
        return self.session

    async def wait_for_session(
        self,
        device_id: str | None = None,
        timeout: float = 3.0,
    ) -> _FakeSession:
        self.device_ids.append(device_id)
        return self.session


class _FakeSystemClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def send_request(self, message: str, data: Any) -> dict[str, Any]:
        self.calls.append((message, data))
        return {"ok": True}


class _FakeSystemGateway:
    def __init__(self) -> None:
        self.client = _FakeSystemClient()

    def get_default_client(self, device_id: str | None = None) -> _FakeSystemClient:
        return self.client


@pytest.fixture(autouse=True)
def _clear_trace_context_between_tests() -> None:
    clear_request_context()
    yield
    clear_request_context()


def test_emit_task_progress_writes_custom_payload(monkeypatch: Any) -> None:
    emitted: list[dict[str, Any]] = []

    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    progress.emit_task_progress(
        label="tap",
        status="running",
        phase="phone_tool",
        message="正在点击屏幕",
        tool_name="tap",
        progress_key="phone-todo-2",
        current_step=2,
        total_steps=5,
    )

    assert emitted == [
        {
            "type": "task_progress",
            "label": "tap",
            "status": "running",
            "phase": "phone_tool",
            "message": "正在点击屏幕",
            "toolName": "tap",
            "progressKey": "phone-todo-2",
            "currentStep": 2,
            "totalSteps": 5,
        }
    ]


def test_emit_task_progress_writes_task_card_payload(monkeypatch: Any) -> None:
    emitted: list[dict[str, Any]] = []

    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    progress.emit_task_progress(
        label="会议通知",
        task_title="为会议通知创建提醒",
        status="waiting_confirmation",
        phase="confirmation",
        step_title="等待确认是否创建会议提醒",
        message="检测到会议通知，是否创建提醒？",
        tool_name="needs_confirmation",
        progress_key="scenario3-demo",
        current_step=2,
        total_steps=3,
        requires_confirmation=True,
        confirmation_id="confirm-123",
        can_cancel=True,
        can_take_over=True,
    )

    assert emitted == [
        {
            "type": "task_progress",
            "label": "会议通知",
            "taskTitle": "为会议通知创建提醒",
            "status": "waiting_confirmation",
            "phase": "confirmation",
            "stepTitle": "等待确认是否创建会议提醒",
            "message": "检测到会议通知，是否创建提醒？",
            "toolName": "needs_confirmation",
            "progressKey": "scenario3-demo",
            "currentStep": 2,
            "totalSteps": 3,
            "requiresConfirmation": True,
            "confirmationId": "confirm-123",
            "canCancel": True,
            "canTakeOver": True,
        }
    ]


def test_emit_task_progress_ignores_missing_stream_context(monkeypatch: Any) -> None:
    def raise_no_stream() -> Any:
        raise RuntimeError("no active stream")

    monkeypatch.setattr(progress, "get_stream_writer", raise_no_stream)

    progress.emit_task_progress(
        label="observe",
        status="running",
        phase="phone_tool",
    )


def test_phone_tool_emits_started_and_completed_progress(monkeypatch: Any) -> None:
    emitted: list[dict[str, Any]] = []
    gateway = _FakePhoneGateway()
    tools = {tool.name: tool for tool in create_phone_tools(gateway)}

    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    asyncio.run(tools["tap"].ainvoke({"x": 10, "y": 20}))

    _assert_controlled_call(gateway.session.calls, "tap", {"x": 10, "y": 20})
    assert emitted[0]["type"] == "task_progress"
    assert emitted[0]["label"] == "tap"
    assert emitted[0]["status"] == "running"
    assert emitted[0]["phase"] == "phone_tool"
    assert emitted[-1]["label"] == "tap"
    assert emitted[-1]["status"] == "completed"


def _assert_controlled_call(
    calls: list[tuple[str, dict[str, Any]]],
    expected_command: str,
    expected_payload: dict[str, Any],
) -> None:
    assert len(calls) == 1
    command, payload = calls[0]
    assert command == expected_command
    assert {key: payload[key] for key in expected_payload} == expected_payload
    assert isinstance(payload["runId"], str) and payload["runId"]
    assert isinstance(payload["actionId"], str) and payload["actionId"]
    assert payload["actionIndex"] == 1
    assert isinstance(payload["deadlineEpochMs"], int)


def test_phone_tool_routes_to_explicit_device_id() -> None:
    gateway = _FakePhoneGateway()
    tools = {tool.name: tool for tool in create_phone_tools(gateway)}

    asyncio.run(tools["tap"].ainvoke({"x": 10, "y": 20, "device_id": "device-uuid-1"}))

    assert gateway.device_ids == ["device-uuid-1"]


def test_scroll_uses_device_relative_safe_area_coordinates() -> None:
    gateway = _FakePhoneGateway()
    tools = {tool.name: tool for tool in create_phone_tools(gateway)}

    asyncio.run(tools["scroll"].ainvoke({"direction": "up", "distance": "medium"}))

    _assert_controlled_call(
        gateway.session.calls,
        "swipe",
        {"startX": 540, "startY": 1800, "endX": 540, "endY": 600},
    )


def test_scroll_adapts_to_different_device_dimensions() -> None:
    gateway = _FakePhoneGateway()
    gateway.session.device_info = SimpleNamespace(width=1440, height=3200)
    tools = {tool.name: tool for tool in create_phone_tools(gateway)}

    asyncio.run(tools["scroll"].ainvoke({"direction": "down", "distance": "short"}))

    _assert_controlled_call(
        gateway.session.calls,
        "swipe",
        {"startX": 720, "startY": 1120, "endX": 720, "endY": 2080},
    )


def test_bound_phone_tool_rejects_device_id_override() -> None:
    gateway = _FakePhoneGateway()
    tools = {
        tool.name: tool
        for tool in create_phone_tools(gateway, default_device_id="device-uuid-1")
    }

    result = json.loads(
        asyncio.run(
            tools["tap"].ainvoke({"x": 10, "y": 20, "device_id": "device-uuid-2"})
        )
    )

    assert result["error"] == "phone_tool_failed"
    assert result["recoverable"] is False


def test_phone_tool_returns_recoverable_error_when_device_does_not_reconnect() -> None:
    class DisconnectedGateway:
        async def wait_for_session(
            self,
            device_id: str | None = None,
            timeout: float = 3.0,
        ) -> None:
            raise DeviceGatewayError("device_not_connected")

    tools = {tool.name: tool for tool in create_phone_tools(DisconnectedGateway())}

    result = json.loads(asyncio.run(tools["home"].ainvoke({"device_id": "device-1"})))

    assert result == {
        "error": "device_not_connected",
        "message": "手机连接已断开，请重新连接后重试",
        "recoverable": True,
    }


def test_return_direct_phone_tools_preserve_device_not_connected_error() -> None:
    class DisconnectedGateway:
        async def wait_for_session(
            self,
            device_id: str | None = None,
            timeout: float = 3.0,
        ) -> None:
            raise DeviceGatewayError("device_not_connected")

    tools = {tool.name: tool for tool in create_phone_tools(DisconnectedGateway())}

    for tool_name in ("interact", "take_over"):
        result = json.loads(
            asyncio.run(tools[tool_name].ainvoke({"message": "请操作手机"}))
        )
        assert result == {
            "error": "device_not_connected",
            "message": "手机连接已断开，请重新连接后重试",
            "recoverable": True,
        }


def test_phone_tool_returns_recoverable_error_when_device_disconnects_during_command() -> None:
    class DisconnectingSession:
        async def send_command(self, message: str, data: Any) -> None:
            del message, data
            raise DeviceGatewayError("Device is disconnected.")

    class DisconnectingGateway:
        async def wait_for_session(
            self,
            device_id: str | None = None,
            timeout: float = 3.0,
        ) -> DisconnectingSession:
            del device_id, timeout
            return DisconnectingSession()

    tools = {tool.name: tool for tool in create_phone_tools(DisconnectingGateway())}

    result = json.loads(asyncio.run(tools["home"].ainvoke({"device_id": "device-1"})))

    assert result == {
        "error": "device_not_connected",
        "message": "手机连接已断开，请重新连接后重试",
        "recoverable": True,
    }


def test_phone_tool_does_not_continue_on_replaced_session() -> None:
    original_session = _FakeSession()
    replacement_session = _FakeSession()

    class ReconnectedGateway:
        async def wait_for_session(
            self,
            device_id: str | None = None,
            timeout: float = 3.0,
        ) -> _FakeSession:
            del device_id, timeout
            return replacement_session

    tools = {
        tool.name: tool
        for tool in create_phone_tools(
            ReconnectedGateway(),
            expected_session=original_session,
        )
    }

    result = json.loads(asyncio.run(tools["home"].ainvoke({"device_id": "device-1"})))

    assert result == {
        "error": "device_not_connected",
        "message": "手机连接已断开，请重新连接后重试",
        "recoverable": True,
    }
    assert replacement_session.calls == []


def test_system_tool_emits_started_and_completed_progress(monkeypatch: Any) -> None:
    emitted: list[dict[str, Any]] = []
    gateway = _FakeSystemGateway()
    tools = {tool.name: tool for tool in create_system_tools(gateway)}

    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    asyncio.run(tools["list_apps"].ainvoke({}))

    assert gateway.client.calls == [("listApps", {"type": "all"})]
    assert emitted[0]["label"] == "list_apps"
    assert emitted[0]["status"] == "running"
    assert emitted[0]["phase"] == "system_tool"
    assert emitted[-1]["label"] == "list_apps"
    assert emitted[-1]["status"] == "completed"


def test_external_tool_emits_started_and_completed_progress(monkeypatch: Any) -> None:
    emitted: list[dict[str, Any]] = []
    tools = {tool.name: tool for tool in create_external_tools()}

    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    asyncio.run(tools["external_tools_status"].ainvoke({}))

    assert emitted[0]["label"] == "external_tools_status"
    assert emitted[0]["status"] == "running"
    assert emitted[0]["phase"] == "external_tool"
    assert emitted[-1]["label"] == "external_tools_status"
    assert emitted[-1]["status"] == "completed"
