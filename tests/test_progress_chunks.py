from __future__ import annotations

import asyncio
from typing import Any

from mobile_agent import progress
from mobile_agent.tools.external import create_external_tools
from mobile_agent.tools.phone import create_phone_tools
from mobile_agent.tools.system import create_system_tools


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

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

    def get_session(self) -> _FakeSession:
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

    def get_default_client(self) -> _FakeSystemClient:
        return self.client


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

    assert gateway.session.calls == [("tap", {"x": 10, "y": 20})]
    assert emitted[0]["type"] == "task_progress"
    assert emitted[0]["label"] == "tap"
    assert emitted[0]["status"] == "running"
    assert emitted[0]["phase"] == "phone_tool"
    assert emitted[-1]["label"] == "tap"
    assert emitted[-1]["status"] == "completed"


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
