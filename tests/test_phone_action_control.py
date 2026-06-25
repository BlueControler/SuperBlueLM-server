from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mobile_agent.action_control import (
    PhoneActionCancelledError,
    PhoneDeviceBusyError,
    PhoneActionLimitError,
    PhoneActionRegistry,
    dispatch_phone_command,
    phone_action_scope,
)
from mobile_agent.trace import request_context


class _Session:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def send_command(self, message: str, data: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((message, data))
        return {"ok": True}


def test_dispatch_attaches_run_scoped_idempotency_metadata() -> None:
    registry = PhoneActionRegistry(max_actions_per_run=3)
    registry.start_run("run-1", "thread-1")
    session = _Session()

    with request_context(thread_id="thread-1", run_id="run-1"), phone_action_scope("tool-1"):
        result = asyncio.run(
            dispatch_phone_command(
                session,
                "launch",
                {"package": "com.tencent.mm"},
                registry=registry,
                device_id="device-1",
            )
        )

    assert result == {"ok": True}
    assert session.calls == [
        (
            "launch",
            {
                "package": "com.tencent.mm",
                "runId": "run-1",
                "actionId": session.calls[0][1]["actionId"],
                "actionIndex": 1,
                "deadlineEpochMs": session.calls[0][1]["deadlineEpochMs"],
            },
        )
    ]
    assert registry.snapshot("run-1")["status"] == "active"
    assert registry.snapshot("run-1")["actions"][0]["status"] == "succeeded"


def test_cancelled_run_is_reactivated_when_reused() -> None:
    """start_run 现在会重置已取消 run 的状态——新的移动请求复用 run_id 不应被旧取消状态阻止。"""
    registry = PhoneActionRegistry(max_actions_per_run=3)
    registry.start_run("run-1", "thread-1")
    registry.cancel_run("run-1", reason="user_cancelled")
    assert registry.snapshot("run-1")["status"] == "cancelled"

    # 模拟新请求复用同一个 run_id
    registry.start_run("run-1", "thread-1")
    assert registry.snapshot("run-1")["status"] == "active"

    session = _Session()
    with request_context(thread_id="thread-1", run_id="run-1"):
        asyncio.run(
            dispatch_phone_command(
                session,
                "launch",
                {"package": "com.tencent.mm"},
                registry=registry,
            )
        )

    assert len(session.calls) == 1
    assert session.calls[0][0] == "launch"
    assert session.calls[0][1]["package"] == "com.tencent.mm"


def test_action_budget_stops_repeated_model_dispatches() -> None:
    registry = PhoneActionRegistry(max_actions_per_run=2, identical_action_limit=2)
    registry.start_run("run-1", "thread-1")
    session = _Session()

    with request_context(thread_id="thread-1", run_id="run-1"):
        asyncio.run(dispatch_phone_command(session, "observe", {}, registry=registry))
        asyncio.run(dispatch_phone_command(session, "observe", {}, registry=registry))
        with pytest.raises(PhoneActionLimitError):
            asyncio.run(dispatch_phone_command(session, "observe", {}, registry=registry))

    assert len(session.calls) == 2


def test_a_recovered_graph_context_without_a_live_mobile_run_cannot_touch_device() -> None:
    from mobile_agent.action_control import PhoneActionUnregisteredRunError

    registry = PhoneActionRegistry()
    session = _Session()

    with request_context(thread_id="thread-1", run_id="old-run"):
        with pytest.raises(PhoneActionUnregisteredRunError, match="当前活动请求"):
            asyncio.run(dispatch_phone_command(session, "launch", {}, registry=registry))

    assert session.calls == []


def test_cancel_run_invokes_its_bound_stream_cancellation_callback_once() -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    cancelled: list[str] = []
    registry.bind_stream_cancellation("run-1", lambda: cancelled.append("stream"))

    registry.cancel_run("run-1", reason="user_cancelled")
    registry.cancel_run("run-1", reason="retry")

    assert cancelled == ["stream"]


def test_registry_keeps_the_real_langgraph_run_identity_for_cancellation_status(tmp_path) -> None:
    registry = PhoneActionRegistry(persist_path=tmp_path / "phone-runs.json")
    registry.start_run("mobile-run-1", "thread-1")

    registry.bind_backend_run("mobile-run-1", "backend-run-1")

    assert registry.backend_run_info("mobile-run-1") == ("thread-1", "backend-run-1")
    assert registry.snapshot("mobile-run-1")["backendStatus"] == "running"

    restored = PhoneActionRegistry(persist_path=tmp_path / "phone-runs.json")

    assert restored.stale_backend_runs() == [
        ("mobile-run-1", "thread-1", "backend-run-1", None),
    ]


def test_an_active_phone_run_exclusively_owns_its_device() -> None:
    registry = PhoneActionRegistry()
    registry.reserve_action(
        run_id="run-1",
        thread_id="thread-1",
        source_id="tool-1",
        command="launch",
        payload={"package": "com.feishu"},
        device_id="device-1",
    )

    with pytest.raises(PhoneDeviceBusyError, match="已有任务"):
        registry.reserve_action(
            run_id="run-2",
            thread_id="thread-2",
            source_id="tool-2",
            command="launch",
            payload={"package": "com.tencent.mm"},
            device_id="device-1",
        )
