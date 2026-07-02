from __future__ import annotations

import asyncio
from typing import Any

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from starlette.testclient import TestClient

from mobile_agent import progress
from mobile_agent.agent.risk_gate import HighRiskActionGateMiddleware
from mobile_agent.http_app import app
from mobile_agent.trace import activate_request_context, clear_request_context, is_high_risk_tool


def _request(
    name: str,
    *,
    args: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args or {}, "id": "call-1", "type": "tool_call"},
        tool=None,
        state={"messages": [], **(state or {})},
        runtime=None,
    )


def test_generic_confirm_endpoint_reuses_scenario3_transaction() -> None:
    client = TestClient(app)
    started = client.post(
        "/mobile/demo/scenario3/notification",
        json={"threadId": "thread-1", "runId": "run-1"},
    ).json()
    confirmation_id = started["events"][-1]["confirmationId"]

    response = client.post(f"/mobile/confirmations/{confirmation_id}/confirm")

    assert response.status_code == 200
    payload = response.json()
    assert payload["confirmationId"] == confirmation_id
    assert payload["status"] == "confirmed"
    assert [event["status"] for event in payload["events"]] == ["running", "completed"]
    assert payload["events"][0]["toolName"] == "create_event"
    assert payload["events"][0]["dryRun"] is True


def test_repeated_generic_confirm_does_not_execute_twice() -> None:
    client = TestClient(app)
    started = client.post(
        "/mobile/demo/scenario3/notification",
        json={"threadId": "thread-1", "runId": "run-1"},
    ).json()
    confirmation_id = started["events"][-1]["confirmationId"]

    first = client.post(f"/mobile/confirmations/{confirmation_id}/confirm")
    second = client.post(f"/mobile/confirmations/{confirmation_id}/confirm")

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["error"] == "confirmation_already_resolved"


def test_generic_reject_endpoint_finishes_scenario3_without_write_tool() -> None:
    client = TestClient(app)
    started = client.post(
        "/mobile/demo/scenario3/notification",
        json={"threadId": "thread-1", "runId": "run-1"},
    ).json()
    confirmation_id = started["events"][-1]["confirmationId"]

    response = client.post(f"/mobile/confirmations/{confirmation_id}/reject")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "rejected"
    assert [event["status"] for event in payload["events"]] == ["cancelled"]
    assert all(event.get("toolName") != "create_event" for event in payload["events"])


def test_high_risk_gate_creates_generic_confirmation_transaction(monkeypatch: Any) -> None:
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)
    context = activate_request_context(thread_id="thread-1", run_id="run-1")
    assert context is not None
    middleware = HighRiskActionGateMiddleware()
    called = False

    async def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return ToolMessage(content="unexpected", tool_call_id="call-1", name="create_event")

    try:
        result = asyncio.run(
            middleware.awrap_tool_call(
                _request(
                    "create_event",
                    args={"event": {"title": "项目会", "dtstart": 1780000000000}},
                ),
                handler,
            )
        )
    finally:
        clear_request_context()

    assert called is False
    assert isinstance(result, Command)
    confirmation_id = result.update["awaiting_confirmation_id"]
    assert isinstance(confirmation_id, str)
    assert confirmation_id
    needs_confirmation = [event for event in emitted if event.get("type") == "needs_confirmation"]
    task_progress = [event for event in emitted if event.get("type") == "task_progress"]
    assert needs_confirmation[-1]["confirmationId"] == confirmation_id
    assert needs_confirmation[-1]["toolName"] == "create_event"
    assert needs_confirmation[-1]["dryRun"] is True
    assert task_progress[-1]["status"] == "waiting_confirmation"
    assert task_progress[-1]["confirmationId"] == confirmation_id
    assert task_progress[-1]["canCancel"] is True
    assert task_progress[-1]["canTakeOver"] is True

    response = TestClient(app).post(f"/mobile/confirmations/{confirmation_id}/confirm")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "confirmed"
    assert payload["dryRun"] is True
    assert [event["status"] for event in payload["events"]] == ["running", "completed"]
    assert all(event["dryRun"] is True for event in payload["events"])


def test_required_write_like_tools_are_high_risk() -> None:
    for tool_name in (
        "create_event",
        "update_reminders",
        "feishu_cli",
        "wecom_cli",
        "archive_file",
        "run_cli_command",
    ):
        assert is_high_risk_tool(tool_name, {}) is True
