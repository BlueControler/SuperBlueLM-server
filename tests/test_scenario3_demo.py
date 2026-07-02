from __future__ import annotations

from starlette.testclient import TestClient

from mobile_agent.http_app import app


def test_scenario3_notification_demo_returns_waiting_confirmation_progress() -> None:
    response = TestClient(app).post(
        "/mobile/demo/scenario3/notification",
        json={"threadId": "thread-1", "runId": "run-1"},
    )

    assert response.status_code == 200
    payload = response.json()
    events = payload["events"]
    assert payload["dryRun"] is True
    assert events[0]["type"] == "task_progress"
    assert events[0]["taskTitle"] == "为会议通知创建提醒"
    assert events[0]["status"] == "running"
    assert events[0]["currentStep"] == 1
    assert events[0]["totalSteps"] == 3
    assert events[0]["stepTitle"] == "检测到会议通知"
    assert events[0]["phase"] == "phone_tool"
    assert events[0]["toolName"] == "list_notifications"

    waiting = events[-1]
    assert waiting["status"] == "waiting_confirmation"
    assert waiting["requiresConfirmation"] is True
    assert waiting["confirmationId"]
    assert waiting["toolName"] == "needs_confirmation"
    assert waiting["canCancel"] is True
    assert waiting["canTakeOver"] is True


def test_scenario3_confirm_demo_returns_create_event_and_completed_progress() -> None:
    client = TestClient(app)
    started = client.post(
        "/mobile/demo/scenario3/notification",
        json={"threadId": "thread-1", "runId": "run-1"},
    ).json()
    confirmation_id = started["events"][-1]["confirmationId"]

    response = client.post(
        "/mobile/demo/scenario3/confirm",
        json={"confirmationId": confirmation_id},
    )

    assert response.status_code == 200
    events = response.json()["events"]
    assert [event["status"] for event in events] == ["running", "completed"]
    assert events[0]["currentStep"] == 3
    assert events[0]["toolName"] == "create_event"
    assert events[0]["dryRun"] is True
    assert events[-1]["phase"] == "finalizing"
    assert events[-1]["message"] == "会议提醒已创建"


def test_scenario3_reject_demo_finishes_without_write_tool() -> None:
    client = TestClient(app)
    started = client.post(
        "/mobile/demo/scenario3/notification",
        json={"threadId": "thread-1", "runId": "run-1"},
    ).json()
    confirmation_id = started["events"][-1]["confirmationId"]

    response = client.post(
        "/mobile/demo/scenario3/reject",
        json={"confirmationId": confirmation_id},
    )

    assert response.status_code == 200
    events = response.json()["events"]
    assert [event["status"] for event in events] == ["cancelled"]
    assert all(event.get("toolName") != "create_event" for event in events)
    assert events[0]["message"] == "已取消创建会议提醒"


def test_scenario3_take_over_demo_returns_taken_over_progress() -> None:
    client = TestClient(app)
    started = client.post(
        "/mobile/demo/scenario3/notification",
        json={"threadId": "thread-1", "runId": "run-1"},
    ).json()
    confirmation_id = started["events"][-1]["confirmationId"]

    response = client.post(
        "/mobile/demo/scenario3/take-over",
        json={"confirmationId": confirmation_id},
    )

    assert response.status_code == 200
    events = response.json()["events"]
    assert [event["status"] for event in events] == ["taken_over"]
    assert events[0]["canCancel"] is False
    assert events[0]["canTakeOver"] is False
