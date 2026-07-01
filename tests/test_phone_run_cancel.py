from __future__ import annotations

import asyncio

from starlette.testclient import TestClient

from mobile_agent.action_control import PhoneActionRegistry
from mobile_agent.http_app import app, cancel_recovered_mobile_runs


class _Gateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.fail = False

    async def cancel_run(self, run_id: str, device_id: str | None = None) -> None:
        if self.fail:
            from mobile_agent.gateways.phone import DeviceGatewayError

            raise DeviceGatewayError("device unavailable")
        self.calls.append((run_id, device_id))


def _stable_cancel_payload(payload: dict[str, object]) -> dict[str, object]:
    assert payload["mobileRunId"] == payload["runId"]
    timestamp = payload.pop("timestamp")
    assert isinstance(timestamp, int)
    assert timestamp > 0
    return payload


def test_cancel_route_blocks_future_actions_and_notifies_the_bound_device(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    registry.reserve_action(
        run_id="run-1",
        thread_id="thread-1",
        source_id="tool-1",
        command="launch",
        payload={"package": "com.tencent.mm"},
        device_id="device-1",
    )
    gateway = _Gateway()

    async def cancel_backend(_: str, __: dict[str, str]) -> str:
        return "not_started"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.phone_gateway", gateway)
    monkeypatch.setattr("mobile_agent.http_app.cancel_upstream_run", cancel_backend)

    response = TestClient(app).post("/mobile/threads/thread-1/runs/run-1/cancel")

    assert response.status_code == 200
    assert _stable_cancel_payload(response.json()) == {
        "runId": "run-1",
        "mobileRunId": "run-1",
        "threadId": "thread-1",
        "status": "local_fenced_only",
        "backendRunId": None,
        "backendStatus": "not_started",
        "deviceStatus": "canceled",
        "localFenced": True,
        "retryable": False,
        "cancelSource": "user",
        "terminalReason": "user",
    }
    assert registry.snapshot("run-1")["status"] == "cancelled"
    assert gateway.calls == [("run-1", "device-1")]


def test_cancel_route_rejects_a_run_from_another_thread(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)

    response = TestClient(app).post("/mobile/threads/thread-2/runs/run-1/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "not_found"


def test_cancel_route_requests_cancellation_of_the_real_langgraph_run(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    registry.bind_backend_run("run-1", "backend-run-1")
    gateway = _Gateway()
    requested: list[tuple[str, dict[str, str]]] = []

    async def cancel_backend(run_id: str, headers: dict[str, str]) -> str:
        requested.append((run_id, headers))
        return "cancel_requested"

    async def read_status(_: str, __: dict[str, str]) -> str:
        return "cancelled"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.phone_gateway", gateway)
    monkeypatch.setattr("mobile_agent.http_app.cancel_upstream_run", cancel_backend)
    monkeypatch.setattr("mobile_agent.http_app.upstream_run_status", read_status)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/run-1/cancel",
        headers={"X-Api-Key": "key-1"},
    )

    assert response.status_code == 200
    assert _stable_cancel_payload(response.json()) == {
        "runId": "run-1",
        "mobileRunId": "run-1",
        "threadId": "thread-1",
        "status": "canceled_confirmed",
        "backendRunId": "backend-run-1",
        "backendStatus": "cancelled",
        "deviceStatus": "not_bound",
        "localFenced": True,
        "retryable": False,
        "cancelSource": "user",
        "terminalReason": "user",
    }
    assert requested == [("run-1", {"x-api-key": "key-1"})]


def test_cancel_route_reports_backend_still_running_when_cancel_is_not_confirmed(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    registry.bind_backend_run("run-1", "backend-run-1")
    gateway = _Gateway()

    async def cancel_backend(_: str, __: dict[str, str]) -> str:
        return "cancel_requested"

    async def read_status(_: str, __: dict[str, str]) -> str:
        return "running"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.phone_gateway", gateway)
    monkeypatch.setattr("mobile_agent.http_app.cancel_upstream_run", cancel_backend)
    monkeypatch.setattr("mobile_agent.http_app.upstream_run_status", read_status)

    response = TestClient(app).post("/mobile/threads/thread-1/runs/run-1/cancel")

    assert response.status_code == 200
    assert _stable_cancel_payload(response.json()) == {
        "runId": "run-1",
        "mobileRunId": "run-1",
        "threadId": "thread-1",
        "status": "backend_still_running",
        "backendRunId": "backend-run-1",
        "backendStatus": "running",
        "deviceStatus": "not_bound",
        "localFenced": True,
        "retryable": True,
        "cancelSource": "user",
        "terminalReason": "user",
    }


def test_cancel_route_retries_backend_cancel_for_locally_cancelled_unconfirmed_run(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    registry.bind_backend_run("run-1", "backend-run-1")
    registry.cancel_run(
        "run-1",
        reason="user",
        cancel_source="user",
        terminal_reason="user",
        cancel_stream=False,
    )
    registry.mark_backend_status("run-1", "running")
    gateway = _Gateway()
    requested: list[str] = []

    async def cancel_backend(run_id: str, __: dict[str, str]) -> str:
        requested.append(run_id)
        return "cancel_requested"

    async def read_status(_: str, __: dict[str, str]) -> str:
        return "running"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.phone_gateway", gateway)
    monkeypatch.setattr("mobile_agent.http_app.cancel_upstream_run", cancel_backend)
    monkeypatch.setattr("mobile_agent.http_app.upstream_run_status", read_status)

    response = TestClient(app).post("/mobile/threads/thread-1/runs/run-1/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "backend_still_running"
    assert response.json()["backendStatus"] == "running"
    assert response.json()["retryable"] is True
    assert requested == ["run-1"]


def test_cancel_route_does_not_treat_stream_closed_as_backend_terminal(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    registry.bind_backend_run("run-1", "backend-run-1")
    registry.mark_terminal("run-1", terminal_reason="stream_closed")
    registry.mark_backend_status("run-1", "stream_closed")
    gateway = _Gateway()
    requested: list[str] = []

    async def cancel_backend(run_id: str, __: dict[str, str]) -> str:
        requested.append(run_id)
        return "cancel_requested"

    async def read_status(_: str, __: dict[str, str]) -> str:
        return "running"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.phone_gateway", gateway)
    monkeypatch.setattr("mobile_agent.http_app.cancel_upstream_run", cancel_backend)
    monkeypatch.setattr("mobile_agent.http_app.upstream_run_status", read_status)

    response = TestClient(app).post("/mobile/threads/thread-1/runs/run-1/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "backend_still_running"
    assert response.json()["backendStatus"] == "running"
    assert response.json()["retryable"] is True
    assert requested == ["run-1"]


def test_cancel_route_records_cancel_source_and_returns_it(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    gateway = _Gateway()

    async def cancel_backend(_: str, __: dict[str, str]) -> str:
        return "cancel_requested"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.phone_gateway", gateway)
    monkeypatch.setattr("mobile_agent.http_app.cancel_upstream_run", cancel_backend)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/run-1/cancel",
        json={"cancelSource": "frontend_timeout"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cancelSource"] == "frontend_timeout"
    assert payload["terminalReason"] == "frontend_timeout"
    snapshot = registry.snapshot("run-1")
    assert snapshot["cancelSource"] == "frontend_timeout"
    assert snapshot["cancellationReason"] == "frontend_timeout"


def test_cancel_route_returns_backend_run_not_bound_when_backend_run_id_is_missing(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    gateway = _Gateway()

    async def cancel_backend(_: str, __: dict[str, str]) -> str:
        return "unknown_not_bound"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.phone_gateway", gateway)
    monkeypatch.setattr("mobile_agent.http_app.cancel_upstream_run", cancel_backend)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/run-1/cancel",
        json={"cancelSource": "frontend_timeout"},
    )

    assert response.status_code == 200
    assert _stable_cancel_payload(response.json()) == {
        "runId": "run-1",
        "mobileRunId": "run-1",
        "threadId": "thread-1",
        "status": "backend_run_not_bound",
        "backendRunId": None,
        "backendStatus": "unknown_not_bound",
        "deviceStatus": "not_bound",
        "localFenced": True,
        "retryable": True,
        "cancelSource": "frontend_timeout",
        "terminalReason": "frontend_timeout",
    }
    assert registry.snapshot("run-1")["status"] == "cancelled"


def test_cancel_route_returns_cancel_unavailable_when_backend_cancel_cannot_be_sent(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    registry.bind_backend_run("run-1", "backend-run-1")
    gateway = _Gateway()

    async def cancel_backend(_: str, __: dict[str, str]) -> str:
        return "cancel_unavailable"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.phone_gateway", gateway)
    monkeypatch.setattr("mobile_agent.http_app.cancel_upstream_run", cancel_backend)

    response = TestClient(app).post("/mobile/threads/thread-1/runs/run-1/cancel")

    assert response.status_code == 200
    assert _stable_cancel_payload(response.json()) == {
        "runId": "run-1",
        "mobileRunId": "run-1",
        "threadId": "thread-1",
        "status": "cancel_unavailable",
        "backendRunId": "backend-run-1",
        "backendStatus": "cancel_unavailable",
        "deviceStatus": "not_bound",
        "localFenced": True,
        "retryable": True,
        "cancelSource": "user",
        "terminalReason": "user",
    }


def test_cancel_route_reports_device_cancel_failed_without_hiding_local_fence(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    registry.reserve_action(
        run_id="run-1",
        thread_id="thread-1",
        source_id="tool-1",
        command="launch",
        payload={"package": "com.tencent.mm"},
        device_id="device-1",
    )
    registry.bind_backend_run("run-1", "backend-run-1")
    gateway = _Gateway()
    gateway.fail = True

    async def cancel_backend(_: str, __: dict[str, str]) -> str:
        return "cancel_requested"

    async def read_status(_: str, __: dict[str, str]) -> str:
        return "cancelled"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.phone_gateway", gateway)
    monkeypatch.setattr("mobile_agent.http_app.cancel_upstream_run", cancel_backend)
    monkeypatch.setattr("mobile_agent.http_app.upstream_run_status", read_status)

    response = TestClient(app).post("/mobile/threads/thread-1/runs/run-1/cancel")

    assert response.status_code == 200
    assert _stable_cancel_payload(response.json()) == {
        "runId": "run-1",
        "mobileRunId": "run-1",
        "threadId": "thread-1",
        "status": "device_cancel_failed",
        "backendRunId": "backend-run-1",
        "backendStatus": "cancelled",
        "deviceStatus": "cancel_failed",
        "localFenced": True,
        "retryable": True,
        "cancelSource": "user",
        "terminalReason": "user",
    }


def test_cancel_route_returns_already_terminal_without_requiring_backend_cancel(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    registry.mark_terminal("run-1", terminal_reason="succeeded")
    called = False

    async def cancel_backend(_: str, __: dict[str, str]) -> str:
        nonlocal called
        called = True
        return "cancel_requested"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.cancel_upstream_run", cancel_backend)

    response = TestClient(app).post("/mobile/threads/thread-1/runs/run-1/cancel")

    assert response.status_code == 200
    assert _stable_cancel_payload(response.json()) == {
        "runId": "run-1",
        "mobileRunId": "run-1",
        "threadId": "thread-1",
        "status": "already_terminal",
        "backendRunId": None,
        "backendStatus": "not_started",
        "deviceStatus": "already_terminal",
        "localFenced": True,
        "retryable": False,
        "cancelSource": None,
        "terminalReason": "succeeded",
    }
    assert called is False


def test_status_route_reports_real_backend_terminal_state(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    registry.bind_backend_run("run-1", "backend-run-1")

    async def read_status(_: str, __: dict[str, str]) -> str:
        return "cancelled"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.upstream_run_status", read_status)

    response = TestClient(app).get("/mobile/threads/thread-1/runs/run-1/status")

    assert response.status_code == 200
    assert response.json() == {
        "runId": "run-1",
        "status": "active",
        "backendStatus": "cancelled",
        "terminal": True,
        "cancelSource": None,
        "terminalReason": None,
    }


def test_status_route_treats_safe_stream_terminal_backend_statuses_consistently(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    registry.bind_backend_run("run-1", "backend-run-1")

    async def read_status(_: str, __: dict[str, str]) -> str:
        return "stream_closed"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app.upstream_run_status", read_status)

    response = TestClient(app).get("/mobile/threads/thread-1/runs/run-1/status")

    assert response.status_code == 200
    assert response.json()["backendStatus"] == "stream_closed"
    assert response.json()["terminal"] is True


def test_status_route_does_not_report_uncertain_cancel_statuses_as_terminal(monkeypatch) -> None:
    for backend_status in (
        "unknown_not_bound",
        "cancel_unavailable",
        "cancel_request_failed",
    ):
        registry = PhoneActionRegistry()
        registry.start_run("run-1", "thread-1")
        registry.bind_backend_run("run-1", "backend-run-1")

        async def read_status(_: str, __: dict[str, str], status: str = backend_status) -> str:
            return status

        monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
        monkeypatch.setattr("mobile_agent.http_app.upstream_run_status", read_status)

        response = TestClient(app).get("/mobile/threads/thread-1/runs/run-1/status")

        assert response.status_code == 200
        assert response.json()["backendStatus"] == backend_status
        assert response.json()["terminal"] is False


def test_startup_cancels_only_recovered_mobile_runs(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("run-1", "thread-1")
    registry.bind_backend_run("run-1", "backend-run-1")
    cancelled: list[tuple[str, str]] = []

    async def cancel_recovered(run_id: str, *, headers: dict[str, str], reason: str) -> str:
        assert headers == {}
        cancelled.append((run_id, reason))
        return "cancel_requested"

    monkeypatch.setattr("mobile_agent.http_app.phone_action_registry", registry)
    monkeypatch.setattr("mobile_agent.http_app._cancel_backend_and_device_run", cancel_recovered)

    asyncio.run(cancel_recovered_mobile_runs())

    assert cancelled == [("run-1", "facade_restarted")]
