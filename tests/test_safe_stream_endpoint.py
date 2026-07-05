from __future__ import annotations

from collections.abc import AsyncIterator
import json

import httpx
from starlette.testclient import TestClient

from mobile_agent.http_app import app
from mobile_agent.action_control import PhoneActionRegistry
from mobile_agent.safe_stream import (
    SafeStreamFilter,
    _bind_backend_run_from_response,
    _encode_frame,
    _forward_headers,
    _with_mobile_run_config,
)
from mobile_agent.trace import MAX_TRACE_EVENT_BYTES


def test_safe_stream_route_is_unconfigured_without_internal_base_url(monkeypatch) -> None:
    monkeypatch.delenv("LANGGRAPH_INTERNAL_BASE_URL", raising=False)
    client = TestClient(app)

    response = client.post("/mobile/threads/thread-1/runs/stream", json={})

    assert response.status_code == 503
    assert response.json() == {
        "error": "safe_stream_unconfigured",
        "message": "安全流服务尚未配置。",
    }


def test_only_required_authentication_headers_are_forwarded_upstream() -> None:
    request = _request_with_headers(
        {
            "x-api-key": "key-1",
            "authorization": "Bearer token-1",
            "x-device-id": "device-1",
            "cookie": "session=secret",
            "x-internal-url": "must-not-forward",
        }
    )

    assert _forward_headers(request) == {
        "content-type": "application/json",
        "x-api-key": "key-1",
        "authorization": "Bearer token-1",
        "x-device-id": "device-1",
    }


def test_upstream_run_config_carries_the_client_owned_mobile_run_id() -> None:
    body = _with_mobile_run_config(
        b'{"input":{"messages":[]},"config":{"configurable":{"device_id":"device-1"}}}',
        run_id="run-1",
        thread_id="thread-1",
    )

    payload = json.loads(body)
    assert payload["config"]["configurable"] == {
        "device_id": "device-1",
        "deviceId": "device-1",
        "mobile_run_id": "run-1",
        "thread_id": "thread-1",
    }
    assert payload["config"]["metadata"] == {
        "device_id": "device-1",
        "deviceId": "device-1",
    }


def test_upstream_run_config_carries_request_device_id() -> None:
    body = _with_mobile_run_config(
        b'{"input":{"messages":[]},"config":{"metadata":{"client":"android"}}}',
        run_id="run-1",
        thread_id="thread-1",
        device_id="device-from-header",
    )

    payload = json.loads(body)

    assert payload["config"]["configurable"] == {
        "mobile_run_id": "run-1",
        "thread_id": "thread-1",
        "device_id": "device-from-header",
        "deviceId": "device-from-header",
    }
    assert payload["config"]["metadata"] == {
        "client": "android",
        "device_id": "device-from-header",
        "deviceId": "device-from-header",
    }


def test_upstream_run_config_reads_device_id_from_metadata() -> None:
    body = _with_mobile_run_config(
        b'{"input":{"messages":[]},"config":{"metadata":{"deviceId":"device-from-metadata"}}}',
        run_id="run-1",
        thread_id="thread-1",
    )

    payload = json.loads(body)

    assert payload["config"]["configurable"]["device_id"] == "device-from-metadata"
    assert payload["config"]["configurable"]["deviceId"] == "device-from-metadata"
    assert payload["config"]["metadata"]["device_id"] == "device-from-metadata"
    assert payload["config"]["metadata"]["deviceId"] == "device-from-metadata"


def test_safe_stream_keeps_langgraph_running_when_its_sse_client_disconnects() -> None:
    body = _with_mobile_run_config(
        b'{"input":{"messages":[]},"on_disconnect":"cancel"}',
        run_id="run-1",
        thread_id="thread-1",
    )

    payload = json.loads(body)

    assert payload["on_disconnect"] == "continue"


def test_safe_stream_rejects_concurrent_thread_runs_and_preserves_config(monkeypatch) -> None:
    monkeypatch.setenv("MOBILE_AGENT_MAX_RECURSION", "9")
    body = _with_mobile_run_config(
        b'{"input":{"messages":[]},"multitask_strategy":"enqueue","config":{"custom_key":"custom_val"}}',
        run_id="run-1",
        thread_id="thread-1",
    )

    payload = json.loads(body)

    assert payload["multitask_strategy"] == "reject"
    # deep_agent 内置 recursion_limit=9999，不再由 _with_mobile_run_config 覆盖
    assert "recursion_limit" not in payload["config"]
    assert payload["config"]["configurable"]["mobile_run_id"] == "run-1"
    assert payload["config"]["custom_key"] == "custom_val"


def test_filter_records_the_terminal_status_that_the_client_observed() -> None:
    from mobile_agent.safe_stream import SafeStreamFilter, SseFrame

    filter_ = SafeStreamFilter()
    frames = filter_.feed(
        SseFrame(
            event="custom",
            data=json.dumps(
                {
                    "type": "trace.v1",
                    "runId": "run-1",
                    "eventId": "event-1",
                    "seq": 1,
                    "event": "run.terminal",
                    "status": "failed",
                }
            ),
        )
    )

    assert frames[0].data["status"] == "failed"
    assert filter_.terminal_status == "failed"


def test_content_location_binds_the_native_langgraph_run_id(monkeypatch) -> None:
    registry = PhoneActionRegistry()
    registry.start_run("mobile-run-1", "thread-1")
    monkeypatch.setattr("mobile_agent.safe_stream.phone_action_registry", registry)

    _bind_backend_run_from_response(
        "mobile-run-1",
        "thread-1",
        httpx.Response(
            200,
            headers={
                "Content-Location": "/threads/thread-1/runs/018f3b48-7b45-7c0f-8ba5-cfc0b6df40b4",
            },
        ),
    )

    assert registry.backend_run_info("mobile-run-1") == (
        "thread-1",
        "018f3b48-7b45-7c0f-8ba5-cfc0b6df40b4",
    )


def test_unexpected_upstream_failure_is_converted_to_a_safe_sse_error(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _BrokenAsyncClient)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/stream",
        json={"input": {"messages": []}},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(response.text)
    assert [event for event, _ in frames] == ["stream.started", "trace.v1", "stream.error"]
    assert [payload["streamSeq"] for _, payload in frames] == [1, 2, 3]
    assert frames[0][1]["type"] == "stream.started"
    assert frames[0][1]["message"] == "已接收请求，正在连接 Agent。"
    assert frames[1][1]["event"] == "run.terminal"
    assert frames[1][1]["status"] == "failed"
    assert frames[1][1]["reason"] == "stream_error"
    assert frames[1][1]["seq"] > 0
    assert frames[2][1]["type"] == "stream.error"
    assert frames[2][1]["message"] == "服务连接中断，请稍后重试。"
    assert frames[2][1]["retryable"] is True
    assert frames[2][1]["terminalStatus"] == "failed"
    assert frames[2][1]["terminalReason"] == "stream_error"
    assert frames[2][1]["streamSeq"] == 3
    assert frames[2][1]["threadId"] == "thread-1"
    assert frames[2][1]["mobileRunId"] == frames[0][1]["runId"]
    assert frames[2][1]["runId"] == frames[0][1]["runId"]
    assert isinstance(frames[2][1]["timestamp"], int)


def test_upstream_http_error_body_is_not_exposed_in_stream_error(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _HttpErrorAsyncClient)
    logged: list[str] = []
    from mobile_agent.safe_stream import logger

    sink_id = logger.add(lambda message: logged.append(str(message)), format="{message}")

    try:
        response = TestClient(app).post(
            "/mobile/threads/thread-1/runs/stream",
            json={"input": {"messages": []}},
            headers={"Authorization": "Bearer client-token"},
        )
    finally:
        logger.remove(sink_id)

    assert response.status_code == 200
    frames = _parse_sse(response.text)
    assert [event for event, _ in frames] == ["stream.started", "trace.v1", "stream.error"]
    error_payload = frames[-1][1]
    assert error_payload["type"] == "stream.error"
    assert error_payload["message"] == "上游服务返回错误，请稍后重试。"
    assert "detail" not in error_payload
    assert "server-secret-token" not in response.text
    assert "Authorization" not in response.text
    assert "Bearer" not in response.text
    assert "/internal/admin" not in response.text
    assert "client-token" not in response.text
    log_text = "\n".join(logged)
    assert "upstream exploded" in log_text
    assert "server-secret-token" not in log_text
    assert "Bearer server-secret-token" not in log_text
    assert '"token":"***"' in log_text
    assert '"Authorization":"***"' in log_text


def test_safe_stream_route_encodes_only_safe_sse_frames(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    _StreamingAsyncClient.lines = [
        "event: custom",
        "data: "
        + json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "run-1",
                "eventId": "event-1",
                "seq": 1,
                "event": "run.terminal",
                "status": "succeeded",
            },
            ensure_ascii=False,
        ),
        "",
        "event: messages-tuple",
        "data: "
        + json.dumps(
            [{"type": "ai", "content": "<think>internal</think>微信已打开"}, {}],
            ensure_ascii=False,
        ),
        "",
    ]
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _StreamingAsyncClient)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/stream",
        json={"input": {"messages": []}},
        headers={
            "X-Mobile-Run-Id": "run-client-1",
            "Authorization": "Bearer must-not-leak",
            "X-Api-Key": "key-must-not-leak",
        },
    )

    assert response.status_code == 200
    frames = _parse_sse(response.text)
    assert [event for event, _ in frames] == [
        "stream.started",
        "trace.v1",
        "assistant.delta",
        "stream.eof",
    ]
    payloads = [payload for _, payload in frames]
    assert [payload["streamSeq"] for payload in payloads] == [1, 2, 3, 4]
    assert payloads[0]["type"] == "stream.started"
    assert payloads[0]["runId"] == "run-client-1"
    assert payloads[0]["threadId"] == "thread-1"
    assert "must-not-leak" not in json.dumps(payloads[0], ensure_ascii=False)
    assert payloads[1]["event"] == "run.terminal"
    assert payloads[1]["status"] == "succeeded"
    assert payloads[2]["type"] == "assistant.delta"
    assert payloads[2]["chunk"] == "微信已打开"
    assert payloads[2]["streamSeq"] == 3
    assert payloads[2]["runId"] == "run-client-1"
    assert payloads[2]["mobileRunId"] == "run-client-1"
    assert payloads[2]["threadId"] == "thread-1"
    assert isinstance(payloads[2]["timestamp"], int)
    assert payloads[3]["type"] == "stream.eof"
    assert payloads[3]["streamSeq"] == 4
    assert payloads[3]["runId"] == "run-client-1"
    assert payloads[3]["mobileRunId"] == "run-client-1"
    assert payloads[3]["threadId"] == "thread-1"
    assert isinstance(payloads[3]["timestamp"], int)
    assert "<think>" not in response.text
    assert "internal" not in response.text


def test_safe_stream_normalizes_trace_run_id_to_client_owned_mobile_run_id(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    _StreamingAsyncClient.lines = [
        "event: custom",
        "data: "
        + json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "trace-upstream-run-1",
                "eventId": "event-1",
                "seq": 1,
                "event": "run.started",
                "threadId": "thread-1",
                "summary": "已接收请求，正在连接 Agent。",
            },
            ensure_ascii=False,
        ),
        "",
        "event: custom",
        "data: "
        + json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "trace-upstream-run-1",
                "eventId": "event-2",
                "seq": 2,
                "event": "run.terminal",
                "status": "succeeded",
            },
            ensure_ascii=False,
        ),
        "",
    ]
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _StreamingAsyncClient)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/stream",
        json={"input": {"messages": []}},
        headers={"X-Mobile-Run-Id": "run-client-1"},
    )

    frames = _parse_sse(response.text)
    assert [event for event, _ in frames] == [
        "stream.started",
        "trace.v1",
        "trace.v1",
        "stream.eof",
    ]
    assert frames[0][1]["runId"] == "run-client-1"
    assert frames[1][1]["runId"] == "run-client-1"
    assert frames[1][1]["threadId"] == "thread-1"
    assert frames[2][1]["runId"] == "run-client-1"
    assert frames[2][1]["event"] == "run.terminal"
    assert frames[2][1]["status"] == "succeeded"


def test_safe_stream_adds_lifecycle_identity_to_safe_events(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    _StreamingAsyncClient.lines = [
        "event: messages-tuple",
        "data: " + json.dumps([{"type": "ai", "content": "ok", "id": "model-final"}, {}], ensure_ascii=False),
        "",
        "event: custom",
        "data: "
        + json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "upstream-run-1",
                "eventId": "event-terminal",
                "seq": 1,
                "event": "run.terminal",
                "status": "succeeded",
            },
            ensure_ascii=False,
        ),
        "",
    ]
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _StreamingAsyncClient)

    response = TestClient(app).post(
        "/mobile/threads/thread-identity/runs/stream",
        json={"input": {"messages": []}},
        headers={"X-Mobile-Run-Id": "run-client-identity"},
    )

    frames = _parse_sse(response.text)
    for event_name, payload in frames:
        assert payload["threadId"] == "thread-identity"
        assert payload["mobileRunId"] == "run-client-identity"
        assert payload["streamSeq"] > 0
        assert isinstance(payload["timestamp"], int)
        if event_name in {"assistant.delta", "stream.eof", "stream.error"}:
            assert payload["runId"] == "run-client-identity"


def test_safe_stream_forwards_assistant_deltas_before_terminal(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    _StreamingAsyncClient.lines = [
        "event: messages-tuple",
        "data: " + json.dumps([{"type": "ai", "content": "第一段", "id": "model-final"}, {}], ensure_ascii=False),
        "",
        "event: messages-tuple",
        "data: " + json.dumps([{"type": "ai", "content": "第二段", "id": "model-final"}, {}], ensure_ascii=False),
        "",
        "event: custom",
        "data: "
        + json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "run-1",
                "eventId": "event-1",
                "seq": 1,
                "event": "run.terminal",
                "status": "succeeded",
            },
            ensure_ascii=False,
        ),
        "",
    ]
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _StreamingAsyncClient)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/stream",
        json={"input": {"messages": []}},
    )

    frames = _parse_sse(response.text)
    assert [event for event, _ in frames] == [
        "stream.started",
        "assistant.delta",
        "assistant.delta",
        "trace.v1",
        "stream.eof",
    ]
    assert [payload["streamSeq"] for _, payload in frames] == [1, 2, 3, 4, 5]
    assert frames[1][1]["chunk"] == "第一段"
    assert frames[2][1]["chunk"] == "第二段"
    assert frames[3][1]["event"] == "run.terminal"


def test_safe_stream_normal_terminal_path_does_not_call_fixed_sleep(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    _StreamingAsyncClient.lines = [
        "event: custom",
        "data: "
        + json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "run-1",
                "eventId": "event-1",
                "seq": 1,
                "event": "run.terminal",
                "status": "succeeded",
            },
            ensure_ascii=False,
        ),
        "",
    ]
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _StreamingAsyncClient)

    async def fail_sleep(_: float) -> None:
        raise AssertionError("normal terminal path must not call fixed sleep")

    monkeypatch.setattr("mobile_agent.safe_stream.asyncio.sleep", fail_sleep)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/stream",
        json={"input": {"messages": []}},
    )

    frames = _parse_sse(response.text)
    assert [event for event, _ in frames] == ["stream.started", "trace.v1", "stream.eof"]


def test_plain_text_without_trace_terminal_gets_synthetic_success_terminal(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    _StreamingAsyncClient.lines = [
        "event: messages-tuple",
        "data: " + json.dumps([{"type": "ai", "content": "ok"}, {}], ensure_ascii=False),
        "",
    ]
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _StreamingAsyncClient)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/stream",
        json={"input": {"messages": []}},
    )

    frames = _parse_sse(response.text)
    assert [event for event, _ in frames] == [
        "stream.started",
        "assistant.delta",
        "trace.v1",
        "stream.eof",
    ]
    assert frames[1][1]["chunk"] == "ok"
    assert frames[2][1]["event"] == "run.terminal"
    assert frames[2][1]["status"] == "succeeded"
    assert frames[2][1]["reason"] == "upstream_completed_without_terminal"
    assert frames[3][1]["type"] == "stream.eof"


def test_heartbeat_only_then_eof_is_interrupted_not_success(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    monkeypatch.setenv("MOBILE_AGENT_STREAM_HEARTBEAT_SECONDS", "1")
    _DelayedStreamingAsyncClient.delays = [1.2]
    _DelayedStreamingAsyncClient.lines = [""]
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _DelayedStreamingAsyncClient)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/stream",
        json={"input": {"messages": []}},
        headers={"X-Mobile-Run-Id": "run-client-1"},
    )

    frames = _parse_sse(response.text)
    assert [event for event, _ in frames] == [
        "stream.started",
        "stream.heartbeat",
        "trace.v1",
        "stream.error",
    ]
    assert frames[1][1]["type"] == "stream.heartbeat"
    assert frames[2][1]["event"] == "run.terminal"
    assert frames[2][1]["status"] == "failed"
    assert frames[3][1]["terminalReason"] == "upstream_ended_without_terminal"


def test_trace_step_without_terminal_fails_without_success_terminal(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    _StreamingAsyncClient.lines = [
        "event: custom",
        "data: "
        + json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "run-1",
                "eventId": "event-1",
                "seq": 1,
                "event": "step.upsert",
                "step": {
                    "stepId": "tool-1",
                    "kind": "tool",
                    "title": "观察屏幕",
                    "summary": "正在读取当前手机屏幕状态。",
                    "status": "running",
                    "visibleToUser": True,
                },
            },
            ensure_ascii=False,
        ),
        "",
    ]
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _StreamingAsyncClient)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/stream",
        json={"input": {"messages": []}},
    )

    frames = _parse_sse(response.text)
    # 有 trace 但无 terminal 时，补充一个可排序的 failed terminal 后再发送错误。
    assert [event for event, _ in frames] == [
        "stream.started",
        "trace.v1",
        "trace.v1",
        "stream.error",
    ]
    assert frames[-2][1]["event"] == "run.terminal"
    assert frames[-2][1]["status"] == "failed"
    assert frames[-2][1]["reason"] == "upstream_ended_without_terminal"
    assert frames[-2][1]["seq"] == 2
    assert frames[-1][1]["terminalStatus"] == "failed"


def test_succeeded_tool_step_without_terminal_gets_synthetic_success_terminal(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    _StreamingAsyncClient.lines = [
        "event: custom",
        "data: "
        + json.dumps(
            {
                "type": "task_progress",
                "label": "打开微信",
                "status": "running",
                "phase": "phone_tool",
                "message": "正在打开微信",
            },
            ensure_ascii=False,
        ),
        "",
        "event: custom",
        "data: "
        + json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "run-1",
                "eventId": "event-1",
                "seq": 1,
                "event": "step.upsert",
                "step": {
                    "stepId": "tool-1",
                    "kind": "tool",
                    "title": "打开微信",
                    "summary": "微信已打开。",
                    "status": "succeeded",
                    "visibleToUser": True,
                },
            },
            ensure_ascii=False,
        ),
        "",
    ]
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _StreamingAsyncClient)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/stream",
        json={"input": {"messages": []}},
    )

    frames = _parse_sse(response.text)
    assert [event for event, _ in frames] == [
        "stream.started",
        "task_progress",
        "trace.v1",
        "trace.v1",
        "stream.eof",
    ]
    assert frames[1][1]["type"] == "task_progress"
    assert frames[2][1]["event"] == "step.upsert"
    assert frames[2][1]["step"]["status"] == "succeeded"
    assert frames[3][1]["event"] == "run.terminal"
    assert frames[3][1]["status"] == "succeeded"
    assert frames[3][1]["reason"] == "upstream_completed_without_terminal"
    assert frames[3][1]["seq"] == 2
    assert frames[4][1]["type"] == "stream.eof"


def test_safe_stream_still_errors_on_empty_stream_without_terminal(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    # 空流(只有 task 事件) 没有实际回复内容 → 应报错
    _StreamingAsyncClient.lines = [
        "event: custom",
        "data: " + json.dumps({"type": "task_progress", "label": "分析请求", "status": "running", "phase": "analysis"}),
        "",
    ]
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _StreamingAsyncClient)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/stream",
        json={"input": {"messages": []}},
    )

    frames = _parse_sse(response.text)
    # 空流无 terminal → 先发业务 failed terminal，再发 transport error。
    assert [event for event, _ in frames] == [
        "stream.started",
        "task_progress",
        "trace.v1",
        "stream.error",
    ]
    assert frames[-2][1]["event"] == "run.terminal"
    assert frames[-2][1]["status"] == "failed"
    assert frames[-2][1]["reason"] == "upstream_ended_without_terminal"
    assert frames[-1][1]["terminalReason"] == "upstream_ended_without_terminal"


def test_safe_stream_emits_heartbeat_while_upstream_is_quiet(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    monkeypatch.setenv("MOBILE_AGENT_STREAM_HEARTBEAT_SECONDS", "1")
    _DelayedStreamingAsyncClient.delays = [1.2, 0.0, 0.0]
    _DelayedStreamingAsyncClient.lines = [
        "event: custom",
        "data: "
        + json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "run-1",
                "eventId": "event-1",
                "seq": 1,
                "event": "run.terminal",
                "status": "succeeded",
            },
            ensure_ascii=False,
        ),
        "",
    ]
    monkeypatch.setattr("mobile_agent.safe_stream.httpx.AsyncClient", _DelayedStreamingAsyncClient)

    response = TestClient(app).post(
        "/mobile/threads/thread-1/runs/stream",
        json={"input": {"messages": []}},
        headers={"X-Mobile-Run-Id": "run-client-1"},
    )

    frames = _parse_sse(response.text)
    assert [event for event, _ in frames] == [
        "stream.started",
        "stream.heartbeat",
        "trace.v1",
        "stream.eof",
    ]
    assert frames[1][1]["type"] == "stream.heartbeat"
    assert frames[1][1]["runId"] == "run-client-1"


def test_encoded_trace_frame_still_fits_after_stream_seq_is_injected() -> None:
    filter_ = SafeStreamFilter()
    frames = filter_.feed(
        _trace_step_with_long_summary(seq=1, summary="正在读取当前手机屏幕状态。" * 500)
    )

    encoded = _encode_frame(frames[0], stream_seq=123_456)
    payload = _parse_sse(encoded)[0][1]

    assert payload["streamSeq"] == 123_456
    assert len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= MAX_TRACE_EVENT_BYTES


class _BrokenAsyncClient:
    def __init__(self, **_: object) -> None:
        pass

    async def __aenter__(self) -> "_BrokenAsyncClient":
        raise RuntimeError("unexpected upstream setup failure")

    async def __aexit__(self, *_: object) -> None:
        pass

    def stream(self, *_: object, **__: object) -> object:
        raise AssertionError("stream must not be reached after setup failure")


class _StreamingAsyncClient:
    lines: list[str] = []

    def __init__(self, **_: object) -> None:
        pass

    async def __aenter__(self) -> "_StreamingAsyncClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    def stream(self, *_: object, **__: object) -> "_StreamingResponse":
        return _StreamingResponse(self.lines)


class _StreamingResponse:
    status_code = 200

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> "_StreamingResponse":
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


class _HttpErrorAsyncClient(_StreamingAsyncClient):
    def stream(self, *_: object, **__: object) -> "_HttpErrorResponse":
        return _HttpErrorResponse()


class _HttpErrorResponse(_StreamingResponse):
    status_code = 502

    def __init__(self) -> None:
        super().__init__([])

    async def aread(self) -> bytes:
        return (
            b'{"error":"upstream exploded",'
            b'"token":"server-secret-token",'
            b'"Authorization":"Bearer server-secret-token",'
            b'"path":"/internal/admin"}'
        )


class _DelayedStreamingAsyncClient(_StreamingAsyncClient):
    delays: list[float] = []

    def stream(self, *_: object, **__: object) -> "_DelayedStreamingResponse":
        return _DelayedStreamingResponse(self.lines, self.delays)


class _DelayedStreamingResponse(_StreamingResponse):
    def __init__(self, lines: list[str], delays: list[float]) -> None:
        super().__init__(lines)
        self._delays = delays

    async def aiter_lines(self) -> AsyncIterator[str]:
        for index, line in enumerate(self._lines):
            delay = self._delays[index] if index < len(self._delays) else 0.0
            if delay > 0:
                import asyncio

                await asyncio.sleep(delay)
            yield line


def _parse_sse(text: str) -> list[tuple[str, dict[str, object]]]:
    parsed: list[tuple[str, dict[str, object]]] = []
    for frame in (item for item in text.strip().split("\n\n") if item):
        event_name = ""
        data = ""
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = line.removeprefix("data:").strip()
        parsed.append((event_name, json.loads(data)))
    return parsed


def _trace_step_with_long_summary(*, seq: int, summary: str):
    from mobile_agent.safe_stream import SseFrame

    return SseFrame(
        event="custom",
        data=json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "run-1",
                "eventId": f"event-{seq}",
                "seq": seq,
                "event": "step.upsert",
                "step": {
                    "stepId": "tool-1",
                    "kind": "tool",
                    "title": "观察屏幕" * 100,
                    "summary": summary,
                    "status": "running",
                    "visibleToUser": True,
                },
            },
            ensure_ascii=False,
        ),
    )


def _request_with_headers(headers: dict[str, str]):
    from starlette.requests import Request

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/mobile/threads/thread-1/runs/stream",
            "headers": [
                (name.encode("latin-1"), value.encode("latin-1"))
                for name, value in {"content-type": "application/json", **headers}.items()
            ],
        }
    )
