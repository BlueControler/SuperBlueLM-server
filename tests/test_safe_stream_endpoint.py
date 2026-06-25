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
        "mobile_run_id": "run-1",
        "thread_id": "thread-1",
    }


def test_safe_stream_forces_langgraph_to_cancel_when_its_sse_client_disconnects() -> None:
    body = _with_mobile_run_config(
        b'{"input":{"messages":[]},"on_disconnect":"continue"}',
        run_id="run-1",
        thread_id="thread-1",
    )

    payload = json.loads(body)

    assert payload["on_disconnect"] == "cancel"


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
    assert [event for event, _ in frames] == ["stream.started", "stream.error"]
    assert [payload["streamSeq"] for _, payload in frames] == [1, 2]
    assert frames[0][1]["type"] == "stream.started"
    assert frames[0][1]["message"] == "已接收请求，正在连接 Agent。"
    assert frames[1][1] == {
        "type": "stream.error",
        "message": "服务连接中断，请稍后重试。",
        "retryable": True,
        "streamSeq": 2,
    }


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
    assert payloads[2] == {"type": "assistant.delta", "chunk": "微信已打开", "streamSeq": 3}
    assert payloads[3] == {"type": "stream.eof", "streamSeq": 4}
    assert "<think>" not in response.text
    assert "internal" not in response.text


def test_safe_stream_does_not_turn_a_missing_terminal_into_success(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERNAL_BASE_URL", "http://internal.example")
    _StreamingAsyncClient.lines = [
        "event: messages-tuple",
        "data: " + json.dumps([{"type": "ai", "content": "处理中"}, {}], ensure_ascii=False),
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
        "stream.eof",
    ]
    assert [payload["streamSeq"] for _, payload in frames] == [1, 2, 3]
    assert "event: stream.error" not in response.text
    assert "run.terminal" not in response.text
    assert "succeeded" not in response.text


def test_trace_step_without_terminal_emits_transport_eof_only(monkeypatch) -> None:
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
    assert [event for event, _ in frames] == [
        "stream.started",
        "trace.v1",
        "stream.eof",
    ]
    assert [payload["streamSeq"] for _, payload in frames] == [1, 2, 3]
    assert "event: stream.error" not in response.text
    assert "run.terminal" not in response.text
    assert "succeeded" not in response.text


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
    assert [event for event, _ in frames] == [
        "stream.started",
        "task_progress",
        "stream.error",
    ]
    assert [payload["streamSeq"] for _, payload in frames] == [1, 2, 3]
    assert frames[-1][1]["message"] == "任务未返回结束状态，已停止后续操作。"


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
