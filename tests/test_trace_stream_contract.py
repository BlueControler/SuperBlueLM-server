from __future__ import annotations

import json

from mobile_agent.safe_stream import SafeStreamFilter, SseFrame
from mobile_agent import trace
from mobile_agent.trace import (
    MAX_TRACE_EVENT_BYTES,
    MAX_TRACE_SUMMARY_CHARS,
    MAX_TRACE_TITLE_CHARS,
)


def test_golden_safe_trace_stream_contract() -> None:
    stream = SafeStreamFilter()
    frames = [
        *stream.feed(_assistant("<think>private plan</think>最终回答")),
        *stream.feed(_trace("step.upsert", seq=1, status="running")),
        *stream.feed(_detail(seq=2)),
        *stream.feed(_trace("step.upsert", seq=3, status="failed")),
        *stream.feed(_trace("step.upsert", seq=4, status="succeeded")),
        *stream.feed(_trace("run.terminal", seq=5, run_status="succeeded")),
        *stream.feed(
            SseFrame(
                event="messages-tuple",
                data=json.dumps([
                    {
                        "type": "ai",
                        "content": "",
                        "tool_calls": [{"name": "tap", "args": {"token": "secret"}}],
                    },
                    {},
                ]),
            ),
        ),
        *stream.feed(
            SseFrame(
                event="custom",
                data=json.dumps({
                    "type": "task_progress",
                    "label": "observe",
                    "status": "running",
                    "phase": "phone_tool",
                    "error": "raw result token=secret",
                }),
            ),
        ),
        *stream.finish(),
    ]

    encoded = [json.dumps(frame.data, ensure_ascii=False) for frame in frames]
    combined = "\n".join(encoded)
    assert "private plan" not in combined
    assert "secret" not in combined
    assert '"args"' not in combined
    assert '"result"' not in combined
    assert "<think>" not in combined
    assert frames[-1].event == "stream.eof"
    assert any(frame.data.get("status") == "succeeded" for frame in frames)

    for frame in frames:
        if frame.event != "trace.v1":
            continue
        assert len(json.dumps(frame.data, ensure_ascii=False).encode("utf-8")) <= MAX_TRACE_EVENT_BYTES
        step = frame.data.get("step")
        if isinstance(step, dict):
            assert len(step["title"]) <= MAX_TRACE_TITLE_CHARS
            assert len(step["summary"]) <= MAX_TRACE_SUMMARY_CHARS
        detail = frame.data.get("detail")
        if isinstance(detail, dict):
            assert len(detail["title"]) <= MAX_TRACE_TITLE_CHARS
            assert len(detail["text"]) <= trace.MAX_TRACE_DETAIL_TEXT_CHARS


def test_eof_without_terminal_remains_transport_only() -> None:
    stream = SafeStreamFilter()
    frames = [
        *stream.feed(_trace("step.upsert", seq=1, status="running")),
        *stream.finish(),
    ]

    assert [frame.event for frame in frames] == ["trace.v1", "stream.eof"]
    assert not any(frame.data.get("event") == "run.terminal" for frame in frames)


def test_waiting_for_user_is_forwarded_as_business_terminal() -> None:
    stream = SafeStreamFilter()
    frames = [
        *stream.feed(_trace("step.upsert", seq=1, status="waiting_for_user")),
        *stream.feed(
            _trace("run.terminal", seq=2, run_status="waiting_for_user"),
        ),
    ]

    assert frames[-1].data["event"] == "run.terminal"
    assert frames[-1].data["status"] == "waiting_for_user"


def _assistant(text: str) -> SseFrame:
    return SseFrame(
        event="messages-tuple",
        data=json.dumps([{"type": "ai", "content": text}, {}]),
    )


def _trace(
    event: str,
    *,
    seq: int,
    status: str = "running",
    run_status: str | None = None,
) -> SseFrame:
    payload: dict[str, object] = {
        "type": "trace.v1",
        "version": 1,
        "runId": "run-1",
        "eventId": f"evt-{seq}",
        "seq": seq,
        "event": event,
    }
    if event == "step.upsert":
        payload["step"] = {
            "stepId": "step-1",
            "kind": "tool",
            "title": "观察屏幕" * 100,
            "summary": "正在读取当前手机屏幕状态。" * 100,
            "status": status,
            "visibleToUser": True,
        }
    else:
        payload["status"] = run_status
    return SseFrame(event="custom", data=json.dumps(payload, ensure_ascii=False))


def _detail(*, seq: int) -> SseFrame:
    return SseFrame(
        event="custom",
        data=json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "run-1",
                "eventId": f"evt-{seq}",
                "seq": seq,
                "event": "step.detail.append",
                "stepId": "step-1",
                "result": {"token": "secret"},
                "detail": {
                    "detailId": "detail-1",
                    "kind": "observation",
                    "title": "观察结果" * 100,
                    "text": "已获取当前屏幕概要，未展示截图或 UI tree。token=secret" + "内容" * 1000,
                    "visibleToUser": True,
                    "raw": {"uiTree": "<node />"},
                },
            },
            ensure_ascii=False,
        ),
    )
