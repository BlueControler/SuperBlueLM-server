from __future__ import annotations

import json

from mobile_agent.safe_stream import SafeStreamFilter, SseFrame
from mobile_agent.trace import MAX_TRACE_EVENT_BYTES


def test_filter_drops_tool_call_arguments_and_tool_results() -> None:
    stream = SafeStreamFilter()
    tool_call = SseFrame(
        event="messages-tuple",
        data=json.dumps([
            {"type": "ai", "content": "", "tool_calls": [{"name": "tap", "args": {"token": "secret"}}]},
            {"langgraph_node": "model"},
        ]),
    )
    tool_result = SseFrame(
        event="messages-tuple",
        data=json.dumps([
            {"type": "tool", "name": "tap", "content": '{"password":"raw"}'},
            {},
        ]),
    )

    assert stream.feed(tool_call) == []
    assert stream.feed(tool_result) == []


def test_filter_drops_tool_call_messages_with_nested_metadata() -> None:
    stream = SafeStreamFilter()

    frames = stream.feed(
        SseFrame(
            event="messages-tuple",
            data=json.dumps([
                {
                    "type": "ai",
                    "content": "I will call weather_query for Beijing.",
                    "additional_kwargs": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "weather_query",
                                    "arguments": '{"city":"Beijing"}',
                                }
                            }
                        ]
                    },
                    "response_metadata": {"finish_reason": "tool_calls"},
                },
                {"langgraph_node": "model", "checkpoint_ns": "model:tool-call"},
            ]),
        )
    )

    assert frames == []


def test_filter_strips_think_across_chunks_and_keeps_safe_answer_text() -> None:
    stream = SafeStreamFilter()
    first = stream.feed(
        SseFrame(event="messages-tuple", data=_assistant_payload("<thi"))
    )
    second = stream.feed(
        SseFrame(event="messages-tuple", data=_assistant_payload("nk>private token=abc</think>最终"))
    )
    third = stream.feed(
        SseFrame(event="messages-tuple", data=_assistant_payload("回答"))
    )
    stream.feed(_terminal_trace(status="succeeded", seq=1))
    completed = stream.finish()

    chunks = [frame.data["chunk"] for frame in [*first, *second, *third, *completed] if frame.event == "assistant.delta"]
    assert "".join(chunks) == "最终回答"
    assert all("private" not in chunk and "abc" not in chunk for chunk in chunks)


def test_filter_buffers_assistant_text_until_stream_finish() -> None:
    stream = SafeStreamFilter()

    frames = stream.feed(
        SseFrame(event="messages-tuple", data=_assistant_payload("Final answer"))
    )
    stream.feed(_terminal_trace(status="succeeded", seq=1))
    completed = stream.finish()

    assert frames == []
    assert [frame.event for frame in completed] == ["assistant.delta", "stream.eof"]
    assert completed[0].data["chunk"] == "Final answer"


def test_filter_discards_intermediate_assistant_text_when_tool_step_arrives() -> None:
    stream = SafeStreamFilter()
    frames = [
        *stream.feed(
            SseFrame(
                event="messages-tuple",
                data=_assistant_payload("I need to call the weather tool."),
            )
        ),
        *stream.feed(_tool_step_trace(status="running", seq=1)),
        *stream.feed(_tool_step_trace(status="succeeded", seq=2)),
        *stream.feed(
            SseFrame(
                event="messages-tuple",
                data=_assistant_payload("Weather result is ready."),
            )
        ),
        *stream.feed(_terminal_trace(status="succeeded", seq=3)),
        *stream.finish(),
    ]

    assistant_chunks = [
        frame.data["chunk"]
        for frame in frames
        if frame.event == "assistant.delta"
    ]
    combined = json.dumps([frame.data for frame in frames], ensure_ascii=False)
    assert assistant_chunks == ["Weather result is ready."]
    assert "I need to call the weather tool" not in combined


def test_filter_drops_structured_reasoning_blocks_and_keeps_public_text() -> None:
    stream = SafeStreamFilter()
    payload = [
        {
            "type": "ai",
            "content": [
                {"type": "reasoning", "text": "private chain token=secret"},
                {"type": "thinking", "text": "private thought"},
                {"type": "text", "text": "Final"},
                {"type": "output_text", "text": " answer"},
            ],
            "additional_kwargs": {
                "reasoning_content": "private additional reasoning",
            },
        },
        {"langgraph_node": "model", "checkpoint_ns": "model:reasoning"},
    ]

    frames = stream.feed(SseFrame(event="messages-tuple", data=json.dumps(payload)))
    stream.feed(_terminal_trace(status="succeeded", seq=1))
    completed = stream.finish()

    assert frames == []
    assert completed[0].event == "assistant.delta"
    assert completed[0].data["chunk"] == "Final answer"
    assert "private" not in json.dumps(completed[0].data)
    assert "secret" not in json.dumps(completed[0].data)


def test_filter_forwards_only_safe_trace_fields_with_a_bounded_payload() -> None:
    stream = SafeStreamFilter()
    payload = {
        "type": "trace.v1",
        "version": 1,
        "runId": "run-1",
        "threadId": "thread-1",
        "eventId": "event-1",
        "seq": 1,
        "event": "step.upsert",
        "rawResult": "token=secret",
        "step": {
            "stepId": "step-1",
            "kind": "tool",
            "title": "观察屏幕",
            "summary": "内容" * 1000,
            "status": "running",
            "visibleToUser": True,
            "args": {"password": "raw"},
        },
    }

    frames = stream.feed(SseFrame(event="custom", data=json.dumps(payload, ensure_ascii=False)))

    assert len(frames) == 1
    safe = frames[0].data
    assert frames[0].event == "trace.v1"
    assert "rawResult" not in safe
    assert "args" not in safe["step"]
    assert len(json.dumps(safe, ensure_ascii=False).encode("utf-8")) <= MAX_TRACE_EVENT_BYTES


def test_filter_preserves_nested_trace_parent_id_without_raw_subagent_fields() -> None:
    stream = SafeStreamFilter()
    payload = {
        "type": "trace.v1",
        "version": 1,
        "runId": "run-1",
        "threadId": "thread-1",
        "eventId": "event-child",
        "seq": 2,
        "event": "step.upsert",
        "step": {
            "stepId": "phone-child-1",
            "parentId": "tool_parent",
            "kind": "phone_action",
            "title": "点击屏幕",
            "summary": "点击操作已执行。",
            "status": "succeeded",
            "visibleToUser": True,
            "rawArgs": {"x": 123, "y": 456},
            "result": {"screenshot": "base64", "uiTree": "<node />"},
        },
    }

    frames = stream.feed(SseFrame(event="custom", data=json.dumps(payload, ensure_ascii=False)))

    assert len(frames) == 1
    step = frames[0].data["step"]
    assert step["parentId"] == "tool_parent"
    assert "rawArgs" not in step
    assert "result" not in step
    encoded = json.dumps(frames[0].data, ensure_ascii=False)
    assert "base64" not in encoded
    assert "uiTree" not in encoded


def test_filter_forwards_only_safe_detail_fields() -> None:
    stream = SafeStreamFilter()
    payload = {
        "type": "trace.v1",
        "version": 1,
        "runId": "run-1",
        "threadId": "thread-1",
        "eventId": "event-2",
        "seq": 2,
        "event": "step.detail.append",
        "stepId": "step-1",
        "args": {"token": "secret"},
        "result": {"screenshot": "base64", "uiTree": "<node />"},
        "detail": {
            "detailId": "detail-1",
            "kind": "tool_result",
            "title": "工具结果",
            "text": "<think>private</think>已完成观察 token=secret 验证码:123456 " + "内容" * 1000,
            "visibleToUser": True,
            "raw": {"password": "raw"},
            "headers": {"Authorization": "Bearer raw"},
        },
    }

    frames = stream.feed(SseFrame(event="custom", data=json.dumps(payload, ensure_ascii=False)))

    assert len(frames) == 1
    safe = frames[0].data
    assert frames[0].event == "trace.v1"
    assert safe["event"] == "step.detail.append"
    assert safe["stepId"] == "step-1"
    assert safe["detail"]["detailId"] == "detail-1"
    assert safe["detail"]["kind"] == "tool_result"
    assert safe["detail"]["visibleToUser"] is True
    encoded = json.dumps(safe, ensure_ascii=False)
    assert "args" not in safe
    assert "result" not in safe
    assert "raw" not in safe["detail"]
    assert "headers" not in safe["detail"]
    assert "<think>" not in encoded
    assert "private" not in encoded
    assert "secret" not in encoded
    assert "验证码" not in encoded
    assert "123456" not in encoded
    assert "screenshot" not in encoded
    assert "uiTree" not in encoded
    assert len(encoded.encode("utf-8")) <= MAX_TRACE_EVENT_BYTES


def test_filter_drops_hidden_or_unknown_detail_events() -> None:
    stream = SafeStreamFilter()
    hidden = _detail_payload(visible=False)
    unknown_kind = _detail_payload(kind="raw_result")

    assert stream.feed(SseFrame(event="custom", data=json.dumps(hidden))) == []
    assert stream.feed(SseFrame(event="custom", data=json.dumps(unknown_kind))) == []


def test_filter_forwards_legacy_progress_without_unknown_fields() -> None:
    stream = SafeStreamFilter()
    payload = {
        "type": "task_progress",
        "label": "observe",
        "status": "running",
        "phase": "phone_tool",
        "message": "Running phone tool",
        "error": "raw result token=secret",
        "toolName": "internal_tool_with_args",
        "secret": "must not escape",
    }

    frames = stream.feed(SseFrame(event="custom", data=json.dumps(payload)))

    assert len(frames) == 1
    assert frames[0].event == "task_progress"
    assert frames[0].data == {
        "type": "task_progress",
        "label": "observe",
        "status": "running",
        "phase": "phone_tool",
        "message": "Running phone tool",
    }


def test_filter_forwards_task_complexity_without_unknown_fields() -> None:
    stream = SafeStreamFilter()
    payload = {
        "type": "task_complexity",
        "complexity": "simple",
        "trackSteps": False,
        "reason": "zero_or_one_tool_call",
        "message": "Short task: skip step tracking",
        "rawPrompt": "token=secret",
    }

    frames = stream.feed(SseFrame(event="custom", data=json.dumps(payload)))

    assert len(frames) == 1
    assert frames[0].event == "task_complexity"
    assert frames[0].data == {
        "type": "task_complexity",
        "complexity": "simple",
        "trackSteps": False,
        "reason": "zero_or_one_tool_call",
        "message": "Short task: skip step tracking",
    }


def test_filter_limits_each_assistant_event_below_four_kib() -> None:
    stream = SafeStreamFilter()

    frames = stream.feed(
        SseFrame(event="messages-tuple", data=_assistant_payload("中" * 10_000))
    )
    stream.feed(_terminal_trace(status="succeeded", seq=1))
    completed = stream.finish()

    assert frames == []
    assert len(json.dumps(completed[0].data, ensure_ascii=False).encode("utf-8")) <= MAX_TRACE_EVENT_BYTES


def _assistant_payload(text: str) -> str:
    return json.dumps([
        {"type": "ai", "content": text},
        {"langgraph_node": "model", "checkpoint_ns": "model:1"},
    ])


def _tool_step_trace(*, status: str, seq: int) -> SseFrame:
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
                    "title": "Weather tool",
                    "summary": "Tool status changed.",
                    "status": status,
                    "visibleToUser": True,
                },
            }
        ),
    )


def _terminal_trace(*, status: str, seq: int) -> SseFrame:
    return SseFrame(
        event="custom",
        data=json.dumps(
            {
                "type": "trace.v1",
                "version": 1,
                "runId": "run-1",
                "eventId": f"event-{seq}",
                "seq": seq,
                "event": "run.terminal",
                "status": status,
            }
        ),
    )


def _detail_payload(
    *,
    visible: bool = True,
    kind: str = "warning",
) -> dict[str, object]:
    return {
        "type": "trace.v1",
        "version": 1,
        "runId": "run-1",
        "eventId": "event-hidden",
        "seq": 3,
        "event": "step.detail.append",
        "stepId": "step-1",
        "detail": {
            "detailId": "detail-hidden",
            "kind": kind,
            "title": "需要确认",
            "text": "该操作需要你确认。",
            "visibleToUser": visible,
        },
    }
