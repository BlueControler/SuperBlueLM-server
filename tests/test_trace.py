from __future__ import annotations

import asyncio
import json
from typing import Any

from mobile_agent import trace


def test_emitter_sequences_events_and_stops_after_terminal() -> None:
    emitted: list[dict[str, Any]] = []
    emitter = trace.TraceEmitter(writer_provider=lambda: emitted.append)

    with trace.request_context(thread_id="thread-1", run_id="run-1"):
        emitter.run_started()
        emitter.step_upsert(
            step_id="step-1",
            kind="tool",
            title="观察屏幕",
            summary="正在读取当前手机屏幕状态。",
            status="running",
        )
        emitter.run_terminal("succeeded")
        emitter.step_upsert(
            step_id="step-2",
            kind="tool",
            title="不应发送",
            summary="终态后禁止发事件。",
            status="running",
        )

    assert [event["seq"] for event in emitted] == [1, 2, 3]
    assert emitted[0]["event"] == "run.started"
    assert emitted[1]["step"]["stepId"] == "step-1"
    assert emitted[2]["event"] == "run.terminal"
    assert emitted[2]["status"] == "succeeded"
    assert len({event["eventId"] for event in emitted}) == 3


def test_step_detail_append_sequences_after_existing_step_only() -> None:
    emitted: list[dict[str, Any]] = []
    emitter = trace.TraceEmitter(writer_provider=lambda: emitted.append)

    with trace.request_context(thread_id="thread-1", run_id="run-1"):
        ignored = emitter.step_detail_append(
            step_id="missing-step",
            kind="tool_call",
            title="不应发送",
            text="不存在的 step 不能创建假详情。",
        )
        emitter.step_upsert(
            step_id="step-1",
            kind="tool",
            title="观察屏幕",
            summary="正在读取当前手机屏幕状态。",
            status="running",
        )
        detail = emitter.step_detail_append(
            step_id="step-1",
            kind="tool_result",
            title="工具结果",
            text="已完成屏幕观察，当前页面显示为微信首页。",
        )

    assert ignored is None
    assert detail is not None
    assert [event["seq"] for event in emitted] == [1, 2]
    assert emitted[1]["event"] == "step.detail.append"
    assert emitted[1]["stepId"] == "step-1"
    assert emitted[1]["detail"]["kind"] == "tool_result"
    assert emitted[1]["detail"]["title"] == "工具结果"
    assert emitted[1]["detail"]["visibleToUser"] is True


def test_each_concurrent_request_has_an_independent_sequence() -> None:
    emitted: list[dict[str, Any]] = []
    emitter = trace.TraceEmitter(writer_provider=lambda: emitted.append)

    async def emit_for(run_id: str) -> None:
        with trace.request_context(thread_id=f"thread-{run_id}", run_id=run_id):
            emitter.run_started()
            await asyncio.sleep(0)
            emitter.run_terminal("succeeded")

    async def scenario() -> None:
        await asyncio.gather(emit_for("run-a"), emit_for("run-b"))

    asyncio.run(scenario())

    by_run: dict[str, list[int]] = {}
    for event in emitted:
        by_run.setdefault(event["runId"], []).append(event["seq"])
    assert by_run == {"run-a": [1, 2], "run-b": [1, 2]}


def test_trace_event_is_limited_by_character_and_byte_budgets() -> None:
    emitted: list[dict[str, Any]] = []
    emitter = trace.TraceEmitter(writer_provider=lambda: emitted.append)
    long_title = "标" * (trace.MAX_TRACE_TITLE_CHARS + 20)
    long_summary = "😀" * (trace.MAX_TRACE_SUMMARY_CHARS + 3000)

    with trace.request_context(thread_id="thread-1", run_id="run-1"):
        emitter.step_upsert(
            step_id="step-1",
            kind="tool",
            title=long_title,
            summary=long_summary,
            status="running",
            parent_id="parent" * 1000,
        )

    event = emitted[0]
    assert len(event["step"]["title"]) <= trace.MAX_TRACE_TITLE_CHARS
    assert len(event["step"]["summary"]) <= trace.MAX_TRACE_SUMMARY_CHARS
    assert len(json.dumps(event, ensure_ascii=False).encode("utf-8")) <= trace.MAX_TRACE_EVENT_BYTES


def test_unknown_tool_never_exposes_its_raw_name_or_arguments() -> None:
    spec = trace.display_spec_for("unsafe_internal_tool")

    assert spec["title"] == "执行受控操作"
    assert spec["safe_args_summary"]({"token": "secret", "password": "raw"}) == "正在执行受控操作。"


def test_phone_todo_risk_classifier_blocks_sensitive_actions_only() -> None:
    assert trace.is_high_risk_phone_todo("打开地图查询路线") is False
    assert trace.is_high_risk_phone_todo("发送消息给张三") is True
    assert trace.is_high_risk_phone_todo("删除这条日程") is True


def test_trace_writer_failure_never_breaks_agent_execution_or_size_limit() -> None:
    def broken_writer(_: dict[str, Any]) -> None:
        raise RuntimeError("custom stream unavailable")

    with trace.request_context(thread_id="线程" * 1_000):
        event = trace.TraceEmitter(lambda: broken_writer).step_upsert(
            step_id="tool-1",
            kind="tool",
            title="标题" * 100,
            summary="结果" * 10_000,
            status="succeeded",
        )

    assert event is not None
    assert len(json.dumps(event, ensure_ascii=False).encode("utf-8")) <= trace.MAX_TRACE_EVENT_BYTES


def test_detail_append_is_bounded_sanitized_and_stops_after_terminal() -> None:
    emitted: list[dict[str, Any]] = []
    emitter = trace.TraceEmitter(writer_provider=lambda: emitted.append)

    with trace.request_context(thread_id="thread-1", run_id="run-1"):
        emitter.step_upsert(
            step_id="step-1",
            kind="tool",
            title="观察屏幕",
            summary="正在读取当前手机屏幕状态。",
            status="running",
        )
        detail = emitter.step_detail_append(
            step_id="step-1",
            kind="observation",
            title="观察结果" * 100,
            text="<think>private plan</think>当前页面 token=secret 验证码:123456 " + "内容" * 1000,
        )
        emitter.run_terminal("succeeded")
        after_terminal = emitter.step_detail_append(
            step_id="step-1",
            kind="tool_result",
            title="不应发送",
            text="终态后禁止追加详情。",
        )

    assert detail is not None
    assert after_terminal is None
    detail_payload = emitted[1]["detail"]
    assert len(detail_payload["title"]) <= trace.MAX_TRACE_TITLE_CHARS
    assert len(detail_payload["text"]) <= trace.MAX_TRACE_DETAIL_TEXT_CHARS
    encoded = json.dumps(emitted[1], ensure_ascii=False)
    assert "<think>" not in encoded
    assert "private plan" not in encoded
    assert "secret" not in encoded
    assert "验证码" not in encoded
    assert "123456" not in encoded
    assert len(encoded.encode("utf-8")) <= trace.MAX_TRACE_EVENT_BYTES


def test_detail_writer_failure_never_breaks_agent_execution_or_size_limit() -> None:
    def broken_writer(_: dict[str, Any]) -> None:
        raise RuntimeError("custom stream unavailable")

    with trace.request_context(thread_id="thread-1", run_id="run-1"):
        emitter = trace.TraceEmitter(lambda: broken_writer)
        emitter.step_upsert(
            step_id="step-1",
            kind="tool",
            title="观察屏幕",
            summary="正在读取当前手机屏幕状态。",
            status="running",
        )
        event = emitter.step_detail_append(
            step_id="step-1",
            kind="tool_result",
            title="工具结果",
            text="结果" * 10_000,
        )

    assert event is not None
    assert len(json.dumps(event, ensure_ascii=False).encode("utf-8")) <= trace.MAX_TRACE_EVENT_BYTES
