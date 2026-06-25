from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from deepagents import create_deep_agent
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from mobile_agent.agent.trace_middleware import TraceMiddleware
from mobile_agent.trace import TraceEmitter


class _BindableFakeChatModel(FakeMessagesListChatModel):
    """Fake model used by create_deep_agent, which always binds tools."""

    def bind_tools(self, *args: object, **kwargs: object) -> "_BindableFakeChatModel":
        del args, kwargs
        return self


class _FailingBindableFakeChatModel(_BindableFakeChatModel):
    def _generate(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("model failed")


def _request(
    name: str,
    state: dict[str, Any],
    args: dict[str, Any] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args or {}, "id": "call-1", "type": "tool_call"},
        tool=None,
        state=state,
        runtime=None,
    )


def test_fatal_tool_error_emits_failed_run_terminal() -> None:
    events: list[dict[str, Any]] = []
    middleware = TraceMiddleware(TraceEmitter(lambda: events.append))
    state: dict[str, Any] = {"messages": []}
    state.update(middleware.before_agent(state, SimpleNamespace(config={})) or {})

    async def failing_handler(_: ToolCallRequest) -> ToolMessage:
        raise RuntimeError("gateway failed")

    with pytest.raises(RuntimeError, match="gateway failed"):
        asyncio.run(middleware.awrap_tool_call(_request("observe", state), failing_handler))

    assert any(
        event.get("step", {}).get("status") == "failed" for event in events
    )
    error_details = [
        event["detail"]
        for event in events
        if event.get("event") == "step.detail.append"
        and event.get("detail", {}).get("kind") == "error"
    ]
    assert error_details
    assert "gateway failed" not in str(error_details)
    assert "Traceback" not in str(error_details)
    assert events[-1]["event"] == "run.terminal"
    assert events[-1]["status"] == "failed"


def test_waiting_command_becomes_waiting_step_and_run_terminal() -> None:
    events: list[dict[str, Any]] = []
    middleware = TraceMiddleware(TraceEmitter(lambda: events.append))
    state: dict[str, Any] = {"messages": []}
    state.update(middleware.before_agent(state, SimpleNamespace(config={})) or {})
    waiting = Command(update={"awaiting_user_action": True, "messages": []})

    async def waiting_handler(_: ToolCallRequest) -> Command:
        return waiting

    result = asyncio.run(middleware.awrap_tool_call(_request("interact", state), waiting_handler))
    assert result is waiting
    state.update(waiting.update)
    middleware.after_agent(state, SimpleNamespace(config={}))

    assert any(
        event.get("step", {}).get("status") == "waiting_for_user" for event in events
    )
    warning_details = [
        event["detail"]
        for event in events
        if event.get("event") == "step.detail.append"
        and event.get("detail", {}).get("kind") == "warning"
    ]
    assert warning_details
    assert "我不会继续自动执行" in warning_details[-1]["text"]
    assert events[-1]["status"] == "waiting_for_user"


def test_tool_call_emits_safe_call_args_and_result_details() -> None:
    events: list[dict[str, Any]] = []
    middleware = TraceMiddleware(TraceEmitter(lambda: events.append))
    state: dict[str, Any] = {"messages": []}
    state.update(middleware.before_agent(state, SimpleNamespace(config={})) or {})

    async def observe_handler(_: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content='{"screenshot":"base64","ui":"<node password=\\"raw\\"/>","token":"secret"}',
            tool_call_id="call-1",
        )

    result = asyncio.run(
        middleware.awrap_tool_call(
            _request("observe", state, {"token": "secret", "includeScreenshot": True}),
            observe_handler,
        )
    )

    assert isinstance(result, ToolMessage)
    details = [
        event["detail"]
        for event in events
        if event.get("event") == "step.detail.append"
        and event.get("stepId") == "tool_call-1"
    ]
    assert [detail["kind"] for detail in details] == [
        "tool_call",
        "tool_args_summary",
        "observation",
    ]
    combined = "\n".join(detail["text"] for detail in details)
    assert "观察当前屏幕" in combined
    assert "不展示截图原图" in combined
    assert "已获取当前屏幕概要" in combined
    assert "secret" not in combined
    assert "base64" not in combined
    assert "<node" not in combined


def test_execute_phone_todo_parent_step_can_receive_child_trace() -> None:
    events: list[dict[str, Any]] = []
    emitter = TraceEmitter(lambda: events.append)
    middleware = TraceMiddleware(emitter)
    state: dict[str, Any] = {"messages": []}
    state.update(middleware.before_agent(state, SimpleNamespace(config={})) or {})

    async def phone_handler(request: ToolCallRequest) -> ToolMessage:
        parent_id = f"tool_{request.tool_call['id']}"
        child_id = f"{parent_id}_phone_1"
        emitter.step_upsert(
            step_id=child_id,
            parent_id=parent_id,
            kind="phone_action",
            title="点击屏幕",
            summary="正在点击目标位置。",
            status="running",
        )
        emitter.step_detail_append(
            step_id=child_id,
            kind="tool_args_summary",
            title="安全参数",
            text="点击目标位置，不展示精确坐标。",
        )
        emitter.step_upsert(
            step_id=child_id,
            parent_id=parent_id,
            kind="phone_action",
            title="点击屏幕",
            summary="点击操作已执行。",
            status="succeeded",
        )
        return ToolMessage(
            content='{"status":"completed","summary":"done"}',
            tool_call_id=request.tool_call["id"],
            name="execute_phone_todo",
            status="success",
        )

    asyncio.run(
        middleware.awrap_tool_call(
            _request("execute_phone_todo", state, {"todo": "点击搜索框"}),
            phone_handler,
        )
    )

    parent_id = "tool_call-1"
    assert any(
        event.get("event") == "step.upsert"
        and event.get("step", {}).get("stepId") == parent_id
        for event in events
    )
    child_steps = [
        event["step"]
        for event in events
        if event.get("event") == "step.upsert"
        and event.get("step", {}).get("parentId") == parent_id
    ]
    assert [step["status"] for step in child_steps] == ["running", "succeeded"]
    assert "不展示精确坐标" in str(events)


def test_failed_child_phone_step_does_not_emit_failed_run_terminal_by_itself() -> None:
    events: list[dict[str, Any]] = []
    emitter = TraceEmitter(lambda: events.append)
    middleware = TraceMiddleware(emitter)
    state: dict[str, Any] = {"messages": []}
    state.update(middleware.before_agent(state, SimpleNamespace(config={})) or {})

    async def phone_handler(request: ToolCallRequest) -> ToolMessage:
        parent_id = f"tool_{request.tool_call['id']}"
        child_id = f"{parent_id}_phone_1"
        emitter.step_upsert(
            step_id=child_id,
            parent_id=parent_id,
            kind="phone_action",
            title="点击屏幕",
            summary="操作未完成，可稍后重试。",
            status="failed",
        )
        return ToolMessage(
            content='{"status":"completed","summary":"parent finished"}',
            tool_call_id=request.tool_call["id"],
            name="execute_phone_todo",
            status="success",
        )

    asyncio.run(
        middleware.awrap_tool_call(
            _request("execute_phone_todo", state, {"todo": "点击搜索框"}),
            phone_handler,
        )
    )

    child_failed = [
        event
        for event in events
        if event.get("step", {}).get("parentId") == "tool_call-1"
        and event.get("step", {}).get("status") == "failed"
    ]
    assert child_failed
    assert not any(
        event.get("event") == "run.terminal" and event.get("status") == "failed"
        for event in events
    )


def test_analysis_step_emits_safe_reasoning_summary_without_raw_think() -> None:
    events: list[dict[str, Any]] = []
    middleware = TraceMiddleware(TraceEmitter(lambda: events.append))
    state: dict[str, Any] = {"messages": []}

    middleware.before_agent(state, SimpleNamespace(config={}))

    details = [
        event["detail"]
        for event in events
        if event.get("event") == "step.detail.append"
        and event.get("detail", {}).get("kind") in {"plan", "reasoning_summary"}
    ]
    assert details
    combined = "\n".join(detail["text"] for detail in details)
    assert "<think>" not in combined
    assert "系统提示词" not in combined


def test_failed_run_state_becomes_failed_trace_terminal() -> None:
    events: list[dict[str, Any]] = []
    middleware = TraceMiddleware(TraceEmitter(lambda: events.append))
    state: dict[str, Any] = {"messages": []}
    state.update(middleware.before_agent(state, SimpleNamespace(config={})) or {})
    state["run_failure_reason"] = "missing_structured_response"

    middleware.after_agent(state, SimpleNamespace(config={}))

    assert events[-2]["step"]["status"] == "failed"
    assert events[-1]["event"] == "run.terminal"
    assert events[-1]["status"] == "failed"


def test_after_agent_resumes_trace_session_across_langgraph_nodes() -> None:
    events: list[dict[str, Any]] = []
    agent = create_deep_agent(
        model=_BindableFakeChatModel(responses=[AIMessage(content="OK")]),
        tools=[],
        middleware=[TraceMiddleware(TraceEmitter(lambda: events.append))],
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "回复我OK"}]})

    assert result["messages"][-1].content == "OK"
    assert [
        (event["seq"], event["event"], event.get("status"), event.get("step", {}).get("status"))
        for event in events
    ] == [
        (1, "run.started", None, None),
        (2, "step.upsert", None, "running"),
        (3, "step.detail.append", None, None),
        (4, "step.upsert", None, "succeeded"),
        (5, "run.terminal", "succeeded", None),
    ]
    assert events[2]["detail"]["kind"] == "plan"


def test_before_agent_uses_client_owned_mobile_run_id_from_runtime_config() -> None:
    events: list[dict[str, Any]] = []
    middleware = TraceMiddleware(TraceEmitter(lambda: events.append))
    state: dict[str, Any] = {"messages": []}
    runtime = SimpleNamespace(
        config={"configurable": {"thread_id": "thread-1", "mobile_run_id": "run-client-1"}},
    )

    update = middleware.before_agent(state, runtime)

    assert update["trace_run_id"] == "run-client-1"
    assert events[0]["runId"] == "run-client-1"
    assert events[0]["threadId"] == "thread-1"
    middleware.after_agent({**state, **update}, runtime)


def test_model_error_resumes_trace_session_and_emits_failed_terminal() -> None:
    events: list[dict[str, Any]] = []
    agent = create_deep_agent(
        model=_FailingBindableFakeChatModel(responses=[]),
        tools=[],
        middleware=[TraceMiddleware(TraceEmitter(lambda: events.append))],
    )

    with pytest.raises(RuntimeError, match="model failed"):
        agent.invoke({"messages": [{"role": "user", "content": "回复我OK"}]})

    assert [
        (event["seq"], event["event"], event.get("status"))
        for event in events
    ] == [
        (1, "run.started", None),
        (2, "step.upsert", None),
        (3, "step.detail.append", None),
        (4, "run.terminal", "failed"),
    ]
    assert events[2]["detail"]["kind"] == "plan"
