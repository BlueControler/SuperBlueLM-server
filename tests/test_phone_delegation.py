from __future__ import annotations

import asyncio
from typing import Any

from langchain.tools import ToolRuntime
from langgraph.graph import END

from mobile_agent import progress
from mobile_agent.agent.phone_delegation import (
    ResetPhoneTodoMiddleware,
    create_phone_delegation_tool,
    execute_tracked_phone_todo,
)
from mobile_agent.agent.phone_subagent import PhoneTodoExecution
from mobile_agent.gateways.phone import DeviceGatewayError


def _execution(
    todo: str,
    *,
    status: str = "completed",
    summary: str = "done",
) -> PhoneTodoExecution:
    return PhoneTodoExecution(
        status=status,
        todo=todo,
        summary=summary,
        phoneState={
            "currentPackage": "com.example",
            "activity": ".MainActivity",
            "hasScreenshot": True,
            "hasUi": True,
        },
        toolCallCount=1,
        needsMainAgentPlan=True,
    )


class _Runner:
    def __init__(self, *results: PhoneTodoExecution) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, bool]] = []
        self.trace_parent_ids: list[str | None] = []

    async def execute(
        self,
        todo: str,
        *,
        allow_short_chain: bool,
        device_id: str | None = None,
        trace_parent_id: str | None = None,
    ) -> PhoneTodoExecution:
        del device_id
        self.calls.append((todo, allow_short_chain))
        self.trace_parent_ids.append(trace_parent_id)
        return self.results.pop(0)


def test_phone_delegation_converts_device_disconnect_to_recoverable_result() -> None:
    class DisconnectedRunner:
        async def execute(
            self,
            todo: str,
            *,
            allow_short_chain: bool,
            device_id: str | None = None,
        ) -> PhoneTodoExecution:
            raise DeviceGatewayError("device_not_connected")

    result, _ = asyncio.run(
        execute_tracked_phone_todo(
            (),
            DisconnectedRunner(),
            "Go home",
            allow_short_chain=False,
            device_id="device-1",
        )
    )

    assert result.status == "failed"
    assert result.error == "device_not_connected"
    assert result.summary == "手机连接已断开，请重新连接后重试"
    assert result.needs_main_agent_plan is True


def test_phone_todo_progress_total_grows_as_main_agent_adds_steps(
    monkeypatch: Any,
) -> None:
    emitted: list[dict[str, Any]] = []
    runner = _Runner(
        _execution("Open the app"),
        _execution("Tap the search box"),
    )
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    first_result, first_steps = asyncio.run(
        execute_tracked_phone_todo(
            (),
            runner,
            "Open the app",
            allow_short_chain=False,
        )
    )
    second_result, second_steps = asyncio.run(
        execute_tracked_phone_todo(
            first_steps,
            runner,
            "Tap the search box",
            allow_short_chain=False,
        )
    )

    assert first_result.status == "completed"
    assert second_result.status == "completed"
    assert first_steps == (
        {
            "index": 1,
            "progressKey": "phone-todo-1",
            "name": "Open the app",
            "status": "completed",
            "summary": "done",
        },
    )
    assert len(second_steps) == 2
    assert len(first_steps) == 1
    assert runner.calls == [
        ("Open the app", False),
        ("Tap the search box", False),
    ]
    assert [event["totalSteps"] for event in emitted] == [1, 1, 2, 2]
    assert emitted[2]["completedSteps"] == [
        {"index": 1, "name": "Open the app", "status": "completed"}
    ]


def test_phone_todo_failure_is_retained_when_main_agent_adds_correction(
    monkeypatch: Any,
) -> None:
    emitted: list[dict[str, Any]] = []
    runner = _Runner(
        _execution("Tap Search", status="failed", summary="wrong screen"),
        _execution("Go back and reopen Search"),
    )
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    _, failed_steps = asyncio.run(
        execute_tracked_phone_todo(
            (),
            runner,
            "Tap Search",
            allow_short_chain=False,
        )
    )
    _, corrected_steps = asyncio.run(
        execute_tracked_phone_todo(
            failed_steps,
            runner,
            "Go back and reopen Search",
            allow_short_chain=True,
        )
    )

    assert failed_steps[0]["status"] == "failed"
    assert corrected_steps[0] == failed_steps[0]
    assert corrected_steps[1]["status"] == "completed"
    assert corrected_steps[1]["progressKey"] == "phone-todo-2"
    assert emitted[-1]["completedSteps"] == [
        {
            "index": 2,
            "name": "Go back and reopen Search",
            "status": "completed",
        }
    ]
    assert runner.calls[-1] == ("Go back and reopen Search", True)


def test_phone_todo_progress_redacts_sensitive_values_without_changing_execution(
    monkeypatch: Any,
) -> None:
    emitted: list[dict[str, Any]] = []
    todo = "Type password=raw and token=secret"
    runner = _Runner(_execution(todo))
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    _, steps = asyncio.run(
        execute_tracked_phone_todo(
            (),
            runner,
            todo,
            allow_short_chain=False,
        )
    )

    assert runner.calls == [(todo, False)]
    assert steps[0]["name"] == "Type password=*** and token=***"
    assert all("raw" not in str(event) for event in emitted)
    assert all("secret" not in str(event) for event in emitted)


def test_phone_todo_history_redacts_runner_summary_and_error(
    monkeypatch: Any,
) -> None:
    emitted: list[dict[str, Any]] = []
    runner = _Runner(
        PhoneTodoExecution(
            status="failed",
            todo="Tap Search",
            summary="password=raw",
            phoneState={
                "currentPackage": "com.example",
                "activity": ".MainActivity",
                "hasScreenshot": True,
                "hasUi": True,
            },
            toolCallCount=1,
            needsMainAgentPlan=True,
            error="token=secret",
        )
    )
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    _, steps = asyncio.run(
        execute_tracked_phone_todo(
            (),
            runner,
            "Tap Search",
            allow_short_chain=False,
        )
    )

    assert steps[0]["summary"] == "password=***"
    assert emitted[-1]["message"] == "password=***"
    assert emitted[-1]["error"] == "token=***"


def test_phone_delegation_tool_schema_exposes_validated_main_agent_inputs() -> None:
    tool = create_phone_delegation_tool(_Runner())
    schema = tool.args_schema.model_json_schema()

    assert schema["properties"].keys() == {"todo", "allow_short_chain"}
    assert schema["properties"]["todo"]["minLength"] == 1
    assert schema["properties"]["todo"]["maxLength"] == 1000


def test_phone_delegation_tool_preserves_injected_runtime_without_restarting_done_step(
    monkeypatch: Any,
) -> None:
    emitted: list[dict[str, Any]] = []
    runner = _Runner(_execution("Open the app"))
    tool = create_phone_delegation_tool(runner)
    runtime = ToolRuntime(
        state={
            "phone_todo_steps": (
                {
                    "index": 1,
                    "progressKey": "phone-todo-1",
                    "name": "old",
                    "status": "completed",
                    "summary": "done",
                },
            )
        },
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="call_1",
        store=None,
    )
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    result = asyncio.run(
        tool.ainvoke(
            {
                "name": "execute_phone_todo",
                "args": {"todo": "Open the app", "runtime": runtime},
                "id": "call_1",
                "type": "tool_call",
            }
        )
    )

    assert "phone_todo_steps" not in result.update
    assert result.update["messages"][0].tool_call_id == "call_1"
    assert '"already_done": true' in result.update["messages"][0].content
    assert runner.calls == []


def test_phone_delegation_ends_run_when_subagent_returns_failed_result() -> None:
    runner = _Runner(
        PhoneTodoExecution(
            status="failed",
            todo="Go home",
            summary="Phone subagent did not return a structured result.",
            phoneState={
                "currentPackage": "com.example",
                "activity": ".MainActivity",
                "hasScreenshot": True,
                "hasUi": True,
            },
            toolCallCount=0,
            needsMainAgentPlan=True,
            error="missing_structured_response",
        )
    )
    tool = create_phone_delegation_tool(runner)
    runtime = ToolRuntime(
        state={"phone_todo_steps": ()},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="call_1",
        store=None,
    )

    command = asyncio.run(
        tool.ainvoke(
            {
                "name": "execute_phone_todo",
                "args": {"todo": "Go home", "runtime": runtime},
                "id": "call_1",
                "type": "tool_call",
            }
        )
    )

    assert command.goto == END
    assert command.update["run_failure_reason"] == "missing_structured_response"
    assert command.update["messages"][0].status == "error"


def test_phone_delegation_ends_run_after_deterministic_phone_success() -> None:
    runner = _Runner(
        PhoneTodoExecution(
            status="completed",
            todo="打开微信",
            summary="已发起打开应用操作。",
            phoneState={
                "currentPackage": "com.tencent.mm",
                "activity": ".ui.LauncherUI",
                "hasScreenshot": True,
                "hasUi": True,
            },
            toolCallCount=1,
            needsMainAgentPlan=False,
        )
    )
    tool = create_phone_delegation_tool(runner)
    runtime = ToolRuntime(
        state={"phone_todo_steps": ()},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="call_1",
        store=None,
    )

    command = asyncio.run(
        tool.ainvoke(
            {
                "name": "execute_phone_todo",
                "args": {"todo": "打开微信", "runtime": runtime},
                "id": "call_1",
                "type": "tool_call",
            }
        )
    )

    assert command.goto == END
    assert command.update["messages"][0].status == "success"
    assert len(command.update["messages"]) == 1
    assert command.update["phone_todo_steps"][0]["summary"] == "已发起打开应用操作。"


def test_phone_delegation_passes_device_id_from_run_metadata() -> None:
    class DeviceRunner:
        def __init__(self) -> None:
            self.device_ids: list[str | None] = []

        async def execute(
            self,
            todo: str,
            *,
            allow_short_chain: bool,
            device_id: str | None = None,
        ) -> PhoneTodoExecution:
            self.device_ids.append(device_id)
            return _execution(todo)

    runner = DeviceRunner()
    tool = create_phone_delegation_tool(runner)
    runtime = ToolRuntime(
        state={},
        context=None,
        config={"metadata": {"deviceId": "device-metadata-1"}},
        stream_writer=lambda _: None,
        tool_call_id="call_1",
        store=None,
    )

    asyncio.run(
        tool.ainvoke(
            {
                "name": "execute_phone_todo",
                "args": {"todo": "Open the app", "runtime": runtime},
                "id": "call_1",
                "type": "tool_call",
            }
        )
    )

    assert runner.device_ids == ["device-metadata-1"]


def test_phone_delegation_passes_trace_parent_id_from_tool_call_id() -> None:
    runner = _Runner(_execution("Open the app"))
    tool = create_phone_delegation_tool(runner)
    runtime = ToolRuntime(
        state={"phone_todo_steps": ()},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="call_parent",
        store=None,
    )

    asyncio.run(
        tool.ainvoke(
            {
                "name": "execute_phone_todo",
                "args": {"todo": "Open the app", "runtime": runtime},
                "id": "call_parent",
                "type": "tool_call",
            }
        )
    )

    assert runner.trace_parent_ids == ["tool_call_parent"]


def test_reset_phone_todo_middleware_returns_fresh_immutable_state() -> None:
    update = ResetPhoneTodoMiddleware().before_agent(
        {
            "phone_todo_steps": (
                {
                    "index": 1,
                    "progressKey": "phone-todo-1",
                    "name": "old",
                    "status": "completed",
                    "summary": "done",
                },
            ),
            "task_complexity_emitted": True,
        },
        runtime=None,
    )

    assert update == {
        "phone_todo_steps": (),
        "task_complexity_emitted": False,
        "awaiting_user_action": False,
        "awaiting_user_reason": "",
        "run_failure_reason": "",
    }
