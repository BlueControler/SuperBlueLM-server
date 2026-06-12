from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from mobile_agent.agent.phone_subagent import (
    PhoneSubagentRunner,
    PhoneToolBudgetExceededError,
    PhoneToolBudgetMiddleware,
    PhoneToolRepeatedActionError,
    PhoneToolSequenceError,
    redact_phone_text,
)
from mobile_agent.agent.middleware import SyncPhoneStateMiddleware


@dataclass
class _DeviceInfo:
    current_package: str = "com.example"
    activity: str = ".MainActivity"
    screenshot: str = "base64-screenshot"
    ui: str = "<node text='private-ui' />"


class _Session:
    device_info = _DeviceInfo()


class _Gateway:
    def get_session(self) -> _Session:
        return _Session()


class _Agent:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        return self.response


class _RaisingAgent:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        raise self.error


def _completed_agent(messages: list[Any] | None = None) -> _Agent:
    return _Agent(
        {
            "messages": messages or [],
            "structured_response": {
                "status": "completed",
                "summary": "done",
                "needsMainAgentPlan": True,
            },
        }
    )


def test_phone_subagent_exposes_only_phone_tools_and_default_budget(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.delenv("PHONE_SUBAGENT_MAX_TOOL_CALLS", raising=False)

    def agent_factory(**kwargs: Any) -> _Agent:
        captured.update(kwargs)
        return _completed_agent()

    runner = PhoneSubagentRunner(
        _Gateway(),
        "openai:phone-small",
        agent_factory=agent_factory,
    )

    asyncio.run(runner.execute("Tap the visible search box", allow_short_chain=False))

    tool_names = {tool.name for tool in captured["tools"]}
    assert {"observe", "tap", "type", "launch"} <= tool_names
    assert "list_apps" not in tool_names
    assert "weather_query" not in tool_names
    budget_middleware = captured["middleware"][-1]
    assert isinstance(budget_middleware, PhoneToolBudgetMiddleware)
    assert budget_middleware.limit == 12
    assert isinstance(captured["middleware"][0], SyncPhoneStateMiddleware)


def test_phone_subagent_binds_tools_to_session_at_todo_start() -> None:
    class CommandSession(_Session):
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def send_command(self, message: str, data: Any) -> dict[str, Any]:
            del data
            self.calls.append(message)
            return {}

    original_session = CommandSession()
    replacement_session = CommandSession()

    class SwitchingGateway:
        current = original_session

        def get_session(self, device_id: str | None = None) -> CommandSession:
            del device_id
            return self.current

        async def wait_for_session(
            self,
            device_id: str | None = None,
        ) -> CommandSession:
            del device_id
            return self.current

    gateway = SwitchingGateway()
    tool_result: list[str] = []

    class ReconnectingAgent:
        def __init__(self, tools: list[Any]) -> None:
            self.tools = {tool.name: tool for tool in tools}

        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            del state
            gateway.current = replacement_session
            tool_result.append(await self.tools["home"].ainvoke({"device_id": "device-1"}))
            return {
                "messages": [],
                "structured_response": {
                    "status": "failed",
                    "summary": "device disconnected",
                    "needsMainAgentPlan": True,
                },
            }

    runner = PhoneSubagentRunner(
        gateway,
        "openai:phone-small",
        agent_factory=lambda **kwargs: ReconnectingAgent(kwargs["tools"]),
    )

    asyncio.run(
        runner.execute(
            "Go home",
            allow_short_chain=False,
            device_id="device-1",
        )
    )

    assert '"error": "device_not_connected"' in tool_result[0]
    assert replacement_session.calls == []


def test_phone_subagent_uses_bounded_short_chain_budget(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setenv("PHONE_SUBAGENT_SHORT_CHAIN_MAX_TOOL_CALLS", "3")

    def agent_factory(**kwargs: Any) -> _Agent:
        captured.update(kwargs)
        return _completed_agent()

    runner = PhoneSubagentRunner(
        _Gateway(),
        "openai:phone-small",
        agent_factory=agent_factory,
    )

    asyncio.run(
        runner.execute(
            "Tap the search box and type noodles",
            allow_short_chain=True,
        )
    )

    budget_middleware = captured["middleware"][-1]
    assert isinstance(budget_middleware, PhoneToolBudgetMiddleware)
    assert budget_middleware.limit == 3


def test_phone_subagent_returns_redacted_server_owned_state_summary() -> None:
    def agent_factory(**kwargs: Any) -> _Agent:
        return _Agent(
            {
                "messages": [
                    ToolMessage(content="ok", tool_call_id="call-1", name="tap")
                ],
                "structured_response": {
                    "status": "needs_user_action",
                    "summary": (
                        "password=raw data:image/png;base64,AAAA "
                        "<node text='private-ui' />"
                    ),
                    "needsMainAgentPlan": True,
                    "error": "token=raw",
                },
            }
        )

    runner = PhoneSubagentRunner(
        _Gateway(),
        "openai:phone-small",
        agent_factory=agent_factory,
    )

    result = asyncio.run(runner.execute("Stop for login", allow_short_chain=False))
    payload = result.model_dump(mode="json", by_alias=True)

    assert payload["status"] == "needs_user_action"
    assert payload["toolCallCount"] == 1
    assert payload["phoneState"] == {
        "currentPackage": "com.example",
        "activity": ".MainActivity",
        "hasScreenshot": True,
        "hasUi": True,
    }
    assert "raw" not in str(payload)
    assert "AAAA" not in str(payload)
    assert "private-ui" not in str(payload)


def test_phone_subagent_rejects_sensitive_todo_before_child_model_invocation() -> None:
    invoked = False

    def agent_factory(**kwargs: Any) -> _Agent:
        nonlocal invoked
        invoked = True
        return _completed_agent()

    runner = PhoneSubagentRunner(
        _Gateway(),
        "openai:phone-small",
        agent_factory=agent_factory,
    )

    result = asyncio.run(
        runner.execute(
            "Type password=raw and token=secret",
            allow_short_chain=False,
        )
    )

    assert invoked is False
    assert result.status == "needs_user_action"
    assert result.todo == "Type password=*** and token=***"
    assert result.error == "phone_todo_contains_sensitive_data"


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (
            PhoneToolBudgetExceededError(executed_count=1, limit=1),
            "budget_exhausted",
        ),
        (PhoneToolSequenceError("parallel"), "failed"),
        (RuntimeError("failed"), "failed"),
    ],
)
def test_phone_subagent_redacts_sensitive_todo_on_execution_error(
    error: Exception,
    status: str,
) -> None:
    runner = PhoneSubagentRunner(
        _Gateway(),
        "openai:phone-small",
        agent_factory=lambda **kwargs: _RaisingAgent(error),
    )

    result = asyncio.run(
        runner.execute(
            "Tap the visible search box",
            allow_short_chain=False,
        )
    )

    assert result.status == status
    assert result.todo == "Tap the visible search box"


def test_phone_subagent_redacts_error_result_todo_defensively() -> None:
    runner = PhoneSubagentRunner(
        _Gateway(),
        "openai:phone-small",
    )

    failed = runner._failed(
        "Type password=raw and token=secret",
        "phone_subagent_execution_failed",
    )
    exhausted = runner._budget_exhausted(
        "Type password=raw and token=secret",
        executed_count=1,
        limit=1,
    )

    assert failed.todo == "Type password=*** and token=***"
    assert exhausted.todo == "Type password=*** and token=***"


def test_phone_text_redaction_covers_extended_credentials_and_images() -> None:
    redacted = redact_phone_text(
        "Authorization: Basic abc123 cookie=session-value session_id=raw "
        "password='two words' "
        "data:image/svg+xml;charset=utf-8;base64,AAAA"
    )

    assert "abc123" not in redacted
    assert "session-value" not in redacted
    assert "raw" not in redacted
    assert "two words" not in redacted
    assert "AAAA" not in redacted


def test_phone_tool_budget_allows_first_sequential_call_and_blocks_second() -> None:
    middleware = PhoneToolBudgetMiddleware({"tap", "type"}, limit=1)

    update = middleware.after_model(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "tap", "args": {"x": 1, "y": 2}, "id": "call-1"}
                    ],
                )
            ]
        },
        runtime=None,
    )

    assert update["phone_tool_call_count"] == 1
    assert update["phone_identical_action_count"] == 1
    with pytest.raises(PhoneToolBudgetExceededError):
        middleware.after_model(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "type", "args": {"text": "noodles"}, "id": "call-2"}
                        ],
                    )
                ],
                "phone_tool_call_count": 1,
            },
            runtime=None,
        )


def test_phone_tool_budget_rejects_parallel_phone_actions() -> None:
    middleware = PhoneToolBudgetMiddleware({"tap", "type"}, limit=4)

    with pytest.raises(PhoneToolSequenceError):
        middleware.after_model(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "tap", "args": {"x": 1, "y": 2}, "id": "call-1"},
                            {"name": "type", "args": {"text": "noodles"}, "id": "call-2"},
                        ],
                    )
                ]
            },
            runtime=None,
        )


def test_phone_tool_budget_rejects_fourth_identical_action() -> None:
    middleware = PhoneToolBudgetMiddleware({"swipe"}, limit=12, identical_action_limit=3)
    action = {
        "name": "swipe",
        "args": {"start_x": 283, "start_y": 1215, "end_x": 283, "end_y": 200},
        "id": "call-1",
    }
    state: dict[str, Any] = {"messages": [AIMessage(content="", tool_calls=[action])]}

    first = middleware.after_model(state, runtime=None)
    second = middleware.after_model({**state, **first}, runtime=None)
    third = middleware.after_model({**state, **second}, runtime=None)

    assert first["phone_identical_action_count"] == 1
    assert second["phone_identical_action_count"] == 2
    assert third["phone_identical_action_count"] == 3
    with pytest.raises(PhoneToolRepeatedActionError):
        middleware.after_model({**state, **third}, runtime=None)


def test_phone_tool_budget_resets_identical_count_when_action_changes() -> None:
    middleware = PhoneToolBudgetMiddleware({"swipe", "observe"}, limit=12)
    swipe = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "swipe",
                "args": {"start_x": 283, "start_y": 1215, "end_x": 283, "end_y": 200},
                "id": "call-1",
            }
        ],
    )
    observe = AIMessage(
        content="",
        tool_calls=[{"name": "observe", "args": {}, "id": "call-2"}],
    )

    first = middleware.after_model({"messages": [swipe]}, runtime=None)
    changed = middleware.after_model(
        {"messages": [observe], **first},
        runtime=None,
    )

    assert changed["phone_identical_action_count"] == 1
    assert changed["phone_last_action_signature"] != first["phone_last_action_signature"]


def test_phone_tool_budget_ignores_structured_output_tool_call() -> None:
    middleware = PhoneToolBudgetMiddleware({"tap"}, limit=1)

    update = middleware.after_model(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "PhoneSubagentDecision",
                            "args": {"status": "completed"},
                            "id": "call-1",
                        }
                    ],
                )
            ]
        },
        runtime=None,
    )

    assert update is None
