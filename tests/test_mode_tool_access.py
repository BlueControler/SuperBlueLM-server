from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage

from mobile_agent.agent.middleware import ModeToolAccessMiddleware


@dataclass(frozen=True)
class _ModelRequest:
    tools: list[dict[str, Any]]

    def override(self, **kwargs: Any) -> _ModelRequest:
        return _ModelRequest(tools=kwargs.get("tools", self.tools))


def _visible_tool_names(
    monkeypatch: Any,
    *,
    mode: str,
) -> list[str]:
    monkeypatch.setattr(
        "mobile_agent.agent.middleware.model_runtime.status",
        lambda: {"mode": mode},
    )
    middleware = ModeToolAccessMiddleware(
        {
            "observe",
            "tap",
            "type",
            "launch",
            "swipe",
            "keyevent",
            "back",
            "home",
            "wait",
            "interact",
            "take_over",
        }
    )
    captured: list[dict[str, Any]] = []

    def handler(request: _ModelRequest) -> None:
        captured.extend(request.tools)

    middleware.wrap_model_call(
        _ModelRequest(
            tools=[
                {"name": "observe"},
                {"name": "tap"},
                {"name": "type"},
                {"name": "launch"},
                {"name": "swipe"},
                {"name": "keyevent"},
                {"name": "back"},
                {"name": "home"},
                {"name": "wait"},
                {"name": "interact"},
                {"name": "take_over"},
                {"name": "execute_phone_todo"},
                {"name": "weather_query"},
                {"name": "task"},
                {"name": "read_file"},
            ]
        ),
        handler,
    )
    return [tool["name"] for tool in captured]


def _tool_call_request(
    name: str,
    *,
    args: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    messages: list[Any] | None = None,
    runtime: Any = None,
) -> ToolCallRequest:
    request_state = {"messages": messages or []}
    if state:
        request_state.update(state)
    return ToolCallRequest(
        tool_call={
            "name": name,
            "args": args or {},
            "id": f"call-{name}",
            "type": "tool_call",
        },
        tool=None,
        state=request_state,
        runtime=runtime,
    )


def test_cloud_main_agent_sees_delegation_and_not_raw_phone_tools(
    monkeypatch: Any,
) -> None:
    assert _visible_tool_names(monkeypatch, mode="cloud") == [
        "execute_phone_todo",
        "weather_query",
    ]


def test_local_main_agent_sees_raw_phone_tools_and_not_delegation(
    monkeypatch: Any,
) -> None:
    assert _visible_tool_names(monkeypatch, mode="local") == [
        "observe",
        "tap",
        "back",
        "home",
        "wait",
        "interact",
        "take_over",
        "weather_query",
    ]


def test_cloud_main_agent_cannot_bypass_delegation_with_raw_phone_call(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "mobile_agent.agent.middleware.model_runtime.status",
        lambda: {"mode": "cloud"},
    )
    middleware = ModeToolAccessMiddleware({"tap"})

    result = middleware.wrap_tool_call(
        _tool_call_request("tap"),
        lambda request: "unexpected",
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "unavailable in cloud mode" in result.content


def test_local_main_agent_cannot_bypass_local_guardrail_with_delegation_call(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "mobile_agent.agent.middleware.model_runtime.status",
        lambda: {"mode": "local"},
    )
    middleware = ModeToolAccessMiddleware({"tap"})

    result = middleware.wrap_tool_call(
        _tool_call_request("execute_phone_todo"),
        lambda request: "unexpected",
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "unavailable in local mode" in result.content


def test_local_main_agent_cannot_call_high_risk_raw_phone_tool(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "mobile_agent.agent.middleware.model_runtime.status",
        lambda: {"mode": "local"},
    )
    middleware = ModeToolAccessMiddleware({"tap", "type"})

    result = middleware.wrap_tool_call(
        _tool_call_request("type"),
        lambda request: "unexpected",
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "unavailable in local mode" in result.content


def test_local_main_agent_cannot_call_second_phone_tool(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "mobile_agent.agent.middleware.model_runtime.status",
        lambda: {"mode": "local"},
    )
    middleware = ModeToolAccessMiddleware({"tap", "back"})

    result = middleware.wrap_tool_call(
        _tool_call_request(
            "back",
            messages=[
                ToolMessage(content="{}", tool_call_id="call-tap", name="tap"),
            ],
        ),
        lambda request: "unexpected",
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "unavailable in local mode" in result.content


def test_local_main_agent_cannot_call_parallel_phone_tools(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "mobile_agent.agent.middleware.model_runtime.status",
        lambda: {"mode": "local"},
    )
    middleware = ModeToolAccessMiddleware({"tap", "back"})

    result = middleware.wrap_tool_call(
        _tool_call_request(
            "tap",
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "tap", "args": {}, "id": "call-tap"},
                        {"name": "back", "args": {}, "id": "call-back"},
                    ],
                ),
            ],
        ),
        lambda request: "unexpected",
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "unavailable in local mode" in result.content


def test_deep_agent_bypass_tools_stay_blocked_in_every_mode(monkeypatch: Any) -> None:
    middleware = ModeToolAccessMiddleware({"tap"})

    for mode in ["cloud", "local"]:
        monkeypatch.setattr(
            "mobile_agent.agent.middleware.model_runtime.status",
            lambda: {"mode": mode},
        )
        for tool_name in ["task", "execute", "read_file", "write_file"]:
            result = middleware.wrap_tool_call(
                _tool_call_request(tool_name),
                lambda request: "unexpected",
            )
            assert isinstance(result, ToolMessage)
            assert result.status == "error"


def test_device_scoped_tool_is_bound_to_thread_device_id(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "mobile_agent.agent.middleware.model_runtime.status",
        lambda: {"mode": "cloud"},
    )
    middleware = ModeToolAccessMiddleware({"tap"}, {"tap", "list_apps"})
    captured: list[ToolCallRequest] = []

    result = middleware.wrap_tool_call(
        _tool_call_request(
            "list_apps",
            args={"device_id": "other-device"},
            state={"deviceId": "thread-device"},
        ),
        lambda request: captured.append(request) or "ok",
    )

    assert result == "ok"
    assert captured[0].tool_call["args"]["device_id"] == "thread-device"


def test_device_scoped_tool_is_bound_to_run_metadata(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "mobile_agent.agent.middleware.model_runtime.status",
        lambda: {"mode": "cloud"},
    )
    middleware = ModeToolAccessMiddleware({"tap"}, {"tap", "list_apps"})
    request = _tool_call_request(
        "list_apps",
        runtime=SimpleNamespace(config={"metadata": {"deviceId": "metadata-device"}}),
    )
    captured: list[ToolCallRequest] = []

    middleware.wrap_tool_call(
        request,
        lambda bound: captured.append(bound) or "ok",
    )

    assert captured[0].tool_call["args"]["device_id"] == "metadata-device"
