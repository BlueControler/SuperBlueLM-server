from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage

from mobile_agent.agent.middleware import (
    RoutedSystemPromptMiddleware,
    build_current_time_message,
)
from mobile_agent.prompt_assets import LOCAL_MODEL_SYSTEM_PROMPT, SYSTEM_PROMPT


@dataclass(frozen=True)
class FakeRequest:
    messages: list[Any]
    state: dict[str, Any]

    def override(self, **kwargs: Any) -> FakeRequest:
        values = {
            "messages": self.messages,
            "state": self.state,
        }
        values.update(kwargs)
        return FakeRequest(**values)


def test_routed_system_prompt_includes_current_time_after_prompt(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "mobile_agent.agent.middleware.model_runtime.status",
        lambda: {"mode": "cloud"},
    )
    fixed = datetime(2026, 5, 9, 10, 30, 5, tzinfo=ZoneInfo("Asia/Shanghai"))
    middleware = RoutedSystemPromptMiddleware(now_provider=lambda: fixed)
    user_message = HumanMessage(content="\u4eca\u5929\u6709\u4ec0\u4e48\u5b89\u6392?")
    request = FakeRequest(
        messages=[user_message],
        state={},
    )
    captured_messages: list[Any] = []

    def handler(next_request: Any) -> None:
        captured_messages.extend(next_request.messages)
        return None

    middleware.wrap_model_call(request, handler)

    assert len(captured_messages) == 3
    assert isinstance(captured_messages[0], SystemMessage)
    assert captured_messages[0].content == SYSTEM_PROMPT
    assert isinstance(captured_messages[1], SystemMessage)
    assert "\u5f53\u524d\u65f6\u95f4" in captured_messages[1].content
    assert "2026-05-09T10:30:05+08:00" in captured_messages[1].content
    assert "UnixMillis: 1778293805000" in captured_messages[1].content
    assert "Asia/Shanghai" in captured_messages[1].content
    for term in [
        "\u4eca\u5929",
        "\u660e\u5929",
        "\u6628\u5929",
        "\u7a0d\u540e",
        "\u672c\u5468",
    ]:
        assert term in captured_messages[1].content
    assert captured_messages[2] is user_message


def test_routed_system_prompt_uses_local_prompt_in_local_mode(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "mobile_agent.agent.middleware.model_runtime.status",
        lambda: {"mode": "local"},
    )
    fixed = datetime(2026, 5, 9, 10, 30, 5, tzinfo=ZoneInfo("Asia/Shanghai"))
    middleware = RoutedSystemPromptMiddleware(now_provider=lambda: fixed)
    user_message = HumanMessage(content="hello")
    request = FakeRequest(
        messages=[user_message],
        state={},
    )
    captured_messages: list[Any] = []

    def handler(next_request: Any) -> None:
        captured_messages.extend(next_request.messages)
        return None

    middleware.wrap_model_call(request, handler)

    assert len(captured_messages) == 3
    assert isinstance(captured_messages[0], SystemMessage)
    assert captured_messages[0].content == LOCAL_MODEL_SYSTEM_PROMPT
    assert isinstance(captured_messages[1], SystemMessage)
    assert "2026-05-09T10:30:05+08:00" in captured_messages[1].content
    assert captured_messages[2] is user_message


def test_routed_system_prompt_replaces_existing_routed_pair(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "mobile_agent.agent.middleware.model_runtime.status",
        lambda: {"mode": "cloud"},
    )
    old_time = datetime(2026, 5, 9, 8, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    new_time = datetime(2026, 5, 9, 10, 30, 5, tzinfo=ZoneInfo("Asia/Shanghai"))
    human_message = HumanMessage(content="keep me")
    middleware = RoutedSystemPromptMiddleware(now_provider=lambda: new_time)
    request = FakeRequest(
        messages=[
            SystemMessage(content=SYSTEM_PROMPT),
            build_current_time_message(old_time),
            human_message,
        ],
        state={},
    )
    captured_messages: list[Any] = []

    def handler(next_request: Any) -> None:
        captured_messages.extend(next_request.messages)
        return None

    middleware.wrap_model_call(request, handler)

    assert len(captured_messages) == 3
    assert captured_messages[0].content == SYSTEM_PROMPT
    assert "2026-05-09T10:30:05+08:00" in captured_messages[1].content
    assert "2026-05-09T08:00:00+08:00" not in captured_messages[1].content
    assert captured_messages[2] is human_message


def test_build_current_time_message_rejects_naive_datetime() -> None:
    naive_now = datetime(2026, 5, 9, 10, 30, 5)

    try:
        build_current_time_message(naive_now)
    except ValueError as exc:
        assert "timezone-aware datetime" in str(exc)
    else:
        raise AssertionError("Expected ValueError for naive datetime")
