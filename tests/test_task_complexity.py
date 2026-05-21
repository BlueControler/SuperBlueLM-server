from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from mobile_agent import progress
from mobile_agent.agent.middleware import TaskComplexityMiddleware
from mobile_agent.agent.task_complexity import classify_task_complexity


def test_text_only_prompt_is_simple() -> None:
    result = classify_task_complexity("解释一下 LangGraph 是什么")

    assert result.complexity == "simple"
    assert result.track_steps is False
    assert result.reason == "text_only_answer"


def test_weather_prompt_is_complex_because_it_requires_tool_or_api() -> None:
    result = classify_task_complexity("查询深圳今天的天气，并给出一句出门建议")

    assert result.complexity == "complex"
    assert result.track_steps is True
    assert result.reason == "tool_or_external_action_required"


def test_phone_operation_prompt_is_complex() -> None:
    result = classify_task_complexity("打开浏览器，搜索蓝心小V")

    assert result.complexity == "complex"
    assert result.track_steps is True
    assert result.reason == "tool_or_external_action_required"


def test_emit_task_complexity_writes_custom_payload(monkeypatch: Any) -> None:
    emitted: list[dict[str, Any]] = []

    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    progress.emit_task_complexity(
        complexity="complex",
        track_steps=True,
        reason="tool_or_external_action_required",
        message="复杂任务，将跟踪步骤进度",
    )

    assert emitted == [
        {
            "type": "task_complexity",
            "complexity": "complex",
            "trackSteps": True,
            "reason": "tool_or_external_action_required",
            "message": "复杂任务，将跟踪步骤进度",
        }
    ]


def test_task_complexity_middleware_emits_once_per_run(monkeypatch: Any) -> None:
    emitted: list[dict[str, Any]] = []
    middleware = TaskComplexityMiddleware()
    state = {"messages": [HumanMessage(content="查询深圳天气")]}

    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    first_update = middleware.before_model(state, runtime=None)
    state.update(first_update or {})
    second_update = middleware.before_model(state, runtime=None)

    assert first_update == {"task_complexity_emitted": True}
    assert second_update is None
    assert emitted == [
        {
            "type": "task_complexity",
            "complexity": "complex",
            "trackSteps": True,
            "reason": "tool_or_external_action_required",
        }
    ]
