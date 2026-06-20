from __future__ import annotations

import asyncio
from typing import Any

from mobile_agent.tools.completion import create_completion_tools


def _finish_tool() -> Any:
    return {tool.name: tool for tool in create_completion_tools()}["finish"]


def test_finish_tool_exposed() -> None:
    tools = {tool.name: tool for tool in create_completion_tools()}
    assert set(tools) == {"finish"}


def test_finish_returns_summary() -> None:
    result = asyncio.run(_finish_tool().ainvoke({"summary": "全部完成"}))
    assert result == "全部完成"


def test_finish_accepts_subtasks() -> None:
    result = asyncio.run(
        _finish_tool().ainvoke(
            {
                "summary": "会议后续已处理",
                "subtasks": [
                    {"name": "消息发送", "status": "completed", "detail": "已通知设计组"},
                    {"name": "文档检索", "status": "skipped"},
                ],
            }
        )
    )
    assert result == "会议后续已处理"


def test_finish_is_return_direct() -> None:
    assert _finish_tool().return_direct is True


def test_finish_subtask_status_schema() -> None:
    schema = _finish_tool().args_schema.model_json_schema()
    subtask_status = schema["$defs"]["FinishSubtask"]["properties"]["status"]
    assert set(subtask_status["enum"]) == {"completed", "failed", "skipped"}


def test_finish_arguments_have_descriptions() -> None:
    schema = _finish_tool().args_schema.model_json_schema()
    for name, property_schema in schema.get("properties", {}).items():
        assert property_schema.get(
            "description"
        ), f"finish.{name} is missing description"
