from __future__ import annotations

from pathlib import Path

from mobile_agent import prompt_assets
from mobile_agent.phone_tools import create_phone_tools
from mobile_agent.system_tools import create_system_tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DummyGateway:
    def get_session(self):  # pragma: no cover - tests only inspect tool metadata
        raise AssertionError("tool metadata tests must not send phone commands")

    def get_default_client(self):  # pragma: no cover - tests only inspect tool metadata
        raise AssertionError("tool metadata tests must not send system commands")


def _tool_by_name(tools):
    return {tool.name: tool for tool in tools}


def test_prompt_assets_do_not_export_duplicate_tool_definitions():
    assert not hasattr(prompt_assets, "TOOL_DEFINITIONS")
    assert not hasattr(prompt_assets, "TOOL_PROMPT")
    assert not hasattr(prompt_assets, "SYSTEM_TOOL_PROMPT")

    assert "finish" not in prompt_assets.SYSTEM_PROMPT
    assert "keyevent" not in prompt_assets.SYSTEM_PROMPT
    assert "list_apps" not in prompt_assets.SYSTEM_PROMPT


def test_custom_deep_agent_uses_only_base_system_prompt():
    source = (PROJECT_ROOT / "mobile_agent" / "custom_deep_agent.py").read_text(
        encoding="utf-8"
    )

    assert "from .prompt_assets import SYSTEM_PROMPT" in source
    assert "system_prompt=SYSTEM_PROMPT" in source
    assert "TOOL_PROMPT" not in source
    assert "SYSTEM_TOOL_PROMPT" not in source


def test_phone_tools_contract_matches_android_protocol():
    tools = _tool_by_name(create_phone_tools(DummyGateway()))

    assert list(tools) == [
        "observe",
        "launch",
        "tap",
        "type",
        "swipe",
        "long_press",
        "double_tap",
        "back",
        "home",
        "keyevent",
        "wait",
        "interact",
        "take_over",
    ]
    assert "finish" not in tools

    keyevent_description = tools["keyevent"].description
    assert "3" in keyevent_description and "HOME" in keyevent_description
    assert "4" in keyevent_description and "BACK" in keyevent_description


def test_system_tools_contract_matches_system_protocol():
    tools = _tool_by_name(create_system_tools(DummyGateway()))

    assert list(tools) == [
        "list_apps",
        "create_event",
        "list_events",
        "update_event",
        "list_reminders",
        "update_reminders",
        "get_location",
    ]
    assert "sensor" not in tools
    assert "{packageName: appLabel}" in tools["list_apps"].description
