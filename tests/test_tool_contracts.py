from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mobile_agent.external_tools import (
    ALLOWED_AMAP_TOOLS,
    SafeCommandRunner,
    _amap_mcp_env,
    call_amap_mcp_tool,
    create_external_tools,
    query_weather,
    validate_readonly_cli_args,
)
from mobile_agent import prompt_assets
from mobile_agent.phone_tools import create_phone_tools
from mobile_agent.system_tools import create_system_tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def anyio_backend():
    return "asyncio"


class DummyGateway:
    def get_session(self):  # pragma: no cover - tests only inspect tool metadata
        raise AssertionError("tool metadata tests must not send phone commands")

    def get_default_client(self):  # pragma: no cover - tests only inspect tool metadata
        raise AssertionError("tool metadata tests must not send system commands")


class FakeCommandRunner:
    def __init__(self) -> None:
        self.calls = []

    async def run(self, command, args, timeout):
        self.calls.append((command, args, timeout))
        return {"ok": True, "command": command, "args": list(args)}


class FakeAmapClient:
    def __init__(self) -> None:
        self.calls = []

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return {"ok": True, "tool_name": tool_name, "arguments": dict(arguments)}


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


def test_external_tools_contract_contains_business_tools():
    tools = _tool_by_name(create_external_tools(FakeCommandRunner(), FakeAmapClient()))

    assert list(tools) == [
        "feishu_cli_readonly",
        "wecom_cli_readonly",
        "amap_mcp_tool",
        "weather_query",
        "external_tools_status",
    ]
    assert "read-only" in tools["feishu_cli_readonly"].description
    assert "maps_weather" in tools["weather_query"].description


def test_readonly_cli_validation_rejects_write_actions():
    error = validate_readonly_cli_args(
        ["im", "+messages-send", "--chat-id", "oc_xxx"],
        frozenset({"im"}),
    )

    assert error is not None
    assert error["error"] == "disallowed_write_action"


def test_readonly_cli_validation_accepts_known_read_domain():
    error = validate_readonly_cli_args(["contact", "get_userlist", "{}"], frozenset({"contact"}))

    assert error is None


def test_readonly_cli_validation_rejects_unknown_domain():
    error = validate_readonly_cli_args(["admin", "list"], frozenset({"contact"}))

    assert error is not None
    assert error["error"] == "disallowed_domain"


def test_amap_tool_whitelist_rejects_unknown_tool():
    assert "maps_weather" in ALLOWED_AMAP_TOOLS


def test_amap_mcp_env_does_not_forward_unrelated_secrets(monkeypatch):
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "old-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("PATH", "node-bin")

    env = _amap_mcp_env("amap-secret")

    assert env["AMAP_MAPS_API_KEY"] == "amap-secret"
    assert env["PATH"] == "node-bin"
    assert "OPENAI_API_KEY" not in env


@pytest.mark.anyio
async def test_weather_query_calls_amap_weather_tool():
    client = FakeAmapClient()

    result = await query_weather(client, "北京")

    assert result["ok"] is True
    assert client.calls == [("maps_weather", {"city": "北京"})]


@pytest.mark.anyio
async def test_amap_mcp_tool_rejects_non_whitelisted_tool():
    result = await call_amap_mcp_tool(FakeAmapClient(), "maps_admin_write", {})

    assert result["error"] == "disallowed_mcp_tool"


@pytest.mark.anyio
async def test_safe_command_runner_uses_exec_and_redacts_sensitive_output(monkeypatch):
    calls = []

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b'{"access_token":"secret-value","ok":true}', b""

    async def fake_create_subprocess_exec(executable, *args, stdout, stderr):
        calls.append((executable, args, stdout, stderr))
        return FakeProcess()

    monkeypatch.setattr("mobile_agent.external_tools._resolve_command", lambda command: "resolved-cli")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = await SafeCommandRunner().run(
        "lark-cli",
        ["contact", "get_userlist", "--token", "secret-value"],
        timeout=3,
    )

    assert calls == [("resolved-cli", ("contact", "get_userlist", "--token", "secret-value"), -1, -1)]
    assert result["args"] == ["contact", "get_userlist", "--token", "***"]
    assert result["stdout"] == {"access_token": "***", "ok": True}


@pytest.mark.anyio
async def test_safe_command_runner_returns_structured_error_for_missing_command():
    result = await SafeCommandRunner().run(
        "definitely-not-installed-superbluelm-tool",
        ["contact", "list"],
        timeout=1,
    )

    assert result["error"] == "command_not_found"
