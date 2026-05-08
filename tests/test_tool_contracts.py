from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from entrypoints import deploy as project_deploy
from entrypoints import setup as project_setup
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


def test_local_model_prompt_keeps_tasks_simple():
    assert "简单任务" in prompt_assets.LOCAL_MODEL_SYSTEM_PROMPT
    assert "零次或一次工具调用" in prompt_assets.LOCAL_MODEL_SYSTEM_PROMPT
    assert "高风险" in prompt_assets.LOCAL_MODEL_SYSTEM_PROMPT
    assert "多步骤" in prompt_assets.LOCAL_MODEL_SYSTEM_PROMPT


def test_custom_deep_agent_uses_only_base_system_prompt():
    source = (PROJECT_ROOT / "mobile_agent" / "custom_deep_agent.py").read_text(
        encoding="utf-8"
    )

    assert "SYSTEM_PROMPT" in source
    assert "system_prompt=SYSTEM_PROMPT" not in source
    assert 'system_prompt=""' in source
    assert "TOOL_PROMPT" not in source
    assert "SYSTEM_TOOL_PROMPT" not in source


def test_custom_deep_agent_routes_cloud_and_local_prompts_separately():
    source = (PROJECT_ROOT / "mobile_agent" / "custom_deep_agent.py").read_text(
        encoding="utf-8"
    )

    assert "LOCAL_MODEL_SYSTEM_PROMPT" in source
    assert "RoutedSystemPromptMiddleware" in source
    assert "SystemMessage" in source
    assert "model_runtime.status().get(\"mode\") == \"local\"" in source


def test_unified_setup_exposes_llama_and_external_tool_actions():
    assert "llama:all" in project_setup.SETUP_ACTIONS
    assert "external:all" in project_setup.SETUP_ACTIONS
    assert "check" in project_setup.SETUP_ACTIONS
    assert "all" in project_setup.SETUP_ACTIONS

    assert project_setup.normalize_legacy_action("llama", "all") == "llama:all"
    assert project_setup.normalize_legacy_action("external", "check") == "external:check"


def test_external_setup_documents_reusable_login_state():
    assert "首次" in project_setup.EXTERNAL_AUTH_NOTICE
    assert "登录态" in project_setup.EXTERNAL_AUTH_NOTICE
    assert "后续" in project_setup.EXTERNAL_AUTH_NOTICE


def test_deploy_plan_profiles_are_explicit_and_ordered():
    core_plan = project_deploy.build_deploy_plan(profile="core", start=False)
    local_plan = project_deploy.build_deploy_plan(profile="local", start=True)
    full_plan = project_deploy.build_deploy_plan(profile="full", start=False)
    default_args = project_deploy.parse_args([])

    assert [step.name for step in core_plan] == [
        "check-python-version",
        "ensure-env-file",
        "install-python-dependencies",
        "check-unified-setup",
    ]
    assert [step.name for step in local_plan] == [
        "check-python-version",
        "ensure-env-file",
        "check-server-port",
        "install-python-dependencies",
        "setup-local-model",
        "check-unified-setup",
        "start-langgraph-server",
        "health-check-server",
    ]
    assert [step.name for step in full_plan] == [
        "check-python-version",
        "ensure-env-file",
        "install-python-dependencies",
        "setup-local-model",
        "setup-external-tools",
        "check-unified-setup",
    ]
    assert core_plan[-1].required is False
    assert local_plan[-2].background is True
    assert local_plan[-1].required is True
    assert default_args.profile == "full"
    assert default_args.start is True


def test_deploy_wrappers_run_full_start_by_default():
    powershell_script = (PROJECT_ROOT / "scripts" / "deploy.ps1").read_text(
        encoding="utf-8"
    )
    shell_script = (PROJECT_ROOT / "scripts" / "deploy.sh").read_text(encoding="utf-8")

    assert "python -m entrypoints.deploy" in powershell_script
    assert "python -m entrypoints.deploy" in shell_script


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
        "feishu_cli",
        "wecom_cli",
        "amap_mcp_tool",
        "weather_query",
        "external_tools_status",
    ]
    assert "read-only" in tools["feishu_cli_readonly"].description
    assert "full-access" in tools["feishu_cli"].description
    assert "write operations" in tools["wecom_cli"].description
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


@pytest.mark.anyio
async def test_full_access_feishu_cli_allows_write_actions():
    runner = FakeCommandRunner()
    tools = _tool_by_name(create_external_tools(runner, FakeAmapClient()))

    raw = await tools["feishu_cli"].ainvoke(
        {"argv": ["im", "+messages-send", "--chat-id", "oc_xxx", "--text", "hello"]}
    )

    result = json.loads(raw)
    assert result["ok"] is True
    assert runner.calls == [
        ("lark-cli", ["im", "+messages-send", "--chat-id", "oc_xxx", "--text", "hello"], 30.0)
    ]


@pytest.mark.anyio
async def test_full_access_wecom_cli_allows_unknown_domains_and_write_actions():
    runner = FakeCommandRunner()
    tools = _tool_by_name(create_external_tools(runner, FakeAmapClient()))

    raw = await tools["wecom_cli"].ainvoke(
        {"argv": ["admin", "delete_user", '{"userid":"zhangsan"}']}
    )

    result = json.loads(raw)
    assert result["ok"] is True
    assert runner.calls == [
        ("wecom-cli", ["admin", "delete_user", '{"userid":"zhangsan"}'], 30.0)
    ]


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
