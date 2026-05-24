from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mobile_agent.tools.external import (
    AmapHttpMcpClient,
    call_amap_mcp_tool,
    external_tools_status_payload,
)
import setup as setup


class FakeHttpCaller:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def __call__(self, url: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((url, tool_name, arguments))
        return {"ok": True, "tool": tool_name, "arguments": arguments}


class RaisingHttpCaller:
    async def __call__(self, url: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom https://mcp.amap.com/mcp?key=test+key")


class RaisingRawKeyHttpCaller:
    async def __call__(self, url: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("provider rejected API key test key")


def test_amap_http_client_uses_official_streamable_http_url(monkeypatch: Any) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")
    monkeypatch.delenv("AMAP_MCP_HTTP_URL", raising=False)
    fake = FakeHttpCaller()

    result = asyncio.run(
        AmapHttpMcpClient(http_caller=fake).call_tool("maps_weather", {"city": "北京"})
    )

    assert result == {"ok": True, "tool": "maps_weather", "arguments": {"city": "北京"}}
    assert fake.calls == [
        ("https://mcp.amap.com/mcp?key=test+key", "maps_weather", {"city": "北京"})
    ]


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("https://example.test/mcp", "https://example.test/mcp?key=test+key"),
        (
            "https://example.test/mcp?tenant=a",
            "https://example.test/mcp?tenant=a&key=test+key",
        ),
        ("https://example.test/mcp?", "https://example.test/mcp?key=test+key"),
        (
            "https://example.test/mcp?tenant=a&",
            "https://example.test/mcp?tenant=a&key=test+key",
        ),
    ],
)
def test_amap_http_url_override_appends_key_with_correct_separator(
    monkeypatch: Any, base_url: str, expected_url: str
) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")
    monkeypatch.setenv("AMAP_MCP_HTTP_URL", base_url)
    fake = FakeHttpCaller()

    asyncio.run(AmapHttpMcpClient(http_caller=fake).call_tool("maps_weather", {}))

    assert fake.calls == [(expected_url, "maps_weather", {})]


def test_amap_http_client_missing_key_returns_json_error(monkeypatch: Any) -> None:
    monkeypatch.delenv("AMAP_MAPS_API_KEY", raising=False)
    fake = FakeHttpCaller()

    result = asyncio.run(
        AmapHttpMcpClient(http_caller=fake).call_tool("maps_weather", {"city": "北京"})
    )

    assert isinstance(result, dict)
    assert result["error"] == "missing_env"
    assert result["env"] == "AMAP_MAPS_API_KEY"
    assert fake.calls == []


def test_amap_http_client_returns_redacted_json_error_when_call_fails(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")
    monkeypatch.delenv("AMAP_MCP_HTTP_URL", raising=False)

    result = asyncio.run(
        AmapHttpMcpClient(http_caller=RaisingHttpCaller()).call_tool("maps_weather", {})
    )

    assert isinstance(result, dict)
    assert result["error"] == "amap_mcp_call_failed"
    assert "test+key" not in result["message"]
    assert "test key" not in result["message"]
    assert "key=***" in result["message"]


def test_amap_http_client_redacts_raw_api_key_when_call_fails(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")

    result = asyncio.run(
        AmapHttpMcpClient(http_caller=RaisingRawKeyHttpCaller()).call_tool("maps_weather", {})
    )

    assert isinstance(result, dict)
    assert result["error"] == "amap_mcp_call_failed"
    assert "test key" not in result["message"]


def test_disallowed_amap_tool_is_rejected_before_http(monkeypatch: Any) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")
    fake = FakeHttpCaller()

    result = asyncio.run(
        call_amap_mcp_tool(AmapHttpMcpClient(http_caller=fake), "maps_secret_write", {})
    )

    assert isinstance(result, dict)
    assert result["error"] == "disallowed_mcp_tool"
    assert fake.calls == []


def test_external_status_reports_http_amap_not_npx(monkeypatch: Any) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")
    monkeypatch.delenv("AMAP_MCP_HTTP_URL", raising=False)

    payload = external_tools_status_payload()

    assert payload["amap_mcp_transport"] == "http"
    assert payload["amap_http_url"].startswith("https://mcp.amap.com/mcp?key=")
    assert "test+key" not in payload["amap_http_url"]
    assert "test key" not in payload["amap_http_url"]
    assert "key=***" in payload["amap_http_url"]
    assert payload["amap_maps_api_key"] is True
    assert "amap_mcp_command" not in payload


def test_setup_external_check_does_not_require_npx_for_amap_http(
    monkeypatch: Any, capsys: Any
) -> None:
    checked: list[str] = []

    def fake_which(command: str) -> str | None:
        checked.append(command)
        if command == "npx":
            return None
        return f"/fake/{command}"

    monkeypatch.setattr(setup.shutil, "which", fake_which)
    monkeypatch.setattr(
        setup.os, "getenv", lambda key: "set" if key == "AMAP_MAPS_API_KEY" else None
    )

    status = setup.check_external_tools()

    output = capsys.readouterr().out
    assert status == 0
    assert "npx" not in checked
    assert "npx:" not in output


def test_setup_external_check_fails_when_amap_key_missing(monkeypatch: Any, capsys: Any) -> None:
    checked: list[str] = []

    def fake_which(command: str) -> str | None:
        checked.append(command)
        if command == "npx":
            return None
        return f"/fake/{command}"

    monkeypatch.setattr(setup.shutil, "which", fake_which)
    monkeypatch.setattr(setup.os, "getenv", lambda key: None)

    status = setup.check_external_tools()

    output = capsys.readouterr().out
    assert status == 1
    assert "npx" not in checked
    assert "npx:" not in output
    assert "AMAP_MAPS_API_KEY: missing" in output
