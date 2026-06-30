from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mobile_agent.tools import external
from mobile_agent.tools.external import (
    AmapHttpMcpClient,
    SafeCommandRunner,
    call_amap_mcp_tool,
    create_external_tools,
    external_tools_status_payload,
    query_weather,
)
import scripts.setup as setup


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


class FakeAmapClient:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((tool_name, arguments))
        return self.result


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


@pytest.mark.parametrize(
    ("city", "expected"),
    [
        ("120000", "120000"),
        ("天津", "120000"),
        ("天津市", "120000"),
        ("北京", "110000"),
        ("北京市", "110000"),
        ("上海", "310000"),
        ("上海市", "310000"),
    ],
)
def test_resolve_amap_city_adcode_handles_adcodes_and_municipalities(
    monkeypatch: Any, city: str, expected: str
) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")

    result = asyncio.run(external.resolve_amap_city_adcode(city))

    assert result == expected


def test_resolve_amap_city_adcode_uses_district_api_for_district_name(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")
    calls: list[tuple[str, dict[str, str]]] = []

    async def fake_get(url: str, params: dict[str, str]) -> dict[str, Any]:
        calls.append((url, params))
        return {"status": "1", "districts": [{"name": "南开区", "adcode": "120104"}]}

    monkeypatch.setattr(external, "_amap_rest_get_json", fake_get, raising=False)

    result = asyncio.run(external.resolve_amap_city_adcode("天津市南开区"))

    assert result == "120104"
    assert calls[0][0].endswith("/v3/config/district")
    assert calls[0][1]["keywords"] == "天津市南开区"


def test_resolve_amap_city_adcode_falls_back_to_geocoding_for_address(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")
    calls: list[str] = []

    async def fake_get(url: str, params: dict[str, str]) -> dict[str, Any]:
        calls.append(url)
        if url.endswith("/v3/config/district"):
            return {"status": "1", "districts": []}
        return {"status": "1", "geocodes": [{"adcode": "120104"}]}

    monkeypatch.setattr(external, "_amap_rest_get_json", fake_get, raising=False)

    result = asyncio.run(external.resolve_amap_city_adcode("南开大学"))

    assert result == "120104"
    assert calls[1].endswith("/v3/geocode/geo")


def test_resolve_amap_city_adcode_falls_back_to_poi_search(monkeypatch: Any) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")
    calls: list[str] = []

    async def fake_get(url: str, params: dict[str, str]) -> dict[str, Any]:
        calls.append(url)
        if url.endswith("/v3/config/district"):
            return {"status": "1", "districts": []}
        if url.endswith("/v3/geocode/geo"):
            return {"status": "1", "geocodes": []}
        return {"status": "1", "pois": [{"name": "南开大学", "adcode": "120104"}]}

    monkeypatch.setattr(external, "_amap_rest_get_json", fake_get, raising=False)

    result = asyncio.run(external.resolve_amap_city_adcode("南开大学"))

    assert result == "120104"
    assert calls[2].endswith("/v3/place/text")


def test_query_weather_resolves_city_before_calling_mcp(monkeypatch: Any) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")
    amap = FakeAmapClient(
        {
            "content": [
                {"type": "text", "text": '{"city":"杭州市","forecasts":[{"date":"2026-06-11"}]}'},
            ]
        }
    )

    async def fake_get(url: str, params: dict[str, str]) -> dict[str, Any]:
        return {"status": "1", "districts": [{"adcode": "330100"}]}

    monkeypatch.setattr(external, "_amap_rest_get_json", fake_get, raising=False)

    result = asyncio.run(query_weather(amap, "杭州"))

    assert result == amap.result
    assert amap.calls == [("maps_weather", {"city": "330100"})]


def test_query_weather_falls_back_to_rest_when_mcp_weather_is_empty(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")
    amap = FakeAmapClient({"content": [{"type": "text", "text": '{"city":null,"forecasts":null}'}]})
    rest_weather = {
        "status": "1",
        "forecasts": [{"city": "天津市", "adcode": "120000", "casts": [{}]}],
    }

    async def fake_get(url: str, params: dict[str, str]) -> dict[str, Any]:
        assert url.endswith("/v3/weather/weatherInfo")
        assert params["city"] == "120000"
        assert params["extensions"] == "all"
        return rest_weather

    monkeypatch.setattr(external, "_amap_rest_get_json", fake_get, raising=False)

    result = asyncio.run(query_weather(amap, "120000"))

    assert result == rest_weather


def test_query_weather_returns_structured_error_when_mcp_and_rest_fail(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test key")
    amap = FakeAmapClient({"error": "amap_mcp_call_failed", "message": "task group failed"})

    async def fake_get(url: str, params: dict[str, str]) -> dict[str, Any]:
        return {"status": "0", "info": "INVALID_USER_KEY"}

    monkeypatch.setattr(external, "_amap_rest_get_json", fake_get, raising=False)

    result = asyncio.run(query_weather(amap, "120000"))

    assert result["error"] == "weather_query_failed"
    assert result["city_input"] == "120000"
    assert result["resolved_adcode"] == "120000"
    assert "INVALID_USER_KEY" in result["message"]
    assert "test key" not in result["message"]


def test_amap_rest_get_retries_once_after_transport_error(monkeypatch: Any) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"status": "1"}

    class FakeAsyncClient:
        calls = 0

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def get(self, url: str, params: dict[str, str]) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                raise external.httpx.ReadTimeout("")
            return FakeResponse()

    client = FakeAsyncClient()
    monkeypatch.setattr(external.httpx, "AsyncClient", lambda **kwargs: client)

    result = asyncio.run(external._amap_rest_get_json("https://example.test", {"key": "test key"}))

    assert result == {"status": "1"}
    assert client.calls == 2


def test_safe_command_runner_resolves_executable_in_thread(monkeypatch: Any) -> None:
    calls: list[tuple[Any, tuple[Any, ...]]] = []

    async def fake_to_thread(function: Any, *args: Any) -> Any:
        calls.append((function, args))
        return None

    monkeypatch.setattr(external.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(SafeCommandRunner().run("missing-command", [], 1))

    assert result["error"] == "command_not_found"
    assert calls == [(external._resolve_command, ("missing-command",))]


def test_external_status_resolves_commands_in_thread(monkeypatch: Any) -> None:
    calls: list[tuple[Any, tuple[Any, ...]]] = []

    async def fake_to_thread(function: Any, *args: Any) -> Any:
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(external.asyncio, "to_thread", fake_to_thread)
    tools = {tool.name: tool for tool in create_external_tools()}

    asyncio.run(tools["external_tools_status"].ainvoke({}))

    assert calls == [(external_tools_status_payload, ())]
