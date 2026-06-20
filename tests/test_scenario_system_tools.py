from __future__ import annotations

import asyncio
import json
from typing import Any

from mobile_agent.gateways.system import SystemToolGateway
from mobile_agent.tools.scenario_system import create_scenario_system_tools


class _FakeSystemClient:
    def __init__(self, result: Any | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._result = result if result is not None else {"ok": True}

    async def send_request(self, message: str, data: Any) -> Any:
        self.calls.append((message, data))
        return self._result


class _FakeSystemGateway:
    def __init__(self, result: Any | None = None) -> None:
        self.client = _FakeSystemClient(result)
        self.device_ids: list[str | None] = []

    def get_default_client(self, device_id: str | None = None) -> _FakeSystemClient:
        self.device_ids.append(device_id)
        return self.client


def _fake_tools(result: Any | None = None) -> tuple[dict[str, Any], _FakeSystemGateway]:
    gateway = _FakeSystemGateway(result)
    tools = {tool.name: tool for tool in create_scenario_system_tools(gateway)}
    return tools, gateway


def _tool_by_name() -> dict[str, Any]:
    return {
        tool.name: tool
        for tool in create_scenario_system_tools(SystemToolGateway())
    }


def test_scenario_tools_exposed() -> None:
    tools = _tool_by_name()
    assert set(tools) == {
        "list_notifications",
        "search_files",
        "archive_file",
        "read_text_file",
    }


def test_list_notifications_omits_empty_data() -> None:
    tools, gateway = _fake_tools({"notifications": []})

    asyncio.run(tools["list_notifications"].ainvoke({}))

    assert gateway.client.calls == [("listNotifications", None)]


def test_list_notifications_passes_filters() -> None:
    tools, gateway = _fake_tools({"notifications": []})

    asyncio.run(
        tools["list_notifications"].ainvoke({"since": 1748500000000, "limit": 5})
    )

    assert gateway.client.calls == [
        ("listNotifications", {"since": 1748500000000, "limit": 5})
    ]


def test_search_files_serializes_payload() -> None:
    tools, gateway = _fake_tools({"files": []})

    asyncio.run(
        tools["search_files"].ainvoke(
            {"keywords": ["评审会", "纪要"], "roots": ["/sdcard/Download"], "limit": 5}
        )
    )

    assert gateway.client.calls == [
        (
            "searchFiles",
            {"keywords": ["评审会", "纪要"], "limit": 5, "roots": ["/sdcard/Download"]},
        )
    ]


def test_search_files_default_limit_without_roots() -> None:
    tools, gateway = _fake_tools({"files": []})

    asyncio.run(tools["search_files"].ainvoke({"keywords": ["纪要"]}))

    assert gateway.client.calls == [("searchFiles", {"keywords": ["纪要"], "limit": 20})]


def test_archive_file_rejects_invalid_mode() -> None:
    tools, gateway = _fake_tools()

    result = asyncio.run(
        tools["archive_file"].ainvoke(
            {"source": "/a/b.pdf", "target_dir": "/a/archive", "mode": "delete"}
        )
    )

    assert json.loads(result)["error"] == "invalid_mode"
    assert gateway.client.calls == []


def test_archive_file_serializes_payload() -> None:
    tools, gateway = _fake_tools({"archivedPath": "/a/archive/b.pdf"})

    asyncio.run(
        tools["archive_file"].ainvoke(
            {"source": "/a/b.pdf", "target_dir": "/a/archive", "mode": "move"}
        )
    )

    assert gateway.client.calls == [
        ("archiveFile", {"source": "/a/b.pdf", "targetDir": "/a/archive", "mode": "move"})
    ]


def test_read_text_file_serializes_payload() -> None:
    tools, gateway = _fake_tools({"content": "hi", "truncated": False})

    asyncio.run(
        tools["read_text_file"].ainvoke({"path": "/a/notes.txt", "max_bytes": 1024})
    )

    assert gateway.client.calls == [
        ("readTextFile", {"path": "/a/notes.txt", "maxBytes": 1024})
    ]


def test_scenario_tool_routes_to_device_id() -> None:
    tools, gateway = _fake_tools({"notifications": []})

    asyncio.run(tools["list_notifications"].ainvoke({"device_id": "device-1"}))

    assert gateway.device_ids == ["device-1"]


def test_all_scenario_tool_arguments_have_descriptions() -> None:
    for tool in _tool_by_name().values():
        schema = tool.args_schema.model_json_schema()
        for name, property_schema in schema.get("properties", {}).items():
            assert property_schema.get(
                "description"
            ), f"{tool.name}.{name} is missing description"
