from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain.tools import ToolRuntime

from mobile_agent.tools.memory import create_memory_tools


class _FakeItem:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value


class _FakeStore:
    def __init__(self) -> None:
        self.data: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.data[(namespace, key)] = value

    async def aget(self, namespace: tuple[str, ...], key: str, **kwargs: Any) -> Any:
        value = self.data.get((namespace, key))
        return _FakeItem(value) if value is not None else None

    async def asearch(self, namespace: tuple[str, ...], **kwargs: Any) -> list[_FakeItem]:
        return [
            _FakeItem(value)
            for (ns, _key), value in self.data.items()
            if ns == namespace
        ]

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        self.data.pop((namespace, key), None)


def _tools() -> dict[str, Any]:
    return {tool.name: tool for tool in create_memory_tools()}


def _runtime(
    store: Any,
    *,
    user: str | None = "user-1",
    device_id: str | None = "device-1",
) -> ToolRuntime:
    configurable: dict[str, Any] = {}
    if user is not None:
        configurable["langgraph_auth_user_id"] = user
    state = {"device_id": device_id} if device_id is not None else {}
    return ToolRuntime(
        state=state,
        context=None,
        config={"configurable": configurable},
        stream_writer=None,
        tool_call_id="test",
        store=store,
        tools=[],
        execution_info=None,
        server_info=None,
    )


def test_memory_tools_exposed() -> None:
    assert set(_tools()) == {
        "save_memory",
        "get_memory",
        "list_memories",
        "delete_memory",
    }


def test_save_and_get_round_trip() -> None:
    tools = _tools()
    store = _FakeStore()
    runtime = _runtime(store)

    save = asyncio.run(
        tools["save_memory"].ainvoke(
            {
                "key": "default_hospital",
                "value": "人民医院",
                "category": "hospital",
                "runtime": runtime,
            }
        )
    )
    assert json.loads(save) == {"ok": True, "key": "default_hospital"}

    got = json.loads(
        asyncio.run(
            tools["get_memory"].ainvoke(
                {"key": "default_hospital", "runtime": runtime}
            )
        )
    )
    assert got["value"] == "人民医院"
    assert got["category"] == "hospital"


def test_get_missing_returns_null_value() -> None:
    tools = _tools()
    runtime = _runtime(_FakeStore())

    got = json.loads(
        asyncio.run(tools["get_memory"].ainvoke({"key": "missing", "runtime": runtime}))
    )
    assert got == {"key": "missing", "value": None}


def test_list_filters_by_category() -> None:
    tools = _tools()
    store = _FakeStore()
    runtime = _runtime(store)

    asyncio.run(
        tools["save_memory"].ainvoke(
            {"key": "h", "value": "人民医院", "category": "hospital", "runtime": runtime}
        )
    )
    asyncio.run(
        tools["save_memory"].ainvoke(
            {"key": "c", "value": "张三", "category": "contact", "runtime": runtime}
        )
    )

    listed = json.loads(
        asyncio.run(
            tools["list_memories"].ainvoke(
                {"category": "hospital", "runtime": runtime}
            )
        )
    )
    assert [m["key"] for m in listed["memories"]] == ["h"]


def test_delete_removes_memory() -> None:
    tools = _tools()
    store = _FakeStore()
    runtime = _runtime(store)

    asyncio.run(
        tools["save_memory"].ainvoke(
            {"key": "k", "value": "v", "runtime": runtime}
        )
    )
    asyncio.run(tools["delete_memory"].ainvoke({"key": "k", "runtime": runtime}))

    got = json.loads(
        asyncio.run(tools["get_memory"].ainvoke({"key": "k", "runtime": runtime}))
    )
    assert got["value"] is None


def test_namespace_isolated_by_user_and_device() -> None:
    tools = _tools()
    store = _FakeStore()

    asyncio.run(
        tools["save_memory"].ainvoke(
            {
                "key": "default_hospital",
                "value": "甲医院",
                "runtime": _runtime(store, user="user-1", device_id="device-1"),
            }
        )
    )

    other_user = json.loads(
        asyncio.run(
            tools["get_memory"].ainvoke(
                {
                    "key": "default_hospital",
                    "runtime": _runtime(store, user="user-2", device_id="device-1"),
                }
            )
        )
    )
    assert other_user["value"] is None

    other_device = json.loads(
        asyncio.run(
            tools["get_memory"].ainvoke(
                {
                    "key": "default_hospital",
                    "runtime": _runtime(store, user="user-1", device_id="device-2"),
                }
            )
        )
    )
    assert other_device["value"] is None


def test_missing_store_returns_unavailable_error() -> None:
    tools = _tools()
    runtime = _runtime(None)

    result = json.loads(
        asyncio.run(
            tools["save_memory"].ainvoke(
                {"key": "k", "value": "v", "runtime": runtime}
            )
        )
    )
    assert result["error"] == "memory_store_unavailable"


def test_memory_arguments_have_descriptions() -> None:
    for tool in _tools().values():
        schema = tool.args_schema.model_json_schema()
        for name, property_schema in schema.get("properties", {}).items():
            assert property_schema.get(
                "description"
            ), f"{tool.name}.{name} is missing description"
