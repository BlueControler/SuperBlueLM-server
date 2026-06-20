from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from ..json_types import JsonObject, JsonValue, to_json_value
from ..progress import emit_task_progress

MEMORY_NAMESPACE_ROOT = "working_memory"
DEFAULT_SEARCH_LIMIT = 20


def _device_id_from_mapping(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    device_id = value.get("device_id") or value.get("deviceId")
    return device_id if isinstance(device_id, str) and device_id else None


class SaveMemoryArgs(BaseModel):
    key: str = Field(
        min_length=1,
        max_length=128,
        description=(
            "Stable identifier for this memory, for example default_hospital, "
            "family_contact_张三, or evening_review_preference."
        ),
    )
    value: str = Field(
        min_length=1,
        max_length=4000,
        description="The memory content to persist as plain text.",
    )
    category: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Optional grouping label such as hospital, contact, preference, or task. "
            "Used to filter when listing memories."
        ),
    )
    runtime: Annotated[Any, SkipJsonSchema()] = None


class GetMemoryArgs(BaseModel):
    key: str = Field(
        min_length=1,
        max_length=128,
        description="The memory key to read.",
    )
    runtime: Annotated[Any, SkipJsonSchema()] = None


class ListMemoryArgs(BaseModel):
    category: str | None = Field(
        default=None,
        max_length=64,
        description="Optional category filter. Omit to list all stored memories.",
    )
    runtime: Annotated[Any, SkipJsonSchema()] = None


class DeleteMemoryArgs(BaseModel):
    key: str = Field(
        min_length=1,
        max_length=128,
        description="The memory key to delete.",
    )
    runtime: Annotated[Any, SkipJsonSchema()] = None


def create_memory_tools() -> list[BaseTool]:
    @tool(
        "save_memory",
        args_schema=SaveMemoryArgs,
        description=(
            "Persist a long-term working memory across conversations, such as the "
            "user's default hospital, frequent contacts, or review preferences. "
            "Reuse the same key to overwrite an existing memory."
        ),
    )
    async def save_memory(
        key: str,
        value: str,
        runtime: ToolRuntime,
        category: str | None = None,
    ) -> str:
        store = _require_store(runtime)
        if store is None:
            return _no_store_result("save_memory")
        emit_task_progress(
            label="save_memory",
            status="running",
            phase="memory",
            message=f"Saving memory: {key}",
            tool_name="save_memory",
        )
        namespace = _namespace(runtime)
        record: JsonObject = {"key": key, "value": value}
        if category:
            record["category"] = category
        await store.aput(namespace, key, record)
        emit_task_progress(
            label="save_memory",
            status="completed",
            phase="memory",
            message=f"Saved memory: {key}",
            tool_name="save_memory",
        )
        return _dump({"ok": True, "key": key})

    @tool(
        "get_memory",
        args_schema=GetMemoryArgs,
        description="Read one long-term working memory by key. Returns null when it does not exist.",
    )
    async def get_memory(key: str, runtime: ToolRuntime) -> str:
        store = _require_store(runtime)
        if store is None:
            return _no_store_result("get_memory")
        namespace = _namespace(runtime)
        item = await store.aget(namespace, key)
        if item is None:
            return _dump({"key": key, "value": None})
        return _dump(_item_value(item))

    @tool(
        "list_memories",
        args_schema=ListMemoryArgs,
        description=(
            "List stored long-term working memories, optionally filtered by category. "
            "Use this to recall what the agent already knows about the user."
        ),
    )
    async def list_memories(runtime: ToolRuntime, category: str | None = None) -> str:
        store = _require_store(runtime)
        if store is None:
            return _no_store_result("list_memories")
        namespace = _namespace(runtime)
        items = await store.asearch(namespace, limit=DEFAULT_SEARCH_LIMIT)
        records = [_item_value(item) for item in items]
        if category:
            records = [
                record for record in records if record.get("category") == category
            ]
        payload: JsonObject = {"memories": list(records)}
        return _dump(payload)

    @tool(
        "delete_memory",
        args_schema=DeleteMemoryArgs,
        description="Delete one long-term working memory by key.",
    )
    async def delete_memory(key: str, runtime: ToolRuntime) -> str:
        store = _require_store(runtime)
        if store is None:
            return _no_store_result("delete_memory")
        namespace = _namespace(runtime)
        await store.adelete(namespace, key)
        return _dump({"ok": True, "key": key})

    return [save_memory, get_memory, list_memories, delete_memory]


def _require_store(runtime: ToolRuntime) -> Any:
    return getattr(runtime, "store", None)


def _namespace(runtime: ToolRuntime) -> tuple[str, ...]:
    state = getattr(runtime, "state", None)
    config = getattr(runtime, "config", None) or {}
    user = _user_identity(config)
    device_id = (
        _device_id_from_mapping(state)
        or _device_id_from_mapping(config.get("metadata"))
        or _device_id_from_mapping(config.get("configurable"))
        or "__default__"
    )
    return (MEMORY_NAMESPACE_ROOT, user, device_id)


def _user_identity(config: Mapping[str, Any]) -> str:
    for container_key in ("configurable", "metadata"):
        container = config.get(container_key)
        if isinstance(container, Mapping):
            identity = container.get("langgraph_auth_user_id") or container.get("owner")
            if isinstance(identity, str) and identity:
                return identity
    return "__shared__"


def _item_value(item: object) -> JsonObject:
    value = getattr(item, "value", None)
    if isinstance(value, Mapping):
        return {str(key): to_json_value(val) for key, val in value.items()}
    return {"value": to_json_value(value)}


def _no_store_result(tool_name: str) -> str:
    return _dump(
        {
            "error": "memory_store_unavailable",
            "message": (
                "Persistent store is not configured for this run; working memory "
                "is unavailable. Ask the user for the needed information instead."
            ),
            "tool": tool_name,
        }
    )


def _dump(data: JsonValue) -> str:
    return json.dumps(to_json_value(data), ensure_ascii=False)


__all__ = ["create_memory_tools"]
