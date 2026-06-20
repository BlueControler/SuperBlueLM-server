from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from ..gateways.system import SystemGatewayError, SystemToolGateway
from ..json_types import JsonValue, to_json_value
from ..progress import emit_task_progress
from .system import DeviceScopedArgs

DEFAULT_FILE_SEARCH_LIMIT = 20
MAX_FILE_SEARCH_LIMIT = 100
DEFAULT_READ_MAX_BYTES = 64 * 1024


class ListNotificationsArgs(DeviceScopedArgs):
    since: int | None = Field(
        default=None,
        description=(
            "Optional lower bound as Unix timestamp in milliseconds. Only return "
            "notifications posted at or after this time."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Optional maximum number of notifications to return.",
    )


class SearchFilesArgs(DeviceScopedArgs):
    keywords: list[str] = Field(
        min_length=1,
        description="Keywords matched against file name or path. All keywords are AND-matched.",
    )
    roots: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of root directories to search under. When omitted the "
            "client searches its default user-document roots."
        ),
    )
    limit: int = Field(
        default=DEFAULT_FILE_SEARCH_LIMIT,
        ge=1,
        le=MAX_FILE_SEARCH_LIMIT,
        description="Maximum number of matched files to return.",
    )


class ArchiveFileArgs(DeviceScopedArgs):
    source: str = Field(description="Absolute path of the source file to archive.")
    target_dir: str = Field(
        description="Absolute path of the destination directory. Created if missing."
    )
    mode: str = Field(
        default="copy",
        description="Archive mode: copy keeps the source, move removes it.",
    )


class ReadTextFileArgs(DeviceScopedArgs):
    path: str = Field(description="Absolute path of the text file to read.")
    max_bytes: int = Field(
        default=DEFAULT_READ_MAX_BYTES,
        ge=1,
        le=512 * 1024,
        description="Maximum number of bytes to read. Content is truncated beyond this.",
    )


def create_scenario_system_tools(gateway: SystemToolGateway) -> list[BaseTool]:
    async def send(
        tool_name: str,
        message: str,
        data: JsonValue,
        device_id: str | None,
        *,
        phase: str = "system_tool",
    ) -> str:
        emit_task_progress(
            label=tool_name,
            status="running",
            phase=phase,
            message=f"Running system tool: {tool_name}",
            tool_name=tool_name,
        )
        try:
            client = (
                gateway.get_default_client(device_id)
                if device_id is not None
                else gateway.get_default_client()
            )
            result = await client.send_request(message, data)
        except SystemGatewayError as exc:
            emit_task_progress(
                label=tool_name,
                status="failed",
                phase=phase,
                message=f"System tool failed: {tool_name}",
                tool_name=tool_name,
                error=str(exc),
            )
            return _dump({"error": str(exc)})
        emit_task_progress(
            label=tool_name,
            status="completed",
            phase=phase,
            message=f"Completed system tool: {tool_name}",
            tool_name=tool_name,
        )
        return _dump(result)

    @tool(
        "list_notifications",
        args_schema=ListNotificationsArgs,
        description=(
            "List recent unhandled system notifications through the /system tool client. "
            "Returns app name, title, text, and post time. Requires notification access "
            "on the device; returns an error when access is not granted."
        ),
    )
    async def list_notifications(
        since: int | None = None,
        limit: int | None = None,
        device_id: str | None = None,
    ) -> str:
        payload: dict[str, JsonValue] = {}
        if since is not None:
            payload["since"] = since
        if limit is not None:
            payload["limit"] = limit
        return await send(
            "list_notifications", "listNotifications", payload or None, device_id
        )

    @tool(
        "search_files",
        args_schema=SearchFilesArgs,
        description=(
            "Search local files on the device by keyword through the /system tool client. "
            "Returns matched file paths with name, size, modified time, and mime type."
        ),
    )
    async def search_files(
        keywords: list[str],
        roots: list[str] | None = None,
        limit: int = DEFAULT_FILE_SEARCH_LIMIT,
        device_id: str | None = None,
    ) -> str:
        payload: dict[str, JsonValue] = {"keywords": list(keywords), "limit": limit}
        if roots:
            payload["roots"] = list(roots)
        return await send("search_files", "searchFiles", payload, device_id)

    @tool(
        "archive_file",
        args_schema=ArchiveFileArgs,
        description=(
            "Archive a local file into a destination directory through the /system tool "
            "client. mode is copy or move. Returns the final archived path."
        ),
    )
    async def archive_file(
        source: str,
        target_dir: str,
        mode: str = "copy",
        device_id: str | None = None,
    ) -> str:
        if mode not in {"copy", "move"}:
            return _dump(
                {"error": "invalid_mode", "message": "mode must be copy or move."}
            )
        return await send(
            "archive_file",
            "archiveFile",
            {"source": source, "targetDir": target_dir, "mode": mode},
            device_id,
        )

    @tool(
        "read_text_file",
        args_schema=ReadTextFileArgs,
        description=(
            "Read a UTF-8 text file from the device through the /system tool client. "
            "Use this to read a reference document before summarizing it. Binary files "
            "are not supported."
        ),
    )
    async def read_text_file(
        path: str,
        max_bytes: int = DEFAULT_READ_MAX_BYTES,
        device_id: str | None = None,
    ) -> str:
        return await send(
            "read_text_file",
            "readTextFile",
            {"path": path, "maxBytes": max_bytes},
            device_id,
        )

    return [list_notifications, search_files, archive_file, read_text_file]


def _dump(data: JsonValue) -> str:
    return json.dumps(to_json_value(data), ensure_ascii=False)


__all__ = ["create_scenario_system_tools"]
