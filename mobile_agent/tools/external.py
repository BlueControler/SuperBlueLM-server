from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

from langchain_core.tools import BaseTool, tool

from ..json_types import JsonObject, JsonValue
from ..progress import emit_task_progress

MAX_OUTPUT_BYTES = 256 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0

SENSITIVE_KEY_PATTERN = re.compile(
    r"(token|secret|key|password|passwd|authorization|credential)",
    re.IGNORECASE,
)
SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)(bearer\s+)[a-z0-9._~+/=-]+|"
    r"([?&](?:token|secret|key|password|api_key|access_token)=)[^&\s]+"
)

CLI_DENY_ACTIONS = {
    "add",
    "append",
    "approve",
    "cancel",
    "complete",
    "create",
    "delete",
    "download",
    "edit",
    "forward",
    "import",
    "invite",
    "patch",
    "post",
    "put",
    "reject",
    "remove",
    "reply",
    "send",
    "transfer",
    "update",
    "upload",
    "write",
}

FEISHU_ALLOWED_DOMAINS = {
    "api",
    "base",
    "calendar",
    "contact",
    "docs",
    "drive",
    "im",
    "mail",
    "minutes",
    "sheets",
    "task",
    "vc",
    "wiki",
}

WECOM_ALLOWED_DOMAINS = {
    "contact",
    "doc",
    "meeting",
    "message",
    "msg",
    "schedule",
    "smartsheet",
    "todo",
}

ALLOWED_AMAP_TOOLS = {
    "maps_geo",
    "maps_regeocode",
    "maps_ip_location",
    "maps_weather",
    "maps_text_search",
    "maps_around_search",
    "maps_direction_walking",
    "maps_direction_driving",
    "maps_direction_transit_integrated",
    "maps_distance",
}


class CommandRunner(Protocol):
    async def run(
        self,
        command: str,
        args: Sequence[str],
        timeout: float,
    ) -> JsonObject: ...


class AmapClient(Protocol):
    async def call_tool(self, tool_name: str, arguments: Mapping[str, JsonValue]) -> JsonValue: ...


class AmapHttpCaller(Protocol):
    async def __call__(
        self,
        url: str,
        tool_name: str,
        arguments: dict[str, JsonValue],
    ) -> JsonValue: ...


@dataclass(frozen=True)
class CliSpec:
    name: str
    env_var: str
    default_command: str
    allowed_domains: frozenset[str]


FEISHU_CLI = CliSpec(
    name="feishu",
    env_var="LARK_CLI_BIN",
    default_command="lark-cli",
    allowed_domains=frozenset(FEISHU_ALLOWED_DOMAINS),
)
WECOM_CLI = CliSpec(
    name="wecom",
    env_var="WECOM_CLI_BIN",
    default_command="wecom-cli",
    allowed_domains=frozenset(WECOM_ALLOWED_DOMAINS),
)


class SafeCommandRunner:
    async def run(
        self,
        command: str,
        args: Sequence[str],
        timeout: float,
    ) -> JsonObject:
        executable = _resolve_command(command)
        if executable is None:
            return {
                "error": "command_not_found",
                "command": command,
                "message": f"Command {command!r} was not found in PATH.",
            }

        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return {"error": "command_start_failed", "command": command, "message": str(exc)}

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            return {
                "error": "command_timeout",
                "command": command,
                "timeout_seconds": timeout,
            }

        stdout_text = _decode_limited(stdout)
        stderr_text = _decode_limited(stderr)
        return {
            "command": command,
            "args": _to_jsonable(_redact_args(args)),
            "returncode": process.returncode,
            "stdout": _to_jsonable(_parse_or_text(_redact_text(stdout_text))),
            "stderr": _to_jsonable(_parse_or_text(_redact_text(stderr_text))),
            "ok": process.returncode == 0,
        }


class AmapHttpMcpClient:
    def __init__(self, http_caller: AmapHttpCaller | None = None) -> None:
        self._http_caller = http_caller or _call_streamable_http_mcp

    async def call_tool(self, tool_name: str, arguments: Mapping[str, JsonValue]) -> JsonValue:
        api_key = os.getenv("AMAP_MAPS_API_KEY")
        if not api_key:
            return {
                "error": "missing_env",
                "env": "AMAP_MAPS_API_KEY",
                "message": "AMAP_MAPS_API_KEY is required for AMap HTTP MCP tools.",
            }

        try:
            return await self._http_caller(_amap_http_url(api_key), tool_name, dict(arguments))
        except Exception as exc:
            return {
                "error": "amap_mcp_call_failed",
                "message": _redact_text(str(exc)).replace(api_key, "***"),
            }


def create_external_tools(
    command_runner: CommandRunner | None = None,
    amap_client: AmapClient | None = None,
) -> list[BaseTool]:
    runner = command_runner or SafeCommandRunner()
    amap = amap_client or AmapHttpMcpClient()

    async def run_external_tool(
        label: str,
        action: Callable[[], Awaitable[JsonValue]],
    ) -> str:
        emit_task_progress(
            label=label,
            status="running",
            phase="external_tool",
            message=f"Running external tool: {label}",
            tool_name=label,
        )
        try:
            result = await action()
        except Exception as exc:
            emit_task_progress(
                label=label,
                status="failed",
                phase="external_tool",
                message=f"External tool failed: {label}",
                tool_name=label,
                error=str(exc),
            )
            raise
        emit_task_progress(
            label=label,
            status="completed",
            phase="external_tool",
            message=f"Completed external tool: {label}",
            tool_name=label,
        )
        return _dump(result)

    @tool(
        "feishu_cli_readonly",
        description=(
            "Run a read-only Feishu/Lark CLI command. argv must be a list of argument "
            "items after lark-cli. Write actions such as send/create/update/delete are rejected."
        ),
    )
    async def feishu_cli_readonly(argv: list[str]) -> str:
        return await run_external_tool(
            "feishu_cli_readonly",
            lambda: _run_readonly_cli(FEISHU_CLI, argv, runner),
        )

    @tool(
        "wecom_cli_readonly",
        description=(
            "Run a read-only WeCom CLI command. argv must be a list of argument items "
            "after wecom-cli. Write actions such as send/create/update/delete are rejected."
        ),
    )
    async def wecom_cli_readonly(argv: list[str]) -> str:
        return await run_external_tool(
            "wecom_cli_readonly",
            lambda: _run_readonly_cli(WECOM_CLI, argv, runner),
        )

    @tool(
        "feishu_cli",
        description=(
            "Run a full-access Feishu/Lark CLI command. argv must be a list of argument "
            "items after lark-cli. This tool allows write operations supported by the CLI."
        ),
    )
    async def feishu_cli(argv: list[str]) -> str:
        return await run_external_tool(
            "feishu_cli",
            lambda: _run_cli(FEISHU_CLI, argv, runner),
        )

    @tool(
        "wecom_cli",
        description=(
            "Run a full-access WeCom CLI command. argv must be a list of argument items "
            "after wecom-cli. This tool allows write operations supported by the CLI."
        ),
    )
    async def wecom_cli(argv: list[str]) -> str:
        return await run_external_tool(
            "wecom_cli",
            lambda: _run_cli(WECOM_CLI, argv, runner),
        )

    @tool(
        "amap_mcp_tool",
        description=(
            "Call a whitelisted AMap MCP tool. Allowed tools include geocode, "
            "reverse geocode, IP location, weather, place search, route, and distance."
        ),
    )
    async def amap_mcp_tool(tool_name: str, arguments: JsonObject) -> str:
        return await run_external_tool(
            "amap_mcp_tool",
            lambda: call_amap_mcp_tool(amap, tool_name, arguments),
        )

    @tool(
        "weather_query",
        description="Query weather through AMap MCP maps_weather. city can be a city name or adcode.",
    )
    async def weather_query(city: str) -> str:
        return await run_external_tool(
            "weather_query",
            lambda: query_weather(amap, city),
        )

    @tool(
        "external_tools_status",
        description="Check local availability of Node, npm, Feishu CLI, WeCom CLI, and AMap HTTP MCP key.",
    )
    async def external_tools_status() -> str:
        async def status_payload() -> JsonObject:
            return external_tools_status_payload()

        return await run_external_tool("external_tools_status", status_payload)

    return [
        feishu_cli_readonly,
        wecom_cli_readonly,
        feishu_cli,
        wecom_cli,
        amap_mcp_tool,
        weather_query,
        external_tools_status,
    ]


async def _run_readonly_cli(
    spec: CliSpec,
    args: Sequence[str],
    runner: CommandRunner,
) -> JsonObject:
    validation_error = validate_readonly_cli_args(args, spec.allowed_domains)
    if validation_error is not None:
        return validation_error

    command = os.getenv(spec.env_var, spec.default_command)
    timeout = _tool_timeout()
    return await runner.run(command=command, args=list(args), timeout=timeout)


async def _run_cli(
    spec: CliSpec,
    args: Sequence[str],
    runner: CommandRunner,
) -> JsonObject:
    validation_error = validate_cli_args(args)
    if validation_error is not None:
        return validation_error

    command = os.getenv(spec.env_var, spec.default_command)
    timeout = _tool_timeout()
    return await runner.run(command=command, args=list(args), timeout=timeout)


def validate_cli_args(args: Sequence[str]) -> JsonObject | None:
    if not isinstance(args, list):
        return {"error": "invalid_args", "message": "argv must be a list of strings."}
    if not args:
        return {"error": "invalid_args", "message": "argv cannot be empty."}
    if not all(isinstance(item, str) for item in args):
        return {"error": "invalid_args", "message": "every argv item must be a string."}
    if any("\x00" in item or "\n" in item or "\r" in item for item in args):
        return {"error": "invalid_args", "message": "argv must not contain control separators."}
    return None


def validate_readonly_cli_args(
    args: Sequence[str],
    allowed_domains: frozenset[str],
) -> JsonObject | None:
    validation_error = validate_cli_args(args)
    if validation_error is not None:
        return validation_error

    domain = _first_non_option(args)
    if domain is not None and domain not in allowed_domains:
        return {
            "error": "disallowed_domain",
            "domain": domain,
            "allowed_domains": _to_jsonable(sorted(allowed_domains)),
        }

    for arg in args:
        action = _dangerous_action(arg)
        if action is not None:
            return {
                "error": "disallowed_write_action",
                "action": action,
                "message": "Only read-only CLI operations are allowed.",
            }

    return None


async def call_amap_mcp_tool(
    amap_client: AmapClient,
    tool_name: str,
    arguments: Mapping[str, JsonValue],
) -> JsonValue:
    if tool_name not in ALLOWED_AMAP_TOOLS:
        return {
            "error": "disallowed_mcp_tool",
            "tool_name": tool_name,
            "allowed_tools": _to_jsonable(sorted(ALLOWED_AMAP_TOOLS)),
        }
    return await amap_client.call_tool(tool_name, arguments)


async def query_weather(amap_client: AmapClient, city: str) -> JsonValue:
    return await call_amap_mcp_tool(amap_client, "maps_weather", {"city": city})


def external_tools_status_payload() -> JsonObject:
    lark_bin = os.getenv("LARK_CLI_BIN", "lark-cli")
    wecom_bin = os.getenv("WECOM_CLI_BIN", "wecom-cli")
    api_key = os.getenv("AMAP_MAPS_API_KEY")
    return {
        "node": _command_status("node"),
        "npm": _command_status("npm"),
        "feishu_cli": _command_status(lark_bin),
        "wecom_cli": _command_status(wecom_bin),
        "amap_mcp_transport": "http",
        "amap_http_url": _redact_text(_amap_http_url(api_key or "***")),
        "amap_maps_api_key": bool(api_key),
    }


def _command_status(command: str) -> JsonObject:
    resolved = _resolve_command(command)
    return {"command": command, "available": resolved is not None, "path": resolved}


def _resolve_command(command: str) -> str | None:
    if os.path.isabs(command) and os.path.exists(command):
        return command
    return shutil.which(command)


def _first_non_option(args: Sequence[str]) -> str | None:
    for arg in args:
        if not arg.startswith("-"):
            return arg.lower()
    return None


def _dangerous_action(arg: str) -> str | None:
    normalized = arg.lower().strip().lstrip("+")
    pieces = set(re.split(r"[^a-z0-9]+", normalized))
    if normalized in CLI_DENY_ACTIONS:
        return normalized
    for action in CLI_DENY_ACTIONS:
        if action in pieces or normalized.startswith(f"{action}_"):
            return action
    return None


def _tool_timeout() -> float:
    raw = os.getenv("EXTERNAL_TOOL_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _amap_http_url(api_key: str) -> str:
    base_url = os.getenv("AMAP_MCP_HTTP_URL", "https://mcp.amap.com/mcp")
    separator = "" if base_url.endswith(("?", "&")) else "&" if "?" in base_url else "?"
    return f"{base_url}{separator}{urlencode({'key': api_key})}"


async def _call_streamable_http_mcp(
    url: str,
    tool_name: str,
    arguments: dict[str, JsonValue],
) -> JsonValue:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:
        return {
            "error": "missing_dependency",
            "dependency": "mcp",
            "message": str(exc),
        }

    async with streamablehttp_client(url) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return _to_jsonable(result)


def _decode_limited(value: bytes) -> str:
    truncated = value[:MAX_OUTPUT_BYTES]
    text = truncated.decode("utf-8", errors="replace")
    if len(value) > MAX_OUTPUT_BYTES:
        return f"{text}\n...[truncated at {MAX_OUTPUT_BYTES} bytes]"
    return text


def _parse_or_text(text: str) -> JsonValue:
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        return _to_jsonable(json.loads(stripped))
    except json.JSONDecodeError:
        return stripped


def _redact_text(text: str) -> str:
    return SENSITIVE_VALUE_PATTERN.sub(lambda match: f"{match.group(1) or match.group(2)}***", text)


def _redact_args(args: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for arg in args:
        if redact_next:
            redacted.append("***")
            redact_next = False
            continue

        key, separator, value = arg.partition("=")
        if SENSITIVE_KEY_PATTERN.search(key):
            if separator:
                redacted.append(f"{key}=***")
            else:
                redacted.append(arg)
                redact_next = True
            continue

        if separator:
            redacted.append(f"{key}={_redact_text(value)}")
        else:
            redacted.append(_redact_text(arg))
    return redacted


def _to_jsonable(value: object) -> JsonValue:
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): ("***" if SENSITIVE_KEY_PATTERN.search(str(key)) else _to_jsonable(item))
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _dump(data: JsonValue) -> str:
    return json.dumps(_to_jsonable(data), ensure_ascii=False)
