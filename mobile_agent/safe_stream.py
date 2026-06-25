"""Safe SSE filtering and proxying for mobile clients."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
import asyncio
import json
import os
import re
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from loguru import logger
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from .action_control import phone_action_registry
from .gateways.phone import DeviceGatewayError
from .runtime import phone_gateway
from .trace import (
    MAX_TRACE_DETAIL_TEXT_CHARS,
    MAX_TRACE_EVENT_BYTES,
    MAX_TRACE_SUMMARY_CHARS,
    MAX_TRACE_TITLE_CHARS,
    TRACE_DETAIL_KINDS,
)

# SafeSseFrame 会额外带上 type/chunk JSON 外壳，预留空间后仍严格小于 4 KiB。
MAX_SAFE_ASSISTANT_DELTA_BYTES = 3_800
_SENSITIVE_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*[^\s,;，。]+"
)
_IMAGE_DATA = re.compile(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE)
_VERIFICATION_CODE = re.compile(r"验证码\s*[:=：]?\s*[A-Za-z0-9]{0,16}")
_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


@dataclass(frozen=True)
class SseFrame:
    event: str
    data: str
    event_id: str | None = None


@dataclass(frozen=True)
class SafeSseFrame:
    event: str
    data: dict[str, Any]


class SafeStreamFilter:
    """Whitelist-only transformation from LangGraph events to mobile events."""

    def __init__(self) -> None:
        self._think = _ThinkStripper()
        self.terminal_status: str | None = None
        self._has_any_text = False
        self._has_trace_event = False

    @property
    def has_any_text(self) -> bool:
        return self._has_any_text

    @property
    def has_trace_event(self) -> bool:
        return self._has_trace_event

    def feed(self, upstream_event: SseFrame) -> list[SafeSseFrame]:
        event = upstream_event.event.lower()
        if "messages" in event:
            text, invocation_id = _assistant_text(upstream_event.data)
            if text is None:
                return []
            safe_text = _sanitize_assistant_text(self._think.feed(text))
            if safe_text:
                self._has_any_text = True
            return [_assistant_delta(safe_text, invocation_id)] if safe_text else []
        if "custom" not in event:
            return []
        payload = _json_object(upstream_event.data)
        if payload is None:
            return []
        payload_type = payload.get("type")
        if payload_type == "trace.v1":
            safe = _safe_trace_payload(payload)
            if safe is not None and safe.get("event") == "run.terminal":
                status = safe.get("status")
                self.terminal_status = status if isinstance(status, str) else None
            if safe is not None:
                self._has_trace_event = True
            return [SafeSseFrame("trace.v1", safe)] if safe is not None else []
        if payload_type == "task_progress":
            safe = _safe_progress_payload(payload)
            return [SafeSseFrame("task_progress", safe)] if safe is not None else []
        return []

    def finish(self) -> list[SafeSseFrame]:
        pending = _sanitize_assistant_text(self._think.finish())
        frames = [_assistant_delta(pending, None)] if pending else []
        frames.append(SafeSseFrame("stream.eof", {"type": "stream.eof"}))
        return frames


async def safe_run_stream(request: Request) -> StreamingResponse | JSONResponse:
    """Relays one existing LangGraph run while removing unsafe event payloads."""
    try:
        base_url = os.getenv("LANGGRAPH_INTERNAL_BASE_URL", "").strip().rstrip("/")
        if not base_url:
            return JSONResponse(
                {"error": "safe_stream_unconfigured", "message": "安全流服务尚未配置。"},
                status_code=503,
            )
        thread_id = request.path_params["thread_id"]
        upstream_url = f"{base_url}/threads/{thread_id}/runs/stream"
        run_id = _mobile_run_id(request)
        phone_action_registry.start_run(run_id, thread_id)
        body = _with_mobile_run_config(await request.body(), run_id=run_id, thread_id=thread_id)
        headers = _forward_headers(request)
    except Exception as exc:
        logger.exception(
            "mobile_safe_run_stream_setup_failed thread_id={}",
            request.path_params.get("thread_id", "unknown"),
        )
        return JSONResponse(
            {
                "error": "stream_setup_failed",
                "exception": type(exc).__name__,
                "detail": str(exc)[:500],
                "message": "请求处理失败，请检查服务配置或稍后重试。",
            },
            status_code=500,
        )

    async def generate() -> AsyncIterator[str]:
        filter_ = SafeStreamFilter()
        stream_seq = 0

        def encode(frame: SafeSseFrame) -> str:
            nonlocal stream_seq
            stream_seq += 1
            return _encode_frame(frame, stream_seq=stream_seq)

        timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
        execution_timeout_seconds = _positive_env(
            "MOBILE_AGENT_MAX_EXECUTION_SECONDS", 120
        )
        current_task = asyncio.current_task()
        if current_task is not None:
            loop = asyncio.get_running_loop()
            phone_action_registry.bind_stream_cancellation(
                run_id,
                lambda: loop.call_soon_threadsafe(current_task.cancel),
            )
        yield encode(_stream_started(run_id=run_id, thread_id=thread_id))
        try:
            async with asyncio.timeout(execution_timeout_seconds):
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST", upstream_url, content=body, headers=headers
                    ) as response:
                        _bind_backend_run_from_response(run_id, thread_id, response)
                        if response.status_code < 200 or response.status_code >= 300:
                            error_body = (await response.aread()).decode("utf-8", errors="ignore")[:500]
                            logger.warning(
                                "mobile_upstream_stream_error run_id={} thread_id={} upstream_status={} upstream_body={}",
                                run_id, thread_id, response.status_code, error_body,
                            )
                            # 区分 409（线程忙）和其他错误，统一发业务终态
                            if response.status_code == 409:
                                terminal_status = "thread_busy"
                                error_message = "当前会话有任务正在执行，请等待完成后再试。"
                                retryable = True
                            else:
                                terminal_status = "failed"
                                error_message = f"上游服务返回 {response.status_code}"
                                retryable = response.status_code in {408, 429} or response.status_code >= 500
                            yield encode(
                                SafeSseFrame(
                                    "stream.error",
                                    {
                                        "type": "stream.error",
                                        "message": error_message,
                                        "detail": error_body,
                                        "retryable": retryable,
                                    },
                                )
                            )
                            yield encode(
                                SafeSseFrame(
                                    "trace.v1",
                                    {
                                        "type": "trace.v1",
                                        "version": 1,
                                        "runId": run_id,
                                        "eventId": f"evt_{uuid4().hex[:12]}",
                                        "seq": 0,
                                        "event": "run.terminal",
                                        "status": terminal_status,
                                    },
                                )
                            )
                            phone_action_registry.mark_terminal(run_id)
                            phone_action_registry.mark_backend_status(run_id, terminal_status)
                            return
                        async for frame in _decode_sse(response.aiter_lines()):
                            for safe_frame in filter_.feed(frame):
                                yield encode(safe_frame)
                        if filter_.terminal_status is None:
                            # 无论有没有 trace/text，缺少 terminal 就是异常终态
                            reason = "upstream_ended_without_terminal"
                            await _cancel_backend_and_device_run(
                                run_id,
                                headers=headers,
                                reason=reason,
                            )
                            if filter_.has_any_text or filter_.has_trace_event:
                                terminal_status = "interrupted"
                                error_message = "任务连接中断，部分结果可能未完成。"
                            else:
                                terminal_status = "failed"
                                error_message = "任务未返回结束状态，已停止后续操作。"
                            yield encode(
                                SafeSseFrame(
                                    "stream.error",
                                    {
                                        "type": "stream.error",
                                        "message": error_message,
                                        "retryable": True,
                                    },
                                )
                            )
                            yield encode(
                                SafeSseFrame(
                                    "trace.v1",
                                    {
                                        "type": "trace.v1",
                                        "version": 1,
                                        "runId": run_id,
                                        "eventId": f"evt_{uuid4().hex[:12]}",
                                        "seq": 0,
                                        "event": "run.terminal",
                                        "status": terminal_status,
                                    },
                                )
                            )
                            phone_action_registry.mark_terminal(run_id)
                            phone_action_registry.mark_backend_status(run_id, terminal_status)
                            return
                        # ✅ 只有 filter_.terminal_status is not None 才走正常终态路径
                        # 内部 LangGraph 流关闭后，等待 500ms 让 trace terminal 等异步事件 flush 到 SSE 通道
                        await asyncio.sleep(0.5)
                        for safe_frame in filter_.finish():
                            yield encode(safe_frame)
                        logger.info(
                            "mobile_run_stream_finished run_id={} thread_id={} terminal_status={}",
                            run_id, thread_id, filter_.terminal_status,
                        )
                        phone_action_registry.mark_terminal(run_id)
                        phone_action_registry.mark_backend_status(
                            run_id,
                            filter_.terminal_status or "missing_terminal",
                        )
        except TimeoutError:
            logger_message = "任务执行超过服务端时限，已请求停止。"
            logger.warning("mobile_run_execution_timeout run_id={} thread_id={}", run_id, thread_id)
            await _cancel_backend_and_device_run(run_id, headers=headers, reason="execution_timeout")
            yield encode(
                SafeSseFrame(
                    "stream.error",
                    {
                        "type": "stream.error",
                        "message": logger_message,
                        "retryable": False,
                    },
                )
            )
        except asyncio.CancelledError:
            logger.info("mobile_run_stream_cancelled run_id={} thread_id={}", run_id, thread_id)
            await _cancel_backend_and_device_run(
                run_id,
                headers=headers,
                reason="stream_cancelled",
            )
            raise
        # 代理边界不允许把内部异常变成半截 SSE 响应；取消信号继承自
        # BaseException，不会被此处捕获，仍可正常终止请求。
        except Exception:
            logger.warning("mobile_run_stream_error run_id={} thread_id={}", run_id, thread_id)
            await _cancel_backend_and_device_run(
                run_id,
                headers=headers,
                reason="stream_error",
            )
            yield encode(
                SafeSseFrame(
                    "stream.error",
                    {
                        "type": "stream.error",
                        "message": "服务连接中断，请稍后重试。",
                        "retryable": True,
                    },
                )
            )
        finally:
            phone_action_registry.clear_stream_cancellation(run_id)
            # 兜底：无论任何路径退出（正常/异常/取消），确保 run 被标记为终态。
            # 如果 TraceEmitter 的 run.terminal 未到达（例如 TRACE_V1_EMIT_ENABLED
            # 曾为 false），这个兜底能防止 LangSmith run 永远悬挂。
            if filter_.terminal_status is None:
                logger.warning(
                    "mobile_run_stream_no_terminal_at_finally run_id={} thread_id={}",
                    run_id, thread_id,
                )
                phone_action_registry.mark_terminal(run_id)
                phone_action_registry.mark_backend_status(run_id, "stream_closed")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _mobile_run_id(request: Request) -> str:
    requested = request.headers.get("x-mobile-run-id", "").strip()
    if requested and _RUN_ID.fullmatch(requested):
        return requested
    return f"run_{uuid4().hex}"


def _with_mobile_run_config(raw: bytes, *, run_id: str, thread_id: str) -> bytes:
    """Inject only server-recognised config; malformed bodies remain upstream errors."""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if not isinstance(payload, dict):
        return raw
    config = payload.get("config")
    config = dict(config) if isinstance(config, Mapping) else {}
    configurable = config.get("configurable")
    configurable = dict(configurable) if isinstance(configurable, Mapping) else {}
    configurable.update({"mobile_run_id": run_id, "thread_id": thread_id})
    config["configurable"] = configurable
    # deep_agent (create_deep_agent) 已内置 recursion_limit=9,999，
    # 不再用 MOBILE_AGENT_MAX_RECURSION 覆盖它——移动端代理的超时安全网
    # 由 MOBILE_AGENT_MAX_EXECUTION_SECONDS 保证。
    payload["config"] = config
    # LangGraph's stream API defaults to `continue`.  That leaves a tool run
    # alive after the mobile SSE client has timed out or disconnected.
    payload["on_disconnect"] = "cancel"
    # No task may silently sit behind a stale run on the same LangGraph
    # thread. The client must receive a deterministic rejection instead.
    payload["multitask_strategy"] = "reject"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _positive_env(name: str, default: int) -> int:
    try:
        return max(int(os.getenv(name, str(default))), 1)
    except ValueError:
        return default


async def _cancel_device_run(
    run_id: str,
    *,
    reason: str,
    cancel_stream: bool,
) -> None:
    device_id = phone_action_registry.cancel_run(
        run_id,
        reason=reason,
        cancel_stream=cancel_stream,
    )
    if device_id is None:
        return
    try:
        await phone_gateway.cancel_run(run_id, device_id)
    except DeviceGatewayError:
        # Registry state remains authoritative; a reconnecting device will
        # reject the cancelled run from its persistent ledger.
        return


async def _cancel_backend_and_device_run(
    run_id: str,
    *,
    headers: Mapping[str, str],
    reason: str,
) -> str:
    """Fence local/device work and ask LangGraph to interrupt its real run."""
    device_id = phone_action_registry.cancel_run(
        run_id,
        reason=reason,
        cancel_stream=False,
    )
    backend_status = await cancel_upstream_run(run_id, headers)
    if device_id is not None:
        try:
            await phone_gateway.cancel_run(run_id, device_id)
        except DeviceGatewayError:
            # A device that reconnects later uses its persisted cancellation
            # ledger, so it cannot accept a later action from this run.
            pass
    return backend_status


async def _cancel_active_run_on_thread(
    thread_id: str,
    base_url: str,
    headers: Mapping[str, str],
) -> bool:
    """Try to cancel any active run on a thread when we don't have the run id."""
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # List runs on the thread, find the active one
            list_url = f"{base_url}/threads/{thread_id}/runs"
            response = await client.get(list_url, headers=dict(headers))
            if response.status_code != 200:
                return False
            runs = response.json()
            if not isinstance(runs, list):
                return False
            # Find the first running/pending run
            active = next(
                (r for r in runs if isinstance(r, dict) and r.get("status") in ("running", "pending")),
                None,
            )
            if active is None:
                return False
            backend_run_id = active.get("run_id") or active.get("id")
            if not backend_run_id:
                return False
            cancel_url = f"{base_url}/threads/{thread_id}/runs/{backend_run_id}/cancel"
            cancel_resp = await client.post(cancel_url, headers=dict(headers))
            return 200 <= cancel_resp.status_code < 300
    except Exception:
        return False


async def cancel_upstream_run(run_id: str, headers: Mapping[str, str]) -> str:
    """Cancel the native LangGraph run associated with a mobile run.

    The public mobile run id is intentionally not passed to the native API as
    its path id: LangGraph creates a distinct UUID for every submitted run.
    """
    run_info = phone_action_registry.backend_run_info(run_id)
    if run_info is None:
        logger.info("mobile_run_backend_cancel_skipped run_id={} reason=not_bound", run_id)
        # 未绑定 backend run id 时，尝试查询并取消 thread 上活跃的 backend run
        thread_id = phone_action_registry.thread_id_for(run_id)
        base_url = os.getenv("LANGGRAPH_INTERNAL_BASE_URL", "").strip().rstrip("/")
        if thread_id and base_url:
            cancelled = await _cancel_active_run_on_thread(thread_id, base_url, headers)
            if cancelled:
                phone_action_registry.mark_backend_status(run_id, "cancel_requested")
                return "cancel_requested"
        phone_action_registry.mark_backend_status(run_id, "unknown_not_bound")
        return "unknown_not_bound"
    thread_id, backend_run_id = run_info
    base_url = os.getenv("LANGGRAPH_INTERNAL_BASE_URL", "").strip().rstrip("/")
    if not base_url or not thread_id:
        logger.warning("mobile_run_backend_cancel_unavailable run_id={}", run_id)
        phone_action_registry.mark_backend_status(run_id, "cancel_unavailable")
        return "cancel_unavailable"
    url = f"{base_url}/threads/{thread_id}/runs/{backend_run_id}/cancel"
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=dict(headers))
    except httpx.HTTPError:
        logger.warning("mobile_run_backend_cancel_request_failed run_id={}", run_id)
        phone_action_registry.mark_backend_status(run_id, "cancel_request_failed")
        return "cancel_request_failed"
    if 200 <= response.status_code < 300:
        phone_action_registry.mark_backend_status(run_id, "cancel_requested")
        return "cancel_requested"
    phone_action_registry.mark_backend_status(run_id, "cancel_request_failed")
    logger.warning(
        "mobile_run_backend_cancel_rejected run_id={} status_code={}",
        run_id,
        response.status_code,
    )
    return "cancel_request_failed"


async def upstream_run_status(run_id: str, headers: Mapping[str, str]) -> str:
    """Return a safe, normalized native LangGraph status for mobile polling."""
    run_info = phone_action_registry.backend_run_info(run_id)
    if run_info is None:
        return str(phone_action_registry.snapshot(run_id).get("backendStatus", "missing"))
    thread_id, backend_run_id = run_info
    base_url = os.getenv("LANGGRAPH_INTERNAL_BASE_URL", "").strip().rstrip("/")
    if not base_url or not thread_id:
        return "unavailable"
    url = f"{base_url}/threads/{thread_id}/runs/{backend_run_id}"
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=dict(headers))
        payload = response.json() if response.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return "unavailable"
    raw_status = payload.get("status") if isinstance(payload, Mapping) else None
    normalized = _normalize_backend_status(raw_status)
    phone_action_registry.mark_backend_status(run_id, normalized)
    return normalized


def _bind_backend_run_from_response(
    run_id: str,
    thread_id: str,
    response: httpx.Response,
) -> None:
    headers = getattr(response, "headers", {})
    content_location = headers.get("content-location") if headers is not None else None
    if not isinstance(content_location, str):
        return
    path = urlsplit(content_location).path.rstrip("/")
    prefix = f"/threads/{thread_id}/runs/"
    if path.startswith(prefix) and (backend_run_id := path.removeprefix(prefix)):
        phone_action_registry.bind_backend_run(run_id, backend_run_id)


def _normalize_backend_status(status: object) -> str:
    mapping = {
        "success": "succeeded",
        "error": "failed",
        "interrupted": "cancelled",
        "timeout": "timeout",
        "pending": "pending",
        "running": "running",
    }
    return mapping.get(status, "unknown") if isinstance(status, str) else "unknown"


async def _decode_sse(lines: AsyncIterator[str]) -> AsyncIterator[SseFrame]:
    event = "message"
    event_id: str | None = None
    data_lines: list[str] = []
    async for line in lines:
        if not line:
            if data_lines:
                yield SseFrame(event=event, data="\n".join(data_lines), event_id=event_id)
            event = "message"
            event_id = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("id:"):
            event_id = line.removeprefix("id:").strip() or None
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())
    if data_lines:
        yield SseFrame(event=event, data="\n".join(data_lines), event_id=event_id)


def _assistant_text(raw: str) -> tuple[str | None, str | None]:
    payload = _json_value(raw)
    message: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    if isinstance(payload, list) and payload:
        message = payload[0] if isinstance(payload[0], Mapping) else None
        metadata = payload[1] if len(payload) > 1 and isinstance(payload[1], Mapping) else None
    elif isinstance(payload, Mapping):
        message = payload
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else None
    if message is None:
        return None, None
    kwargs = message.get("kwargs") if isinstance(message.get("kwargs"), Mapping) else {}
    message_type = str(message.get("type") or message.get("role") or kwargs.get("type") or "").lower()
    if message_type == "tool" or "toolmessage" in message_type:
        return None, None
    if message.get("tool_calls") or kwargs.get("tool_calls") or message.get("tool_call_chunks") or kwargs.get("tool_call_chunks"):
        return None, None
    content = message.get("content", kwargs.get("content"))
    text = _content_to_text(content)
    if not text:
        return None, None
    invocation_id = str(
        message.get("id")
        or (metadata or {}).get("checkpoint_ns")
        or (metadata or {}).get("langgraph_checkpoint_ns")
        or ""
    ).strip() or None
    return text, invocation_id


def _content_to_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(value, Mapping):
        text = value.get("text") or value.get("content")
        return text if isinstance(text, str) else ""
    return ""


def _safe_trace_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    event_name = payload.get("event")
    run_id = payload.get("runId")
    event_id = payload.get("eventId")
    sequence = payload.get("seq")
    if not all(isinstance(value, str) and value for value in (event_name, run_id, event_id)):
        return None
    if not isinstance(sequence, int) or sequence < 1:
        return None
    safe: dict[str, Any] = {
        "type": "trace.v1",
        "version": 1,
        "runId": _bounded(str(run_id), 128),
        "eventId": _bounded(str(event_id), 128),
        "seq": sequence,
        "event": _bounded(str(event_name), 64),
    }
    thread_id = payload.get("threadId")
    if isinstance(thread_id, str) and thread_id:
        safe["threadId"] = _bounded(thread_id, 128)
    if event_name == "run.terminal":
        status = payload.get("status")
        if status in {"succeeded", "failed", "cancelled", "waiting_for_user"}:
            safe["status"] = status
        else:
            return None
    elif event_name == "run.started":
        summary = payload.get("summary")
        if isinstance(summary, str) and summary:
            safe["summary"] = _safe_description(summary, MAX_TRACE_SUMMARY_CHARS)
    elif event_name == "step.upsert":
        step = payload.get("step")
        if not isinstance(step, Mapping) or step.get("visibleToUser") is not True:
            return None
        if not all(isinstance(step.get(key), str) and step.get(key) for key in ("stepId", "kind", "title", "summary", "status")):
            return None
        if step["status"] not in {
            "queued",
            "running",
            "succeeded",
            "failed",
            "cancelled",
            "waiting_for_user",
        }:
            return None
        safe_step: dict[str, Any] = {
            "stepId": _bounded(str(step["stepId"]), 128),
            "kind": _bounded(str(step["kind"]), 64),
            "title": _bounded(str(step["title"]), MAX_TRACE_TITLE_CHARS),
            "summary": _safe_description(
                str(step["summary"]), MAX_TRACE_SUMMARY_CHARS
            ),
            "status": step["status"],
            "visibleToUser": True,
        }
        parent_id = step.get("parentId")
        if isinstance(parent_id, str) and parent_id:
            safe_step["parentId"] = _bounded(parent_id, 128)
        safe["step"] = safe_step
    elif event_name == "step.detail.append":
        step_id = payload.get("stepId")
        detail = payload.get("detail")
        if not isinstance(step_id, str) or not step_id:
            return None
        if not isinstance(detail, Mapping):
            return None
        if detail.get("visibleToUser", True) is not True:
            return None
        if not all(
            isinstance(detail.get(key), str) and detail.get(key)
            for key in ("detailId", "kind", "title", "text")
        ):
            return None
        if detail["kind"] not in TRACE_DETAIL_KINDS:
            return None
        safe["stepId"] = _bounded(step_id, 128)
        safe["detail"] = {
            "detailId": _bounded(str(detail["detailId"]), 128),
            "kind": _bounded(str(detail["kind"]), 64),
            "title": _safe_description(str(detail["title"]), MAX_TRACE_TITLE_CHARS),
            "text": _safe_description(
                str(detail["text"]), MAX_TRACE_DETAIL_TEXT_CHARS
            ),
            "visibleToUser": True,
        }
    else:
        return None
    return _fit_payload(safe)


def _safe_progress_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    label = payload.get("label")
    status = payload.get("status")
    phase = payload.get("phase")
    if not all(isinstance(value, str) and value for value in (label, status, phase)):
        return None
    safe: dict[str, Any] = {
        "type": "task_progress",
        "label": _bounded(label, MAX_TRACE_TITLE_CHARS),
        "status": _bounded(status, 32),
        "phase": _bounded(phase, 64),
    }
    for key, limit in (("message", MAX_TRACE_SUMMARY_CHARS), ("progressKey", 128)):
        value = payload.get(key)
        if isinstance(value, str) and value:
            safe[key] = _safe_description(value, limit)
    for key in ("currentStep", "totalSteps"):
        value = payload.get(key)
        if isinstance(value, int) and value >= 0:
            safe[key] = value
    return _fit_payload(safe)


def _assistant_delta(text: str, invocation_id: str | None) -> SafeSseFrame:
    data: dict[str, Any] = {"type": "assistant.delta", "chunk": _limit_bytes(text, MAX_SAFE_ASSISTANT_DELTA_BYTES)}
    if invocation_id:
        data["invocationId"] = _bounded(invocation_id, 128)
    return SafeSseFrame("assistant.delta", data)


def _stream_started(*, run_id: str, thread_id: str) -> SafeSseFrame:
    return SafeSseFrame(
        "stream.started",
        {
            "type": "stream.started",
            "runId": _bounded(run_id, 128),
            "threadId": _bounded(thread_id, 128),
            "message": "已接收请求，正在连接 Agent。",
        },
    )


def _sanitize_assistant_text(text: str) -> str:
    if not text:
        return ""
    redacted = _SENSITIVE_VALUE.sub(lambda match: f"{match.group(1)}=***", text)
    redacted = _IMAGE_DATA.sub("[已隐藏图片数据]", redacted)
    redacted = _VERIFICATION_CODE.sub("[已隐藏敏感内容]", redacted)
    return _limit_bytes(redacted, MAX_SAFE_ASSISTANT_DELTA_BYTES)


def _safe_description(text: str, limit: int) -> str:
    without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return _bounded(_sanitize_assistant_text(without_think), limit)


def _fit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if _payload_size(payload) <= MAX_TRACE_EVENT_BYTES:
        return payload
    detail = payload.get("detail")
    if isinstance(detail, dict):
        detail["text"] = _bounded(str(detail.get("text", "")), 256)
        detail["title"] = _bounded(str(detail.get("title", "")), 32)
        if _payload_size(payload) <= MAX_TRACE_EVENT_BYTES:
            return payload
        detail["text"] = "处理详情已省略。"
        detail["title"] = "详情"
        if _payload_size(payload) <= MAX_TRACE_EVENT_BYTES:
            return payload
    step = payload.get("step")
    if isinstance(step, dict):
        step["summary"] = _bounded(str(step.get("summary", "")), 64)
    payload.pop("summary", None)
    payload.pop("message", None)
    payload.pop("error", None)
    if _payload_size(payload) <= MAX_TRACE_EVENT_BYTES:
        return payload
    if isinstance(step, dict):
        step["title"] = "执行过程更新"
        step["summary"] = "处理详情已省略。"
        step["kind"] = "generic"
    return payload


def _payload_size(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _json_value(raw: str) -> object | None:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _json_object(raw: str) -> Mapping[str, Any] | None:
    value = _json_value(raw)
    return value if isinstance(value, Mapping) else None


def _bounded(value: str, limit: int) -> str:
    text = value.strip()
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _limit_bytes(value: str, limit: int) -> str:
    if len(value.encode("utf-8")) <= limit:
        return value
    kept: list[str] = []
    total = 0
    for char in value:
        size = len(char.encode("utf-8"))
        if total + size + len("…".encode("utf-8")) > limit:
            break
        kept.append(char)
        total += size
    return "".join(kept) + "…"


def _forward_headers(request: Request) -> dict[str, str]:
    headers = {"content-type": request.headers.get("content-type", "application/json")}
    for name in ("x-api-key", "authorization", "x-device-id"):
        value = request.headers.get(name)
        if value:
            headers[name] = value
    return headers


def _encode_frame(frame: SafeSseFrame, *, stream_seq: int | None = None) -> str:
    data = dict(frame.data)
    if stream_seq is not None:
        data["streamSeq"] = stream_seq
    data = _fit_safe_frame_payload(data)
    return f"event: {frame.event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _fit_safe_frame_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fitted = _fit_payload(dict(payload))
    if _payload_size(fitted) <= MAX_TRACE_EVENT_BYTES:
        return fitted
    for key, limit in (
        ("chunk", MAX_SAFE_ASSISTANT_DELTA_BYTES - 256),
        ("detail", 512),
        ("message", MAX_TRACE_SUMMARY_CHARS),
        ("error", MAX_TRACE_SUMMARY_CHARS),
    ):
        value = fitted.get(key)
        if isinstance(value, str):
            fitted[key] = _limit_bytes(value, max(limit, 64))
    if _payload_size(fitted) <= MAX_TRACE_EVENT_BYTES:
        return fitted
    for key in ("detail", "error"):
        fitted.pop(key, None)
    if _payload_size(fitted) <= MAX_TRACE_EVENT_BYTES:
        return fitted
    if fitted.get("type") == "stream.error":
        fitted["message"] = "服务连接中断，请稍后重试。"
    elif isinstance(fitted.get("chunk"), str):
        fitted["chunk"] = _limit_bytes(str(fitted["chunk"]), 256)
    return fitted


class _ThinkStripper:
    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buffer += text
        output: list[str] = []
        while self._buffer:
            marker = self._CLOSE if self._inside else self._OPEN
            index = self._buffer.lower().find(marker)
            if index < 0:
                suffix_length = _partial_suffix_length(self._buffer, marker)
                stable = self._buffer[:-suffix_length] if suffix_length else self._buffer
                if not self._inside:
                    output.append(stable)
                self._buffer = self._buffer[-suffix_length:] if suffix_length else ""
                break
            if not self._inside:
                output.append(self._buffer[:index])
                self._inside = True
            else:
                self._inside = False
            self._buffer = self._buffer[index + len(marker):]
        return "".join(output)

    def finish(self) -> str:
        if self._inside:
            self._buffer = ""
            return ""
        pending = self._buffer
        self._buffer = ""
        return pending


def _partial_suffix_length(value: str, marker: str) -> int:
    lower_value = value.lower()
    lower_marker = marker.lower()
    for size in range(min(len(value), len(marker) - 1), 0, -1):
        if lower_marker.startswith(lower_value[-size:]):
            return size
    return 0


__all__ = ["SafeSseFrame", "SafeStreamFilter", "SseFrame", "safe_run_stream"]
