"""Safe SSE filtering and proxying for mobile clients."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
import asyncio
import json
import os
import re
import time
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
    r"(?i)\b(api[_-]?key|app[_-]?key|appkey|x-api-key|token|password|secret|authorization|cookie)\s*[:=]\s*[^\s,;，。]+"
)
_SENSITIVE_JSON_VALUE = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|app[_-]?key|appkey|x-api-key|token|password|secret|authorization|cookie)[\"']?\s*:\s*[\"'])[^\"'\r\n]+([\"'])"
)
_AUTHORIZATION_VALUE = re.compile(
    r"(?i)\bauthorization\s*[:=]\s*(?:[A-Za-z]+\s+)?[^\s,;，。\r\n]+"
)
_COOKIE_VALUE = re.compile(r"(?i)\bcookie\s*[:=]\s*[^\r\n]+")
_IMAGE_DATA = re.compile(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE)
_LONG_BASE64 = re.compile(r"\b[A-Za-z0-9+/]{120,}={0,2}\b")
_VERIFICATION_CODE = re.compile(r"验证码\s*[:=：]?\s*[A-Za-z0-9]{0,16}")
_TRACEBACK_TEXT = re.compile(r"(?is)Traceback\s*\(most recent call last\):.*")
_UI_TREE_TEXT = re.compile(r"(?is)<hierarchy\b.*?</hierarchy>|<node\b[^>]*(?:/>|>.*?</node>)")
_RAW_TOOL_TEXT = re.compile(
    r"(?is)\b(?:raw\s+tool\s+(?:args?|result)|raw[_-]?(?:args?|result)|tool[_\s-]+(?:args?|result))\s*[:=]\s*(?:\{.*?\}|\[.*?\]|[^\r\n]+)"
)
_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_PUBLIC_TEXT_BLOCK_TYPES = frozenset({"text", "output_text", "input_text"})
_REASONING_BLOCK_TOKENS = ("reasoning", "thinking", "thought")
_COMPLETED_WORK_STATUSES = frozenset({"succeeded", "completed", "done"})
_INCOMPLETE_WORK_STATUSES = frozenset(
    {"queued", "pending", "running", "waiting_for_user", "waiting_confirmation"}
)
_FAILED_WORK_STATUSES = frozenset({"failed", "cancelled", "taken_over", "error", "timeout"})
_THINK_BLOCK_TEXT = re.compile(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>")
_THINK_TAG_TEXT = re.compile(r"(?is)<\s*/?\s*think\s*>")


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

    def __init__(
        self,
        *,
        run_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        self._think = _ThinkStripper()
        self._run_id = run_id
        self._thread_id = thread_id
        self.terminal_status: str | None = None
        self._has_any_text = False
        self._has_trace_event = False
        self._step_statuses: dict[str, str] = {}
        self._progress_statuses: dict[str, str] = {}
        self._last_trace_seq = 0

    @property
    def has_any_text(self) -> bool:
        return self._has_any_text

    @property
    def has_trace_event(self) -> bool:
        return self._has_trace_event

    @property
    def next_trace_seq(self) -> int:
        return self._last_trace_seq + 1

    @property
    def can_synthesize_success_terminal(self) -> bool:
        if self.terminal_status is not None:
            return False
        if self._has_failed_visible_work or self._has_incomplete_visible_work:
            return False
        return self._has_any_text or self._has_completed_visible_work

    def feed(self, upstream_event: SseFrame) -> list[SafeSseFrame]:
        event = upstream_event.event.lower()
        if "messages" in event:
            if _assistant_has_tool_call_signal(upstream_event.data):
                self._discard_pending_assistant_text()
                return []
            text, invocation_id = _assistant_text(upstream_event.data)
            if text is None:
                return []
            safe_text = _sanitize_assistant_text(self._think.feed(text))
            if safe_text:
                self._has_any_text = True
                return [_assistant_delta(safe_text, invocation_id)]
            return []
        if "custom" not in event:
            return []
        payload = _json_object(upstream_event.data)
        if payload is None:
            return []
        payload_type = payload.get("type")
        if payload_type == "trace.v1":
            safe = _safe_trace_payload(
                payload,
                mobile_run_id=self._run_id,
                thread_id=self._thread_id,
            )
            if _starts_non_summary_step(safe):
                self._discard_pending_assistant_text()
            if safe is not None:
                sequence = safe.get("seq")
                if isinstance(sequence, int) and sequence > self._last_trace_seq:
                    self._last_trace_seq = sequence
            if safe is not None and safe.get("event") == "run.terminal":
                status = safe.get("status")
                self.terminal_status = status if isinstance(status, str) else None
            if safe is not None:
                self._record_trace_status(safe)
                self._has_trace_event = True
            return [SafeSseFrame("trace.v1", safe)] if safe is not None else []
        if payload_type == "task_progress":
            safe = _safe_progress_payload(payload)
            if _starts_non_analysis_progress(safe):
                self._discard_pending_assistant_text()
            if safe is not None:
                self._record_progress_status(safe)
            return [SafeSseFrame("task_progress", safe)] if safe is not None else []
        if payload_type == "needs_confirmation":
            safe = _safe_needs_confirmation_payload(payload)
            return [SafeSseFrame("needs_confirmation", safe)] if safe is not None else []
        if payload_type == "task_complexity":
            safe = _safe_task_complexity_payload(payload)
            return [SafeSseFrame("task_complexity", safe)] if safe is not None else []
        return []

    def finish(self, *, allow_pending_text: bool = False) -> list[SafeSseFrame]:
        pending = _sanitize_assistant_text(self._think.finish())
        frames = (
            [_assistant_delta(pending, None)]
            if pending and (self.terminal_status == "succeeded" or allow_pending_text)
            else []
        )
        self._discard_pending_assistant_text()
        frames.append(SafeSseFrame("stream.eof", {"type": "stream.eof"}))
        return frames

    def _discard_pending_assistant_text(self) -> None:
        self._think = _ThinkStripper()

    @property
    def _has_completed_visible_work(self) -> bool:
        if self._step_statuses:
            return any(status in _COMPLETED_WORK_STATUSES for status in self._step_statuses.values())
        return any(status in _COMPLETED_WORK_STATUSES for status in self._progress_statuses.values())

    @property
    def _has_incomplete_visible_work(self) -> bool:
        if self._step_statuses:
            return any(status in _INCOMPLETE_WORK_STATUSES for status in self._step_statuses.values())
        return any(status in _INCOMPLETE_WORK_STATUSES for status in self._progress_statuses.values())

    @property
    def _has_failed_visible_work(self) -> bool:
        if self._step_statuses:
            return any(status in _FAILED_WORK_STATUSES for status in self._step_statuses.values())
        return any(status in _FAILED_WORK_STATUSES for status in self._progress_statuses.values())

    def _record_trace_status(self, payload: Mapping[str, Any]) -> None:
        if payload.get("event") != "step.upsert":
            return
        step = payload.get("step")
        if not isinstance(step, Mapping):
            return
        step_id = step.get("stepId")
        status = step.get("status")
        if isinstance(step_id, str) and isinstance(status, str):
            self._step_statuses[step_id] = status

    def _record_progress_status(self, payload: Mapping[str, Any]) -> None:
        status = payload.get("status")
        if not isinstance(status, str):
            return
        label = payload.get("label")
        phase = payload.get("phase")
        progress_key = payload.get("progressKey")
        key = progress_key if isinstance(progress_key, str) and progress_key else f"{phase}:{label}"
        self._progress_statuses[str(key)] = status


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
        raw_body = await request.body()
        body = _with_mobile_run_config(
            raw_body,
            run_id=run_id,
            thread_id=thread_id,
            device_id=_request_device_id(request, raw_body),
        )
        headers = _forward_headers(request)
    except Exception:
        logger.warning(
            "mobile_safe_run_stream_setup_failed thread_id={} error_category=setup_failed",
            request.path_params.get("thread_id", "unknown"),
        )
        return JSONResponse(
            {
                "error": "stream_setup_failed",
                "message": "请求处理失败，请检查服务配置或稍后重试。",
            },
            status_code=500,
        )

    async def generate() -> AsyncIterator[str]:
        filter_ = SafeStreamFilter(run_id=run_id, thread_id=thread_id)
        stream_seq = 0
        terminal_emitted = False

        def encode(frame: SafeSseFrame) -> str:
            nonlocal stream_seq
            stream_seq += 1
            backend_run_id: str | None = None
            run_info = phone_action_registry.backend_run_info(run_id)
            if run_info is not None:
                backend_run_id = run_info[1]
            _log_safe_frame_emit(
                frame,
                run_id=run_id,
                thread_id=thread_id,
                stream_seq=stream_seq,
            )
            return _encode_frame(
                frame,
                stream_seq=stream_seq,
                run_id=run_id,
                thread_id=thread_id,
                backend_run_id=backend_run_id,
                timestamp=_timestamp_ms(),
            )

        def synthetic_terminal(status: str, reason: str) -> SafeSseFrame:
            nonlocal terminal_emitted
            terminal_emitted = True
            filter_.terminal_status = status
            return _terminal_frame(
                run_id,
                status=status,
                reason=reason,
                seq=filter_.next_trace_seq,
            )

        timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
        current_task = asyncio.current_task()
        if current_task is not None:
            loop = asyncio.get_running_loop()
            phone_action_registry.bind_stream_cancellation(
                run_id,
                lambda: loop.call_soon_threadsafe(current_task.cancel),
            )
        yield encode(_stream_started(run_id=run_id, thread_id=thread_id))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", upstream_url, content=body, headers=headers
                ) as response:
                    _bind_backend_run_from_response(run_id, thread_id, response)
                    if response.status_code < 200 or response.status_code >= 300:
                        error_body = await response.aread()
                        error_detail = _safe_upstream_error_detail(error_body)
                        error_category = _upstream_error_category(response.status_code)
                        logger.warning(
                            "mobile_upstream_stream_error run_id={} thread_id={} upstream_status={} error_category={} upstream_error_detail={}",
                            run_id, thread_id, response.status_code, error_category, error_detail,
                        )
                        # 区分 409（线程忙）和其他错误，统一发业务终态
                        if response.status_code == 409:
                            terminal_status = "failed"
                            terminal_reason = "thread_busy"
                            error_message = "当前会话有任务正在执行，请等待完成后再试。"
                            retryable = True
                        else:
                            terminal_status = "failed"
                            terminal_reason = f"upstream_http_{response.status_code}"
                            error_message = "上游服务返回错误，请稍后重试。"
                            retryable = response.status_code in {408, 429} or response.status_code >= 500
                        yield encode(
                            synthetic_terminal(terminal_status, terminal_reason)
                        )
                        yield encode(
                            _stream_error(
                                error_message,
                                retryable=retryable,
                                terminal_status=terminal_status,
                                terminal_reason=terminal_reason,
                            )
                        )
                        phone_action_registry.mark_terminal(run_id, terminal_reason=terminal_reason)
                        phone_action_registry.mark_backend_status(run_id, terminal_reason)
                        return
                    heartbeat_seconds = _positive_float_env(
                        "MOBILE_AGENT_STREAM_HEARTBEAT_SECONDS",
                        15.0,
                    )
                    async for frame in _stream_with_heartbeats(
                        response.aiter_lines(),
                        heartbeat_seconds=heartbeat_seconds,
                        run_id=run_id,
                    ):
                        if isinstance(frame, SafeSseFrame):
                            yield encode(frame)
                            continue
                        for safe_frame in filter_.feed(frame):
                            yield encode(safe_frame)
                    if filter_.terminal_status is None:
                        if filter_.can_synthesize_success_terminal:
                            reason = "upstream_completed_without_terminal"
                            yield encode(
                                synthetic_terminal("succeeded", reason)
                            )
                            for safe_frame in filter_.finish(allow_pending_text=True):
                                yield encode(safe_frame)
                            logger.info(
                                "mobile_run_stream_synthesized_success_terminal run_id={} thread_id={} reason={}",
                                run_id, thread_id, reason,
                            )
                            phone_action_registry.mark_terminal(run_id)
                            phone_action_registry.mark_backend_status(run_id, "succeeded")
                            return
                        # 没有任何可确认完成信号的 EOF 不能伪造成成功。
                        reason = "upstream_ended_without_terminal"
                        await _cancel_backend_and_device_run(
                            run_id,
                            headers=headers,
                            reason=reason,
                        )
                        terminal_status = "failed"
                        error_message = (
                            "任务连接中断，部分结果可能未完成。"
                            if filter_.has_any_text or filter_.has_trace_event
                            else "任务未返回结束状态，已停止后续操作。"
                        )
                        yield encode(
                            synthetic_terminal(terminal_status, reason)
                        )
                        yield encode(
                            _stream_error(
                                error_message,
                                retryable=True,
                                terminal_status=terminal_status,
                                terminal_reason=reason,
                            )
                        )
                        phone_action_registry.mark_terminal(run_id, terminal_reason=reason)
                        phone_action_registry.mark_backend_status(run_id, terminal_status)
                        return
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
        except asyncio.CancelledError:
            logger.info("mobile_run_stream_cancelled run_id={} thread_id={}", run_id, thread_id)
            # A cancelled proxy task can be caused by Android recreating the SSE
            # connection. Only explicit /cancel should interrupt the native
            # LangGraph run.
            raise
        # 代理边界不允许把内部异常变成半截 SSE 响应；取消信号继承自
        # BaseException，不会被此处捕获，仍可正常终止请求。
        except Exception:
            logger.warning("mobile_run_stream_error run_id={} thread_id={}", run_id, thread_id)
            reason = "stream_error"
            await _cancel_backend_and_device_run(
                run_id,
                headers=headers,
                reason=reason,
            )
            yield encode(
                synthetic_terminal("failed", reason)
            )
            yield encode(
                _stream_error(
                    "服务连接中断，请稍后重试。",
                    retryable=True,
                    terminal_status="failed",
                    terminal_reason=reason,
                )
            )
        finally:
            phone_action_registry.clear_stream_cancellation(run_id)
            # 兜底：无论任何路径退出（正常/异常/取消），确保 run 被标记为终态。
            # 如果 TraceEmitter 的 run.terminal 未到达（例如 TRACE_V1_EMIT_ENABLED
            # 曾为 false），这个兜底能防止 LangSmith run 永远悬挂。
            if filter_.terminal_status is None and not terminal_emitted:
                logger.warning(
                    "mobile_run_stream_no_terminal_at_finally run_id={} thread_id={}",
                    run_id, thread_id,
                )
                phone_action_registry.mark_terminal(run_id, terminal_reason="stream_closed")
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


def _request_device_id(request: Request, raw: bytes) -> str | None:
    header_value = _normal_device_id(request.headers.get("x-device-id"))
    if header_value is not None:
        return header_value
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return _payload_device_id(payload)


def _with_mobile_run_config(
    raw: bytes,
    *,
    run_id: str,
    thread_id: str,
    device_id: str | None = None,
) -> bytes:
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
    selected_device_id = _normal_device_id(device_id) or _payload_device_id(payload)
    if selected_device_id is not None:
        configurable["device_id"] = selected_device_id
        configurable["deviceId"] = selected_device_id
    configurable.update({"mobile_run_id": run_id, "thread_id": thread_id})
    config["configurable"] = configurable
    metadata = config.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    if selected_device_id is not None:
        metadata["device_id"] = selected_device_id
        metadata["deviceId"] = selected_device_id
    if metadata:
        config["metadata"] = metadata
    # deep_agent (create_deep_agent) 已内置 recursion_limit=9,999，
    # 不再用 MOBILE_AGENT_MAX_RECURSION 覆盖它。
    payload["config"] = config
    # Keep the native LangGraph run alive if the mobile SSE proxy disconnects.
    # Explicit /cancel remains the authoritative cancel path.
    payload["on_disconnect"] = "continue"
    # No task may silently sit behind a stale run on the same LangGraph
    # thread. The client must receive a deterministic rejection instead.
    payload["multitask_strategy"] = "reject"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _payload_device_id(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return None
    configurable = config.get("configurable")
    metadata = config.get("metadata")
    configurable_id = (
        _normal_device_id(configurable.get("device_id"))
        or _normal_device_id(configurable.get("deviceId"))
        if isinstance(configurable, Mapping)
        else None
    )
    metadata_id = (
        _normal_device_id(metadata.get("device_id"))
        or _normal_device_id(metadata.get("deviceId"))
        if isinstance(metadata, Mapping)
        else None
    )
    return configurable_id or metadata_id


def _normal_device_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _positive_float_env(name: str, default: float) -> float:
    try:
        return max(float(os.getenv(name, str(default))), 0.1)
    except ValueError:
        return default


async def _stream_with_heartbeats(
    lines: AsyncIterator[str],
    *,
    heartbeat_seconds: float,
    run_id: str,
) -> AsyncIterator[SseFrame | SafeSseFrame]:
    sentinel = object()
    queue: asyncio.Queue[SseFrame | BaseException | object] = asyncio.Queue()

    async def read_upstream() -> None:
        try:
            async for frame in _decode_sse(lines):
                await queue.put(frame)
        except BaseException as exc:
            await queue.put(exc)
        finally:
            await queue.put(sentinel)

    reader = asyncio.create_task(read_upstream())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield _stream_heartbeat(run_id)
                continue
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        if not reader.done():
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass


def _terminal_frame(
    run_id: str,
    *,
    status: str,
    reason: str,
    seq: int,
    cancel_source: str | None = None,
) -> SafeSseFrame:
    data: dict[str, Any] = {
        "type": "trace.v1",
        "version": 1,
        "runId": run_id,
        "eventId": f"evt_{uuid4().hex[:12]}",
        "seq": max(seq, 1),
        "event": "run.terminal",
        "status": status,
        "reason": _bounded(reason, 128),
    }
    if cancel_source:
        data["cancelSource"] = _bounded(cancel_source, 64)
    return SafeSseFrame("trace.v1", data)


def _stream_error(
    message: str,
    *,
    retryable: bool,
    terminal_status: str | None = None,
    terminal_reason: str | None = None,
    cancel_source: str | None = None,
) -> SafeSseFrame:
    data: dict[str, Any] = {
        "type": "stream.error",
        "message": _safe_error_message(message),
        "retryable": retryable,
    }
    if terminal_status:
        data["terminalStatus"] = terminal_status
    if terminal_reason:
        data["terminalReason"] = _bounded(terminal_reason, 128)
    if cancel_source:
        data["cancelSource"] = _bounded(cancel_source, 64)
    return SafeSseFrame("stream.error", data)


def _upstream_error_category(status_code: int) -> str:
    if status_code == 409:
        return "thread_busy"
    if status_code == 408:
        return "timeout"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "server_error"
    if status_code >= 400:
        return "client_error"
    return "unexpected_status"


def _stream_heartbeat(run_id: str) -> SafeSseFrame:
    return SafeSseFrame(
        "stream.heartbeat",
        {
            "type": "stream.heartbeat",
            "runId": _bounded(run_id, 128),
            "timestamp": int(time.time() * 1000),
        },
    )


async def _cancel_device_run(
    run_id: str,
    *,
    reason: str,
    cancel_source: str | None = None,
    cancel_stream: bool,
) -> None:
    device_id = phone_action_registry.cancel_run(
        run_id,
        reason=reason,
        cancel_source=cancel_source or reason,
        terminal_reason=reason,
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
    cancel_source: str | None = None,
) -> str:
    """Fence local/device work and ask LangGraph to interrupt its real run."""
    device_id = phone_action_registry.cancel_run(
        run_id,
        reason=reason,
        cancel_source=cancel_source or reason,
        terminal_reason=reason,
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


def _assistant_has_tool_call_signal(raw: str) -> bool:
    payload = _json_value(raw)
    message: Mapping[str, Any] | None = None
    if isinstance(payload, list) and payload:
        message = payload[0] if isinstance(payload[0], Mapping) else None
    elif isinstance(payload, Mapping):
        message = payload
    if message is None:
        return False
    kwargs = message.get("kwargs") if isinstance(message.get("kwargs"), Mapping) else {}
    return _has_tool_call_signal(message, kwargs)


def _has_tool_call_signal(message: Mapping[str, Any], kwargs: Mapping[str, Any]) -> bool:
    additional_kwargs = _mapping_value(message, "additional_kwargs")
    response_metadata = _mapping_value(message, "response_metadata")
    kwargs_additional = _mapping_value(kwargs, "additional_kwargs")
    kwargs_response_metadata = _mapping_value(kwargs, "response_metadata")
    finish_reason = str(
        response_metadata.get("finish_reason")
        or kwargs_response_metadata.get("finish_reason")
        or ""
    ).lower()
    return bool(
        message.get("tool_calls")
        or kwargs.get("tool_calls")
        or message.get("tool_call_chunks")
        or kwargs.get("tool_call_chunks")
        or additional_kwargs.get("tool_calls")
        or kwargs_additional.get("tool_calls")
        or additional_kwargs.get("function_call")
        or kwargs_additional.get("function_call")
        or finish_reason == "tool_calls"
    )


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else {}



def _content_to_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _content_block_to_public_text(item)
            if text:
                parts.append(text)
        return "".join(parts)
    return _content_block_to_public_text(value)


def _content_block_to_public_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return ""
    block_type = _normalized_block_type(value)
    if _is_reasoning_block_type(block_type):
        return ""
    if block_type and block_type not in _PUBLIC_TEXT_BLOCK_TYPES:
        return ""
    text = value.get("text")
    if isinstance(text, str):
        return text
    content = value.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _content_to_text(content)
    return ""


def _normalized_block_type(value: Mapping[str, Any]) -> str:
    raw_type = value.get("type") or value.get("kind")
    return str(raw_type).strip().lower().replace("-", "_") if raw_type else ""


def _is_reasoning_block_type(block_type: str) -> bool:
    return any(token in block_type for token in _REASONING_BLOCK_TOKENS)


def _safe_trace_payload(
    payload: Mapping[str, Any],
    *,
    mobile_run_id: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any] | None:
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
        "runId": _bounded(str(mobile_run_id or run_id), 128),
        "eventId": _bounded(str(event_id), 128),
        "seq": sequence,
        "event": _bounded(str(event_name), 64),
    }
    safe_thread_id = thread_id if isinstance(thread_id, str) and thread_id else payload.get("threadId")
    if isinstance(safe_thread_id, str) and safe_thread_id:
        safe["threadId"] = _bounded(safe_thread_id, 128)
    if event_name == "run.terminal":
        status = payload.get("status")
        if status in {"succeeded", "failed", "cancelled", "waiting_for_user"}:
            safe["status"] = status
        else:
            return None
        reason = payload.get("reason")
        if isinstance(reason, str) and reason:
            safe["reason"] = _bounded(reason, 128)
        cancel_source = payload.get("cancelSource")
        if isinstance(cancel_source, str) and cancel_source:
            safe["cancelSource"] = _bounded(cancel_source, 64)
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
    if not isinstance(label, str) or not label:
        label = payload.get("stepTitle") or payload.get("taskTitle")
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
    for key, limit in (
        ("runId", 128),
        ("threadId", 128),
        ("taskTitle", MAX_TRACE_TITLE_CHARS),
        ("stepTitle", MAX_TRACE_TITLE_CHARS),
        ("message", MAX_TRACE_SUMMARY_CHARS),
        ("toolName", 128),
        ("progressKey", 128),
        ("confirmationId", 128),
    ):
        value = payload.get(key)
        if isinstance(value, str) and value:
            safe[key] = _safe_description(value, limit)
    for key in ("currentStep", "totalSteps"):
        value = payload.get(key)
        if isinstance(value, int) and value >= 0:
            safe[key] = value
    for key in ("requiresConfirmation", "canCancel", "canTakeOver", "dryRun"):
        value = payload.get(key)
        if isinstance(value, bool):
            safe[key] = value
    completed_steps = _safe_completed_steps(payload.get("completedSteps"))
    if completed_steps:
        safe["completedSteps"] = completed_steps
    return _fit_payload(safe)


def _safe_needs_confirmation_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    confirmation_id = payload.get("confirmationId")
    operation = payload.get("operation")
    tool_name = payload.get("toolName")
    payload_preview = payload.get("payloadPreview")
    if not all(
        isinstance(value, str) and value
        for value in (confirmation_id, operation, tool_name, payload_preview)
    ):
        return None
    safe: dict[str, Any] = {
        "type": "needs_confirmation",
        "confirmationId": _bounded(str(confirmation_id), 128),
        "operation": _safe_description(str(operation), MAX_TRACE_TITLE_CHARS),
        "toolName": _safe_description(str(tool_name), 128),
        "payloadPreview": _safe_description(str(payload_preview), MAX_TRACE_SUMMARY_CHARS),
    }
    for key, limit in (
        ("runId", 128),
        ("threadId", 128),
        ("taskTitle", MAX_TRACE_TITLE_CHARS),
        ("targetApp", MAX_TRACE_TITLE_CHARS),
        ("riskLevel", 32),
        ("confirmText", 32),
        ("cancelText", 32),
    ):
        value = payload.get(key)
        if isinstance(value, str) and value:
            safe[key] = _safe_description(value, limit)
    dry_run = payload.get("dryRun")
    if isinstance(dry_run, bool):
        safe["dryRun"] = dry_run
    return _fit_payload(safe)


def _safe_completed_steps(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    safe_steps: list[dict[str, Any]] = []
    for item in value[:20]:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        status = item.get("status")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(status, str) or not status:
            continue
        step: dict[str, Any] = {
            "name": _safe_description(name, MAX_TRACE_TITLE_CHARS),
            "status": _bounded(status, 32),
        }
        index = item.get("index")
        if isinstance(index, int) and index >= 0:
            step["index"] = index
        tool_name = item.get("toolName")
        if isinstance(tool_name, str) and tool_name:
            step["toolName"] = _safe_description(tool_name, 128)
        safe_steps.append(step)
    return safe_steps


def _safe_task_complexity_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    complexity = payload.get("complexity")
    track_steps = payload.get("trackSteps")
    reason = payload.get("reason")
    if complexity not in {"simple", "complex"}:
        return None
    if not isinstance(track_steps, bool):
        return None
    if not isinstance(reason, str) or not reason:
        return None
    safe: dict[str, Any] = {
        "type": "task_complexity",
        "complexity": complexity,
        "trackSteps": track_steps,
        "reason": _bounded(reason, 64),
    }
    message = payload.get("message")
    if isinstance(message, str) and message:
        safe["message"] = _safe_description(message, MAX_TRACE_SUMMARY_CHARS)
    return _fit_payload(safe)


def _starts_non_summary_step(payload: Mapping[str, Any] | None) -> bool:
    if payload is None or payload.get("event") != "step.upsert":
        return False
    step = payload.get("step")
    return (
        isinstance(step, Mapping)
        and step.get("status") == "running"
        and step.get("kind") != "summary"
    )


def _starts_non_analysis_progress(payload: Mapping[str, Any] | None) -> bool:
    if payload is None:
        return False
    return payload.get("status") == "running" and payload.get("phase") != "analysis"


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
    redacted = _strip_think_markup(text)
    redacted = _SENSITIVE_JSON_VALUE.sub(lambda match: f"{match.group(1)}***{match.group(2)}", redacted)
    redacted = _AUTHORIZATION_VALUE.sub("Authorization=***", redacted)
    redacted = _COOKIE_VALUE.sub("Cookie=***", redacted)
    redacted = _RAW_TOOL_TEXT.sub("[已隐藏工具原始数据]", redacted)
    redacted = _UI_TREE_TEXT.sub("[已隐藏UI树]", redacted)
    redacted = _SENSITIVE_VALUE.sub(lambda match: f"{match.group(1)}=***", redacted)
    redacted = _IMAGE_DATA.sub("[已隐藏图片数据]", redacted)
    redacted = _LONG_BASE64.sub("[已隐藏base64数据]", redacted)
    redacted = _VERIFICATION_CODE.sub("[已隐藏敏感内容]", redacted)
    return _limit_bytes(redacted, MAX_SAFE_ASSISTANT_DELTA_BYTES)


def _safe_error_message(text: str) -> str:
    without_traceback = _TRACEBACK_TEXT.sub("错误详情已隐藏。", text)
    return _safe_description(without_traceback, MAX_TRACE_SUMMARY_CHARS) or "服务连接中断，请稍后重试。"


def _safe_upstream_error_detail(body: bytes) -> str:
    if not body:
        return ""
    text = body[:8192].decode("utf-8", errors="replace")
    return _safe_description(text, MAX_TRACE_SUMMARY_CHARS)


def _safe_description(text: str, limit: int) -> str:
    return _bounded(_sanitize_assistant_text(text), limit)


def _strip_think_markup(text: str) -> str:
    without_blocks = _THINK_BLOCK_TEXT.sub("", text)
    return _THINK_TAG_TEXT.sub("", without_blocks)


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


def _log_safe_frame_emit(
    frame: SafeSseFrame,
    *,
    run_id: str,
    thread_id: str,
    stream_seq: int,
) -> None:
    trace_event = frame.data.get("event") if frame.event == "trace.v1" else None
    terminal_status = frame.data.get("status") if trace_event == "run.terminal" else None
    logger.debug(
        "mobile_safe_stream_emit run_id={} thread_id={} stream_seq={} event={} trace_event={} terminal_status={}",
        run_id,
        _short_log_id(thread_id),
        stream_seq,
        frame.event,
        trace_event,
        terminal_status,
    )


def _encode_frame(
    frame: SafeSseFrame,
    *,
    stream_seq: int | None = None,
    run_id: str | None = None,
    thread_id: str | None = None,
    backend_run_id: str | None = None,
    timestamp: int | None = None,
) -> str:
    data = dict(frame.data)
    if stream_seq is not None:
        data["streamSeq"] = stream_seq
    if run_id:
        data.setdefault("mobileRunId", _bounded(run_id, 128))
        if frame.event != "trace.v1":
            data.setdefault("runId", _bounded(run_id, 128))
    if thread_id:
        data.setdefault("threadId", _bounded(thread_id, 128))
    if backend_run_id:
        data.setdefault("backendRunId", _bounded(backend_run_id, 128))
    if timestamp is not None:
        data.setdefault("timestamp", timestamp)
    data = _fit_safe_frame_payload(data)
    return f"event: {frame.event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _short_log_id(value: str | None) -> str:
    if not value:
        return "none"
    return value if len(value) <= 12 else f"{value[:6]}…{value[-4:]}"


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
    _MARKERS = (_OPEN, _CLOSE)

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buffer += text
        output: list[str] = []
        while self._buffer:
            markers = (self._CLOSE,) if self._inside else self._MARKERS
            marker, index = _find_first_marker(self._buffer, markers)
            if index < 0:
                suffix_length = _partial_suffix_length(self._buffer, markers)
                stable = self._buffer[:-suffix_length] if suffix_length else self._buffer
                if not self._inside:
                    output.append(stable)
                self._buffer = self._buffer[-suffix_length:] if suffix_length else ""
                break
            if not self._inside:
                output.append(self._buffer[:index])
                self._inside = marker == self._OPEN
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


def _find_first_marker(value: str, markers: tuple[str, ...]) -> tuple[str, int]:
    lower_value = value.lower()
    first_marker = markers[0]
    first_index = -1
    for marker in markers:
        index = lower_value.find(marker.lower())
        if index >= 0 and (first_index < 0 or index < first_index):
            first_marker = marker
            first_index = index
    return first_marker, first_index


def _partial_suffix_length(value: str, markers: tuple[str, ...]) -> int:
    lower_value = value.lower()
    longest = 0
    for marker in markers:
        lower_marker = marker.lower()
        for size in range(min(len(value), len(marker) - 1), 0, -1):
            if lower_marker.startswith(lower_value[-size:]):
                longest = max(longest, size)
                break
    return longest


__all__ = ["SafeSseFrame", "SafeStreamFilter", "SseFrame", "safe_run_stream"]
