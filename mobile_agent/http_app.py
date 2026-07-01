"""Custom LangGraph HTTP app routes for mobile and system tool websockets."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from .action_control import phone_action_registry
from .asr.config import DEFAULT_MAX_AUDIO_BYTES, AsrConfigError
from .asr.provider import AliyunNlsProvider, AsrProviderError, AsrRequest
from .gateways.phone import DeviceGatewayError
from .gateways.system import SystemGatewayError
from .local_model_runtime import LocalModelRuntimeError, model_runtime
from .runtime import phone_gateway, system_gateway
from .safe_stream import (
    _cancel_backend_and_device_run,
    _forward_headers,
    cancel_upstream_run,
    safe_run_stream,
    upstream_run_status,
)


async def cancel_recovered_mobile_runs() -> None:
    """On facade startup, stop only persisted mobile runs from a prior process."""
    recovered = phone_action_registry.stale_backend_runs()
    for run_id, thread_id, backend_run_id, _ in recovered:
        logger.warning(
            "mobile_run_recovery_cancel run_id={} thread_id={} backend_run_id={}",
            run_id,
            thread_id,
            backend_run_id,
        )
        await _cancel_backend_and_device_run(
            run_id,
            headers={},
            reason="facade_restarted",
        )


@asynccontextmanager
async def _lifespan(_: Starlette) -> AsyncIterator[None]:
    await cancel_recovered_mobile_runs()
    yield


async def adb_websocket(websocket: WebSocket) -> None:
    await phone_gateway.starlette_handler(websocket)


async def system_websocket(websocket: WebSocket) -> None:
    await system_gateway.starlette_handler(websocket)


async def adb_status(request: Request) -> JSONResponse:
    device_id = request.path_params.get("device_id")
    try:
        session = phone_gateway.get_session(device_id)
    except DeviceGatewayError:
        return JSONResponse({"connected": False})

    device_info = session.device_info
    return JSONResponse(
        {
            "connected": True,
            "width": device_info.width if device_info else None,
            "height": device_info.height if device_info else None,
            "currentPackage": device_info.current_package if device_info else None,
            "activity": device_info.activity if device_info else None,
        }
    )


async def system_status(request: Request) -> JSONResponse:
    device_id = request.path_params.get("device_id")
    try:
        client = system_gateway.get_default_client(device_id)
    except SystemGatewayError:
        return JSONResponse({"connected": False})

    return JSONResponse(
        {
            "connected": True,
            "path": client.info.path,
            "remoteAddress": str(client.info.remote_address),
        }
    )


async def network_status(request: Request) -> JSONResponse:
    return JSONResponse(model_runtime.status())


def asr_provider_factory() -> AliyunNlsProvider:
    """创建 ASR provider（配置由 load_aliyun_nls_config 缓存，无 I/O）。"""
    return AliyunNlsProvider()


async def transcribe_audio(request: Request) -> JSONResponse:
    try:
        form = await request.form()
    except Exception:
        return JSONResponse(
            {"error": "invalid_multipart", "message": "请使用 multipart/form-data 上传音频。"},
            status_code=400,
        )

    upload = form.get("audio")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse(
            {"error": "missing_audio", "message": "缺少 audio 文件字段。"},
            status_code=400,
        )

    audio = await upload.read()
    if not audio:
        return JSONResponse(
            {"error": "empty_audio", "message": "音频内容为空。"},
            status_code=400,
        )
    if len(audio) > DEFAULT_MAX_AUDIO_BYTES:
        return JSONResponse(
            {"error": "audio_too_large", "message": "音频文件过大。"},
            status_code=413,
        )

    sample_rate = _parse_positive_int(form.get("sampleRate"), default=16000)
    if sample_rate is None:
        return JSONResponse(
            {"error": "invalid_sample_rate", "message": "sampleRate 必须是正整数。"},
            status_code=400,
        )

    asr_request = AsrRequest(
        audio_format=_form_string(form.get("format"), default="pcm").lower(),
        sample_rate=sample_rate,
        language=_form_string(form.get("language"), default="zh-CN"),
    )

    try:
        provider = asr_provider_factory()
        max_audio_bytes = getattr(getattr(provider, "config", None), "max_audio_bytes", DEFAULT_MAX_AUDIO_BYTES)
        if len(audio) > max_audio_bytes:
            return JSONResponse(
                {"error": "audio_too_large", "message": "音频文件过大。"},
                status_code=413,
            )
        result = await provider.transcribe(audio, asr_request)
    except AsrConfigError as exc:
        return JSONResponse(
            {"error": "asr_config_missing", "message": str(exc)},
            status_code=503,
        )
    except AsrProviderError as exc:
        logger.warning("asr_transcribe_failed message={}", str(exc))
        return JSONResponse(
            {"error": "asr_provider_failed", "message": str(exc)},
            status_code=502,
        )
    except Exception:
        logger.exception("asr_transcribe_unhandled_exception")
        return JSONResponse(
            {"error": "asr_internal_error", "message": "语音识别服务暂时不可用"},
            status_code=500,
        )

    return JSONResponse(
        {
            "text": result.text,
            "provider": result.provider,
            "requestId": result.request_id,
            "durationMs": result.duration_ms,
        }
    )


def _form_string(value: object, *, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _parse_positive_int(value: object, *, default: int) -> int | None:
    if value is None or value == "":
        return default
    try:
        parsed = int(str(value))
    except ValueError:
        return None
    return parsed if parsed > 0 else None


async def cancel_mobile_run(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    run_id = request.path_params["run_id"]
    if phone_action_registry.thread_id_for(run_id) != thread_id:
        return JSONResponse(
            {
                "runId": run_id,
                "mobileRunId": run_id,
                "threadId": thread_id,
                "status": "not_found",
                "backendRunId": None,
                "backendStatus": "missing",
                "cancelSource": None,
                "terminalReason": "not_found",
                "timestamp": int(time.time() * 1000),
            }
        )
    cancel_source = await _cancel_source_from_request(request)
    before_snapshot = phone_action_registry.snapshot(run_id)
    if _snapshot_has_confirmed_backend_terminal(before_snapshot):
        return JSONResponse(
            {
                "runId": run_id,
                "mobileRunId": run_id,
                "threadId": thread_id,
                "status": "already_terminal",
                "backendRunId": before_snapshot.get("backendRunId"),
                "backendStatus": before_snapshot.get("backendStatus", "not_started"),
                "deviceStatus": "already_terminal",
                "localFenced": True,
                "retryable": False,
                "cancelSource": before_snapshot.get("cancelSource"),
                "terminalReason": before_snapshot.get("terminalReason"),
                "timestamp": int(time.time() * 1000),
            }
        )

    # Fence device commands before doing network I/O, but do not cancel the
    # proxy task until the native LangGraph cancellation request has been sent.
    # safe_stream keeps disconnects from becoming implicit upstream cancels, so
    # this explicit endpoint is the user-driven cancellation path.
    device_id = phone_action_registry.cancel_run(
        run_id,
        reason=cancel_source,
        cancel_source=cancel_source,
        terminal_reason=cancel_source,
        cancel_stream=False,
    )
    cancellation_headers = _forward_headers(request)
    cancellation_headers.pop("content-type", None)
    logger.info("mobile_run_cancel_requested run_id={} thread_id={}", run_id, thread_id)
    backend_status = await cancel_upstream_run(run_id, cancellation_headers)
    after_snapshot = phone_action_registry.snapshot(run_id)
    backend_run_id = after_snapshot.get("backendRunId")
    if backend_status == "cancel_requested" and backend_run_id is not None:
        backend_status = await upstream_run_status(run_id, cancellation_headers)
        after_snapshot = phone_action_registry.snapshot(run_id)
        backend_run_id = after_snapshot.get("backendRunId")
    device_status = "not_bound"
    if device_id is not None:
        try:
            await phone_gateway.cancel_run(run_id, device_id)
            device_status = "canceled"
        except DeviceGatewayError:
            # The ledger is already cancelled.  A disconnected device cannot
            # receive a later command because every incoming command is also
            # rejected by its persisted run cancellation state.
            device_status = "cancel_failed"
    response_status, retryable = _cancel_response_status(
        backend_status,
        device_status=device_status,
        backend_run_id=backend_run_id,
    )
    phone_action_registry.trigger_stream_cancellation(run_id)
    logger.info(
        "mobile_run_cancel_dispatched run_id={} thread_id={} backend_status={}",
        run_id,
        thread_id,
        backend_status,
    )
    return JSONResponse(
        {
            "runId": run_id,
            "mobileRunId": run_id,
            "threadId": thread_id,
            "status": response_status,
            "backendRunId": backend_run_id,
            "backendStatus": backend_status,
            "deviceStatus": device_status,
            "localFenced": True,
            "retryable": retryable,
            "cancelSource": cancel_source,
            "terminalReason": cancel_source,
            "timestamp": int(time.time() * 1000),
        }
    )


async def mobile_run_status(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    run_id = request.path_params["run_id"]
    if phone_action_registry.thread_id_for(run_id) != thread_id:
        return JSONResponse({"error": "run_not_found"}, status_code=404)

    status_headers = _forward_headers(request)
    status_headers.pop("content-type", None)
    backend_status = await upstream_run_status(run_id, status_headers)
    snapshot = phone_action_registry.snapshot(run_id)
    terminal = backend_status in _TERMINAL_BACKEND_STATUSES
    return JSONResponse(
        {
            "runId": run_id,
            "status": snapshot["status"],
            "backendStatus": backend_status,
            "terminal": terminal,
            "cancelSource": snapshot.get("cancelSource"),
            "terminalReason": snapshot.get("terminalReason"),
        }
    )


_ALLOWED_CANCEL_SOURCES = {
    "user",
    "frontend_timeout",
    "stream_error",
    "client_disconnected",
    "session_deleted",
    "server_timeout",
}

_TERMINAL_BACKEND_STATUSES = {
    "succeeded",
    "failed",
    "cancelled",
    "timeout",
    "server_timeout",
    "stream_closed",
    "thread_busy",
    "not_started",
}

_CONFIRMED_ALREADY_TERMINAL_BACKEND_STATUSES = _TERMINAL_BACKEND_STATUSES - {
    # 本地 SSE proxy 关闭不代表 LangGraph/backend run 已停止。
    "stream_closed",
}

_CONFIRMED_CANCEL_BACKEND_STATUSES = {
    "cancelled",
    "not_started",
}

_UNCONFIRMED_CANCEL_BACKEND_STATUSES = {
    "running",
    "pending",
    "cancel_requested",
    "unavailable",
}


def _snapshot_has_confirmed_backend_terminal(snapshot: Mapping[str, object]) -> bool:
    if snapshot.get("status") not in {"terminal", "cancelled"}:
        return False
    backend_status = str(snapshot.get("backendStatus", "not_started"))
    return backend_status in _CONFIRMED_ALREADY_TERMINAL_BACKEND_STATUSES


def _cancel_response_status(
    backend_status: str,
    *,
    device_status: str,
    backend_run_id: object,
) -> tuple[str, bool]:
    if device_status == "cancel_failed":
        return "device_cancel_failed", True
    if backend_status == "unknown_not_bound":
        return "backend_run_not_bound", True
    if backend_status in {"cancel_unavailable", "cancel_request_failed"}:
        return backend_status, True
    if backend_status in _CONFIRMED_CANCEL_BACKEND_STATUSES:
        return (
            "canceled_confirmed" if backend_run_id is not None else "local_fenced_only",
            False,
        )
    if backend_status in _UNCONFIRMED_CANCEL_BACKEND_STATUSES:
        return "backend_still_running", True
    return "local_fenced_only", False


async def _cancel_source_from_request(request: Request) -> str:
    try:
        payload = await request.json()
    except ValueError:
        payload = {}
    value = payload.get("cancelSource") if isinstance(payload, dict) else None
    if not isinstance(value, str) or not value.strip():
        value = request.headers.get("x-cancel-source")
    source = str(value or "user").strip().lower()
    return source if source in _ALLOWED_CANCEL_SOURCES else "user"


async def update_network_status(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    connected = payload.get("connected") if isinstance(payload, dict) else None
    if not isinstance(connected, bool):
        return JSONResponse(
            {"error": "invalid_connected", "message": "`connected` must be a boolean."},
            status_code=400,
        )

    try:
        status = await asyncio.to_thread(model_runtime.set_network_connected, connected)
    except LocalModelRuntimeError as exc:
        return JSONResponse(
            {"error": "local_model_start_failed", "message": str(exc)},
            status_code=500,
        )

    return JSONResponse(status)


app = Starlette(
    lifespan=_lifespan,
    routes=[
        Route("/mobile/asr/transcribe", transcribe_audio, methods=["POST"]),
        Route("/mobile/threads/{thread_id}/runs/stream", safe_run_stream, methods=["POST"]),
        Route(
            "/mobile/threads/{thread_id}/runs/{run_id}/cancel",
            cancel_mobile_run,
            methods=["POST"],
        ),
        Route(
            "/mobile/threads/{thread_id}/runs/{run_id}/status",
            mobile_run_status,
            methods=["GET"],
        ),
        WebSocketRoute("/adb", adb_websocket),
        WebSocketRoute("/adb/{device_id}", adb_websocket),
        WebSocketRoute("/system", system_websocket),
        WebSocketRoute("/system/{device_id}", system_websocket),
        Route("/adb/status", adb_status, methods=["GET"]),
        Route("/adb/{device_id}/status", adb_status, methods=["GET"]),
        Route("/system/status", system_status, methods=["GET"]),
        Route("/system/{device_id}/status", system_status, methods=["GET"]),
        Route("/network/status", network_status, methods=["GET"]),
        Route("/network/{device_id}/status", network_status, methods=["GET"]),
        Route("/network/status", update_network_status, methods=["POST"]),
        Route("/network/{device_id}/status", update_network_status, methods=["POST"]),
    ]
)
