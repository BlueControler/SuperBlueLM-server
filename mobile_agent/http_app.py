"""Custom LangGraph HTTP app routes for mobile and system tool websockets."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from .action_control import phone_action_registry
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


async def cancel_mobile_run(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    run_id = request.path_params["run_id"]
    if phone_action_registry.thread_id_for(run_id) != thread_id:
        return JSONResponse({"error": "run_not_found"}, status_code=404)

    # Fence device commands before doing network I/O, but do not cancel the
    # proxy task until the native LangGraph cancellation request has been sent.
    # The stream disconnect is a second cancellation path (`on_disconnect` is
    # forced to `cancel` by safe_stream).
    device_id = phone_action_registry.cancel_run(
        run_id,
        reason="user_cancelled",
        cancel_stream=False,
    )
    cancellation_headers = _forward_headers(request)
    cancellation_headers.pop("content-type", None)
    logger.info("mobile_run_cancel_requested run_id={} thread_id={}", run_id, thread_id)
    backend_status = await cancel_upstream_run(run_id, cancellation_headers)
    if device_id is not None:
        try:
            await phone_gateway.cancel_run(run_id, device_id)
        except DeviceGatewayError:
            # The ledger is already cancelled.  A disconnected device cannot
            # receive a later command because every incoming command is also
            # rejected by its persisted run cancellation state.
            pass
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
            "status": "cancellation_requested",
            "backendStatus": backend_status,
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
    terminal = backend_status in {"succeeded", "failed", "cancelled", "timeout"}
    return JSONResponse(
        {
            "runId": run_id,
            "status": snapshot["status"],
            "backendStatus": backend_status,
            "terminal": terminal,
        }
    )


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
