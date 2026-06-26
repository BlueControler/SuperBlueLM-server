from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from loguru import logger
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket

from ..json_types import JsonObject, JsonValue, to_json_object, to_json_value
from .rpc import (
    JsonLineEnvelope,
    JsonLineProtocolViolation,
    JsonLineRpcSession,
    JsonLineWebSocket,
    StarletteWebSocketConnection,
    normalized_path,
)

VERBOSE_HEARTBEAT = os.getenv("VERBOSE_HEARTBEAT", "").lower() in {"1", "true", "yes"}
DEFAULT_SESSION_RECONNECT_WAIT_SECONDS = 3.0
DEFAULT_PHONE_COMMAND_TIMEOUT_SECONDS = 45.0
DEVICE_NOT_CONNECTED_CODE = "device_not_connected"
DEVICE_NOT_CONNECTED_MESSAGE = "手机连接已断开，请重新连接后重试"


class DeviceGatewayError(RuntimeError):
    pass


class DeviceNotConnectedError(DeviceGatewayError):
    def __init__(self) -> None:
        super().__init__(DEVICE_NOT_CONNECTED_CODE)


class ProtocolViolation(DeviceGatewayError, JsonLineProtocolViolation):
    pass


class ConnectData(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    screenshot: str | None = None
    screenshot_mime_type: str | None = Field(default=None, alias="screenshotMimeType")
    ui: str | None = None
    current_package: str | None = Field(default=None, alias="currentPackage")
    activity: str | None = None
    token: str | None = None

    model_config = {"populate_by_name": True}


class ErrorData(BaseModel):
    message: str
    screenshot: str | None = None
    screenshot_mime_type: str | None = Field(default=None, alias="screenshotMimeType")
    ui: str | None = None
    current_package: str | None = Field(default=None, alias="currentPackage")
    activity: str | None = None

    model_config = {"populate_by_name": True}


@dataclass
class DeviceInfo:
    width: int
    height: int
    screenshot: str | None
    screenshot_mime_type: str
    ui: str | None
    current_package: str | None
    activity: str | None
    token: str | None = None


class ConnectedDeviceSession(JsonLineRpcSession):
    def __init__(self, websocket: JsonLineWebSocket) -> None:
        super().__init__(
            websocket,
            request_id_start=2,
            disconnect_error=lambda message: DeviceGatewayError(
                message.replace("WebSocket client", "Device")
            ),
        )
        self.device_info: DeviceInfo | None = None
        self.ready = asyncio.Event()

    async def wait_ready(self, timeout: float = 10.0) -> None:
        await asyncio.wait_for(self.ready.wait(), timeout=timeout)

    async def send_command(
        self,
        message: str,
        data: JsonValue,
        timeout: float | None = None,
    ) -> JsonObject:
        if not self.ready.is_set():
            raise DeviceGatewayError("Device has not completed connect.")

        logger.info(f"-> message={message} data={_sanitize_log_payload(data)}")
        response = await self.send_rpc_request(
            message,
            data,
            timeout=_phone_command_timeout_seconds() if timeout is None else timeout,
        )
        response_data = to_json_value(response.data)
        logger.info(
            f"<- requestId={response.request_id} message={response.message} "
            f"data={_sanitize_log_payload(response_data)}"
        )

        if response.message == "error":
            error = ErrorData.model_validate(response_data)
            self._update_device_info(
                screenshot=error.screenshot,
                screenshot_mime_type=error.screenshot_mime_type,
                ui=error.ui,
                current_package=error.current_package,
                activity=error.activity,
            )
            raise DeviceGatewayError(error.message)

        if response.message != "actionResult":
            raise DeviceGatewayError(f"Expected 'actionResult', got {response.message!r}.")

        action_result = to_json_object(response_data)
        self._update_device_info_from_payload(action_result)
        return action_result

    async def cancel_run(self, run_id: str) -> None:
        """Ask the Android executor to stop queued work for one logical run."""
        if not self.ready.is_set():
            raise DeviceGatewayError("Device has not completed connect.")
        response = await self.send_rpc_request(
            "cancel",
            {"runId": run_id},
            timeout=min(_phone_command_timeout_seconds(), 5.0),
        )
        if response.message == "error":
            error = ErrorData.model_validate(to_json_value(response.data))
            raise DeviceGatewayError(error.message)
        if response.message != "cancelled":
            raise DeviceGatewayError(
                f"Expected 'cancelled' response, got {response.message!r}."
            )

    async def _handle_client_request(self, envelope: JsonLineEnvelope) -> None:
        if not self.ready.is_set():
            if envelope.message != "connect":
                raise ProtocolViolation("The first client request must be 'connect'.")
            self._handle_connect(envelope)
            return

        if envelope.message == "connect":
            raise ProtocolViolation("Duplicate 'connect' is not allowed on one websocket session.")

        if envelope.message == "ping":
            if envelope.request_id is not None:
                raise ProtocolViolation("'ping' must not carry requestId.")
            if VERBOSE_HEARTBEAT:
                logger.debug("<- heartbeat")
            await self.send_response("pong")
            return

        await self.send_response(
            "error",
            {
                "message": f"Unsupported client request: {envelope.message}",
            },
            request_id=envelope.request_id,
        )

    def _handle_client_response(self, envelope: JsonLineEnvelope) -> None:
        if not self.ready.is_set():
            raise ProtocolViolation("Received response before 'connect' completed.")
        if envelope.message == "pong":
            raise ProtocolViolation("Client must not send 'pong'.")
        super()._handle_client_response(envelope)

    def _handle_connect(self, envelope: JsonLineEnvelope) -> None:
        if envelope.request_id != 1:
            raise ProtocolViolation("'connect' must carry fixed requestId=1.")

        connect_data = ConnectData.model_validate(envelope.data)
        self.device_info = DeviceInfo(
            width=connect_data.width,
            height=connect_data.height,
            screenshot=connect_data.screenshot,
            screenshot_mime_type=_screenshot_mime_type(connect_data.screenshot_mime_type),
            ui=connect_data.ui,
            current_package=connect_data.current_package,
            activity=connect_data.activity,
            token=connect_data.token,
        )
        self.ready.set()
        logger.info(
            f"device connected: size={connect_data.width}x{connect_data.height} "
            f"requestIdStart=2"
        )

    def _update_device_info(
        self,
        *,
        screenshot: str | None,
        screenshot_mime_type: str | None,
        ui: str | None,
        current_package: str | None,
        activity: str | None,
    ) -> None:
        if self.device_info is None:
            return
        if screenshot is not None:
            self.device_info.screenshot = screenshot
        if screenshot_mime_type is not None:
            self.device_info.screenshot_mime_type = _screenshot_mime_type(screenshot_mime_type)
        if ui is not None:
            self.device_info.ui = ui
        if current_package is not None:
            self.device_info.current_package = current_package
        if activity is not None:
            self.device_info.activity = activity

    def _update_device_info_from_payload(self, payload: JsonObject) -> None:
        self._update_device_info(
            screenshot=_optional_str(payload.get("screenshot")),
            screenshot_mime_type=_optional_str(payload.get("screenshotMimeType")),
            ui=_optional_str(payload.get("ui")),
            current_package=_optional_str(payload.get("currentPackage")),
            activity=_optional_str(payload.get("activity")),
        )


class DeviceGateway:
    DEFAULT_DEVICE_ID = "__default__"

    def __init__(self, path_prefix: str = "/adb") -> None:
        self.path_prefix = path_prefix.rstrip("/")
        self._sessions: dict[str, ConnectedDeviceSession] = {}
        self._session_changed = asyncio.Condition()

    def get_session(self, device_id: str | None = None) -> ConnectedDeviceSession:
        if device_id is not None:
            session = self._sessions.get(device_id)
            if session is not None and not session.closed.is_set():
                return session
            raise DeviceNotConnectedError()

        active_sessions = self._active_sessions()
        if len(active_sessions) == 1:
            return active_sessions[0]
        if len(active_sessions) > 1:
            raise DeviceGatewayError(
                "Multiple devices are connected; device_id is required."
            )
        raise DeviceNotConnectedError()

    async def wait_for_session(
        self,
        device_id: str | None = None,
        timeout: float | None = None,
    ) -> ConnectedDeviceSession:
        wait_seconds = _session_reconnect_wait_seconds() if timeout is None else max(timeout, 0)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_seconds

        async with self._session_changed:
            while True:
                try:
                    return self.get_session(device_id)
                except DeviceNotConnectedError:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise DeviceNotConnectedError() from None
                    try:
                        await asyncio.wait_for(
                            self._session_changed.wait(),
                            timeout=remaining,
                        )
                    except TimeoutError:
                        raise DeviceNotConnectedError() from None

    async def cancel_run(self, run_id: str, device_id: str | None = None) -> None:
        session = self.get_session(device_id)
        await session.cancel_run(run_id)

    async def handler(self, websocket: JsonLineWebSocket) -> None:
        request = websocket.request
        if request is None:
            raise DeviceGatewayError("Missing websocket request metadata.")
        device_id = self._device_id_from_path(request.path)
        session = ConnectedDeviceSession(websocket)

        await session.start()
        await session.wait_ready()
        logger.info(
            f"Registering device session device_id={device_id!r} "
            f"remote={websocket.remote_address!r}."
        )
        async with self._session_changed:
            old_session = self._sessions.get(device_id)
            if old_session is not None and not old_session.closed.is_set():
                logger.warning(
                    f"Rejecting duplicate device session for device_id={device_id!r}; "
                    f"active_remote={old_session.websocket.remote_address!r}, "
                    f"duplicate_remote={websocket.remote_address!r}."
                )
                await websocket.close(
                    code=1008,
                    reason="device connection already active",
                )
                await session.stop()
                return
            self._sessions[device_id] = session
            self._session_changed.notify_all()
        try:
            await session.closed.wait()
        finally:
            logger.info(f"Device session closed for device_id={device_id!r}.")
            async with self._session_changed:
                if self._sessions.get(device_id) is session:
                    self._sessions.pop(device_id, None)
                    self._session_changed.notify_all()
            await session.stop()

    async def starlette_handler(self, websocket: WebSocket) -> None:
        await websocket.accept()
        await self.handler(StarletteWebSocketConnection(websocket))

    def _validate_path(self, path: str) -> None:
        self._device_id_from_path(path)

    def _device_id_from_path(self, path: str) -> str:
        normalized = normalized_path(path)
        if normalized == self.path_prefix:
            return self.DEFAULT_DEVICE_ID
        prefix = f"{self.path_prefix}/"
        if normalized.startswith(prefix) and "/" not in normalized[len(prefix) :]:
            device_id = normalized[len(prefix) :]
            if device_id:
                return device_id
        raise DeviceGatewayError(
            f"Invalid device path {path!r}. Expected {self.path_prefix!r} "
            f"or {self.path_prefix}/{{device_id}}."
        )

    def _active_sessions(self) -> list[ConnectedDeviceSession]:
        return [session for session in self._sessions.values() if not session.closed.is_set()]


def _optional_str(value: JsonValue) -> str | None:
    return value if isinstance(value, str) else None


def _screenshot_mime_type(value: str | None) -> str:
    if isinstance(value, str) and value.startswith("image/"):
        return value
    return "image/png"


def _session_reconnect_wait_seconds() -> float:
    raw = os.getenv(
        "PHONE_SESSION_RECONNECT_WAIT_SECONDS",
        str(DEFAULT_SESSION_RECONNECT_WAIT_SECONDS),
    )
    try:
        return max(float(raw), 0)
    except ValueError:
        return DEFAULT_SESSION_RECONNECT_WAIT_SECONDS


def _phone_command_timeout_seconds() -> float:
    raw = os.getenv(
        "PHONE_COMMAND_TIMEOUT_SECONDS",
        str(DEFAULT_PHONE_COMMAND_TIMEOUT_SECONDS),
    )
    try:
        return max(float(raw), 1)
    except ValueError:
        return DEFAULT_PHONE_COMMAND_TIMEOUT_SECONDS


def _sanitize_log_payload(payload: JsonValue) -> JsonValue:
    if isinstance(payload, dict):
        return {
            key: "<omitted>" if key in {"screenshot", "ui"} else _sanitize_log_payload(value)
            for key, value in payload.items()
        }

    if isinstance(payload, list):
        return [_sanitize_log_payload(item) for item in payload]

    return payload
