from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

VERBOSE_HEARTBEAT = os.getenv("VERBOSE_HEARTBEAT", "").lower() in {"1", "true", "yes"}


class DeviceGatewayError(RuntimeError):
    pass


class MessageEnvelope(BaseModel):
    type: Literal["request", "response"]
    message: str
    data: Any = None
    requestId: int | None = None


class ConnectData(BaseModel):
    token: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    screenshot: str | None = None
    ui: str | None = None
    currentPackage: str | None = None
    activity: str | None = None


class ActionResultData(BaseModel):
    screenshot: str | None = None
    ui: str | None = None
    currentPackage: str | None = None
    activity: str | None = None


class ErrorData(BaseModel):
    message: str
    screenshot: str | None = None
    ui: str | None = None
    currentPackage: str | None = None
    activity: str | None = None


@dataclass
class DeviceInfo:
    device_id: str
    width: int
    height: int
    token: str
    screenshot: str | None
    ui: str | None
    current_package: str | None
    activity: str | None


class ConnectedDeviceSession:
    def __init__(self, websocket: ServerConnection, device_id: str) -> None:
        self.websocket = websocket
        self.device_id = device_id
        self.device_info: DeviceInfo | None = None
        self.ready = asyncio.Event()
        self.closed = asyncio.Event()
        self._request_lock = asyncio.Lock()
        self._pending_response: asyncio.Future[MessageEnvelope] | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._next_request_id: int | None = None

    async def start(self) -> None:
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def stop(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

    async def wait_ready(self, timeout: float = 10.0) -> None:
        await asyncio.wait_for(self.ready.wait(), timeout=timeout)

    async def send_command(self, message: str, data: Any, timeout: float = 20.0) -> dict[str, Any]:
        if self.closed.is_set():
            raise DeviceGatewayError(f"Device {self.device_id} is disconnected.")
        if not self.ready.is_set():
            raise DeviceGatewayError(f"Device {self.device_id} has not completed connect.")

        async with self._request_lock:
            loop = asyncio.get_running_loop()
            self._pending_response = loop.create_future()
            request_id = self._consume_next_request_id()
            payload = MessageEnvelope(
                type="request",
                message=message,
                data=data,
                requestId=request_id,
            )
            print(
                f"[server] -> device={self.device_id} requestId={request_id} "
                f"message={payload.message} data={payload.data}"
            )
            await self.websocket.send(payload.model_dump_json(exclude_none=True) + "\n")
            try:
                response = await asyncio.wait_for(self._pending_response, timeout=timeout)
            finally:
                self._pending_response = None

        print(
            f"[server] <- device={self.device_id} requestId={response.requestId} "
            f"message={response.message} data={response.data}"
        )
        if response.requestId != request_id:
            raise DeviceGatewayError(
                f"Expected response requestId {request_id}, got {response.requestId}."
            )

        if response.message == "error":
            error = ErrorData.model_validate(response.data)
            self._update_device_info(
                screenshot=error.screenshot,
                ui=error.ui,
                current_package=error.currentPackage,
                activity=error.activity,
            )
            raise DeviceGatewayError(error.message)

        if response.message != "actionResult":
            raise DeviceGatewayError(
                f"Expected 'actionResult', got {response.message!r}."
            )

        action_result = ActionResultData.model_validate(response.data)
        self._update_device_info(
            screenshot=action_result.screenshot,
            ui=action_result.ui,
            current_package=action_result.currentPackage,
            activity=action_result.activity,
        )
        return action_result.model_dump()

    async def _reader_loop(self) -> None:
        try:
            async for raw in self.websocket:
                envelope = self._parse(raw)
                if envelope.type == "request":
                    await self._handle_client_request(envelope)
                elif self._pending_response is not None and not self._pending_response.done():
                    self._pending_response.set_result(envelope)
        except ConnectionClosed:
            pass
        finally:
            self.closed.set()
            if self._pending_response is not None and not self._pending_response.done():
                self._pending_response.set_exception(
                    DeviceGatewayError(f"Device {self.device_id} disconnected.")
                )

    async def _handle_client_request(self, envelope: MessageEnvelope) -> None:
        if envelope.message == "connect":
            connect_data = ConnectData.model_validate(envelope.data)
            self.device_info = DeviceInfo(
                device_id=self.device_id,
                width=connect_data.width,
                height=connect_data.height,
                token=connect_data.token,
                screenshot=connect_data.screenshot,
                ui=connect_data.ui,
                current_package=connect_data.currentPackage,
                activity=connect_data.activity,
            )
            self._next_request_id = (envelope.requestId or 0) + 1
            self.ready.set()
            print(
                f"[server] device connected: deviceId={self.device_id} "
                f"size={connect_data.width}x{connect_data.height} "
                f"requestIdStart={self._next_request_id}"
            )
            return

        if envelope.message == "ping":
            if VERBOSE_HEARTBEAT:
                print(f"[server] <- heartbeat from device={self.device_id}")
            response = MessageEnvelope(type="response", message="pong", data=None)
            await self.websocket.send(response.model_dump_json(exclude_none=True) + "\n")
            return

        response = MessageEnvelope(
            type="response",
            message="error",
            data=ErrorData(message=f"Unsupported client request: {envelope.message}").model_dump(),
            requestId=envelope.requestId,
        )
        await self.websocket.send(response.model_dump_json(exclude_none=True) + "\n")

    def _consume_next_request_id(self) -> int:
        if self._next_request_id is None:
            raise DeviceGatewayError(
                f"Device {self.device_id} has not initialized the requestId sequence."
            )
        current = self._next_request_id
        self._next_request_id += 1
        return current

    def _update_device_info(
        self,
        screenshot: str | None,
        ui: str | None,
        current_package: str | None,
        activity: str | None,
    ) -> None:
        if self.device_info is None:
            return
        if screenshot is not None:
            self.device_info.screenshot = screenshot
        if ui is not None:
            self.device_info.ui = ui
        if current_package is not None:
            self.device_info.current_package = current_package
        if activity is not None:
            self.device_info.activity = activity

    @staticmethod
    def _parse(raw: str) -> MessageEnvelope:
        line = raw.strip()
        if not line:
            raise DeviceGatewayError("Received empty JSONL message.")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DeviceGatewayError(f"Invalid JSON message: {exc}") from exc
        return MessageEnvelope.model_validate(payload)


class DeviceGateway:
    def __init__(self, path_prefix: str = "/ws/devices") -> None:
        self.path_prefix = path_prefix.rstrip("/")
        self._devices: dict[str, ConnectedDeviceSession] = {}
        self._lock = asyncio.Lock()

    def get_device(self, device_id: str) -> ConnectedDeviceSession:
        session = self._devices.get(device_id)
        if session is None or session.closed.is_set():
            raise DeviceGatewayError(f"Device {device_id!r} is not connected.")
        return session

    def get_default_device(self) -> ConnectedDeviceSession:
        for session in self._devices.values():
            if not session.closed.is_set():
                return session
        raise DeviceGatewayError("No connected device is available.")

    async def wait_for_device(self, device_id: str, timeout: float = 15.0) -> ConnectedDeviceSession:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            try:
                return self.get_device(device_id)
            except DeviceGatewayError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise
                await asyncio.sleep(0.1)

    async def handler(self, websocket: ServerConnection) -> None:
        device_id = self._extract_device_id(websocket.request.path)
        session = ConnectedDeviceSession(websocket, device_id)
        await session.start()
        await session.wait_ready()
        async with self._lock:
            self._devices[device_id] = session
        try:
            await session.closed.wait()
        finally:
            async with self._lock:
                if self._devices.get(device_id) is session:
                    self._devices.pop(device_id, None)
            await session.stop()

    def _extract_device_id(self, path: str) -> str:
        prefix = f"{self.path_prefix}/"
        if not path.startswith(prefix):
            raise DeviceGatewayError(
                f"Invalid device path {path!r}. Expected prefix {self.path_prefix!r}."
            )
        device_id = path[len(prefix) :]
        if not device_id:
            raise DeviceGatewayError("Missing deviceId in websocket path.")
        return device_id
