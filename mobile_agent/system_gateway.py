from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed


class SystemGatewayError(RuntimeError):
    pass


class SystemProtocolViolation(SystemGatewayError):
    pass


class SystemMessageEnvelope(BaseModel):
    type: Literal["request", "response"]
    message: str
    data: Any = None
    requestId: int | None = Field(default=None, ge=1)


@dataclass
class SystemClientInfo:
    path: str
    remote_address: Any


class ConnectedSystemClient:
    def __init__(self, websocket: ServerConnection, path: str) -> None:
        self.websocket = websocket
        self.info = SystemClientInfo(path=path, remote_address=websocket.remote_address)
        self.closed = asyncio.Event()
        self._request_id_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._pending_responses: dict[int, asyncio.Future[SystemMessageEnvelope]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._next_request_id = 1

    async def start(self) -> None:
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def stop(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

    async def send_request(
        self,
        message: str,
        data: Any,
        timeout: float = 20.0,
    ) -> Any:
        if self.closed.is_set():
            raise SystemGatewayError("System tool client is disconnected.")

        loop = asyncio.get_running_loop()
        async with self._request_id_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            pending_response = loop.create_future()
            self._pending_responses[request_id] = pending_response
            envelope = SystemMessageEnvelope(
                type="request",
                message=message,
                data=data,
                requestId=request_id,
            )

        async with self._send_lock:
            await self.websocket.send(envelope.model_dump_json(exclude_none=True) + "\n")

        try:
            response = await asyncio.wait_for(pending_response, timeout=timeout)
        finally:
            self._pending_responses.pop(request_id, None)

        if response.message != message:
            raise SystemGatewayError(
                f"Expected response message {message!r}, got {response.message!r}."
            )
        if isinstance(response.data, dict) and "error" in response.data:
            raise SystemGatewayError(str(response.data["error"]))
        return response.data

    async def _reader_loop(self) -> None:
        try:
            async for raw in self.websocket:
                text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    envelope = self._parse(line)
                    if envelope.type == "request":
                        await self._handle_client_request(envelope)
                    else:
                        self._handle_client_response(envelope)
        except SystemProtocolViolation as exc:
            await self.websocket.close(code=1008, reason=str(exc))
        except ConnectionClosed:
            pass
        finally:
            self.closed.set()
            for future in list(self._pending_responses.values()):
                if not future.done():
                    future.set_exception(SystemGatewayError("System tool client disconnected."))
            self._pending_responses.clear()

    async def _handle_client_request(self, envelope: SystemMessageEnvelope) -> None:
        if envelope.message != "ping":
            raise SystemProtocolViolation(
                f"Unsupported client request from system tool client: {envelope.message!r}."
            )
        if envelope.requestId is not None:
            raise SystemProtocolViolation("ping must not carry requestId.")

        response = SystemMessageEnvelope(type="response", message="pong", data=None)
        await self.websocket.send(response.model_dump_json(exclude_none=True) + "\n")

    def _handle_client_response(self, envelope: SystemMessageEnvelope) -> None:
        if envelope.requestId is None:
            raise SystemProtocolViolation("Business response must carry requestId.")
        pending_response = self._pending_responses.get(envelope.requestId)
        if pending_response is None or pending_response.done():
            raise SystemProtocolViolation(
                f"Received unexpected response {envelope.message!r} "
                f"with requestId={envelope.requestId}."
            )
        pending_response.set_result(envelope)

    @staticmethod
    def _parse(raw: str) -> SystemMessageEnvelope:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemProtocolViolation(f"Invalid JSON message: {exc}") from exc
        try:
            return SystemMessageEnvelope.model_validate(payload)
        except Exception as exc:
            raise SystemProtocolViolation(f"Invalid envelope: {exc}") from exc


class SystemToolGateway:
    def __init__(self, path: str = "/system") -> None:
        self.path = path
        self._client: ConnectedSystemClient | None = None
        self._lock = asyncio.Lock()

    def get_default_client(self) -> ConnectedSystemClient:
        if self._client is None or self._client.closed.is_set():
            raise SystemGatewayError("No connected system tool client is available.")
        return self._client

    async def handler(self, websocket: ServerConnection) -> None:
        request = websocket.request
        path = request.path.split("?", 1)[0] if request is not None else ""
        if path != self.path:
            raise SystemGatewayError(f"Invalid system tool path {path!r}. Expected {self.path!r}.")

        client = ConnectedSystemClient(websocket, path=path)
        await client.start()
        async with self._lock:
            old_client = self._client
            self._client = client
        if old_client is not None and not old_client.closed.is_set():
            await old_client.websocket.close(code=1000, reason="replaced by a new system client")

        try:
            await client.closed.wait()
        finally:
            async with self._lock:
                if self._client is client:
                    self._client = None
            await client.stop()
