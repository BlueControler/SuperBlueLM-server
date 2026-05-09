from __future__ import annotations

import asyncio
from dataclasses import dataclass

from starlette.websockets import WebSocket

from ..json_types import JsonValue, to_json_value
from .rpc import (
    JsonLineEnvelope,
    JsonLineProtocolViolation,
    JsonLineRpcSession,
    JsonLineWebSocket,
    StarletteWebSocketConnection,
    normalized_path,
)


class SystemGatewayError(RuntimeError):
    pass


class SystemProtocolViolation(SystemGatewayError, JsonLineProtocolViolation):
    pass


@dataclass(frozen=True)
class SystemClientInfo:
    path: str
    remote_address: object


class ConnectedSystemClient(JsonLineRpcSession):
    def __init__(self, websocket: JsonLineWebSocket, path: str) -> None:
        super().__init__(
            websocket,
            request_id_start=1,
            disconnect_error=lambda message: SystemGatewayError(
                message.replace("WebSocket client", "System tool client")
            ),
        )
        self.info = SystemClientInfo(path=path, remote_address=websocket.remote_address)

    async def send_request(
        self,
        message: str,
        data: JsonValue,
        timeout: float = 20.0,
    ) -> JsonValue:
        response = await self.send_rpc_request(message, data, timeout=timeout)
        if response.message != message:
            raise SystemGatewayError(
                f"Expected response message {message!r}, got {response.message!r}."
            )
        response_data = to_json_value(response.data)
        if isinstance(response_data, dict) and "error" in response_data:
            raise SystemGatewayError(str(response_data["error"]))
        return response_data

    async def _handle_client_request(self, envelope: JsonLineEnvelope) -> None:
        try:
            await super()._handle_client_request(envelope)
        except JsonLineProtocolViolation as exc:
            raise SystemProtocolViolation(
                f"Unsupported client request from system tool client: {envelope.message!r}."
            ) from exc


class SystemToolGateway:
    def __init__(self, path: str = "/system") -> None:
        self.path = path
        self._client: ConnectedSystemClient | None = None
        self._lock = asyncio.Lock()

    def get_default_client(self) -> ConnectedSystemClient:
        if self._client is None or self._client.closed.is_set():
            raise SystemGatewayError("No connected system tool client is available.")
        return self._client

    async def handler(self, websocket: JsonLineWebSocket) -> None:
        request = websocket.request
        path = normalized_path(request.path) if request is not None else ""
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

    async def starlette_handler(self, websocket: WebSocket) -> None:
        await websocket.accept()
        await self.handler(StarletteWebSocketConnection(websocket))
