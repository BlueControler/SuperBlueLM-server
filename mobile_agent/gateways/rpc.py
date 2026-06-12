from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from ..json_types import JsonValue

SocketMessage = str | bytes
GatewayErrorFactory = Callable[[str], Exception]


@dataclass
class RequestMetadata:
    path: str


class RequestInfo(Protocol):
    path: str


class JsonLineWebSocket(Protocol):
    request: RequestInfo | None
    remote_address: object

    async def send(self, message: str) -> None: ...

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...

    def __aiter__(self) -> AsyncIterator[SocketMessage]: ...


class JsonLineProtocolViolation(RuntimeError):
    pass


class JsonLineEnvelope(BaseModel):
    type: Literal["request", "response"]
    message: str
    data: object = None
    request_id: int | None = Field(default=None, alias="requestId", ge=1)

    model_config = {"populate_by_name": True}


class StarletteWebSocketConnection:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.request: RequestInfo | None = RequestMetadata(path=websocket.url.path)
        self.remote_address: object = websocket.client

    async def send(self, message: str) -> None:
        await self.websocket.send_text(message)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        await self.websocket.close(code=code, reason=reason)

    def __aiter__(self) -> StarletteWebSocketConnection:
        return self

    async def __anext__(self) -> str:
        try:
            return await self.websocket.receive_text()
        except WebSocketDisconnect:
            raise StopAsyncIteration from None


class JsonLineRpcSession:
    def __init__(
        self,
        websocket: JsonLineWebSocket,
        *,
        request_id_start: int,
        disconnect_error: GatewayErrorFactory,
    ) -> None:
        self.websocket = websocket
        self.closed = asyncio.Event()
        self._request_id_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._pending_responses: dict[int, asyncio.Future[JsonLineEnvelope]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._next_request_id = request_id_start
        self._disconnect_error = disconnect_error

    async def start(self) -> None:
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def stop(self) -> None:
        if self._reader_task is None:
            return

        self._reader_task.cancel()
        try:
            await self._reader_task
        except asyncio.CancelledError:
            pass

    async def send_rpc_request(
        self,
        message: str,
        data: JsonValue,
        timeout: float = 20.0,
    ) -> JsonLineEnvelope:
        if self.closed.is_set():
            raise self._disconnect_error("WebSocket client is disconnected.")

        loop = asyncio.get_running_loop()
        async with self._request_id_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            pending_response = loop.create_future()
            self._pending_responses[request_id] = pending_response
            envelope = JsonLineEnvelope(
                type="request",
                message=message,
                data=data,
                requestId=request_id,
            )

        try:
            try:
                await self._send_envelope(envelope)
            except (OSError, ConnectionClosed, RuntimeError) as exc:
                self.closed.set()
                logger.warning(f"JSONL WebSocket send failed: {exc!r}")
                raise self._disconnect_error("WebSocket client send failed.") from exc
            return await asyncio.wait_for(pending_response, timeout=timeout)
        finally:
            self._pending_responses.pop(request_id, None)

    async def send_response(
        self,
        message: str,
        data: JsonValue = None,
        request_id: int | None = None,
    ) -> None:
        await self._send_envelope(
            JsonLineEnvelope(
                type="response",
                message=message,
                data=data,
                requestId=request_id,
            )
        )

    async def _send_envelope(self, envelope: JsonLineEnvelope) -> None:
        async with self._send_lock:
            await self.websocket.send(
                envelope.model_dump_json(by_alias=True, exclude_none=True) + "\n"
            )

    async def _reader_loop(self) -> None:
        try:
            async for raw in self.websocket:
                for line in _json_lines(raw):
                    envelope = parse_envelope(line)
                    if envelope.type == "request":
                        await self._handle_client_request(envelope)
                    else:
                        self._handle_client_response(envelope)
        except JsonLineProtocolViolation as exc:
            logger.warning(f"Closing JSONL WebSocket after protocol violation: {exc}")
            await self.websocket.close(code=1008, reason=str(exc))
        except ConnectionClosed as exc:
            logger.info(f"JSONL WebSocket connection closed: {exc}")
        except Exception as exc:
            logger.warning(f"JSONL WebSocket reader stopped unexpectedly: {exc!r}")
        finally:
            self.closed.set()
            for future in list(self._pending_responses.values()):
                if not future.done():
                    future.set_exception(self._disconnect_error("WebSocket client disconnected."))
            self._pending_responses.clear()

    async def _handle_client_request(self, envelope: JsonLineEnvelope) -> None:
        if envelope.message != "ping":
            raise JsonLineProtocolViolation(f"Unsupported client request: {envelope.message!r}.")
        if envelope.request_id is not None:
            raise JsonLineProtocolViolation("ping must not carry requestId.")
        await self.send_response("pong")

    def _handle_client_response(self, envelope: JsonLineEnvelope) -> None:
        if envelope.request_id is None:
            raise JsonLineProtocolViolation("Business response must carry requestId.")

        pending_response = self._pending_responses.get(envelope.request_id)
        if pending_response is None or pending_response.done():
            logger.warning(
                f"Received unexpected response {envelope.message!r} "
                f"with requestId={envelope.request_id}; ignoring stale response."
            )
            return
        pending_response.set_result(envelope)


def parse_envelope(raw: str) -> JsonLineEnvelope:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JsonLineProtocolViolation(f"Invalid JSON message: {exc}") from exc

    try:
        return JsonLineEnvelope.model_validate(payload)
    except Exception as exc:
        raise JsonLineProtocolViolation(f"Invalid envelope: {exc}") from exc


def normalized_path(path: str) -> str:
    return path.split("?", 1)[0]


def _json_lines(raw: SocketMessage) -> list[str]:
    text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]
