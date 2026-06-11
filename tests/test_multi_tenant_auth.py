from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from starlette.routing import Match

from mobile_agent import auth as auth_module
from mobile_agent.gateways.phone import DeviceGateway, DeviceGatewayError
from mobile_agent.gateways.system import SystemGatewayError, SystemToolGateway
from mobile_agent.http_app import app


class _Request:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeWebSocket:
    def __init__(self, path: str) -> None:
        self.request = _Request(path)
        self.remote_address = ("test", 1)
        self.sent: list[str] = []
        self.close_calls: list[tuple[int, str | None]] = []
        self._connect_sent = False
        self._closed = asyncio.Event()

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.close_calls.append((code, reason))
        self._closed.set()

    def __aiter__(self) -> "_FakeWebSocket":
        return self

    async def __anext__(self) -> str:
        if not self._connect_sent:
            self._connect_sent = True
            return json.dumps(
                {
                    "type": "request",
                    "message": "connect",
                    "requestId": 1,
                    "data": {
                        "width": 1080,
                        "height": 2400,
                    },
                }
            )

        await self._closed.wait()
        raise StopAsyncIteration


class _FakeSystemWebSocket(_FakeWebSocket):
    async def __anext__(self) -> str:
        await self._closed.wait()
        raise StopAsyncIteration


def test_auth_identity_uses_x_api_key_without_storing_raw_secret() -> None:
    user = asyncio.run(
        auth_module.authenticate(
            headers={
                b"x-api-key": b"alice-secret",
            },
            authorization=None,
        )
    )

    assert user["identity"].startswith("api-key:")
    assert "alice-secret" not in user["identity"]


def test_auth_identity_falls_back_to_bearer_token() -> None:
    user = asyncio.run(
        auth_module.authenticate(
            headers={},
            authorization="Bearer bob-token",
        )
    )

    assert user["identity"].startswith("bearer:")


def test_thread_create_adds_owner_metadata() -> None:
    value = {"metadata": {"graph_id": "agent"}}
    ctx = SimpleNamespace(user=SimpleNamespace(identity="user-1"), action="create")

    result = asyncio.run(auth_module.authorize_threads(ctx, value))

    assert result == {"owner": "user-1"}
    assert value["metadata"] == {"graph_id": "agent", "owner": "user-1"}


def test_thread_read_filters_by_owner_without_overwriting_metadata() -> None:
    value = {"metadata": {"owner": "other"}}
    ctx = SimpleNamespace(user=SimpleNamespace(identity="user-1"), action="read")

    result = asyncio.run(auth_module.authorize_threads(ctx, value))

    assert result == {"owner": "user-1"}
    assert value["metadata"] == {"owner": "other"}


def test_device_gateway_accepts_plain_and_device_scoped_adb_paths() -> None:
    gateway = DeviceGateway()

    gateway._validate_path("/adb")
    gateway._validate_path("/adb/device-1")

    try:
        gateway._validate_path("/adb/device-1/extra")
    except DeviceGatewayError:
        pass
    else:
        raise AssertionError("nested device paths must be rejected")


def test_system_gateway_accepts_plain_and_device_scoped_paths() -> None:
    gateway = SystemToolGateway()

    assert gateway._normalized_client_path("/system") == "/system"
    assert gateway._normalized_client_path("/system/device-1") == "/system/device-1"

    try:
        gateway._normalized_client_path("/system/device-1/extra")
    except SystemGatewayError:
        pass
    else:
        raise AssertionError("nested system paths must be rejected")


async def _wait_for_system_client(
    gateway: SystemToolGateway,
    device_id: str,
) -> object:
    for _ in range(50):
        try:
            return gateway.get_default_client(device_id)
        except SystemGatewayError:
            await asyncio.sleep(0.01)
    raise AssertionError(f"system client {device_id!r} was not registered")


def test_system_gateway_tracks_multiple_device_ids() -> None:
    async def run() -> None:
        gateway = SystemToolGateway()
        first_socket = _FakeSystemWebSocket("/system/device-1")
        second_socket = _FakeSystemWebSocket("/system/device-2")
        first_task = asyncio.create_task(gateway.handler(first_socket))
        second_task = asyncio.create_task(gateway.handler(second_socket))

        first_client = await _wait_for_system_client(gateway, "device-1")
        second_client = await _wait_for_system_client(gateway, "device-2")

        assert first_client is not second_client
        assert gateway.get_default_client() is second_client

        await first_socket.close()
        await second_socket.close()
        await asyncio.gather(first_task, second_task)

    asyncio.run(run())


def test_http_app_registers_device_scoped_websocket_routes() -> None:
    adb_scope = {"type": "websocket", "path": "/adb/device-1"}
    system_scope = {"type": "websocket", "path": "/system/device-1"}

    assert any(route.matches(adb_scope)[0] == Match.FULL for route in app.routes)
    assert any(route.matches(system_scope)[0] == Match.FULL for route in app.routes)


async def _wait_for_session(gateway: DeviceGateway, device_id: str) -> object:
    for _ in range(50):
        try:
            return gateway.get_session(device_id)
        except DeviceGatewayError:
            await asyncio.sleep(0.01)
    raise AssertionError(f"session {device_id!r} was not registered")


async def _wait_for_replaced_session(
    gateway: DeviceGateway,
    device_id: str,
    old_session: object,
) -> object:
    for _ in range(50):
        try:
            session = gateway.get_session(device_id)
        except DeviceGatewayError:
            await asyncio.sleep(0.01)
            continue
        if session is not old_session:
            return session
        await asyncio.sleep(0.01)
    raise AssertionError(f"session {device_id!r} was not replaced")


def test_device_gateway_replaces_same_device_connection_without_asgi_error() -> None:
    async def run() -> None:
        gateway = DeviceGateway()
        first_socket = _FakeWebSocket("/adb/device-1")
        first_task = asyncio.create_task(gateway.handler(first_socket))
        first_session = await _wait_for_session(gateway, "device-1")

        second_socket = _FakeWebSocket("/adb/device-1")
        second_task = asyncio.create_task(gateway.handler(second_socket))
        second_session = await _wait_for_replaced_session(
            gateway,
            "device-1",
            first_session,
        )

        assert second_session is not first_session
        assert first_socket.close_calls == [
            (1000, "replaced by a new device connection")
        ]

        await second_socket.close()
        await asyncio.gather(first_task, second_task)

    asyncio.run(run())


def test_device_gateway_tracks_multiple_device_ids() -> None:
    async def run() -> None:
        gateway = DeviceGateway()
        first_socket = _FakeWebSocket("/adb/device-1")
        second_socket = _FakeWebSocket("/adb/device-2")
        first_task = asyncio.create_task(gateway.handler(first_socket))
        second_task = asyncio.create_task(gateway.handler(second_socket))

        first_session = await _wait_for_session(gateway, "device-1")
        second_session = await _wait_for_session(gateway, "device-2")

        assert first_session is not second_session
        assert gateway.get_session() is second_session

        await first_socket.close()
        await second_socket.close()
        await asyncio.gather(first_task, second_task)

    asyncio.run(run())
