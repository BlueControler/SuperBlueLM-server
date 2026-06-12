from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from starlette.routing import Match
from starlette.requests import Request

from mobile_agent import auth as auth_module
from mobile_agent import http_app
from mobile_agent.gateways.phone import DeviceGateway, DeviceGatewayError
from mobile_agent.gateways.system import SystemGatewayError, SystemToolGateway
from mobile_agent.http_app import app
from mobile_agent.gateways.phone import ConnectData


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


def test_connect_data_accepts_optional_screenshot_mime_type() -> None:
    payload = ConnectData.model_validate(
        {
            "width": 1080,
            "height": 2400,
            "screenshot": "base64",
            "screenshotMimeType": "image/webp",
        }
    )

    assert payload.screenshot_mime_type == "image/webp"


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
        try:
            gateway.get_default_client()
        except SystemGatewayError as exc:
            assert "device_id" in str(exc)
        else:
            raise AssertionError("multiple clients must require an explicit device_id")

        await first_socket.close()
        await second_socket.close()
        await asyncio.gather(first_task, second_task)

    asyncio.run(run())


def test_http_app_registers_device_scoped_websocket_routes() -> None:
    adb_scope = {"type": "websocket", "path": "/adb/device-1"}
    system_scope = {"type": "websocket", "path": "/system/device-1"}
    adb_status_scope = {"type": "http", "path": "/adb/device-1/status", "method": "GET"}
    system_status_scope = {
        "type": "http",
        "path": "/system/device-1/status",
        "method": "GET",
    }
    network_status_scope = {
        "type": "http",
        "path": "/network/device-1/status",
        "method": "GET",
    }

    assert any(route.matches(adb_scope)[0] == Match.FULL for route in app.routes)
    assert any(route.matches(system_scope)[0] == Match.FULL for route in app.routes)
    assert any(route.matches(adb_status_scope)[0] == Match.FULL for route in app.routes)
    assert any(route.matches(system_status_scope)[0] == Match.FULL for route in app.routes)
    assert any(route.matches(network_status_scope)[0] == Match.FULL for route in app.routes)


def test_adb_status_uses_device_id_from_path(monkeypatch: object) -> None:
    requested: list[str | None] = []
    session = SimpleNamespace(
        device_info=SimpleNamespace(
            width=1080,
            height=2400,
            current_package="com.example",
            activity=".MainActivity",
        )
    )

    def get_session(device_id: str | None = None) -> object:
        requested.append(device_id)
        return session

    monkeypatch.setattr(http_app.phone_gateway, "get_session", get_session)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/adb/device-uuid-1/status",
            "headers": [],
            "path_params": {"device_id": "device-uuid-1"},
        }
    )

    response = asyncio.run(http_app.adb_status(request))

    assert response.status_code == 200
    assert requested == ["device-uuid-1"]


def test_plain_adb_status_keeps_single_device_compatibility(monkeypatch: object) -> None:
    requested: list[str | None] = []
    session = SimpleNamespace(device_info=None)

    def get_session(device_id: str | None = None) -> object:
        requested.append(device_id)
        return session

    monkeypatch.setattr(http_app.phone_gateway, "get_session", get_session)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/adb/status",
            "headers": [],
            "path_params": {},
        }
    )

    response = asyncio.run(http_app.adb_status(request))

    assert response.status_code == 200
    assert requested == [None]


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
        try:
            gateway.get_session()
        except DeviceGatewayError as exc:
            assert "device_id" in str(exc)
        else:
            raise AssertionError("multiple devices must require an explicit device_id")

        await first_socket.close()
        await second_socket.close()
        await asyncio.gather(first_task, second_task)

    asyncio.run(run())


def test_device_gateway_waits_for_device_reconnect() -> None:
    async def run() -> None:
        gateway = DeviceGateway()
        waiting = asyncio.create_task(
            gateway.wait_for_session("device-1", timeout=0.5)
        )
        await asyncio.sleep(0)
        socket = _FakeWebSocket("/adb/device-1")
        handler = asyncio.create_task(gateway.handler(socket))

        session = await waiting

        assert session is gateway.get_session("device-1")
        await socket.close()
        await handler

    asyncio.run(run())


def test_device_gateway_wait_timeout_returns_device_not_connected_error() -> None:
    async def run() -> None:
        gateway = DeviceGateway()

        try:
            await gateway.wait_for_session("device-1", timeout=0.01)
        except DeviceGatewayError as exc:
            assert exc.args[0] == "device_not_connected"
        else:
            raise AssertionError("missing device must time out")

    asyncio.run(run())


def test_system_gateway_requires_device_id_when_multiple_clients_are_connected() -> None:
    async def run() -> None:
        gateway = SystemToolGateway()
        first_socket = _FakeSystemWebSocket("/system/device-1")
        second_socket = _FakeSystemWebSocket("/system/device-2")
        first_task = asyncio.create_task(gateway.handler(first_socket))
        second_task = asyncio.create_task(gateway.handler(second_socket))

        await _wait_for_system_client(gateway, "device-1")
        await _wait_for_system_client(gateway, "device-2")

        try:
            gateway.get_default_client()
        except SystemGatewayError as exc:
            assert "device_id" in str(exc)
        else:
            raise AssertionError("multiple clients must require an explicit device_id")

        await first_socket.close()
        await second_socket.close()
        await asyncio.gather(first_task, second_task)

    asyncio.run(run())
