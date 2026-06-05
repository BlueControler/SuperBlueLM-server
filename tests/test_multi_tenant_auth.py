from __future__ import annotations

import asyncio
from types import SimpleNamespace

from starlette.routing import Match

from mobile_agent import auth as auth_module
from mobile_agent.gateways.phone import DeviceGateway, DeviceGatewayError
from mobile_agent.gateways.system import SystemGatewayError, SystemToolGateway
from mobile_agent.http_app import app


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


def test_http_app_registers_device_scoped_websocket_routes() -> None:
    adb_scope = {"type": "websocket", "path": "/adb/device-1"}
    system_scope = {"type": "websocket", "path": "/system/device-1"}

    assert any(route.matches(adb_scope)[0] == Match.FULL for route in app.routes)
    assert any(route.matches(system_scope)[0] == Match.FULL for route in app.routes)
