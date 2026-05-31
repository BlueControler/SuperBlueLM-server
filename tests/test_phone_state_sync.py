from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from mobile_agent.agent.middleware import SyncPhoneStateMiddleware


@dataclass
class _DeviceInfo:
    width: int = 1080
    height: int = 2400
    screenshot: str = "base64-screenshot"
    ui: str = "<node text='Search' />"
    current_package: str = "com.example"
    activity: str = ".MainActivity"


class _Session:
    def __init__(self) -> None:
        self.device_info = _DeviceInfo()


class _Gateway:
    def __init__(self) -> None:
        self.session = _Session()

    def get_session(self) -> _Session:
        return self.session


@dataclass(frozen=True)
class _ModelRequest:
    messages: list[Any]
    state: dict[str, Any]

    def override(self, **kwargs: Any) -> _ModelRequest:
        return _ModelRequest(
            messages=kwargs.get("messages", self.messages),
            state=kwargs.get("state", self.state),
        )


def test_sync_phone_state_refreshes_snapshot_when_ui_changes() -> None:
    gateway = _Gateway()
    middleware = SyncPhoneStateMiddleware(gateway)

    first = middleware.before_model({}, runtime=None)
    gateway.session.device_info.ui = "<node text='Results' />"
    second = middleware.before_model(first or {}, runtime=None)

    assert first is not None
    assert first["phone_snapshot"]["ui"] == "<node text='Search' />"
    assert second is not None
    assert second["phone_snapshot"]["ui"] == "<node text='Results' />"


def test_sync_phone_state_injects_latest_ui_tree_and_screenshot_into_model() -> None:
    gateway = _Gateway()
    middleware = SyncPhoneStateMiddleware(gateway)
    captured: list[Any] = []

    middleware.wrap_model_call(
        _ModelRequest(messages=[HumanMessage(content="next")], state={}),
        lambda request: captured.extend(request.messages),
    )

    phone_state = captured[-1]
    assert isinstance(phone_state, HumanMessage)
    assert "ui=<node text='Search' />" in phone_state.content[0]["text"]
    assert phone_state.content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,base64-screenshot"},
    }
