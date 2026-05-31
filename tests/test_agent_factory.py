from __future__ import annotations

from typing import Any

import mobile_agent.agent.factory as factory
from mobile_agent.agent.middleware import (
    ModeToolAccessMiddleware,
    SyncPhoneStateMiddleware,
)
from mobile_agent.agent.phone_delegation import ResetPhoneTodoMiddleware


class _Gateway:
    pass


def test_factory_wires_main_agent_to_restricted_phone_subagent(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}
    main_model = object()
    phone_model = object()

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(factory, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(factory, "build_cloud_model", lambda: main_model)
    monkeypatch.setattr(
        factory,
        "build_phone_subagent_model",
        lambda model: phone_model,
    )
    monkeypatch.setattr(factory, "create_system_tools", lambda gateway: [])
    monkeypatch.setattr(factory, "create_external_tools", lambda: [])

    factory.build_agent(_Gateway(), _Gateway())

    assert captured["model"] is main_model
    assert {tool.name for tool in captured["tools"]} >= {
        "observe",
        "tap",
        "type",
        "launch",
        "execute_phone_todo",
    }
    assert any(
        isinstance(item, ResetPhoneTodoMiddleware)
        for item in captured["middleware"]
    )
    assert any(
        isinstance(item, ModeToolAccessMiddleware)
        for item in captured["middleware"]
    )
    assert any(
        isinstance(item, SyncPhoneStateMiddleware)
        for item in captured["middleware"]
    )
