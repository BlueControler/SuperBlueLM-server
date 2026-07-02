from __future__ import annotations

from typing import Any

import mobile_agent.agent.factory as factory
from mobile_agent.agent.middleware import (
    ModeToolAccessMiddleware,
    SyncPhoneStateMiddleware,
)
from mobile_agent.agent.medical_travel_sop import MedicalTravelSopMiddleware
from mobile_agent.agent.meeting_minutes_sop import MeetingMinutesSopMiddleware


class _Gateway:
    pass


def test_factory_wires_main_agent_to_raw_phone_tools(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}
    main_model = object()

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(factory, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(factory, "build_cloud_model", lambda: main_model)
    monkeypatch.setattr(factory, "create_system_tools", lambda gateway: [])
    monkeypatch.setattr(factory, "create_external_tools", lambda: [])

    factory.build_agent(_Gateway(), _Gateway())

    assert captured["model"] is main_model
    assert {tool.name for tool in captured["tools"]} >= {
        "observe",
        "tap",
        "type",
        "launch",
    }
    assert "execute_phone_todo" not in {tool.name for tool in captured["tools"]}
    assert any(
        item.__class__.__name__ == "ResetAgentRunStateMiddleware"
        for item in captured["middleware"]
    )
    assert any(
        isinstance(item, MeetingMinutesSopMiddleware)
        for item in captured["middleware"]
    )
    assert any(
        isinstance(item, MedicalTravelSopMiddleware)
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
