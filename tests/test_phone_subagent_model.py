from __future__ import annotations

import logging
from typing import Any

import mobile_agent.local_model_runtime as runtime


def _clear_phone_model_env(monkeypatch: Any) -> None:
    for name in [
        "PHONE_SUBAGENT_MODEL",
        "PHONE_SUBAGENT_BASE_URL",
        "PHONE_SUBAGENT_API_KEY",
        "PHONE_SUBAGENT_MAX_TOKENS",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_phone_subagent_model_reuses_main_model_when_not_configured(
    monkeypatch: Any,
) -> None:
    _clear_phone_model_env(monkeypatch)
    main_model = object()

    assert runtime.build_phone_subagent_model(main_model) is main_model


def test_phone_subagent_model_warns_when_reusing_main_model(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    _clear_phone_model_env(monkeypatch)
    caplog.set_level(logging.WARNING, logger=runtime.__name__)

    runtime.build_phone_subagent_model(object())

    assert "PHONE_SUBAGENT_MODEL is not configured" in caplog.text


def test_phone_subagent_model_uses_independent_openai_compatible_config(
    monkeypatch: Any,
) -> None:
    _clear_phone_model_env(monkeypatch)
    captured: dict[str, Any] = {}

    def fake_chat_openai(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runtime, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setenv("PHONE_SUBAGENT_MODEL", "phone-small")
    monkeypatch.setenv("PHONE_SUBAGENT_BASE_URL", "http://phone-model.example/v1")
    monkeypatch.setenv("PHONE_SUBAGENT_API_KEY", "phone-secret")
    monkeypatch.setenv("PHONE_SUBAGENT_MAX_TOKENS", "1536")

    runtime.build_phone_subagent_model("openai:gpt-5.4")

    assert captured["model"] == "phone-small"
    assert captured["base_url"] == "http://phone-model.example/v1"
    assert captured["max_tokens"] == 1536
    assert captured["api_key"].get_secret_value() == "phone-secret"


def test_phone_subagent_model_reuses_main_agent_api_key(
    monkeypatch: Any,
) -> None:
    _clear_phone_model_env(monkeypatch)
    captured: dict[str, Any] = {}

    def fake_chat_openai(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runtime, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setenv("PHONE_SUBAGENT_MODEL", "gpt-5-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "main-agent-secret")

    runtime.build_phone_subagent_model("openai:gpt-5.4")

    assert captured["model"] == "gpt-5-mini"
    assert captured["api_key"].get_secret_value() == "main-agent-secret"


def test_phone_subagent_model_uses_provider_model_string_without_client_config(
    monkeypatch: Any,
) -> None:
    _clear_phone_model_env(monkeypatch)
    monkeypatch.setattr(runtime, "load_dotenv", lambda: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PHONE_SUBAGENT_MODEL", "gpt-5-mini")

    assert (
        runtime.build_phone_subagent_model("openai:gpt-5.4")
        == "openai:gpt-5-mini"
    )
