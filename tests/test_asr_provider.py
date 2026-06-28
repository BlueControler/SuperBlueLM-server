from __future__ import annotations

import json
import sys
import types

import pytest

from mobile_agent.asr.config import AliyunNlsConfig, AsrConfigError, load_aliyun_nls_config
from mobile_agent.asr.provider import AliyunNlsProvider, AsrProviderError, AsrRequest
import mobile_agent.asr.config as _cfg


def test_aliyun_nls_config_requires_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg._config_cache = None  # 重置缓存
    for key in (
        "ALIYUN_NLS_ACCESS_KEY_ID",
        "ALIYUN_NLS_ACCESS_KEY_SECRET",
        "ALIYUN_NLS_APP_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(AsrConfigError, match="ALIYUN_NLS_ACCESS_KEY_ID"):
        load_aliyun_nls_config()


def test_aliyun_nls_config_uses_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg._config_cache = None  # 重置缓存
    monkeypatch.setenv("ALIYUN_NLS_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("ALIYUN_NLS_ACCESS_KEY_SECRET", "secret")
    monkeypatch.setenv("ALIYUN_NLS_APP_KEY", "app")
    monkeypatch.delenv("ALIYUN_NLS_REGION", raising=False)
    monkeypatch.delenv("ALIYUN_NLS_GATEWAY_URL", raising=False)
    monkeypatch.delenv("ALIYUN_NLS_TOKEN_DOMAIN", raising=False)
    monkeypatch.delenv("ALIYUN_NLS_TOKEN_URL", raising=False)

    config = load_aliyun_nls_config()

    assert config.access_key_id == "ak"
    assert config.access_key_secret == "secret"
    assert config.app_key == "app"
    assert config.region == "cn-shanghai"
    assert config.gateway_url == "wss://nls-gateway.cn-shanghai.aliyuncs.com/ws/v1"
    assert config.token_domain == "cn-shanghai"
    assert config.token_url == "nls-meta.cn-shanghai.aliyuncs.com"
    assert config.default_sample_rate == 16000
    assert config.max_audio_bytes > 0


def test_aliyun_provider_streams_pcm_audio_and_extracts_final_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeRecognizer:
        def __init__(self, **kwargs: object) -> None:
            calls["recognizer_kwargs"] = kwargs
            self.on_start = kwargs.get("on_start")
            self.on_completed = kwargs["on_completed"]
            self.sent_audio: list[bytes] = []

        def start(self, **kwargs: object) -> None:
            calls["start_kwargs"] = kwargs
            if self.on_start is not None:
                self.on_start("started")

        def send_audio(self, data: bytes) -> None:
            self.sent_audio.append(data)

        def stop(self, timeout: int) -> None:
            calls["stop_timeout"] = timeout
            calls["sent_audio"] = self.sent_audio
            self.on_completed(
                json.dumps(
                    {
                        "header": {"task_id": "task-1"},
                        "payload": {"result": "打开微信"},
                    }
                )
            )

        def shutdown(self) -> None:
            calls["shutdown"] = True

    def fake_get_token(**kwargs: object) -> str:
        calls["token_kwargs"] = kwargs
        return "token-1"

    fake_nls = types.SimpleNamespace(
        token=types.SimpleNamespace(getToken=fake_get_token),
        NlsSpeechRecognizer=FakeRecognizer,
    )
    monkeypatch.setitem(sys.modules, "nls", fake_nls)
    monkeypatch.setattr("mobile_agent.asr.provider.time.sleep", lambda _: None)

    provider = AliyunNlsProvider(
        AliyunNlsConfig(
            access_key_id="ak",
            access_key_secret="secret",
            app_key="app",
            request_timeout_seconds=7,
        )
    )

    result = provider._transcribe_blocking(b"\x01" * 641, AsrRequest())

    assert result.text == "打开微信"
    assert result.provider == "aliyun-nls"
    assert result.request_id == "task-1"
    assert calls["token_kwargs"] == {
        "akid": "ak",
        "aksecret": "secret",
        "domain": "cn-shanghai",
        "version": "2019-02-28",
        "url": "nls-meta.cn-shanghai.aliyuncs.com",
    }
    assert calls["recognizer_kwargs"] == {
        "url": "wss://nls-gateway.cn-shanghai.aliyuncs.com/ws/v1",
        "token": "token-1",
        "appkey": "app",
        "on_start": calls["recognizer_kwargs"]["on_start"],
        "on_completed": calls["recognizer_kwargs"]["on_completed"],
        "on_error": calls["recognizer_kwargs"]["on_error"],
        "on_close": calls["recognizer_kwargs"]["on_close"],
    }
    assert calls["start_kwargs"] == {
        "aformat": "pcm",
        "sample_rate": 16000,
        "enable_intermediate_result": False,
        "enable_punctuation_prediction": True,
        "enable_inverse_text_normalization": True,
        "timeout": 7,
    }
    assert calls["sent_audio"] == [b"\x01" * 640, b"\x01"]
    assert calls["stop_timeout"] == 7
    assert calls["shutdown"] is True


def test_aliyun_provider_redacts_token_error_details(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_token(**kwargs: object) -> str:
        raise RuntimeError(
            "url=/?AccessKeyId=secret-ak&Signature=secret-signature&SignatureNonce=nonce"
        )

    fake_nls = types.SimpleNamespace(
        token=types.SimpleNamespace(getToken=fail_get_token),
        NlsSpeechRecognizer=object,
    )
    monkeypatch.setitem(sys.modules, "nls", fake_nls)
    provider = AliyunNlsProvider(
        AliyunNlsConfig(
            access_key_id="ak",
            access_key_secret="secret",
            app_key="app",
        )
    )

    with pytest.raises(AsrProviderError) as exc_info:
        provider._transcribe_blocking(b"\x01" * 640, AsrRequest())

    message = str(exc_info.value)
    assert "阿里云 NLS token 获取失败" in message
    assert "secret-ak" not in message
    assert "secret-signature" not in message
    assert "AccessKeyId=" not in message
    assert "Signature=" not in message
    assert exc_info.value.__cause__ is None
