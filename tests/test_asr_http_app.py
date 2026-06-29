from __future__ import annotations

from starlette.testclient import TestClient

from mobile_agent import http_app
from mobile_agent.asr.provider import AsrRequest, AsrTranscriptionResult
from mobile_agent.http_app import app


def test_transcribe_route_uploads_audio_to_provider(monkeypatch) -> None:
    calls: list[tuple[bytes, AsrRequest]] = []

    class FakeProvider:
        async def transcribe(self, audio: bytes, request: AsrRequest) -> AsrTranscriptionResult:
            calls.append((audio, request))
            return AsrTranscriptionResult(
                text="打开微信",
                provider="aliyun-nls",
                request_id="req-1",
                duration_ms=12,
            )

    monkeypatch.setattr(http_app, "asr_provider_factory", lambda: FakeProvider())

    response = TestClient(app).post(
        "/mobile/asr/transcribe",
        data={"format": "pcm", "sampleRate": "16000", "language": "zh-CN"},
        files={"audio": ("speech.pcm", b"\x01\x02\x03\x04", "audio/pcm")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "打开微信",
        "provider": "aliyun-nls",
        "requestId": "req-1",
        "durationMs": 12,
    }
    assert calls == [
        (
            b"\x01\x02\x03\x04",
            AsrRequest(audio_format="pcm", sample_rate=16000, language="zh-CN"),
        )
    ]


def test_transcribe_route_rejects_empty_audio(monkeypatch) -> None:
    class UnusedProvider:
        async def transcribe(self, audio: bytes, request: AsrRequest) -> AsrTranscriptionResult:
            raise AssertionError("empty audio must be rejected before provider call")

    monkeypatch.setattr(http_app, "asr_provider_factory", lambda: UnusedProvider())

    response = TestClient(app).post(
        "/mobile/asr/transcribe",
        data={"format": "pcm", "sampleRate": "16000"},
        files={"audio": ("empty.pcm", b"", "audio/pcm")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "empty_audio"
