from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import re
import threading
import time
from typing import Any

from .config import AliyunNlsConfig, load_aliyun_nls_config

_logger = logging.getLogger(__name__)


class AsrProviderError(RuntimeError):
    """Raised when the speech-to-text provider cannot return usable text."""


@dataclass(frozen=True)
class AsrRequest:
    audio_format: str = "pcm"
    sample_rate: int = 16000
    language: str = "zh-CN"


@dataclass(frozen=True)
class AsrTranscriptionResult:
    text: str
    provider: str
    request_id: str | None = None
    duration_ms: int | None = None


class AliyunNlsProvider:
    provider_name = "aliyun-nls"

    def __init__(self, config: AliyunNlsConfig | None = None) -> None:
        self.config = config or load_aliyun_nls_config()

    async def transcribe(self, audio: bytes, request: AsrRequest) -> AsrTranscriptionResult:
        if not audio:
            raise AsrProviderError("音频内容为空")
        return await asyncio.to_thread(self._transcribe_blocking, audio, request)

    def _transcribe_blocking(self, audio: bytes, request: AsrRequest) -> AsrTranscriptionResult:
        """同步转写（运行在 asyncio.to_thread 线程中）。

        关键时序：
        1. start() 发起 WebSocket 连接（异步）→ 等待 on_start
        2. on_start 后 → 逐块 send_audio()
        3. stop() 通知 SDK 音频结束 → 等待 on_completed / on_error
        4. 任意阶段 on_error → 立即终止
        """
        nls = _load_nls_module()
        token = _get_nls_token(nls, self.config)
        timeout = self.config.request_timeout_seconds

        _logger.info(
            "nls_transcribe_start token_len=%d appkey=%s gateway=%s timeout=%d audio_bytes=%d",
            len(token), _mask(self.config.app_key), self.config.gateway_url, timeout, len(audio),
        )

        # ---- 线程安全状态 ----
        lock = threading.Lock()
        started_event = threading.Event()
        completed_event = threading.Event()
        completed_messages: list[Any] = []
        error_messages: list[Any] = []
        close_messages: list[Any] = []

        def _record_error(message: Any) -> None:
            with lock:
                error_messages.append(message)
            _logger.warning("nls_on_error message=%s", str(message)[:500])
            started_event.set()    # 解除 start 等待
            completed_event.set()  # 解除 complete 等待

        def on_start(*args: Any) -> None:
            msg = args[0] if args else None
            _logger.info("nls_on_start message=%s", str(msg)[:200] if msg else "none")
            started_event.set()

        def on_completed(*args: Any) -> None:
            msg = args[0] if args else None
            _logger.info("nls_on_completed message=%s", str(msg)[:200] if msg else "none")
            with lock:
                if msg is not None:
                    completed_messages.append(msg)
            completed_event.set()

        def on_error(*args: Any) -> None:
            msg = args[0] if args else None
            _record_error(msg)

        def on_close(*args: Any) -> None:
            msg = args[0] if args else None
            with lock:
                if msg is not None:
                    close_messages.append(msg)
            _logger.info("nls_on_close message=%s", str(msg)[:200] if msg else "none")

        recognizer = nls.NlsSpeechRecognizer(
            url=self.config.gateway_url,
            token=token,
            appkey=self.config.app_key,
            on_start=on_start,
            on_completed=on_completed,
            on_error=on_error,
            on_close=on_close,
        )
        _logger.info("nls_recognizer_created")

        started_at = time.monotonic()
        recognizer_started = False
        try:
            _logger.info("nls_start_calling")
            recognizer.start(
                aformat=request.audio_format,
                sample_rate=request.sample_rate,
                enable_intermediate_result=False,
                enable_punctuation_prediction=True,
                enable_inverse_text_normalization=True,
                timeout=timeout,
            )
            recognizer_started = True
            _logger.info("nls_start_returned waiting_on_start")

            # --- 阶段 1：等待 WebSocket 连接就绪 ---
            if not started_event.wait(timeout=timeout):
                with lock:
                    err_snap = list(error_messages)
                    close_snap = list(close_messages)
                _logger.error(
                    "nls_on_start_timeout timeout=%d errors=%d close=%d err_detail=%s close_detail=%s",
                    timeout, len(err_snap), len(close_snap),
                    str(err_snap[-1])[:500] if err_snap else "none",
                    str(close_snap[-1])[:500] if close_snap else "none",
                )
                raise AsrProviderError("阿里云语音识别连接超时（未收到 on_start）")
            with lock:
                if error_messages:
                    raise AsrProviderError(f"阿里云语音识别连接失败：{_format_error(error_messages[-1])}")

            # --- 阶段 2：发送音频 ---
            chunks = _audio_chunks(audio, request.sample_rate, request.audio_format)
            for chunk in chunks:
                with lock:
                    if error_messages:
                        break
                recognizer.send_audio(chunk.data)
                if chunk.delay_seconds > 0:
                    time.sleep(chunk.delay_seconds)

            # --- 阶段 3：结束发送，等待识别结果 ---
            with lock:
                send_error = bool(error_messages)
            if not send_error:
                recognizer.stop(timeout=timeout)
                if not completed_event.wait(timeout=timeout):
                    raise AsrProviderError("阿里云语音识别等待结果超时")

        except AsrProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK raises custom runtime exceptions.
            raise AsrProviderError(f"阿里云语音识别失败：{exc}") from exc
        finally:
            self._shutdown_recognizer(recognizer, recognizer_started)

        # --- 结果处理 ---
        with lock:
            errors_snapshot = list(error_messages)
            completed_snapshot = list(completed_messages)

        if errors_snapshot:
            raise AsrProviderError(f"阿里云语音识别失败：{_format_error(errors_snapshot[-1])}")
        if not completed_snapshot:
            raise AsrProviderError("阿里云语音识别未返回结果")

        message = completed_snapshot[-1]
        text = _extract_text(message)
        if not text:
            raise AsrProviderError("阿里云语音识别结果为空")

        return AsrTranscriptionResult(
            text=text,
            provider=self.provider_name,
            request_id=_extract_request_id(message),
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )

    @staticmethod
    def _shutdown_recognizer(recognizer: Any, was_started: bool) -> None:
        if not was_started:
            return
        shutdown = getattr(recognizer, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass


@dataclass(frozen=True)
class _AudioChunk:
    data: bytes
    delay_seconds: float


def _load_nls_module() -> Any:
    try:
        import nls  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise AsrProviderError("缺少阿里云 NLS Python SDK，请安装 alibabacloud-nls-python-sdk") from exc
    return nls


def _get_nls_token(nls: Any, config: AliyunNlsConfig) -> str:
    token_module = getattr(nls, "token", None)
    get_token = getattr(token_module, "getToken", None) or getattr(nls, "getToken", None)
    if not callable(get_token):
        raise AsrProviderError("阿里云 NLS SDK 缺少 token.getToken")
    try:
        token = _token_value(
            get_token(
                akid=config.access_key_id,
                aksecret=config.access_key_secret,
                domain=config.token_domain,
                version="2019-02-28",
                url=config.token_url,
            )
        )
    except Exception as exc:  # noqa: BLE001 - SDK raises custom token exceptions.
        message = _safe_exception_message(exc, config.access_key_id, config.access_key_secret)
        raise AsrProviderError(f"阿里云 NLS token 获取失败：{message}") from None
    if not token:
        raise AsrProviderError("阿里云 NLS token 获取失败")
    return token


def _format_error(message: Any) -> str:
    """将 NLS SDK 错误消息格式化为可读字符串。"""
    if isinstance(message, str):
        return message
    parsed = _parse_message(message)
    if isinstance(parsed, dict):
        msg = parsed.get("message") or parsed.get("status_text") or ""
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    return str(message)


def _mask(value: str) -> str:
    """脱敏：只保留首尾各两位。"""
    if len(value) <= 4:
        return "***"
    return value[:2] + "***" + value[-2:]


def _safe_exception_message(exc: Exception, *secrets: str) -> str:
    message = str(exc)
    message = re.sub(
        r"(?i)([?&])(?:AccessKeyId|AccessKeySecret|Signature|SignatureNonce|SecurityToken)=[^&\s)]+",
        r"\1<redacted>",
        message,
    )
    message = re.sub(
        r"(?i)\b(?:AccessKeyId|AccessKeySecret|Signature|SignatureNonce|SecurityToken)=[^&\s)]+",
        "<redacted>",
        message,
    )
    for secret in secrets:
        if secret and len(secret) >= 4:
            message = message.replace(secret, "<redacted>")
    return message


def _token_value(raw_token: Any) -> str:
    if isinstance(raw_token, str):
        return raw_token.strip()
    if isinstance(raw_token, dict):
        for key in ("token", "Token", "id", "Id"):
            value = raw_token.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    get_id = getattr(raw_token, "getId", None)
    if callable(get_id):
        value = get_id()
        if isinstance(value, str):
            return value.strip()
    for attr in ("token", "id"):
        value = getattr(raw_token, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _audio_chunks(audio: bytes, sample_rate: int, audio_format: str) -> list[_AudioChunk]:
    chunk_size = 640 if audio_format.lower() == "pcm" and sample_rate == 16000 else 4096
    delay = 0.02 if audio_format.lower() == "pcm" and sample_rate == 16000 else 0.0
    return [
        _AudioChunk(data=audio[index : index + chunk_size], delay_seconds=delay)
        for index in range(0, len(audio), chunk_size)
    ]


def _extract_text(message: Any) -> str:
    parsed = _parse_message(message)
    if isinstance(parsed, dict):
        for path in (
            ("payload", "result"),
            ("payload", "text"),
            ("payload", "result", "text"),
            ("result",),
            ("text",),
        ):
            value = _nested_get(parsed, path)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return message.strip() if isinstance(message, str) and not message.lstrip().startswith("{") else ""


def _extract_request_id(message: Any) -> str | None:
    parsed = _parse_message(message)
    if isinstance(parsed, dict):
        for path in (
            ("header", "task_id"),
            ("header", "namespace"),
            ("request_id",),
            ("requestId",),
        ):
            value = _nested_get(parsed, path)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _parse_message(message: Any) -> Any:
    if isinstance(message, dict):
        return message
    if not isinstance(message, str):
        return None
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        return None


def _nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current
