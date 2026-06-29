from __future__ import annotations

from dataclasses import dataclass
import os


DEFAULT_GATEWAY_URL = "wss://nls-gateway.cn-shanghai.aliyuncs.com/ws/v1"
DEFAULT_TOKEN_URL = "nls-meta.cn-shanghai.aliyuncs.com"
DEFAULT_REGION = "cn-shanghai"
DEFAULT_MAX_AUDIO_BYTES = 5 * 1024 * 1024
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TIMEOUT_SECONDS = 30


class AsrConfigError(RuntimeError):
    """Raised when speech-to-text provider configuration is incomplete."""


@dataclass(frozen=True)
class AliyunNlsConfig:
    access_key_id: str
    access_key_secret: str
    app_key: str
    region: str = DEFAULT_REGION
    gateway_url: str = DEFAULT_GATEWAY_URL
    token_domain: str = DEFAULT_REGION
    token_url: str = DEFAULT_TOKEN_URL
    default_sample_rate: int = DEFAULT_SAMPLE_RATE
    max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES
    request_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


_config_cache: AliyunNlsConfig | None = None
_config_keys = (
    "ALIYUN_NLS_ACCESS_KEY_ID",
    "ALIYUN_NLS_ACCESS_KEY_SECRET",
    "ALIYUN_NLS_APP_KEY",
    "ALIYUN_NLS_REGION",
    "ALIYUN_NLS_GATEWAY_URL",
    "ALIYUN_NLS_TOKEN_DOMAIN",
    "ALIYUN_NLS_TOKEN_URL",
    "ALIYUN_NLS_DEFAULT_SAMPLE_RATE",
    "ALIYUN_NLS_MAX_AUDIO_BYTES",
    "ALIYUN_NLS_TIMEOUT_SECONDS",
)


def _config_fingerprint() -> str:
    """返回所有 NLS 配置项拼接后的指纹，用于检测 .env 变更。"""
    return "|".join(_env(k) for k in _config_keys)


def load_aliyun_nls_config() -> AliyunNlsConfig:
    """读取阿里云 NLS 配置（仅从 os.environ，不含 I/O）。

    .env 文件由 langgraph 在启动阶段加载。缓存会在环境变量变更时
    自动失效，因此 .env 修改 + 进程重启即可生效。
    """
    global _config_cache
    fp = _config_fingerprint()
    if _config_cache is not None:
        # 检查缓存是否仍然有效（关键环境变量未变）
        if _config_cache.access_key_id == _env("ALIYUN_NLS_ACCESS_KEY_ID") and \
           _config_cache.access_key_secret == _env("ALIYUN_NLS_ACCESS_KEY_SECRET") and \
           _config_cache.app_key == _env("ALIYUN_NLS_APP_KEY") and \
           _config_cache.gateway_url == _env("ALIYUN_NLS_GATEWAY_URL", DEFAULT_GATEWAY_URL):
            return _config_cache

    missing = [
        key
        for key in (
            "ALIYUN_NLS_ACCESS_KEY_ID",
            "ALIYUN_NLS_ACCESS_KEY_SECRET",
            "ALIYUN_NLS_APP_KEY",
        )
        if not _env(key)
    ]
    if missing:
        raise AsrConfigError("缺少阿里云 NLS 配置：" + ", ".join(missing))

    _config_cache = AliyunNlsConfig(
        access_key_id=_env("ALIYUN_NLS_ACCESS_KEY_ID"),
        access_key_secret=_env("ALIYUN_NLS_ACCESS_KEY_SECRET"),
        app_key=_env("ALIYUN_NLS_APP_KEY"),
        region=_env("ALIYUN_NLS_REGION", DEFAULT_REGION),
        gateway_url=_env("ALIYUN_NLS_GATEWAY_URL", DEFAULT_GATEWAY_URL),
        token_domain=_env("ALIYUN_NLS_TOKEN_DOMAIN", DEFAULT_REGION),
        token_url=_env("ALIYUN_NLS_TOKEN_URL", DEFAULT_TOKEN_URL),
        default_sample_rate=_env_int("ALIYUN_NLS_DEFAULT_SAMPLE_RATE", DEFAULT_SAMPLE_RATE),
        max_audio_bytes=_env_int("ALIYUN_NLS_MAX_AUDIO_BYTES", DEFAULT_MAX_AUDIO_BYTES),
        request_timeout_seconds=_env_int("ALIYUN_NLS_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
    )
    return _config_cache


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise AsrConfigError(f"{key} 必须是整数") from exc
    if parsed <= 0:
        raise AsrConfigError(f"{key} 必须大于 0")
    return parsed
