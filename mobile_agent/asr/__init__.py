"""Speech-to-text provider integration."""

from .config import AliyunNlsConfig, AsrConfigError, load_aliyun_nls_config
from .provider import (
    AliyunNlsProvider,
    AsrProviderError,
    AsrRequest,
    AsrTranscriptionResult,
)

__all__ = [
    "AliyunNlsConfig",
    "AliyunNlsProvider",
    "AsrConfigError",
    "AsrProviderError",
    "AsrRequest",
    "AsrTranscriptionResult",
    "load_aliyun_nls_config",
]
