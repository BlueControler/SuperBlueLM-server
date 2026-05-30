"""Runtime switching between cloud models and a local llama.cpp server."""

from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .json_types import JsonObject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REPO = "ggml-org/gemma-4-E2B-it-GGUF"
DEFAULT_MODEL_FILE = "gemma-4-E2B-it-Q8_0.gguf"
DEFAULT_MODEL_NAME = "gemma-4-E2B-it"


class LocalModelRuntimeError(RuntimeError):
    """Raised when the local model runtime cannot be started or configured."""


@dataclass(frozen=True)
class LocalModelConfig:
    server_binary: Path
    model_path: Path
    host: str
    port: int
    model_name: str
    context_size: int
    extra_args: tuple[str, ...]
    startup_timeout_seconds: float

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"


class LocalModelRuntime:
    """Owns local llama.cpp process state and model routing flags."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._network_connected = True
        self._process: subprocess.Popen[bytes] | None = None
        self._local_model: ChatOpenAI | None = None
        self._last_error: str | None = None
        self._config: LocalModelConfig | None = None

    def set_network_connected(self, connected: bool) -> JsonObject:
        with self._lock:
            self._network_connected = connected
            if connected:
                self._stop_local_server_locked()
                self._local_model = None
                self._last_error = None
                return self.status()

            try:
                config = self._load_config()
                self._ensure_local_server_locked(config)
                self._local_model = self._build_local_model(config)
                self._config = config
                self._last_error = None
            except Exception as exc:
                self._stop_local_server_locked()
                self._local_model = None
                self._last_error = str(exc)
                raise LocalModelRuntimeError(str(exc)) from exc

            return self.status()

    def get_model_override(self) -> ChatOpenAI | None:
        with self._lock:
            if self._network_connected:
                return None
            if self._local_model is None:
                raise LocalModelRuntimeError(
                    self._last_error or "local model is not ready for offline mode."
                )
            return self._local_model

    def status(self) -> JsonObject:
        with self._lock:
            process = self._process
            running = process is not None and process.poll() is None
            return {
                "networkConnected": self._network_connected,
                "mode": "cloud" if self._network_connected else "local",
                "localServerRunning": running,
                "localBaseUrl": self._config.base_url if self._config else None,
                "localModelName": self._config.model_name if self._config else None,
                "localModelPath": str(self._config.model_path) if self._config else None,
                "lastError": self._last_error,
            }

    def _load_config(self) -> LocalModelConfig:
        load_dotenv()
        server_binary = _resolve_server_binary(os.getenv("LLAMA_CPP_SERVER_BINARY"))
        model_path = _resolve_model_path(os.getenv("LLAMA_CPP_MODEL_PATH"))
        host = os.getenv("LLAMA_CPP_HOST", "127.0.0.1")
        port = int(os.getenv("LLAMA_CPP_PORT", "8080"))
        model_name = os.getenv("LLAMA_CPP_MODEL_NAME", DEFAULT_MODEL_NAME)
        context_size = int(os.getenv("LLAMA_CPP_CONTEXT_SIZE", "8192"))
        timeout = float(os.getenv("LLAMA_CPP_STARTUP_TIMEOUT", "120"))
        extra_args = tuple(_split_extra_args(os.getenv("LLAMA_CPP_SERVER_ARGS", "")))

        if not server_binary.exists():
            raise LocalModelRuntimeError(
                f"llama.cpp server binary not found: {server_binary}. "
                "Run `python -m scripts.setup llama:all` first, or set "
                "LLAMA_CPP_SERVER_BINARY."
            )
        if not model_path.exists():
            raise LocalModelRuntimeError(
                f"local model file not found: {model_path}. "
                "Run `python -m scripts.setup llama:all` first, or set "
                "LLAMA_CPP_MODEL_PATH."
            )

        return LocalModelConfig(
            server_binary=server_binary,
            model_path=model_path,
            host=host,
            port=port,
            model_name=model_name,
            context_size=context_size,
            extra_args=extra_args,
            startup_timeout_seconds=timeout,
        )

    def _ensure_local_server_locked(self, config: LocalModelConfig) -> None:
        if self._process is not None and self._process.poll() is None:
            if _server_healthy(config.health_url, timeout=1):
                self._config = config
                return
            self._stop_local_server_locked()

        if not _port_available(config.host, config.port):
            if _server_healthy(config.health_url, timeout=2):
                self._config = config
                return
            raise LocalModelRuntimeError(
                f"{config.host}:{config.port} is already in use, but llama.cpp health "
                "check did not pass."
            )

        command = [
            str(config.server_binary),
            "--host",
            config.host,
            "--port",
            str(config.port),
            "-m",
            str(config.model_path),
            "-c",
            str(config.context_size),
            "--jinja",
            *config.extra_args,
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(config.server_binary.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_until_healthy_locked(config)
        self._config = config

    def _wait_until_healthy_locked(self, config: LocalModelConfig) -> None:
        deadline = time.monotonic() + config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise LocalModelRuntimeError(
                    f"llama.cpp server exited early with code {self._process.returncode}."
                )
            if _server_healthy(config.health_url, timeout=2):
                return
            time.sleep(0.5)

        raise LocalModelRuntimeError(
            f"llama.cpp server did not become healthy within "
            f"{config.startup_timeout_seconds:g} seconds."
        )

    def _stop_local_server_locked(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def _build_local_model(self, config: LocalModelConfig) -> ChatOpenAI:
        max_tokens = int(os.getenv("LLAMA_CPP_MAX_TOKENS", "2048"))
        return ChatOpenAI(
            api_key=SecretStr(os.getenv("LLAMA_CPP_API_KEY", "sk-local")),
            base_url=config.base_url,
            model=config.model_name,
            max_tokens=max_tokens,  # type: ignore[call-arg]
        )


def build_cloud_model() -> ChatOpenAI | str:
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
    openai_base_url = os.getenv("OPENAI_BASE_URL") or None
    if openai_key:
        return ChatOpenAI(
            api_key=SecretStr(openai_key),
            base_url=openai_base_url,
            model=openai_model,
            max_tokens=openai_max_tokens,  # type: ignore[call-arg]
        )

    return "openai:gpt-5.4"


def build_phone_subagent_model(main_cloud_model: ChatOpenAI | str) -> ChatOpenAI | str:
    load_dotenv()
    model_name = os.getenv("PHONE_SUBAGENT_MODEL")
    if not model_name:
        return main_cloud_model

    api_key = os.getenv("PHONE_SUBAGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("PHONE_SUBAGENT_BASE_URL") or None
    max_tokens = int(os.getenv("PHONE_SUBAGENT_MAX_TOKENS", "2048"))
    if api_key or base_url:
        return ChatOpenAI(
            api_key=SecretStr(api_key or "sk-local"),
            base_url=base_url,
            model=model_name,
            max_tokens=max_tokens,  # type: ignore[call-arg]
        )

    return f"openai:{model_name}"


def _resolve_server_binary(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()

    executable_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    local_root = PROJECT_ROOT / ".local" / "llama.cpp"
    candidates = sorted(local_root.rglob(executable_name)) if local_root.exists() else []
    if candidates:
        return candidates[0].resolve()
    return (local_root / executable_name).resolve()


def _resolve_model_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return (PROJECT_ROOT / ".local" / "models" / DEFAULT_MODEL_FILE).resolve()


def _split_extra_args(value: str) -> list[str]:
    return [part for part in value.split() if part]


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) != 0


def _server_healthy(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


model_runtime = LocalModelRuntime()

__all__ = [
    "DEFAULT_MODEL_FILE",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_REPO",
    "LocalModelRuntimeError",
    "build_cloud_model",
    "build_phone_subagent_model",
    "model_runtime",
]
