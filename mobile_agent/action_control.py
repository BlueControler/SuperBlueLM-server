"""Run-scoped, idempotent dispatch control for phone side effects.

The WebSocket request id is a transport correlation id only.  It cannot tell a
device whether a retry belongs to an active task, so every device command also
carries a stable run/action identity managed here.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock
from time import monotonic, time
from typing import Callable, Generator, Literal, Mapping, Protocol
from uuid import uuid4

from loguru import logger

from .json_types import JsonObject, JsonValue
from .trace import current_context

RunStatus = Literal["active", "cancelled", "terminal"]
ActionStatus = Literal["queued", "succeeded", "failed", "cancelled"]


class PhoneActionControlError(RuntimeError):
    """Base class for a locally rejected phone action."""


class PhoneActionCancelledError(PhoneActionControlError):
    pass


class PhoneActionTerminalError(PhoneActionControlError):
    pass


class PhoneActionUnregisteredRunError(PhoneActionControlError):
    pass


class PhoneActionLimitError(PhoneActionControlError):
    pass


class PhoneDeviceBusyError(PhoneActionControlError):
    pass


class PhoneCommandSession(Protocol):
    async def send_command(self, message: str, data: JsonValue) -> JsonObject: ...


@dataclass
class PhoneAction:
    action_id: str
    index: int
    source_id: str
    command: str
    fingerprint: str
    status: ActionStatus = "queued"


@dataclass
class PhoneRun:
    run_id: str
    thread_id: str | None
    status: RunStatus = "active"
    device_id: str | None = None
    cancellation_reason: str | None = None
    cancel_source: str | None = None
    terminal_reason: str | None = None
    next_action_index: int = 1
    actions: list[PhoneAction] = field(default_factory=list)
    cancel_stream: Callable[[], None] | None = None
    backend_run_id: str | None = None
    backend_status: str = "not_started"
    last_accessed_at: float = field(default_factory=monotonic)


class PhoneActionRegistry:
    """Small in-process ledger that blocks stale and runaway phone commands."""

    _FINISHED_RUN_TTL_SECONDS = 600.0
    _IDLE_ACTIVE_RUN_TTL_SECONDS = 900.0

    def __init__(
        self,
        *,
        max_actions_per_run: int | None = None,
        identical_action_limit: int | None = None,
        action_timeout_seconds: int | None = None,
        persist_path: Path | None = None,
    ) -> None:
        self.max_actions_per_run = (
            _positive_env("PHONE_MAX_ACTIONS_PER_RUN", 16)
            if max_actions_per_run is None
            else max(max_actions_per_run, 1)
        )
        self.identical_action_limit = (
            _positive_env("PHONE_MAX_IDENTICAL_ACTIONS_PER_RUN", 3)
            if identical_action_limit is None
            else max(identical_action_limit, 1)
        )
        self.action_timeout_seconds = (
            _positive_env("PHONE_ACTION_TIMEOUT_SECONDS", 60)
            if action_timeout_seconds is None
            else max(action_timeout_seconds, 1)
        )
        self._lock = RLock()
        self._persist_path = persist_path
        self._runs: dict[str, PhoneRun] = self._load_persisted_runs()

    def start_run(self, run_id: str, thread_id: str | None) -> None:
        with self._lock:
            self._prune_stale_locked()
            existing = self._runs.get(run_id)
            if existing is None:
                self._runs[run_id] = PhoneRun(run_id=run_id, thread_id=thread_id)
                logger.info("phone_run_created run_id={} thread_id={}", run_id, thread_id)
                self._persist_locked()
            else:
                # 复用已有 run 时必须重置为 active——之前的请求可能已将其
                # 标记为 cancelled / terminal。
                if existing.status != "active":
                    logger.info(
                        "phone_run_reactivated run_id={} thread_id={} previous_status={}",
                        run_id, thread_id, existing.status,
                    )
                    existing.status = "active"
                    existing.cancellation_reason = None
                    existing.cancel_source = None
                    existing.terminal_reason = None
                    existing.cancel_stream = None
                if existing.thread_id is None and thread_id is not None:
                    existing.thread_id = thread_id
                existing.last_accessed_at = monotonic()
                self._persist_locked()

    def has_registered_run(self, run_id: str) -> bool:
        """Only a mobile run registered by this server instance may act."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                run.last_accessed_at = monotonic()
            return run is not None

    def reserve_action(
        self,
        *,
        run_id: str,
        thread_id: str | None,
        source_id: str,
        command: str,
        payload: Mapping[str, object],
        device_id: str | None,
    ) -> tuple[PhoneAction, int]:
        with self._lock:
            self.start_run(run_id, thread_id)
            run = self._runs[run_id]
            run.last_accessed_at = monotonic()
            if run.status == "cancelled":
                raise PhoneActionCancelledError("该任务已取消，已拒绝执行手机操作。")
            if run.status != "active":
                raise PhoneActionTerminalError("该任务已结束，已拒绝执行手机操作。")
            if device_id:
                conflicting_run = next(
                    (
                        candidate
                        for candidate in self._runs.values()
                        if candidate.run_id != run_id
                        and candidate.status == "active"
                        and candidate.device_id == device_id
                    ),
                    None,
                )
                if conflicting_run is not None:
                    logger.warning(
                        "phone_action_rejected run_id={} device_id={} reason=device_busy active_run_id={}",
                        run_id,
                        device_id,
                        conflicting_run.run_id,
                    )
                    raise PhoneDeviceBusyError(
                        "该设备已有任务正在执行，请先停止原任务后再试。"
                    )
            if len(run.actions) >= self.max_actions_per_run:
                raise PhoneActionLimitError("手机操作次数达到安全上限，任务已停止。")
            fingerprint = _fingerprint(command, payload)
            if sum(action.fingerprint == fingerprint for action in run.actions) >= self.identical_action_limit:
                raise PhoneActionLimitError("检测到重复手机操作，任务已停止。")
            action = PhoneAction(
                action_id=f"act_{uuid4().hex}",
                index=run.next_action_index,
                source_id=source_id,
                command=command,
                fingerprint=fingerprint,
            )
            run.next_action_index += 1
            run.actions.append(action)
            if device_id:
                run.device_id = device_id
            deadline_ms = int((time() + self.action_timeout_seconds) * 1000)
            self._persist_locked()
            return action, deadline_ms

    def mark_result(self, run_id: str, action_id: str, *, succeeded: bool) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.last_accessed_at = monotonic()
            action = next((item for item in run.actions if item.action_id == action_id), None)
            if action is None or action.status == "cancelled":
                return
            action.status = "succeeded" if succeeded else "failed"
            self._persist_locked()

    def cancel_run(
        self,
        run_id: str,
        *,
        reason: str,
        cancel_source: str | None = None,
        terminal_reason: str | None = None,
        cancel_stream: bool = True,
    ) -> str | None:
        callback: Callable[[], None] | None = None
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            run.last_accessed_at = monotonic()
            if run.status == "terminal":
                run.cancel_source = cancel_source or run.cancel_source
                run.terminal_reason = terminal_reason or reason or run.terminal_reason
                self._persist_locked()
                return run.device_id
            if run.status == "cancelled":
                run.cancel_source = cancel_source or run.cancel_source
                run.terminal_reason = terminal_reason or reason or run.terminal_reason
                self._persist_locked()
                return run.device_id
            run.status = "cancelled"
            run.cancellation_reason = reason
            run.cancel_source = cancel_source or reason
            run.terminal_reason = terminal_reason or reason
            for action in run.actions:
                if action.status == "queued":
                    action.status = "cancelled"
            if cancel_stream:
                callback = run.cancel_stream
            device_id = run.device_id
            self._persist_locked()
        if callback is not None:
            callback()
        logger.info(
            "phone_run_cancelled run_id={} reason={} cancel_source={}",
            run_id,
            reason,
            cancel_source or reason,
        )
        return device_id

    def bind_stream_cancellation(
        self,
        run_id: str,
        cancel_callback: Callable[[], None],
    ) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.cancel_stream = cancel_callback
            run.last_accessed_at = monotonic()

    def clear_stream_cancellation(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                run.cancel_stream = None
                run.last_accessed_at = monotonic()

    def trigger_stream_cancellation(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            callback = run.cancel_stream if run is not None else None
        if callback is not None:
            callback()

    def bind_backend_run(self, run_id: str, backend_run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.backend_run_id = backend_run_id
            run.backend_status = "running"
            run.last_accessed_at = monotonic()
            self._persist_locked()
        logger.info(
            "phone_run_backend_bound run_id={} backend_run_id={}",
            run_id,
            backend_run_id,
        )

    def backend_run_info(self, run_id: str) -> tuple[str | None, str] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.backend_run_id is None:
                return None
            run.last_accessed_at = monotonic()
            return run.thread_id, run.backend_run_id

    def mark_backend_status(self, run_id: str, status: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.backend_status = status
            run.last_accessed_at = monotonic()
            self._persist_locked()
        logger.info("phone_run_backend_status run_id={} status={}", run_id, status)

    def thread_id_for(self, run_id: str) -> str | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                run.last_accessed_at = monotonic()
            return run.thread_id if run is not None else None

    def mark_terminal(self, run_id: str, *, terminal_reason: str | None = None) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status == "cancelled":
                if run is not None and terminal_reason:
                    run.terminal_reason = terminal_reason
                    self._persist_locked()
                return
            run.status = "terminal"
            run.terminal_reason = terminal_reason or run.terminal_reason
            run.last_accessed_at = monotonic()
            for action in run.actions:
                if action.status == "queued":
                    action.status = "cancelled"
            self._persist_locked()
        logger.info("phone_run_terminal run_id={}", run_id)

    def is_cancelled(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                run.last_accessed_at = monotonic()
            return run is not None and run.status == "cancelled"

    def snapshot(self, run_id: str) -> dict[str, object]:
        with self._lock:
            self._prune_stale_locked()
            run = self._runs.get(run_id)
            if run is None:
                return {"status": "missing", "actions": []}
            return {
                "status": run.status,
                "deviceId": run.device_id,
                "backendRunId": run.backend_run_id,
                "backendStatus": run.backend_status,
                "cancellationReason": run.cancellation_reason,
                "cancelSource": run.cancel_source,
                "terminalReason": run.terminal_reason,
                "actions": [
                    {
                        "actionId": action.action_id,
                        "index": action.index,
                        "command": action.command,
                        "status": action.status,
                    }
                    for action in run.actions
                ],
            }

    def _prune_stale_locked(self) -> None:
        now = monotonic()
        stale_run_ids = [
            run_id
            for run_id, run in self._runs.items()
            if (
                run.status != "active"
                and now - run.last_accessed_at > self._FINISHED_RUN_TTL_SECONDS
            )
            or (
                run.status == "active"
                and run.cancel_stream is None
                and now - run.last_accessed_at > self._IDLE_ACTIVE_RUN_TTL_SECONDS
            )
        ]
        for run_id in stale_run_ids:
            self._runs.pop(run_id, None)

        self._persist_locked()

    def stale_backend_runs(self) -> list[tuple[str, str, str, str | None]]:
        """Known unfinished mobile runs recovered after a facade restart."""
        with self._lock:
            return [
                (run.run_id, run.thread_id, run.backend_run_id, run.device_id)
                for run in self._runs.values()
                if run.status == "active"
                and run.thread_id is not None
                and run.backend_run_id is not None
            ]

    def _load_persisted_runs(self) -> dict[str, PhoneRun]:
        if self._persist_path is None or not self._persist_path.exists():
            return {}
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("phone_run_registry_load_failed path={}", self._persist_path)
            return {}
        records = raw.get("runs") if isinstance(raw, Mapping) else None
        if not isinstance(records, list):
            return {}
        restored: dict[str, PhoneRun] = {}
        for item in records:
            if not isinstance(item, Mapping):
                continue
            run_id = item.get("runId")
            thread_id = item.get("threadId")
            backend_run_id = item.get("backendRunId")
            if not all(isinstance(value, str) and value for value in (run_id, thread_id, backend_run_id)):
                continue
            status = item.get("status")
            if status not in {"active", "cancelled", "terminal"}:
                continue
            restored[run_id] = PhoneRun(
                run_id=run_id,
                thread_id=thread_id,
                status=status,
                device_id=item.get("deviceId") if isinstance(item.get("deviceId"), str) else None,
                backend_run_id=backend_run_id,
                backend_status=item.get("backendStatus") if isinstance(item.get("backendStatus"), str) else "unknown",
                cancellation_reason=item.get("cancellationReason") if isinstance(item.get("cancellationReason"), str) else None,
                cancel_source=item.get("cancelSource") if isinstance(item.get("cancelSource"), str) else None,
                terminal_reason=item.get("terminalReason") if isinstance(item.get("terminalReason"), str) else None,
            )
        return restored

    def _persist_locked(self) -> None:
        if self._persist_path is None:
            return
        records = [
            {
                "runId": run.run_id,
                "threadId": run.thread_id,
                "status": run.status,
                "deviceId": run.device_id,
                "backendRunId": run.backend_run_id,
                "backendStatus": run.backend_status,
                "cancellationReason": run.cancellation_reason,
                "cancelSource": run.cancel_source,
                "terminalReason": run.terminal_reason,
            }
            for run in self._runs.values()
            if run.thread_id is not None and run.backend_run_id is not None
        ]
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._persist_path.with_suffix(f"{self._persist_path.suffix}.tmp")
            temp_path.write_text(json.dumps({"runs": records}), encoding="utf-8")
            temp_path.replace(self._persist_path)
        except OSError:
            logger.warning("phone_run_registry_persist_failed path={}", self._persist_path)
        except BaseException:
            # langgraph dev 的 BlockingError 等非标准异常：持久化失败不影响功能
            logger.debug("phone_run_registry_persist_skipped path={}", self._persist_path)


_phone_action_source: ContextVar[str | None] = ContextVar(
    "mobile_agent_phone_action_source",
    default=None,
)


@contextmanager
def phone_action_scope(source_id: str) -> Generator[None, None, None]:
    token = _phone_action_source.set(source_id)
    try:
        yield
    finally:
        _phone_action_source.reset(token)


async def dispatch_phone_command(
    session: PhoneCommandSession,
    command: str,
    data: JsonValue,
    *,
    registry: PhoneActionRegistry | None = None,
    device_id: str | None = None,
) -> JsonObject:
    """Dispatch one command only while its run remains active.

    Unscoped callers receive a generated compatibility run id.  Production
    agent paths always have a TraceMiddleware context; the fallback preserves
    existing isolated tool tests without weakening a mobile request's control
    path.
    """

    registry = registry or phone_action_registry
    context = current_context()
    run_id = context.run_id if context is not None else f"compat_{uuid4().hex}"
    thread_id = context.thread_id if context is not None else None
    if context is None:
        logger.warning("Unscoped phone command received; creating compatibility action run.")
    elif not registry.has_registered_run(run_id):
        logger.warning(
            "phone_action_rejected run_id={} reason=run_not_registered",
            run_id,
        )
        raise PhoneActionUnregisteredRunError(
            "任务不属于当前活动请求，已拒绝执行手机操作。"
        )
    payload = dict(data) if isinstance(data, Mapping) else {}
    source_id = _phone_action_source.get() or f"unscoped_{uuid4().hex}"
    action, deadline_ms = registry.reserve_action(
        run_id=run_id,
        thread_id=thread_id,
        source_id=source_id,
        command=command,
        payload=payload,
        device_id=device_id,
    )
    controlled_payload: JsonObject = {
        **payload,
        "runId": run_id,
        "actionId": action.action_id,
        "actionIndex": action.index,
        "deadlineEpochMs": deadline_ms,
    }
    logger.info(
        "phone_action_dispatch run_id={} action_id={} index={} command={}",
        run_id,
        action.action_id,
        action.index,
        command,
    )
    try:
        result = await session.send_command(command, controlled_payload)
    except Exception:
        registry.mark_result(run_id, action.action_id, succeeded=False)
        logger.warning(
            "phone_action_finished run_id={} action_id={} command={} status=failed",
            run_id,
            action.action_id,
            command,
        )
        raise
    registry.mark_result(run_id, action.action_id, succeeded=True)
    logger.info(
        "phone_action_finished run_id={} action_id={} command={} status=succeeded",
        run_id,
        action.action_id,
        command,
    )
    return result


def _fingerprint(command: str, payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        {"command": command, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _positive_env(name: str, default: int) -> int:
    try:
        return max(int(os.getenv(name, str(default))), 1)
    except ValueError:
        return default


def _default_registry_path() -> Path:
    raw = os.getenv("MOBILE_AGENT_RUN_REGISTRY_PATH", ".mobile_agent_runs.json").strip()
    return Path(raw or ".mobile_agent_runs.json")


phone_action_registry = PhoneActionRegistry(persist_path=_default_registry_path())


__all__ = [
    "PhoneActionCancelledError",
    "PhoneActionControlError",
    "PhoneDeviceBusyError",
    "PhoneActionLimitError",
    "PhoneActionRegistry",
    "PhoneActionTerminalError",
    "PhoneActionUnregisteredRunError",
    "dispatch_phone_command",
    "phone_action_registry",
    "phone_action_scope",
]
