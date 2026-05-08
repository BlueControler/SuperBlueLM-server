from __future__ import annotations

import argparse
import os
import socket
import shutil
import subprocess
import sys
import sysconfig
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mobile_agent.local_model_runtime import PROJECT_ROOT


@dataclass(frozen=True)
class DeployStep:
    name: str
    command: tuple[str, ...] = ()
    required: bool = True
    background: bool = False


def build_deploy_plan(
    *,
    profile: str,
    start: bool,
    port: int = 2024,
    install_deps: bool = True,
    check_python: bool = True,
) -> list[DeployStep]:
    if profile not in {"core", "local", "full"}:
        raise ValueError(f"unsupported deploy profile: {profile}")

    steps: list[DeployStep] = []
    if check_python:
        steps.append(DeployStep("check-python-version", ("internal:check-python",)))
    steps.append(DeployStep("ensure-env-file", ("internal:ensure-env",)))
    if start:
        steps.append(DeployStep("check-server-port", ("internal:check-port", str(port))))
    if install_deps:
        steps.append(DeployStep("install-python-dependencies", _dependency_command()))

    if profile in {"local", "full"}:
        steps.append(
            DeployStep(
                "setup-local-model",
                (sys.executable, "-m", "entrypoints.setup", "llama:all"),
            )
        )
    if profile == "full":
        steps.append(
            DeployStep(
                "setup-external-tools",
                (sys.executable, "-m", "entrypoints.setup", "external:all"),
            )
        )

    steps.append(
        DeployStep(
            "check-unified-setup",
            (sys.executable, "-m", "entrypoints.setup", "check"),
            required=profile != "core",
        )
    )

    if start:
        steps.append(
            DeployStep(
                "start-langgraph-server",
                _langgraph_command(port),
                background=True,
            )
        )
        steps.append(
            DeployStep(
                "health-check-server",
                ("internal:health-check", f"http://127.0.0.1:{port}/network/status"),
            )
        )

    return steps


def _dependency_command() -> tuple[str, ...]:
    uv = shutil.which("uv")
    if uv:
        return (uv, "sync", "--group", "dev")
    return (
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "anyio>=4.7.0",
        "langgraph-cli[inmem]>=0.4.14",
        "pytest>=8.3.5",
        "ruff>=0.8.2",
        "mypy>=1.13.0",
    )


def _langgraph_command(port: int) -> tuple[str, ...]:
    executable_name = "langgraph.exe" if os.name == "nt" else "langgraph"
    venv_scripts = PROJECT_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    venv_executable = venv_scripts / executable_name
    if shutil.which("uv") or venv_executable.exists():
        return (str(venv_executable), "dev", "--port", str(port))

    scripts_path = Path(sysconfig.get_path("scripts")) / executable_name
    if scripts_path.exists() or not shutil.which("langgraph"):
        return (str(scripts_path), "dev", "--port", str(port))

    return ("langgraph", "dev", "--port", str(port))


def run_plan(plan: Sequence[DeployStep], *, dry_run: bool = False) -> int:
    background_processes: list[subprocess.Popen[bytes]] = []
    for step in plan:
        print(f"\n== {step.name} ==")
        print(f"> {' '.join(step.command)}")
        if dry_run:
            continue

        status = _run_step(step, background_processes)
        if status != 0:
            if not step.required:
                print(f"warning: {step.name} reported status {status}")
                continue
            _stop_background_processes(background_processes)
            return int(status)
    return 0


def _run_step(step: DeployStep, background_processes: list[subprocess.Popen[bytes]]) -> int:
    if step.command and step.command[0].startswith("internal:"):
        return _run_internal_step(step)

    if step.background:
        process = _start_background(step.command)
        background_processes.append(process)
        print(f"started background process pid={process.pid}")
        return 0

    completed = subprocess.run(list(step.command), check=False)
    return int(completed.returncode)


def _run_internal_step(step: DeployStep) -> int:
    action = step.command[0]
    if action == "internal:check-python":
        return _check_python_version()
    if action == "internal:ensure-env":
        return _ensure_env_file()
    if action == "internal:check-port":
        return _check_port_available(int(step.command[1]))
    if action == "internal:health-check":
        return _health_check(step.command[1])
    print(f"unknown internal step: {action}")
    return 1


def _check_python_version() -> int:
    required = (3, 14)
    current = sys.version_info[:2]
    if current >= required:
        print(f"python: {sys.version.split()[0]}")
        return 0
    print(
        "error: Python >= 3.14 is required by pyproject.toml; "
        f"current is {sys.version.split()[0]}."
    )
    return 1


def _ensure_env_file() -> int:
    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"
    if env_path.exists():
        print(f"env: {env_path}")
        return 0
    if not example_path.exists():
        print(f"error: missing {example_path}")
        return 1
    shutil.copyfile(example_path, env_path)
    print(f"created env file: {env_path}")
    return 0


def _check_port_available(port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            print(f"error: 127.0.0.1:{port} is already in use.")
            return 1
    print(f"port available: 127.0.0.1:{port}")
    return 0


def _health_check(url: str, *, timeout_seconds: float = 90) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if 200 <= response.status < 500:
                    print(f"healthy: {url}")
                    return 0
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    print(f"error: server did not become healthy: {url}")
    return 1


def _start_background(command: tuple[str, ...]) -> subprocess.Popen[bytes]:
    kwargs: dict[str, object] = {"cwd": str(PROJECT_ROOT)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(list(command), **kwargs)


def _stop_background_processes(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command deployment helper.")
    parser.add_argument(
        "--profile",
        choices=["core", "local", "full"],
        default="full",
        help=(
            "core installs Python dependencies and checks setup; local also installs "
            "llama.cpp/model; full also installs and initializes external tools."
        ),
    )
    parser.add_argument(
        "--start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start LangGraph after setup.",
    )
    parser.add_argument("--port", type=int, default=2024, help="LangGraph dev server port.")
    parser.add_argument(
        "--no-install-deps",
        action="store_true",
        help="Skip Python dependency installation.",
    )
    parser.add_argument(
        "--allow-unsupported-python",
        action="store_true",
        help="Skip the Python >= 3.14 guard for local diagnostics.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print deployment steps only.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plan = build_deploy_plan(
        profile=args.profile,
        start=args.start,
        port=args.port,
        install_deps=not args.no_install_deps,
        check_python=not args.allow_unsupported_python,
    )
    raise SystemExit(run_plan(plan, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
