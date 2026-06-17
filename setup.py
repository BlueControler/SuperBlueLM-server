from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Sequence

from mobile_agent.local_model_runtime import (
    DEFAULT_MODEL_FILE,
    PROJECT_ROOT,
)

GITHUB_LATEST_RELEASE_API = (
    "https://gh-proxy.com/https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
)
DEFAULT_INSTALL_ROOT = PROJECT_ROOT / ".local"

SETUP_ACTIONS = {
    "check",
    "all",
    "external:check",
    "external:install-feishu",
    "external:install-wecom",
    "external:all",
    "llama:check",
    "llama:download-llama",
    "llama:all",
}
DEPLOY_ACTIONS = {
    "local": [("setup-local-model", "llama:all")],
    "external": [("setup-external-tools", "external:all")],
    "full": [("setup-local-model", "llama:all"), ("setup-external-tools", "external:all")],
}


class SetupError(RuntimeError):
    pass


def _run(command: list[str]) -> int:
    print(f"> {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def _executable(name: str) -> str:
    if platform.system().lower() == "windows":
        return f"{name}.cmd"
    return name


def _cli_installed(name: str) -> bool:
    return shutil.which(_executable(name)) is not None


def check_external_tools() -> int:
    commands = [
        "node",
        _executable("npm"),
        _executable("lark-cli"),
        _executable("wecom-cli"),
    ]
    failed = False
    for command in commands:
        resolved = shutil.which(command)
        print(f"{command}: {resolved or 'not found'}")
        failed = failed or resolved is None
    amap_key = os.getenv("AMAP_MAPS_API_KEY")
    print(f"AMAP_MAPS_API_KEY: {'set' if amap_key else 'missing'}")
    failed = failed or not amap_key
    return 1 if failed else 0


def install_feishu() -> int:
    if _cli_installed("lark-cli"):
        print("cli already installed: lark-cli")
        return 0
    return _run([_executable("npm"), "install", "-g", "@larksuite/cli"])


def install_wecom() -> int:
    if _cli_installed("wecom-cli"):
        print("cli already installed: wecom-cli")
        return 0
    return _run([_executable("npm"), "install", "-g", "@wecom/cli"])


def install_external_tools_all() -> int:
    for step in (install_feishu, install_wecom):
        status = step()
        if status != 0:
            return status
    return 0


def _request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": "IWebsocket-server setup"}
    token = os.getenv("GITHUB_TOKEN")
    if "api.github.com" in url and token:
        headers["Authorization"] = f"Bearer {token}"
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if "huggingface.co" in url and hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    return urllib.request.Request(url, headers=headers)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"download: {url}")
    with urllib.request.urlopen(_request(url)) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        received = 0
        with destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                if total:
                    percent = received * 100 / total
                    print(f"\r  {percent:5.1f}% {received / 1024 / 1024:.1f} MiB", end="")
    print()


def _fetch_latest_release() -> dict[str, Any]:
    with urllib.request.urlopen(_request(GITHUB_LATEST_RELEASE_API)) as response:
        return json.loads(response.read().decode("utf-8"))


def _asset_for_target(release: dict[str, Any], target: str) -> dict[str, Any]:
    assets = release.get("assets", [])
    if not isinstance(assets, list):
        raise SetupError("llama.cpp release response did not contain assets.")
    _validate_asset_target(target)

    scored_assets: list[tuple[int, str, dict[str, Any]]] = []
    for asset in assets:
        name = asset.get("name")
        if not isinstance(name, str):
            continue
        score = _asset_score_for_target(name, target)
        if score is not None:
            scored_assets.append((score, name, asset))

    if not scored_assets:
        available = ", ".join(str(asset.get("name")) for asset in assets[:50])
        raise SetupError(f"no llama.cpp binary asset matched {target}. Available: {available}")

    return sorted(scored_assets, key=lambda item: (item[0], item[1]))[0][2]


def _asset_score_for_target(name: str, target: str) -> int | None:
    lower = name.lower()
    if target == "windows-x64":
        if not (
            lower.startswith("llama-")
            and lower.endswith(".zip")
            and "bin-win" in lower
            and "x64" in lower
        ):
            return None
        if "cpu" in lower:
            return 0
        if "cuda-12.4" in lower:
            return 10
        if "cuda" in lower or "cudart" in lower:
            return 20
        return 30
    if target == "linux-x64":
        if not (lower.endswith((".tar.gz", ".tgz")) and "bin-ubuntu" in lower and "x64" in lower):
            return None
        return 20 if _asset_name_has_accelerator(lower) else 0
    if target == "android-arm64":
        if lower.endswith((".tar.gz", ".tgz")) and "bin-android-arm64" in lower:
            return 0
        return None
    _validate_asset_target(target)


def _validate_asset_target(target: str) -> None:
    if target not in {"windows-x64", "linux-x64", "android-arm64"}:
        raise SetupError(f"unsupported target: {target}")


def _asset_name_has_accelerator(name: str) -> bool:
    lower = name.lower()
    accelerator_markers = (
        "cuda",
        "cudart",
        "vulkan",
        "rocm",
        "hip",
        "sycl",
        "openvino",
        "kompute",
    )
    return any(marker in lower for marker in accelerator_markers)


def _extract(archive: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    name = archive.name.lower()
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temp_dir:
        staging = Path(temp_dir) / "content"
        staging.mkdir()
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zip_file:
                _safe_extract_zip(zip_file, staging)
        elif name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive) as tar_file:
                _safe_extract_tar(tar_file, staging)
        else:
            raise SetupError(f"unsupported archive format: {archive}")

        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(staging), destination)


def _safe_extract_zip(zip_file: zipfile.ZipFile, destination: Path) -> None:
    destination_root = destination.resolve()
    for member in zip_file.infolist():
        target = (destination / member.filename).resolve()
        if not _is_relative_to(target, destination_root):
            raise SetupError(f"archive member escapes destination: {member.filename}")
    zip_file.extractall(destination)


def _safe_extract_tar(tar_file: tarfile.TarFile, destination: Path) -> None:
    destination_root = destination.resolve()
    for member in tar_file.getmembers():
        target = (destination / member.name).resolve()
        if not _is_relative_to(target, destination_root):
            raise SetupError(f"archive member escapes destination: {member.name}")
    tar_file.extractall(destination, filter="data")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def install_llama_cpp(target: str, install_root: Path) -> Path:
    (install_root / "models").mkdir(parents=True, exist_ok=True)
    destination = install_root / "llama.cpp" / target
    existing = _find_existing_llama_server(destination, target)
    if existing is not None:
        print(f"llama.cpp already installed: {existing}")
        return existing

    release = _fetch_latest_release()
    tag_name = release.get("tag_name") or "latest"
    asset = _asset_for_target(release, target)
    asset_name = str(asset["name"])
    download_url = f"https://gh-proxy.com/{asset['browser_download_url']}"

    with tempfile.TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / asset_name
        _download(download_url, archive)
        _extract(archive, destination)

    server = _find_llama_server(destination, target)
    _make_executable(server)
    print(f"installed llama.cpp {tag_name}: {server}")
    return server


def _find_existing_llama_server(root: Path, target: str) -> Path | None:
    if not root.exists():
        return None
    try:
        server = _find_llama_server(root, target)
    except SetupError:
        return None
    if server.exists() and server.stat().st_size > 0:
        return server
    return None


def _find_llama_server(root: Path, target: str) -> Path:
    names = ["llama-server.exe"] if target == "windows-x64" else ["llama-server"]
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    raise SetupError(f"llama-server binary not found under {root}")


def _make_executable(path: Path) -> None:
    if platform.system().lower() == "windows":
        return
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def check_llama_cpp(install_root: Path) -> int:
    server_matches = sorted((install_root / "llama.cpp").rglob("llama-server*"))
    model_path = install_root / "models" / DEFAULT_MODEL_FILE
    print(f"llama-server: {server_matches[0] if server_matches else 'not found'}")
    print(f"model: {model_path if model_path.exists() else 'not found'}")
    print(
        f"HF token: {'set' if os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') else 'missing'}"
    )
    return 0 if server_matches and model_path.exists() else 1


def detect_target() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" and machine in {"amd64", "x86_64"}:
        return "windows-x64"
    if system == "linux" and machine in {"amd64", "x86_64"}:
        return "linux-x64"
    if system in {"linux", "android"} and machine in {"aarch64", "arm64"}:
        return "android-arm64"
    raise SetupError(f"cannot auto-detect supported target from {system}/{machine}")


def run_action(
    action: str,
    *,
    target: str = "auto",
    install_root: Path = DEFAULT_INSTALL_ROOT,
) -> int:
    if action not in SETUP_ACTIONS:
        raise SetupError(f"unsupported setup action: {action}")

    install_root = install_root.expanduser().resolve()
    resolved_target = detect_target() if target == "auto" else target

    if action == "check":
        external_status = check_external_tools()
        llama_status = check_llama_cpp(install_root)
        return 0 if external_status == 0 and llama_status == 0 else 1
    if action == "all":
        llama_status = run_action(
            "llama:all",
            target=target,
            install_root=install_root,
        )
        if llama_status != 0:
            return llama_status
        return run_action(
            "external:all",
            target=target,
            install_root=install_root,
        )

    external_actions = {
        "external:check": check_external_tools,
        "external:install-feishu": install_feishu,
        "external:install-wecom": install_wecom,
        "external:all": install_external_tools_all,
    }
    if action in external_actions:
        return external_actions[action]()

    if action == "llama:check":
        return check_llama_cpp(install_root)
    if action in {"llama:download-llama", "llama:all"}:
        install_llama_cpp(resolved_target, install_root)
    return 0


def deploy_actions(profile: str) -> list[tuple[str, str]]:
    actions = DEPLOY_ACTIONS.get(profile)
    if actions is None:
        raise SetupError(f"unsupported deploy profile: {profile}")
    return actions.copy()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified setup for local llama.cpp and external business tools.",
        epilog="Use `python -m scripts.setup deploy --help` for deployment profiles.",
    )
    parser.add_argument("action", choices=sorted(SETUP_ACTIONS), help="Setup action to run.")
    parser.add_argument(
        "--target",
        choices=["auto", "windows-x64", "linux-x64", "android-arm64"],
        default="auto",
        help="llama.cpp binary target.",
    )
    parser.add_argument(
        "--install-root",
        type=Path,
        default=DEFAULT_INSTALL_ROOT,
        help="Directory used for llama.cpp and model files.",
    )
    return parser.parse_args(argv)


def parse_deploy_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command deployment helper.")
    parser.add_argument(
        "--profile",
        choices=sorted(DEPLOY_ACTIONS),
        default="full",
        help=(
            "local sets up llama.cpp/model; external sets up business tools; "
            "full runs local then external."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print deployment steps only.")
    parser.add_argument(
        "--target",
        choices=["auto", "windows-x64", "linux-x64", "android-arm64"],
        default="auto",
        help="llama.cpp binary target.",
    )
    parser.add_argument(
        "--install-root",
        type=Path,
        default=DEFAULT_INSTALL_ROOT,
        help="Directory used for llama.cpp and model files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args[:1] == ["deploy"]:
        main_deploy(raw_args[1:])
        return

    args = parse_args(raw_args)
    try:
        status = run_action(
            args.action,
            target=args.target,
            install_root=args.install_root,
        )
    except SetupError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc
    raise SystemExit(status)


def main_deploy(argv: Sequence[str] | None = None) -> None:
    args = parse_deploy_args(argv)
    try:
        for name, action in deploy_actions(args.profile):
            print(f"\n== {name} ==")
            print(f"> {sys.executable} -m scripts.setup {action}")
            if args.dry_run:
                continue

            status = run_action(
                action,
                target=args.target,
                install_root=args.install_root,
            )
            if status != 0:
                raise SystemExit(status)
    except SetupError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc
    raise SystemExit(0)


if __name__ == "__main__":
    main()
