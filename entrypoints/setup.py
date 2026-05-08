from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Sequence

from mobile_agent.local_model_runtime import (
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_REPO,
    PROJECT_ROOT,
)


NPM_REGISTRY = "https://registry.npmmirror.com"
GITHUB_LATEST_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
DEFAULT_INSTALL_ROOT = PROJECT_ROOT / ".local"
EXTERNAL_AUTH_NOTICE = (
    "外部工具首次初始化可能需要在飞书/企微 CLI 中人工登录或授权；"
    "授权完成后 CLI 通常会在本机保存登录态，后续再次执行一键部署会复用已有登录态，"
    "一般不需要重复手动登录。"
)

SETUP_ACTIONS = {
    "check",
    "all",
    "external:check",
    "external:install-feishu",
    "external:auth-feishu",
    "external:install-wecom",
    "external:init-wecom",
    "external:verify",
    "external:all",
    "llama:check",
    "llama:download-llama",
    "llama:download-model",
    "llama:all",
}


class SetupError(RuntimeError):
    pass


def _run(command: list[str]) -> int:
    print(f"> {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def check_external_tools() -> int:
    commands = ["node", "npm", "npx", "lark-cli", "wecom-cli"]
    failed = False
    for command in commands:
        resolved = shutil.which(command)
        print(f"{command}: {resolved or 'not found'}")
        failed = failed or resolved is None
    print(f"AMAP_MAPS_API_KEY: {'set' if os.getenv('AMAP_MAPS_API_KEY') else 'missing'}")
    return 1 if failed else 0


def install_feishu() -> int:
    return _run(["npm", "install", "-g", "@larksuite/cli", f"--registry={NPM_REGISTRY}"])


def auth_feishu() -> int:
    status = _run(["lark-cli", "config", "init", "--new"])
    if status != 0:
        return status
    return _run(["lark-cli", "auth", "login", "--recommend"])


def install_wecom() -> int:
    return _run(["npm", "install", "-g", "@wecom/cli", f"--registry={NPM_REGISTRY}"])


def init_wecom() -> int:
    return _run(["wecom-cli", "init"])


def verify_external_tools() -> int:
    status = _run(["lark-cli", "auth", "status"])
    if status != 0:
        return status
    return _run(["wecom-cli", "contact", "get_userlist", "{}"])


def install_external_tools_all() -> int:
    print(EXTERNAL_AUTH_NOTICE)
    for step in (install_feishu, auth_feishu, install_wecom, init_wecom, verify_external_tools):
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

    def names_containing(*parts: str) -> list[dict[str, Any]]:
        matched = []
        for asset in assets:
            name = asset.get("name")
            if isinstance(name, str) and all(part in name for part in parts):
                matched.append(asset)
        return matched

    if target == "windows-x64":
        candidates = names_containing("bin-win-x64")
    elif target == "linux-x64":
        candidates = names_containing("bin-ubuntu-x64")
    elif target == "android-arm64":
        candidates = names_containing("bin-android-arm64")
    else:
        raise SetupError(f"unsupported target: {target}")

    preferred = [
        asset
        for asset in candidates
        if not _asset_name_has_accelerator(str(asset.get("name", "")))
    ]
    selected = (preferred or candidates)[0] if (preferred or candidates) else None
    if selected is None:
        available = ", ".join(str(asset.get("name")) for asset in assets[:20])
        raise SetupError(f"no llama.cpp binary asset matched {target}. Available: {available}")
    return selected


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
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(destination)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as tar_file:
            tar_file.extractall(destination)
    else:
        raise SetupError(f"unsupported archive format: {archive}")


def install_llama_cpp(target: str, install_root: Path) -> Path:
    release = _fetch_latest_release()
    tag_name = release.get("tag_name") or "latest"
    asset = _asset_for_target(release, target)
    asset_name = str(asset["name"])
    download_url = str(asset["browser_download_url"])
    destination = install_root / "llama.cpp" / target

    with tempfile.TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / asset_name
        _download(download_url, archive)
        _extract(archive, destination)

    server = _find_llama_server(destination, target)
    _make_executable(server)
    print(f"installed llama.cpp {tag_name}: {server}")
    return server


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


def download_model(repo: str, filename: str, install_root: Path) -> Path:
    destination = install_root / "models" / filename
    if destination.exists() and destination.stat().st_size > 0:
        print(f"model already exists: {destination}")
        return destination

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"
    try:
        _download(url, destination)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise SetupError(
                "Hugging Face refused the model download. Accept the model license "
                "and set HF_TOKEN or HUGGINGFACE_TOKEN before rerunning."
            ) from exc
        raise

    print(f"downloaded model: {destination}")
    return destination


def check_llama_cpp(install_root: Path) -> int:
    server_matches = sorted((install_root / "llama.cpp").rglob("llama-server*"))
    model_path = install_root / "models" / DEFAULT_MODEL_FILE
    print(f"llama-server: {server_matches[0] if server_matches else 'not found'}")
    print(f"model: {model_path if model_path.exists() else 'not found'}")
    print(f"HF token: {'set' if os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') else 'missing'}")
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


def normalize_legacy_action(kind: str, action: str) -> str:
    normalized = f"{kind}:{action}"
    if normalized not in SETUP_ACTIONS:
        raise SetupError(f"unsupported setup action: {normalized}")
    return normalized


def run_action(
    action: str,
    *,
    target: str = "auto",
    install_root: Path = DEFAULT_INSTALL_ROOT,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_file: str = DEFAULT_MODEL_FILE,
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
            model_repo=model_repo,
            model_file=model_file,
        )
        if llama_status != 0:
            return llama_status
        return run_action(
            "external:all",
            target=target,
            install_root=install_root,
            model_repo=model_repo,
            model_file=model_file,
        )

    external_actions = {
        "external:check": check_external_tools,
        "external:install-feishu": install_feishu,
        "external:auth-feishu": auth_feishu,
        "external:install-wecom": install_wecom,
        "external:init-wecom": init_wecom,
        "external:verify": verify_external_tools,
        "external:all": install_external_tools_all,
    }
    if action in external_actions:
        return external_actions[action]()

    if action == "llama:check":
        return check_llama_cpp(install_root)
    if action in {"llama:download-llama", "llama:all"}:
        install_llama_cpp(resolved_target, install_root)
    if action in {"llama:download-model", "llama:all"}:
        download_model(model_repo, model_file, install_root)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified setup for local llama.cpp and external business tools."
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
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO, help="Hugging Face repo.")
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE, help="GGUF file name.")
    return parser.parse_args(argv)


def parse_external_tools_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install and initialize external business tools.")
    parser.add_argument(
        "action",
        choices=[
            "check",
            "install-feishu",
            "auth-feishu",
            "install-wecom",
            "init-wecom",
            "verify",
            "all",
        ],
        help="Setup action to run.",
    )
    return parser.parse_args(argv)


def parse_llama_cpp_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install llama.cpp and download the default Gemma 4 GGUF model."
    )
    parser.add_argument(
        "action",
        choices=["check", "download-llama", "download-model", "all"],
        help="Setup action to run.",
    )
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
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO, help="Hugging Face repo.")
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE, help="GGUF file name.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        status = run_action(
            args.action,
            target=args.target,
            install_root=args.install_root,
            model_repo=args.model_repo,
            model_file=args.model_file,
        )
    except SetupError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc
    raise SystemExit(status)


def main_external_tools(argv: Sequence[str] | None = None) -> None:
    args = parse_external_tools_args(argv)
    try:
        action = normalize_legacy_action("external", args.action)
        status = run_action(action)
    except SetupError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc
    raise SystemExit(status)


def main_llama_cpp(argv: Sequence[str] | None = None) -> None:
    args = parse_llama_cpp_args(argv)
    try:
        action = normalize_legacy_action("llama", args.action)
        status = run_action(
            action,
            target=args.target,
            install_root=args.install_root,
            model_repo=args.model_repo,
            model_file=args.model_file,
        )
    except SetupError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
