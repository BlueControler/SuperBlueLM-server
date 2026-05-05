from __future__ import annotations

import argparse
import os
import shutil
import subprocess


NPM_REGISTRY = "https://registry.npmmirror.com"


def _run(command: list[str]) -> int:
    print(f"> {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def check() -> int:
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


def verify() -> int:
    status = _run(["lark-cli", "auth", "status"])
    if status != 0:
        return status
    return _run(["wecom-cli", "contact", "get_userlist", "{}"])


def install_all() -> int:
    for step in (install_feishu, auth_feishu, install_wecom, init_wecom, verify):
        status = step()
        if status != 0:
            return status
    return 0


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = {
        "check": check,
        "install-feishu": install_feishu,
        "auth-feishu": auth_feishu,
        "install-wecom": install_wecom,
        "init-wecom": init_wecom,
        "verify": verify,
        "all": install_all,
    }
    raise SystemExit(actions[args.action]())


if __name__ == "__main__":
    main()
