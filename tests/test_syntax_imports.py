from __future__ import annotations

import py_compile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_core_python_files_compile() -> None:
    files = [
        "scripts/setup.py",
        "mobile_agent/local_model_runtime.py",
        "mobile_agent/tools/external.py",
        "mobile_agent/tools/system.py",
        "mobile_agent/agent/middleware.py",
        "mobile_agent/agent/phone_subagent.py",
        "mobile_agent/agent/phone_delegation.py",
    ]

    for relative_path in files:
        py_compile.compile(str(PROJECT_ROOT / relative_path), doraise=True)
