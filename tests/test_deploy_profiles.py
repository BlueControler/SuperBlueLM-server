import sys

from scripts.deploy import build_deploy_plan


def _step_names(profile: str) -> list[str]:
    return [
        step.name
        for step in build_deploy_plan(
            profile=profile,
            start=False,
            install_deps=False,
            check_python=False,
        )
    ]


def _commands(profile: str) -> list[tuple[str, ...]]:
    return [
        step.command
        for step in build_deploy_plan(
            profile=profile,
            start=False,
            install_deps=False,
            check_python=False,
        )
    ]


def test_core_profile_runs_warning_style_core_check_only() -> None:
    plan = build_deploy_plan(
        profile="core",
        start=False,
        install_deps=False,
        check_python=False,
    )
    commands = [step.command for step in plan]

    assert plan[-1].name == "check-core-setup"
    assert plan[-1].command == (sys.executable, "-m", "scripts.setup", "check")
    assert plan[-1].required is False
    assert (sys.executable, "-m", "scripts.setup", "llama:check") not in commands
    assert (sys.executable, "-m", "scripts.setup", "external:check") not in commands


def test_local_profile_sets_up_only_local_model() -> None:
    names = _step_names("local")
    commands = _commands("local")

    assert "setup-local-model" in names
    assert "setup-external-tools" not in names
    assert (sys.executable, "-m", "scripts.setup", "llama:all") in commands
    assert (sys.executable, "-m", "scripts.setup", "external:all") not in commands
    assert (sys.executable, "-m", "scripts.setup", "llama:check") in commands
    assert (sys.executable, "-m", "scripts.setup", "external:check") not in commands


def test_external_profile_sets_up_only_external_tools() -> None:
    names = _step_names("external")
    commands = _commands("external")

    assert "setup-external-tools" in names
    assert "setup-local-model" not in names
    assert (sys.executable, "-m", "scripts.setup", "external:all") in commands
    assert (sys.executable, "-m", "scripts.setup", "llama:all") not in commands
    assert (sys.executable, "-m", "scripts.setup", "external:check") in commands
    assert (sys.executable, "-m", "scripts.setup", "llama:check") not in commands


def test_full_profile_is_local_plus_external() -> None:
    names = _step_names("full")
    commands = _commands("full")

    assert names == [
        "ensure-env-file",
        "setup-local-model",
        "check-local-model-setup",
        "setup-external-tools",
        "check-external-tools-setup",
    ]
    assert (sys.executable, "-m", "scripts.setup", "llama:all") in commands
    assert (sys.executable, "-m", "scripts.setup", "external:all") in commands
    assert (sys.executable, "-m", "scripts.setup", "llama:check") in commands
    assert (sys.executable, "-m", "scripts.setup", "external:check") in commands
