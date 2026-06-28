from scripts.setup import deploy_actions


def _actions(profile: str) -> list[str]:
    return [action for _, action in deploy_actions(profile)]


def test_local_profile_sets_up_only_local_model() -> None:
    actions = _actions("local")

    assert actions == ["llama:all"]


def test_external_profile_sets_up_only_external_tools() -> None:
    actions = _actions("external")

    assert actions == ["external:all"]


def test_full_profile_is_local_plus_external() -> None:
    actions = _actions("full")

    assert actions == ["llama:all", "external:all"]
