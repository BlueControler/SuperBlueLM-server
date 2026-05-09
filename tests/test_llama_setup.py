import pytest

from scripts.setup import SetupError, _asset_for_target


def _asset(name):
    return {"name": name, "browser_download_url": f"https://example.invalid/{name}"}


def test_windows_x64_prefers_installable_cpu_asset_over_cuda_and_cudart_assets():
    release = {
        "assets": [
            _asset("llama-b9089-bin-win-cuda-12.4-x64.zip"),
            _asset("cudart-llama-bin-win-cuda-12.4-x64.zip"),
            _asset("llama-b9089-bin-win-cpu-x64.zip"),
            _asset("llama-b9089-bin-win-cuda-13.1-x64.zip"),
        ]
    }

    asset = _asset_for_target(release, "windows-x64")

    assert asset["name"] == "llama-b9089-bin-win-cpu-x64.zip"


def test_windows_x64_falls_back_to_installable_cuda_asset_not_cudart_support_archive():
    release = {
        "assets": [
            _asset("cudart-llama-bin-win-cuda-12.4-x64.zip"),
            _asset("llama-b9089-bin-win-cuda-12.4-x64.zip"),
        ]
    }

    asset = _asset_for_target(release, "windows-x64")

    assert asset["name"] == "llama-b9089-bin-win-cuda-12.4-x64.zip"


def test_windows_x64_rejects_cudart_support_archive_without_installable_binary_asset():
    release = {
        "assets": [
            _asset("cudart-llama-bin-win-cuda-12.4-x64.zip"),
        ]
    }

    with pytest.raises(SetupError, match="no llama.cpp binary asset matched windows-x64"):
        _asset_for_target(release, "windows-x64")


def test_linux_x64_prefers_cpu_asset_over_accelerated_asset():
    release = {
        "assets": [
            _asset("llama-b9085-bin-ubuntu-vulkan-x64.tar.gz"),
            _asset("llama-b9085-bin-ubuntu-x64.tar.gz"),
        ]
    }

    asset = _asset_for_target(release, "linux-x64")

    assert asset["name"] == "llama-b9085-bin-ubuntu-x64.tar.gz"


def test_android_arm64_matches_android_asset():
    release = {
        "assets": [
            _asset("llama-b9085-bin-android-arm64.tar.gz"),
        ]
    }

    asset = _asset_for_target(release, "android-arm64")

    assert asset["name"] == "llama-b9085-bin-android-arm64.tar.gz"


def test_asset_for_target_rejects_unsupported_target_before_scanning_assets():
    with pytest.raises(SetupError, match="unsupported target: bad-target"):
        _asset_for_target({"assets": []}, "bad-target")
