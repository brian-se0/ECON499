from __future__ import annotations

from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_neural_model_defaults_to_cuda() -> None:
    payload = yaml.safe_load(
        (_repo_root() / "configs" / "models" / "neural_surface.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert payload["device"] == "cuda"


def test_lightgbm_defaults_to_windows_gpu_mode() -> None:
    payload = yaml.safe_load(
        (_repo_root() / "configs" / "models" / "lightgbm.yaml").read_text(encoding="utf-8")
    )
    assert payload["device_type"] == "gpu"
