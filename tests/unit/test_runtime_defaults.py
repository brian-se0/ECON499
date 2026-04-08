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


def test_raw_data_defaults_include_official_thesis_window() -> None:
    payload = yaml.safe_load(
        (_repo_root() / "configs" / "data" / "raw.yaml").read_text(encoding="utf-8")
    )
    assert payload["sample_start_date"] == "2004-01-02"
    assert payload["sample_end_date"] == "2021-04-09"
