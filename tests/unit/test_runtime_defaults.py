from __future__ import annotations

from pathlib import Path

import yaml

from ivsurf.config import (
    CleaningConfig,
    FeatureConfig,
    HedgingConfig,
    RawDataConfig,
    SurfaceGridConfig,
    load_yaml_config,
)


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


def test_committed_data_configs_parse_into_typed_models() -> None:
    repo_root = _repo_root()
    raw_config = RawDataConfig.model_validate(
        load_yaml_config(repo_root / "configs" / "data" / "raw.yaml")
    )
    cleaning_config = CleaningConfig.model_validate(
        load_yaml_config(repo_root / "configs" / "data" / "cleaning.yaml")
    )
    surface_config = SurfaceGridConfig.model_validate(
        load_yaml_config(repo_root / "configs" / "data" / "surface.yaml")
    )
    feature_config = FeatureConfig.model_validate(
        load_yaml_config(repo_root / "configs" / "data" / "features.yaml")
    )

    assert raw_config.target_symbol == "^SPX"
    assert cleaning_config.require_positive_iv is True
    assert surface_config.total_variance_floor == 1.0e-8
    assert feature_config.lag_windows == (1, 5, 22)


def test_committed_hedging_config_has_no_dead_keys() -> None:
    hedging_config = HedgingConfig.model_validate(
        load_yaml_config(_repo_root() / "configs" / "eval" / "hedging.yaml")
    )

    assert hedging_config.hedge_maturity_days == 30
