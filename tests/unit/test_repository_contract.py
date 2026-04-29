from __future__ import annotations

from pathlib import Path

import torch

from ivsurf.config import (
    CleaningConfig,
    FeatureConfig,
    HedgingConfig,
    HpoProfileConfig,
    NeuralModelConfig,
    RawDataConfig,
    SurfaceGridConfig,
    TrainingProfileConfig,
    load_yaml_config,
)
from ivsurf.hedging.validation import require_hedging_config_in_surface_domain
from ivsurf.models.neural_surface import NeuralSurfaceRegressor
from ivsurf.models.penalties import (
    calendar_monotonicity_penalty,
    convexity_penalty,
    roughness_penalty,
)
from ivsurf.surfaces.grid import SurfaceGrid


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_required_committed_config_assets_exist_and_parse() -> None:
    repo_root = _repo_root()
    required_paths = (
        repo_root / "configs" / "data" / "raw.yaml",
        repo_root / "configs" / "data" / "cleaning.yaml",
        repo_root / "configs" / "data" / "surface.yaml",
        repo_root / "configs" / "data" / "features.yaml",
        repo_root / "configs" / "official_smoke" / "data" / "surface.yaml",
        repo_root / "configs" / "official_smoke" / "data" / "features.yaml",
        repo_root / "configs" / "official_smoke" / "workflow" / "hpo_smoke.yaml",
        repo_root / "configs" / "official_smoke" / "workflow" / "train_smoke.yaml",
    )

    missing_paths = [path for path in required_paths if not path.exists()]
    assert not missing_paths, (
        "Missing required committed config assets: "
        f"{[path.relative_to(repo_root).as_posix() for path in missing_paths]}"
    )

    raw_config = RawDataConfig.model_validate(load_yaml_config(required_paths[0]))
    cleaning_config = CleaningConfig.model_validate(load_yaml_config(required_paths[1]))
    production_surface_config = SurfaceGridConfig.model_validate(
        load_yaml_config(required_paths[2])
    )
    production_feature_config = FeatureConfig.model_validate(load_yaml_config(required_paths[3]))
    smoke_surface_config = SurfaceGridConfig.model_validate(load_yaml_config(required_paths[4]))
    smoke_feature_config = FeatureConfig.model_validate(load_yaml_config(required_paths[5]))
    smoke_hpo_config = HpoProfileConfig.model_validate(load_yaml_config(required_paths[6]))
    smoke_train_config = TrainingProfileConfig.model_validate(load_yaml_config(required_paths[7]))

    assert raw_config.target_symbol == "^SPX"
    assert cleaning_config.require_positive_iv is True
    assert len(production_surface_config.moneyness_points) == 9
    assert len(production_surface_config.maturity_days) == 9
    assert production_feature_config.lag_windows == (1, 5, 22)
    assert len(smoke_surface_config.moneyness_points) == 3
    assert len(smoke_surface_config.maturity_days) == 3
    assert smoke_feature_config.lag_windows == (1, 5, 22)
    assert smoke_hpo_config.seed == 1
    assert smoke_hpo_config.n_trials == 1
    assert smoke_train_config.epochs >= 5
    assert smoke_train_config.neural_min_epochs_before_early_stop >= 2


def test_official_smoke_grid_is_valid_for_enabled_neural_penalties() -> None:
    repo_root = _repo_root()
    smoke_surface_config = SurfaceGridConfig.model_validate(
        load_yaml_config(repo_root / "configs" / "official_smoke" / "data" / "surface.yaml")
    )
    neural_config = NeuralModelConfig.model_validate(
        {
            **load_yaml_config(repo_root / "configs" / "models" / "neural_surface.yaml"),
            "device": "cpu",
        }
    )

    NeuralSurfaceRegressor(
        config=neural_config,
        grid_shape=(
            len(smoke_surface_config.maturity_days),
            len(smoke_surface_config.moneyness_points),
        ),
        moneyness_points=smoke_surface_config.moneyness_points,
    )
    grid_shape = (
        len(smoke_surface_config.maturity_days),
        len(smoke_surface_config.moneyness_points),
    )
    surface = torch.full((1, grid_shape[0] * grid_shape[1]), 0.04, dtype=torch.float32)
    penalties = (
        calendar_monotonicity_penalty(surface, grid_shape),
        convexity_penalty(
            surface,
            grid_shape,
            moneyness_points=smoke_surface_config.moneyness_points,
        ),
        roughness_penalty(surface, grid_shape),
    )
    assert all(bool(torch.isfinite(value).item()) for value in penalties)


def test_official_smoke_hedging_config_stays_inside_surface_domain_after_target_gap() -> None:
    repo_root = _repo_root()
    smoke_surface_config = SurfaceGridConfig.model_validate(
        load_yaml_config(repo_root / "configs" / "official_smoke" / "data" / "surface.yaml")
    )
    smoke_hedging_config = HedgingConfig.model_validate(
        load_yaml_config(repo_root / "configs" / "official_smoke" / "eval" / "hedging.yaml")
    )

    require_hedging_config_in_surface_domain(
        smoke_hedging_config,
        SurfaceGrid.from_config(smoke_surface_config),
        max_target_gap_days=4,
    )
