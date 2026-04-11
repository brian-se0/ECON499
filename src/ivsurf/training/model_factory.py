"""Shared model construction helpers for tuning and walk-forward training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optuna

from ivsurf.config import NeuralModelConfig
from ivsurf.models.elasticnet import ElasticNetSurfaceModel
from ivsurf.models.har_factor import HarFactorSurfaceModel
from ivsurf.models.lightgbm_model import LightGBMSurfaceModel
from ivsurf.models.neural_surface import NeuralSurfaceRegressor
from ivsurf.models.no_change import NoChangeSurfaceModel
from ivsurf.models.random_forest import RandomForestSurfaceModel
from ivsurf.models.ridge import RidgeSurfaceModel

TUNABLE_MODEL_NAMES: tuple[str, ...] = (
    "ridge",
    "elasticnet",
    "har_factor",
    "lightgbm",
    "random_forest",
    "neural_surface",
)


def _float_param(params: Mapping[str, object], key: str) -> float:
    value = params[key]
    if not isinstance(value, int | float | str):
        message = f"Expected {key} to be numeric-like, found {type(value).__name__}."
        raise TypeError(message)
    return float(value)


def _int_param(params: Mapping[str, object], key: str) -> int:
    value = params[key]
    if not isinstance(value, int | float | str):
        message = f"Expected {key} to be integer-like, found {type(value).__name__}."
        raise TypeError(message)
    return int(value)


def _bool_param(params: Mapping[str, object], key: str) -> bool:
    value = params[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    message = f"Expected {key} to be boolean-like, found {type(value).__name__}."
    raise TypeError(message)


def suggest_model_from_trial(
    *,
    model_name: str,
    trial: optuna.Trial,
    target_dim: int,
    grid_shape: tuple[int, int],
    base_neural_config: NeuralModelConfig,
    base_lightgbm_params: Mapping[str, object] | None = None,
) -> Any:
    """Construct a tunable model by sampling a documented Optuna search space."""

    if model_name == "ridge":
        return RidgeSurfaceModel(
            alpha=trial.suggest_float("alpha", 1.0e-4, 100.0, log=True),
            clip_predictions_to_train_log_range=True,
        )
    if model_name == "elasticnet":
        return ElasticNetSurfaceModel(
            alpha=trial.suggest_float("alpha", 1.0e-2, 10.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.01, 0.95),
            max_iter=50_000,
            tol=1.0e-4,
        )
    if model_name == "har_factor":
        return HarFactorSurfaceModel(
            n_factors=trial.suggest_int("n_factors", 2, min(12, target_dim)),
            alpha=trial.suggest_float("alpha", 1.0e-4, 100.0, log=True),
            target_dim=target_dim,
        )
    if model_name == "lightgbm":
        params = {
            key: value for key, value in (base_lightgbm_params or {}).items() if key != "model_name"
        }
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 60, step=5),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 1.0e-4, 10.0, log=True),
                "n_factors": trial.suggest_int("n_factors", 2, min(12, target_dim)),
            }
        )
        params.setdefault("device_type", "gpu")
        params.setdefault("random_state", 7)
        params.setdefault("objective", "regression")
        params.setdefault("metric", "l2")
        params.setdefault("verbosity", -1)
        return LightGBMSurfaceModel(**params)
    if model_name == "random_forest":
        return RandomForestSurfaceModel(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 16),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=7,
            n_jobs=-1,
        )
    if model_name == "neural_surface":
        config = base_neural_config.model_copy(
            update={
                "hidden_width": trial.suggest_int("hidden_width", 64, 512, step=64),
                "depth": trial.suggest_int("depth", 2, 5),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3),
                "learning_rate": trial.suggest_float("learning_rate", 1.0e-4, 1.0e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1.0e-6, 1.0e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "calendar_penalty_weight": trial.suggest_float(
                    "calendar_penalty_weight", 1.0e-5, 5.0e-2, log=True
                ),
                "convexity_penalty_weight": trial.suggest_float(
                    "convexity_penalty_weight", 1.0e-5, 5.0e-2, log=True
                ),
                "roughness_penalty_weight": trial.suggest_float(
                    "roughness_penalty_weight", 1.0e-5, 0.05, log=True
                ),
            }
        )
        return NeuralSurfaceRegressor(config=config, grid_shape=grid_shape)
    message = f"Unsupported model_name for tuning: {model_name}"
    raise ValueError(message)


def make_model_from_params(
    *,
    model_name: str,
    params: Mapping[str, object],
    target_dim: int,
    grid_shape: tuple[int, int],
    base_neural_config: NeuralModelConfig,
) -> Any:
    """Construct a model from persisted tuned parameters."""

    if model_name == "no_change":
        return NoChangeSurfaceModel()
    if model_name == "ridge":
        return RidgeSurfaceModel(
            alpha=_float_param(params, "alpha"),
            clip_predictions_to_train_log_range=_bool_param(
                params,
                "clip_predictions_to_train_log_range",
            ),
        )
    if model_name == "elasticnet":
        return ElasticNetSurfaceModel(
            alpha=_float_param(params, "alpha"),
            l1_ratio=_float_param(params, "l1_ratio"),
            max_iter=_int_param(params, "max_iter"),
            tol=_float_param(params, "tol"),
        )
    if model_name == "har_factor":
        return HarFactorSurfaceModel(
            n_factors=_int_param(params, "n_factors"),
            alpha=_float_param(params, "alpha"),
            target_dim=target_dim,
        )
    if model_name == "lightgbm":
        return LightGBMSurfaceModel(**dict(params))
    if model_name == "random_forest":
        return RandomForestSurfaceModel(**dict(params))
    if model_name == "neural_surface":
        config = base_neural_config.model_copy(update=dict(params))
        return NeuralSurfaceRegressor(config=config, grid_shape=grid_shape)
    message = f"Unsupported model_name for model construction: {model_name}"
    raise ValueError(message)
