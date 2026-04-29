"""Shared model construction helpers for tuning and walk-forward training."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import optuna

from ivsurf.config import NeuralModelConfig
from ivsurf.models.elasticnet import ElasticNetSurfaceModel
from ivsurf.models.har_factor import HarFactorSurfaceModel
from ivsurf.models.lightgbm_model import LightGBMSurfaceModel
from ivsurf.models.naive import NaiveSurfaceModel
from ivsurf.models.neural_surface import NeuralSurfaceRegressor
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

NEURAL_MODEL_PARAM_KEYS: set[str] = set(NeuralModelConfig.model_fields) - {"model_name"}


def _float_param(params: Mapping[str, object], key: str) -> float:
    value = _required_param(params, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"Expected {key} to be a real numeric value, found {type(value).__name__}."
        raise TypeError(message)
    normalized = float(value)
    if not isfinite(normalized):
        message = f"Expected {key} to be finite, found {value!r}."
        raise ValueError(message)
    return normalized


def _int_param(params: Mapping[str, object], key: str) -> int:
    value = _required_param(params, key)
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"Expected {key} to be an integer, found {type(value).__name__}."
        raise TypeError(message)
    return value


def _str_param(params: Mapping[str, object], key: str) -> str:
    value = _required_param(params, key)
    if not isinstance(value, str) or not value:
        message = f"Expected {key} to be a non-empty string, found {type(value).__name__}."
        raise TypeError(message)
    return value


def _required_param(params: Mapping[str, object], key: str) -> object:
    if key not in params:
        message = f"Missing required model parameter {key!r}."
        raise KeyError(message)
    return params[key]


def _reject_extra_params(
    params: Mapping[str, object],
    *,
    allowed_keys: set[str],
    model_name: str,
) -> None:
    extra_keys = sorted(set(params) - allowed_keys)
    if extra_keys:
        message = f"Unexpected {model_name} parameters: {extra_keys!r}."
        raise ValueError(message)


def _factor_upper_bound(
    *,
    target_dim: int,
    max_factor_count: int | None,
    model_name: str,
) -> int:
    upper_bound = target_dim if max_factor_count is None else min(target_dim, max_factor_count)
    if upper_bound < 2 and model_name in {"har_factor", "lightgbm"}:
        message = (
            f"{model_name} tuning requires max_factor_count >= 2; "
            f"found {upper_bound}."
        )
        raise ValueError(message)
    return upper_bound


def _require_suggested_factor_count_in_bounds(
    *,
    n_factors: int,
    upper_bound: int,
    model_name: str,
) -> int:
    if 2 <= n_factors <= upper_bound:
        return n_factors
    message = (
        f"{model_name} tuning suggested n_factors={n_factors}, outside the admissible "
        f"range [2, {upper_bound}]."
    )
    raise ValueError(message)


def _lightgbm_params(params: Mapping[str, object]) -> dict[str, object]:
    allowed_keys = {
        "n_factors",
        "device_type",
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "feature_fraction",
        "lambda_l2",
        "objective",
        "metric",
        "verbosity",
        "n_jobs",
        "random_state",
    }
    _reject_extra_params(params, allowed_keys=allowed_keys, model_name="lightgbm")
    return {
        "n_factors": _int_param(params, "n_factors"),
        "device_type": _str_param(params, "device_type"),
        "n_estimators": _int_param(params, "n_estimators"),
        "learning_rate": _float_param(params, "learning_rate"),
        "num_leaves": _int_param(params, "num_leaves"),
        "max_depth": _int_param(params, "max_depth"),
        "min_child_samples": _int_param(params, "min_child_samples"),
        "feature_fraction": _float_param(params, "feature_fraction"),
        "lambda_l2": _float_param(params, "lambda_l2"),
        "objective": _str_param(params, "objective"),
        "metric": _str_param(params, "metric"),
        "verbosity": _int_param(params, "verbosity"),
        "n_jobs": _int_param(params, "n_jobs"),
        "random_state": _int_param(params, "random_state"),
    }


def _random_forest_params(params: Mapping[str, object]) -> dict[str, object]:
    allowed_keys = {"n_estimators", "max_depth", "min_samples_leaf", "random_state", "n_jobs"}
    _reject_extra_params(params, allowed_keys=allowed_keys, model_name="random_forest")
    return {
        "n_estimators": _int_param(params, "n_estimators"),
        "max_depth": _int_param(params, "max_depth"),
        "min_samples_leaf": _int_param(params, "min_samples_leaf"),
        "random_state": _int_param(params, "random_state"),
        "n_jobs": _int_param(params, "n_jobs"),
    }


def suggest_model_from_trial(
    *,
    model_name: str,
    trial: optuna.Trial,
    target_dim: int,
    max_factor_count: int | None = None,
    grid_shape: tuple[int, int],
    moneyness_points: tuple[float, ...],
    base_neural_config: NeuralModelConfig,
    base_lightgbm_params: Mapping[str, object] | None = None,
) -> Any:
    """Construct a tunable model by sampling a documented Optuna search space."""

    factor_upper_bound = _factor_upper_bound(
        target_dim=target_dim,
        max_factor_count=max_factor_count,
        model_name=model_name,
    )
    if model_name == "ridge":
        return RidgeSurfaceModel(alpha=trial.suggest_float("alpha", 1.0e-4, 100.0, log=True))
    if model_name == "elasticnet":
        return ElasticNetSurfaceModel(
            alpha=trial.suggest_float("alpha", 1.0e-2, 10.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.01, 0.95),
            max_iter=50_000,
            tol=1.0e-4,
        )
    if model_name == "har_factor":
        n_factors = trial.suggest_int("n_factors", 2, min(12, factor_upper_bound))
        return HarFactorSurfaceModel(
            n_factors=_require_suggested_factor_count_in_bounds(
                n_factors=n_factors,
                upper_bound=factor_upper_bound,
                model_name="har_factor",
            ),
            alpha=trial.suggest_float("alpha", 1.0e-4, 100.0, log=True),
            target_dim=target_dim,
        )
    if model_name == "lightgbm":
        if base_lightgbm_params is None:
            message = "LightGBM tuning requires explicit base_lightgbm_params."
            raise ValueError(message)
        params = {key: value for key, value in base_lightgbm_params.items() if key != "model_name"}
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 60, step=5),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 1.0e-4, 10.0, log=True),
            }
        )
        n_factors = trial.suggest_int("n_factors", 2, min(12, factor_upper_bound))
        params["n_factors"] = _require_suggested_factor_count_in_bounds(
            n_factors=n_factors,
            upper_bound=factor_upper_bound,
            model_name="lightgbm",
        )
        return LightGBMSurfaceModel(**_lightgbm_params(params))
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
        return NeuralSurfaceRegressor(
            config=config,
            grid_shape=grid_shape,
            moneyness_points=moneyness_points,
        )
    message = f"Unsupported model_name for tuning: {model_name}"
    raise ValueError(message)


def make_model_from_params(
    *,
    model_name: str,
    params: Mapping[str, object],
    target_dim: int,
    grid_shape: tuple[int, int],
    moneyness_points: tuple[float, ...],
    base_neural_config: NeuralModelConfig,
) -> Any:
    """Construct a model from persisted tuned parameters."""

    if model_name == "naive":
        _reject_extra_params(params, allowed_keys=set(), model_name="naive")
        return NaiveSurfaceModel()
    if model_name == "ridge":
        _reject_extra_params(params, allowed_keys={"alpha"}, model_name="ridge")
        return RidgeSurfaceModel(alpha=_float_param(params, "alpha"))
    if model_name == "elasticnet":
        _reject_extra_params(
            params,
            allowed_keys={"alpha", "l1_ratio", "max_iter", "tol"},
            model_name="elasticnet",
        )
        return ElasticNetSurfaceModel(
            alpha=_float_param(params, "alpha"),
            l1_ratio=_float_param(params, "l1_ratio"),
            max_iter=_int_param(params, "max_iter"),
            tol=_float_param(params, "tol"),
        )
    if model_name == "har_factor":
        _reject_extra_params(params, allowed_keys={"n_factors", "alpha"}, model_name="har_factor")
        return HarFactorSurfaceModel(
            n_factors=_int_param(params, "n_factors"),
            alpha=_float_param(params, "alpha"),
            target_dim=target_dim,
        )
    if model_name == "lightgbm":
        return LightGBMSurfaceModel(**_lightgbm_params(params))
    if model_name == "random_forest":
        return RandomForestSurfaceModel(**_random_forest_params(params))
    if model_name == "neural_surface":
        _reject_extra_params(
            params,
            allowed_keys=NEURAL_MODEL_PARAM_KEYS,
            model_name="neural_surface",
        )
        config = NeuralModelConfig.model_validate(
            {
                **base_neural_config.model_dump(mode="python"),
                **dict(params),
            },
            strict=True,
        )
        return NeuralSurfaceRegressor(
            config=config,
            grid_shape=grid_shape,
            moneyness_points=moneyness_points,
        )
    message = f"Unsupported model_name for model construction: {model_name}"
    raise ValueError(message)
