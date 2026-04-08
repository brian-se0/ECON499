from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
import orjson
import polars as pl
import typer

from ivsurf.config import NeuralModelConfig, RawDataConfig, SurfaceGridConfig, load_yaml_config
from ivsurf.evaluation.metrics import weighted_rmse
from ivsurf.models.base import DatasetMatrices, dataset_to_matrices
from ivsurf.models.elasticnet import ElasticNetSurfaceModel
from ivsurf.models.har_factor import HarFactorSurfaceModel
from ivsurf.models.lightgbm_model import LightGBMSurfaceModel
from ivsurf.models.neural_surface import NeuralSurfaceRegressor
from ivsurf.models.random_forest import RandomForestSurfaceModel
from ivsurf.models.ridge import RidgeSurfaceModel
from ivsurf.splits.manifests import WalkforwardSplit, load_splits
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.training.fit_sklearn import fit_and_predict
from ivsurf.training.fit_torch import fit_and_predict_neural

app = typer.Typer(add_completion=False)


def _indices_for_dates(all_dates: np.ndarray, subset: tuple[str, ...]) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(all_dates)}
    return np.asarray([lookup[item] for item in subset], dtype=np.int64)


def _validation_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    observed_masks: np.ndarray,
    vega_weights: np.ndarray,
) -> float:
    weights = observed_masks * np.maximum(vega_weights, 0.0)
    return weighted_rmse(
        y_true=y_true.reshape(-1),
        y_pred=y_pred.reshape(-1),
        weights=weights.reshape(-1),
    )


def _make_model(
    model_name: str,
    trial: optuna.Trial,
    target_dim: int,
    grid_shape: tuple[int, int],
    base_neural_config: NeuralModelConfig,
) -> Any:
    if model_name == "ridge":
        return RidgeSurfaceModel(alpha=trial.suggest_float("alpha", 1.0e-4, 100.0, log=True))
    if model_name == "elasticnet":
        return ElasticNetSurfaceModel(
            alpha=trial.suggest_float("alpha", 1.0e-4, 10.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.01, 0.95),
            max_iter=10_000,
        )
    if model_name == "har_factor":
        return HarFactorSurfaceModel(
            n_factors=trial.suggest_int("n_factors", 2, min(12, target_dim)),
            alpha=trial.suggest_float("alpha", 1.0e-4, 100.0, log=True),
            target_dim=target_dim,
        )
    if model_name == "lightgbm":
        return LightGBMSurfaceModel(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 60, step=5),
            feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
            lambda_l2=trial.suggest_float("lambda_l2", 1.0e-4, 10.0, log=True),
            random_state=7,
        )
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
                    "calendar_penalty_weight", 1.0e-4, 0.5, log=True
                ),
                "convexity_penalty_weight": trial.suggest_float(
                    "convexity_penalty_weight", 1.0e-4, 0.5, log=True
                ),
                "roughness_penalty_weight": trial.suggest_float(
                    "roughness_penalty_weight", 1.0e-5, 0.05, log=True
                ),
            }
        )
        return NeuralSurfaceRegressor(config=config, grid_shape=grid_shape)
    message = f"Unsupported model_name for tuning: {model_name}"
    raise ValueError(message)


def _objective_factory(
    model_name: str,
    matrices: DatasetMatrices,
    tuning_splits: list[WalkforwardSplit],
    grid: SurfaceGrid,
    base_neural_config: NeuralModelConfig,
):
    def objective(trial: optuna.Trial) -> float:
        scores: list[float] = []
        for split in tuning_splits:
            train_index = _indices_for_dates(matrices.quote_dates, split.train_dates)
            validation_index = _indices_for_dates(matrices.quote_dates, split.validation_dates)
            model = _make_model(
                model_name=model_name,
                trial=trial,
                target_dim=matrices.targets.shape[1],
                grid_shape=grid.shape,
                base_neural_config=base_neural_config,
            )
            if model_name == "neural_surface":
                predictions = fit_and_predict_neural(
                    model,
                    train_index,
                    validation_index,
                    matrices,
                )
            else:
                predictions = fit_and_predict(model, train_index, validation_index, matrices)
            scores.append(
                _validation_score(
                    y_true=matrices.targets[validation_index],
                    y_pred=predictions,
                    observed_masks=matrices.observed_masks[validation_index],
                    vega_weights=matrices.vega_weights[validation_index],
                )
            )
        return float(np.mean(scores))

    return objective


@app.command()
def main(
    model_name: str,
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    neural_config_path: Path = Path("configs/models/neural_surface.yaml"),
    n_trials: int = 20,
    tuning_splits_count: int = 3,
) -> None:
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    neural_config = NeuralModelConfig.model_validate(load_yaml_config(neural_config_path))
    grid = SurfaceGrid.from_config(surface_config)

    feature_frame = pl.read_parquet(
        raw_config.gold_dir / "daily_features.parquet"
    ).sort("quote_date")
    matrices = dataset_to_matrices(feature_frame)
    splits = load_splits(raw_config.manifests_dir / "walkforward_splits.json")
    if tuning_splits_count <= 0:
        message = "tuning_splits_count must be positive."
        raise ValueError(message)
    tuning_splits = splits[:tuning_splits_count]
    if not tuning_splits:
        message = "No walk-forward splits available for tuning."
        raise ValueError(message)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        _objective_factory(
            model_name=model_name,
            matrices=matrices,
            tuning_splits=tuning_splits,
            grid=grid,
            base_neural_config=neural_config,
        ),
        n_trials=n_trials,
    )

    payload = {
        "model_name": model_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "tuning_splits_count": tuning_splits_count,
    }
    output_path = raw_config.manifests_dir / "tuning" / f"{model_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    typer.echo(f"Saved tuning results to {output_path}")


if __name__ == "__main__":
    app()
