from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import typer

from ivsurf.config import NeuralModelConfig, RawDataConfig, SurfaceGridConfig, load_yaml_config
from ivsurf.evaluation.forecast_store import write_forecasts
from ivsurf.models.base import dataset_to_matrices
from ivsurf.models.elasticnet import ElasticNetSurfaceModel
from ivsurf.models.har_factor import HarFactorSurfaceModel
from ivsurf.models.lightgbm_model import LightGBMSurfaceModel
from ivsurf.models.neural_surface import NeuralSurfaceRegressor
from ivsurf.models.no_change import NoChangeSurfaceModel
from ivsurf.models.random_forest import RandomForestSurfaceModel
from ivsurf.models.ridge import RidgeSurfaceModel
from ivsurf.splits.manifests import load_splits
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.training.fit_sklearn import fit_and_predict
from ivsurf.training.fit_torch import fit_and_predict_neural

app = typer.Typer(add_completion=False)


def _indices_for_dates(all_dates: np.ndarray, subset: tuple[str, ...]) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(all_dates)}
    return np.asarray([lookup[item] for item in subset], dtype=np.int64)


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    ridge_config_path: Path = Path("configs/models/ridge.yaml"),
    elasticnet_config_path: Path = Path("configs/models/elasticnet.yaml"),
    har_config_path: Path = Path("configs/models/har_factor.yaml"),
    lightgbm_config_path: Path = Path("configs/models/lightgbm.yaml"),
    random_forest_config_path: Path = Path("configs/models/random_forest.yaml"),
    neural_config_path: Path = Path("configs/models/neural_surface.yaml"),
) -> None:
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    grid = SurfaceGrid.from_config(surface_config)

    feature_frame = pl.read_parquet(
        raw_config.gold_dir / "daily_features.parquet"
    ).sort("quote_date")
    matrices = dataset_to_matrices(feature_frame)
    splits = load_splits(raw_config.manifests_dir / "walkforward_splits.json")

    ridge_params = load_yaml_config(ridge_config_path)
    elasticnet_params = load_yaml_config(elasticnet_config_path)
    har_params = load_yaml_config(har_config_path)
    lightgbm_params = load_yaml_config(lightgbm_config_path)
    random_forest_params = load_yaml_config(random_forest_config_path)
    neural_config = NeuralModelConfig.model_validate(load_yaml_config(neural_config_path))

    model_factories = {
        "no_change": lambda: NoChangeSurfaceModel(),
        "ridge": lambda: RidgeSurfaceModel(alpha=float(ridge_params["alpha"])),
        "elasticnet": lambda: ElasticNetSurfaceModel(
            alpha=float(elasticnet_params["alpha"]),
            l1_ratio=float(elasticnet_params["l1_ratio"]),
            max_iter=int(elasticnet_params["max_iter"]),
        ),
        "har_factor": lambda: HarFactorSurfaceModel(
            n_factors=int(har_params["n_factors"]),
            alpha=float(har_params["alpha"]),
            target_dim=matrices.targets.shape[1],
        ),
        "lightgbm": lambda: LightGBMSurfaceModel(
            **{key: value for key, value in lightgbm_params.items() if key != "model_name"}
        ),
        "random_forest": lambda: RandomForestSurfaceModel(
            **{key: value for key, value in random_forest_params.items() if key != "model_name"}
        ),
        "neural_surface": lambda: NeuralSurfaceRegressor(
            config=neural_config,
            grid_shape=grid.shape,
        ),
    }

    for model_name, factory in model_factories.items():
        prediction_blocks: list[np.ndarray] = []
        quote_date_blocks: list[np.ndarray] = []
        target_date_blocks: list[np.ndarray] = []
        for split in splits:
            train_dates = split.train_dates + split.validation_dates
            train_index = _indices_for_dates(matrices.quote_dates, train_dates)
            test_index = _indices_for_dates(matrices.quote_dates, split.test_dates)
            model = factory()
            if model_name == "neural_surface":
                predictions = fit_and_predict_neural(model, train_index, test_index, matrices)  # type: ignore[arg-type]
            else:
                predictions = fit_and_predict(model, train_index, test_index, matrices)  # type: ignore[arg-type]
            prediction_blocks.append(predictions)
            quote_date_blocks.append(matrices.quote_dates[test_index])
            target_date_blocks.append(matrices.target_dates[test_index])

        write_forecasts(
            output_path=raw_config.gold_dir / "forecasts" / f"{model_name}.parquet",
            model_name=model_name,
            quote_dates=np.concatenate(quote_date_blocks),
            target_dates=np.concatenate(target_date_blocks),
            predictions=np.vstack(prediction_blocks),
            grid=grid,
        )


if __name__ == "__main__":
    app()
