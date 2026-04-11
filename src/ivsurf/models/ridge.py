"""Ridge surface model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ivsurf.models.base import SurfaceForecastModel
from ivsurf.models.positive_target import LogPositiveTargetAdapter


class RidgeSurfaceModel(SurfaceForecastModel):
    """Multi-output ridge regression on the daily feature matrix."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        )
        self.target_adapter = LogPositiveTargetAdapter("RidgeSurfaceModel")

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
        training_weights: np.ndarray | None = None,
    ) -> RidgeSurfaceModel:
        transformed_targets = self.target_adapter.transform_targets(
            targets,
            array_name="training targets",
        )
        self.pipeline.fit(features, transformed_targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        log_predictions = np.asarray(self.pipeline.predict(features), dtype=np.float64)
        return self.target_adapter.inverse_predictions(log_predictions)
