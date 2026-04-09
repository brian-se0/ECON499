"""Elastic-net surface model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ivsurf.models.base import SurfaceForecastModel
from ivsurf.models.positive_target import LogPositiveTargetAdapter


class ElasticNetSurfaceModel(SurfaceForecastModel):
    """Multi-output elastic net on the daily feature matrix."""

    def __init__(self, alpha: float, l1_ratio: float, max_iter: int) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter),
                ),
            ]
        )
        self.target_adapter = LogPositiveTargetAdapter("ElasticNetSurfaceModel")

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
    ) -> ElasticNetSurfaceModel:
        self.pipeline.fit(
            features,
            self.target_adapter.transform_targets(targets, array_name="training targets"),
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.target_adapter.inverse_predictions(self.pipeline.predict(features))
