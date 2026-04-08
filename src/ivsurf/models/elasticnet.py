"""Elastic-net surface model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ivsurf.models.base import SurfaceForecastModel


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

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
    ) -> ElasticNetSurfaceModel:
        self.pipeline.fit(features, targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.pipeline.predict(features), dtype=np.float64)

