"""Ridge surface model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ivsurf.models.base import SurfaceForecastModel


class RidgeSurfaceModel(SurfaceForecastModel):
    """Multi-output ridge regression on the daily feature matrix."""

    def __init__(self, alpha: float) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        )

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
    ) -> RidgeSurfaceModel:
        self.pipeline.fit(features, targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.pipeline.predict(features), dtype=np.float64)

