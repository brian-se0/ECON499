"""Random-forest surface model."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ivsurf.models.base import SurfaceForecastModel


class RandomForestSurfaceModel(SurfaceForecastModel):
    """Multi-output random forest on compact daily features."""

    def __init__(self, **params: int) -> None:
        self.model = RandomForestRegressor(**params)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
    ) -> RandomForestSurfaceModel:
        self.model.fit(features, targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(features), dtype=np.float64)

