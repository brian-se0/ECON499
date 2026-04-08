"""LightGBM surface model."""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from ivsurf.models.base import SurfaceForecastModel


class LightGBMSurfaceModel(SurfaceForecastModel):
    """Multi-output LightGBM regression."""

    def __init__(self, **params: Any) -> None:
        self.model = MultiOutputRegressor(LGBMRegressor(**params))

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
    ) -> LightGBMSurfaceModel:
        self.model.fit(features, targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(features), dtype=np.float64)
