"""No-change benchmark."""

from __future__ import annotations

import numpy as np

from ivsurf.models.base import SurfaceForecastModel


class NoChangeSurfaceModel(SurfaceForecastModel):
    """Forecast tomorrow's surface as today's completed surface."""

    def __init__(self, lag_prefix: str = "feature_surface_mean_01_") -> None:
        self.lag_prefix = lag_prefix
        self._column_count: int | None = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
    ) -> NoChangeSurfaceModel:
        self._column_count = targets.shape[1]
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._column_count is None:
            message = "NoChangeSurfaceModel must be fit before predict."
            raise ValueError(message)
        return features[:, : self._column_count].copy()

