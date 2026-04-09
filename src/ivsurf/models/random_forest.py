"""Random-forest surface model."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ivsurf.models.base import SurfaceForecastModel
from ivsurf.models.positive_target import LogPositiveTargetAdapter


class RandomForestSurfaceModel(SurfaceForecastModel):
    """Multi-output random forest on compact daily features."""

    def __init__(self, **params: Any) -> None:
        self.model = RandomForestRegressor(**params)
        self.target_adapter = LogPositiveTargetAdapter("RandomForestSurfaceModel")

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
    ) -> RandomForestSurfaceModel:
        self.model.fit(
            features,
            self.target_adapter.transform_targets(targets, array_name="training targets"),
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.target_adapter.inverse_predictions(self.model.predict(features))
