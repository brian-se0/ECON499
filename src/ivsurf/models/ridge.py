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

    def __init__(self, alpha: float, clip_predictions_to_train_log_range: bool) -> None:
        self.alpha = alpha
        self.clip_predictions_to_train_log_range = clip_predictions_to_train_log_range
        self.training_log_target_min: np.ndarray | None = None
        self.training_log_target_max: np.ndarray | None = None
        self.last_clipped_prediction_count: int = 0
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
        self.training_log_target_min = np.min(transformed_targets, axis=0)
        self.training_log_target_max = np.max(transformed_targets, axis=0)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        log_predictions = np.asarray(self.pipeline.predict(features), dtype=np.float64)
        self.last_clipped_prediction_count = 0
        if self.clip_predictions_to_train_log_range:
            if self.training_log_target_min is None or self.training_log_target_max is None:
                message = "RidgeSurfaceModel clipping bounds are unavailable before fit."
                raise ValueError(message)
            clipped_predictions = np.clip(
                log_predictions,
                self.training_log_target_min,
                self.training_log_target_max,
            )
            self.last_clipped_prediction_count = int(
                np.count_nonzero(~np.isclose(clipped_predictions, log_predictions))
            )
            log_predictions = clipped_predictions
        return self.target_adapter.inverse_predictions(log_predictions)
