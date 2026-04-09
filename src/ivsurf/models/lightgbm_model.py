"""LightGBM surface model."""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from ivsurf.config import TrainingProfileConfig
from ivsurf.models.base import SurfaceForecastModel
from ivsurf.models.positive_target import LogPositiveTargetAdapter


class LightGBMSurfaceModel(SurfaceForecastModel):
    """Multi-output LightGBM regression."""

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.estimators: list[LGBMRegressor] = []
        self.best_iterations: tuple[int, ...] = ()
        self.target_adapter = LogPositiveTargetAdapter("LightGBMSurfaceModel")

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
        training_weights: np.ndarray | None = None,
        *,
        validation_features: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
        training_profile: TrainingProfileConfig | None = None,
    ) -> LightGBMSurfaceModel:
        estimators: list[LGBMRegressor] = []
        best_iterations: list[int] = []
        transformed_targets = self.target_adapter.transform_targets(
            targets,
            array_name="training targets",
        )
        transformed_validation_targets = (
            None
            if validation_targets is None
            else self.target_adapter.transform_targets(
                validation_targets,
                array_name="validation targets",
            )
        )
        for target_index in range(targets.shape[1]):
            estimator = LGBMRegressor(**self.params)
            fit_kwargs: dict[str, Any] = {}
            if (
                validation_features is not None
                and transformed_validation_targets is not None
                and training_profile is not None
            ):
                fit_kwargs["callbacks"] = [log_evaluation(period=0)]
                fit_kwargs["eval_set"] = [
                    (validation_features, transformed_validation_targets[:, target_index]),
                ]
                fit_kwargs["eval_metric"] = "l2"
                fit_kwargs["callbacks"].append(
                    early_stopping(
                        stopping_rounds=training_profile.lightgbm_early_stopping_rounds,
                        first_metric_only=training_profile.lightgbm_first_metric_only,
                        min_delta=training_profile.lightgbm_early_stopping_min_delta,
                        verbose=False,
                    )
                )
            estimator.fit(features, transformed_targets[:, target_index], **fit_kwargs)
            estimators.append(estimator)
            best_iteration = estimator.best_iteration_
            if best_iteration is None:
                n_estimators = self.params.get("n_estimators")
                if not isinstance(n_estimators, int):
                    message = "LightGBM params must include integer n_estimators."
                    raise TypeError(message)
                best_iteration = n_estimators
            best_iterations.append(int(best_iteration))
        self.estimators = estimators
        self.best_iterations = tuple(best_iterations)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.estimators:
            message = "LightGBMSurfaceModel must be fit before predict."
            raise ValueError(message)
        predictions = np.column_stack(
            [
                estimator.predict(
                    features,
                    num_iteration=best_iteration,
                )
                for estimator, best_iteration in zip(
                    self.estimators,
                    self.best_iterations,
                    strict=True,
                )
            ]
        )
        return self.target_adapter.inverse_predictions(predictions)
