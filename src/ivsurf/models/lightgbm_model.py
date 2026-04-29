"""LightGBM surface model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from ivsurf.config import TrainingProfileConfig
from ivsurf.features.factors import SurfacePCAFactors
from ivsurf.models.base import SurfaceForecastModel
from ivsurf.models.positive_target import LogPositiveTargetAdapter


class LightGBMSurfaceModel(SurfaceForecastModel):
    """Factor-space LightGBM regression on log total variance."""

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        if "n_factors" not in self.params:
            message = "LightGBM params must include integer n_factors."
            raise KeyError(message)
        raw_n_factors = self.params.pop("n_factors")
        if isinstance(raw_n_factors, bool) or not isinstance(raw_n_factors, int):
            message = (
                "LightGBM n_factors must be an integer, "
                f"found {type(raw_n_factors).__name__}."
            )
            raise TypeError(message)
        self.n_factors = raw_n_factors
        if self.n_factors <= 0:
            message = f"LightGBM n_factors must be positive, found {self.n_factors}."
            raise ValueError(message)
        self.estimators: list[LGBMRegressor] = []
        self.best_iterations: tuple[int, ...] = ()
        self.factorizer: SurfacePCAFactors | None = None
        self.inner_fit_count: int = 0
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
        on_factor_complete: Callable[[int, int], None] | None = None,
    ) -> LightGBMSurfaceModel:
        estimators: list[LGBMRegressor] = []
        best_iterations: list[int] = []
        transformed_targets = self.target_adapter.transform_targets(
            targets,
            array_name="training targets",
        )
        max_factor_count = min(transformed_targets.shape[0], transformed_targets.shape[1])
        if self.n_factors > max_factor_count:
            message = (
                "LightGBMSurfaceModel n_factors exceeds the admissible PCA rank for the "
                f"training window ({self.n_factors} > {max_factor_count})."
            )
            raise ValueError(message)

        factorizer = SurfacePCAFactors(n_components=self.n_factors).fit(transformed_targets)
        target_factors = factorizer.transform(transformed_targets)
        transformed_validation_targets = (
            None
            if validation_targets is None
            else factorizer.transform(
                self.target_adapter.transform_targets(
                    validation_targets,
                    array_name="validation targets",
                )
            )
        )

        for factor_index in range(self.n_factors):
            estimator = LGBMRegressor(**self.params)
            fit_kwargs: dict[str, Any] = {}
            if (
                validation_features is not None
                and transformed_validation_targets is not None
                and training_profile is not None
            ):
                fit_kwargs["callbacks"] = [log_evaluation(period=0)]
                fit_kwargs["eval_set"] = [
                    (validation_features, transformed_validation_targets[:, factor_index]),
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
            estimator.fit(features, target_factors[:, factor_index], **fit_kwargs)
            estimators.append(estimator)
            best_iteration = estimator.best_iteration_
            if best_iteration is None:
                n_estimators = self.params.get("n_estimators")
                if not isinstance(n_estimators, int):
                    message = "LightGBM params must include integer n_estimators."
                    raise TypeError(message)
                best_iteration = n_estimators
            best_iterations.append(int(best_iteration))
            if on_factor_complete is not None:
                on_factor_complete(factor_index + 1, self.n_factors)

        self.estimators = estimators
        self.best_iterations = tuple(best_iterations)
        self.factorizer = factorizer
        self.inner_fit_count = len(estimators)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.estimators or self.factorizer is None:
            message = "LightGBMSurfaceModel must be fit before predict."
            raise ValueError(message)
        predict_kwargs: dict[str, Any] = {}
        n_jobs = self.params.get("n_jobs")
        if n_jobs is not None:
            if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
                message = f"LightGBM n_jobs must be an integer, found {type(n_jobs).__name__}."
                raise TypeError(message)
            predict_kwargs["num_threads"] = n_jobs
        factor_predictions = np.column_stack(
            [
                np.asarray(
                    estimator.booster_.predict(
                        features,
                        num_iteration=best_iteration,
                        **predict_kwargs,
                    ),
                    dtype=np.float64,
                )
                for estimator, best_iteration in zip(
                    self.estimators,
                    self.best_iterations,
                    strict=True,
                )
            ]
        )
        reconstructed_log_targets = self.factorizer.inverse_transform(factor_predictions)
        return self.target_adapter.inverse_predictions(reconstructed_log_targets)
