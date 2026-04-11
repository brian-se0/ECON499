"""Elastic-net surface model."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ivsurf.exceptions import ModelConvergenceError
from ivsurf.models.base import SurfaceForecastModel
from ivsurf.models.positive_target import LogPositiveTargetAdapter


class ElasticNetSurfaceModel(SurfaceForecastModel):
    """Multi-output elastic net on the daily feature matrix."""

    def __init__(self, alpha: float, l1_ratio: float, max_iter: int, tol: float) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.last_convergence_warning_message: str | None = None
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MultiTaskElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=max_iter,
                        tol=tol,
                    ),
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
        training_weights: np.ndarray | None = None,
    ) -> ElasticNetSurfaceModel:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                self.pipeline.fit(
                    features,
                    self.target_adapter.transform_targets(targets, array_name="training targets"),
                )
        except ConvergenceWarning as exc:
            self.last_convergence_warning_message = str(exc)
            message = (
                "ElasticNetSurfaceModel emitted a convergence warning under explicit fit "
                f"settings alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
                f"max_iter={self.max_iter}, tol={self.tol}: {exc}"
            )
            raise ModelConvergenceError(message) from exc
        self.last_convergence_warning_message = None
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.target_adapter.inverse_predictions(self.pipeline.predict(features))
