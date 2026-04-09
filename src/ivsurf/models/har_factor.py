"""HAR-style factor surface model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ivsurf.features.factors import SurfacePCAFactors
from ivsurf.models.base import SurfaceForecastModel
from ivsurf.models.positive_target import LogPositiveTargetAdapter


class HarFactorSurfaceModel(SurfaceForecastModel):
    """Project surfaces to factors, run HAR-style regression, then reconstruct."""

    def __init__(self, n_factors: int, alpha: float, target_dim: int) -> None:
        self.n_factors = n_factors
        self.alpha = alpha
        self.target_dim = target_dim
        self.feature_scaler = StandardScaler()
        self.factor_model = Ridge(alpha=alpha)
        self.factorizer = SurfacePCAFactors(n_components=n_factors)
        self.target_adapter = LogPositiveTargetAdapter("HarFactorSurfaceModel")

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
    ) -> HarFactorSurfaceModel:
        lag_1 = features[:, : self.target_dim]
        lag_5 = features[:, self.target_dim : (2 * self.target_dim)]
        lag_22 = features[:, (2 * self.target_dim) : (3 * self.target_dim)]
        transformed_targets = self.target_adapter.transform_targets(
            targets,
            array_name="training targets",
        )
        transformed_lag_1 = self.target_adapter.transform_targets(
            lag_1,
            array_name="lag_1 features",
        )
        transformed_lag_5 = self.target_adapter.transform_targets(
            lag_5,
            array_name="lag_5 features",
        )
        transformed_lag_22 = self.target_adapter.transform_targets(
            lag_22,
            array_name="lag_22 features",
        )

        self.factorizer.fit(transformed_targets)
        target_factors = self.factorizer.transform(transformed_targets)
        har_features = np.concatenate(
            [
                self.factorizer.transform(transformed_lag_1),
                self.factorizer.transform(transformed_lag_5),
                self.factorizer.transform(transformed_lag_22),
            ],
            axis=1,
        )
        scaled_features = self.feature_scaler.fit_transform(har_features)
        self.factor_model.fit(scaled_features, target_factors)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        lag_1 = features[:, : self.target_dim]
        lag_5 = features[:, self.target_dim : (2 * self.target_dim)]
        lag_22 = features[:, (2 * self.target_dim) : (3 * self.target_dim)]
        har_features = np.concatenate(
            [
                self.factorizer.transform(
                    self.target_adapter.transform_targets(lag_1, array_name="lag_1 features")
                ),
                self.factorizer.transform(
                    self.target_adapter.transform_targets(lag_5, array_name="lag_5 features")
                ),
                self.factorizer.transform(
                    self.target_adapter.transform_targets(lag_22, array_name="lag_22 features")
                ),
            ],
            axis=1,
        )
        predicted_factors = self.factor_model.predict(self.feature_scaler.transform(har_features))
        return self.target_adapter.inverse_predictions(
            self.factorizer.inverse_transform(predicted_factors)
        )
