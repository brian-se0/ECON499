"""HAR-style factor surface model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ivsurf.features.factors import SurfacePCAFactors
from ivsurf.models.base import SurfaceForecastModel


class HarFactorSurfaceModel(SurfaceForecastModel):
    """Project surfaces to factors, run HAR-style regression, then reconstruct."""

    def __init__(self, n_factors: int, alpha: float, target_dim: int) -> None:
        self.n_factors = n_factors
        self.alpha = alpha
        self.target_dim = target_dim
        self.feature_scaler = StandardScaler()
        self.factor_model = Ridge(alpha=alpha)
        self.factorizer = SurfacePCAFactors(n_components=n_factors)

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

        self.factorizer.fit(targets)
        target_factors = self.factorizer.transform(targets)
        har_features = np.concatenate(
            [
                self.factorizer.transform(lag_1),
                self.factorizer.transform(lag_5),
                self.factorizer.transform(lag_22),
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
                self.factorizer.transform(lag_1),
                self.factorizer.transform(lag_5),
                self.factorizer.transform(lag_22),
            ],
            axis=1,
        )
        predicted_factors = self.factor_model.predict(self.feature_scaler.transform(har_features))
        return self.factorizer.inverse_transform(predicted_factors)

