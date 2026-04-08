"""Compact factor features for tree and HAR-style models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(slots=True)
class SurfacePCAFactors:
    """Fit/transform helper for surface factors."""

    n_components: int
    pca: PCA | None = None

    def fit(self, surfaces: np.ndarray) -> SurfacePCAFactors:
        self.pca = PCA(n_components=self.n_components, svd_solver="full")
        self.pca.fit(surfaces)
        return self

    def transform(self, surfaces: np.ndarray) -> np.ndarray:
        if self.pca is None:
            message = "SurfacePCAFactors must be fit before transform."
            raise ValueError(message)
        return np.asarray(self.pca.transform(surfaces), dtype=np.float64)

    def inverse_transform(self, factors: np.ndarray) -> np.ndarray:
        if self.pca is None:
            message = "SurfacePCAFactors must be fit before inverse_transform."
            raise ValueError(message)
        return np.asarray(self.pca.inverse_transform(factors), dtype=np.float64)
