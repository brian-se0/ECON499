"""Generic walk-forward fitting utilities for numpy-based models."""

from __future__ import annotations

import numpy as np

from ivsurf.models.base import DatasetMatrices, SurfaceForecastModel


def fit_and_predict(
    model: SurfaceForecastModel,
    train_index: np.ndarray,
    test_index: np.ndarray,
    matrices: DatasetMatrices,
) -> np.ndarray:
    """Fit on the training slice and return test predictions."""

    model.fit(
        features=matrices.features[train_index],
        targets=matrices.targets[train_index],
        observed_masks=matrices.observed_masks[train_index],
        vega_weights=matrices.vega_weights[train_index],
        training_weights=matrices.training_weights[train_index],
    )
    return model.predict(matrices.features[test_index])
