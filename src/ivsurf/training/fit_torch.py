"""Torch-specific training utility."""

from __future__ import annotations

import numpy as np

from ivsurf.models.base import DatasetMatrices
from ivsurf.models.neural_surface import NeuralSurfaceRegressor


def fit_and_predict_neural(
    model: NeuralSurfaceRegressor,
    train_index: np.ndarray,
    test_index: np.ndarray,
    matrices: DatasetMatrices,
) -> np.ndarray:
    """Fit the neural model and return test predictions."""

    model.fit(
        features=matrices.features[train_index],
        targets=matrices.targets[train_index],
        observed_masks=matrices.observed_masks[train_index],
        vega_weights=matrices.vega_weights[train_index],
    )
    return model.predict(matrices.features[test_index])

