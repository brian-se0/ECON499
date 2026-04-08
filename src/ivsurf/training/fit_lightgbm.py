"""Validation-aware LightGBM fitting utilities."""

from __future__ import annotations

import numpy as np

from ivsurf.config import TrainingProfileConfig
from ivsurf.models.base import DatasetMatrices
from ivsurf.models.lightgbm_model import LightGBMSurfaceModel


def fit_and_predict_lightgbm(
    model: LightGBMSurfaceModel,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    predict_index: np.ndarray,
    matrices: DatasetMatrices,
    training_profile: TrainingProfileConfig,
) -> np.ndarray:
    """Fit LightGBM with validation-aware early stopping and return predictions."""

    model.fit(
        features=matrices.features[train_index],
        targets=matrices.targets[train_index],
        observed_masks=matrices.observed_masks[train_index],
        vega_weights=matrices.vega_weights[train_index],
        validation_features=matrices.features[validation_index],
        validation_targets=matrices.targets[validation_index],
        training_profile=training_profile,
    )
    return model.predict(matrices.features[predict_index])
