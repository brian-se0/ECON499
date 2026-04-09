"""Torch-specific training utilities."""

from __future__ import annotations

import numpy as np
import optuna

from ivsurf.config import TrainingProfileConfig
from ivsurf.models.base import DatasetMatrices
from ivsurf.models.neural_surface import NeuralSurfaceRegressor


def fit_and_predict_neural(
    model: NeuralSurfaceRegressor,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    predict_index: np.ndarray,
    matrices: DatasetMatrices,
    training_profile: TrainingProfileConfig,
    *,
    trial: optuna.Trial | None = None,
    trial_step_offset: int = 0,
) -> np.ndarray:
    """Fit the neural model with validation-aware stopping and return predictions."""

    model.fit(
        features=matrices.features[train_index],
        targets=matrices.targets[train_index],
        observed_masks=matrices.observed_masks[train_index],
        vega_weights=matrices.vega_weights[train_index],
        training_weights=matrices.training_weights[train_index],
        validation_features=matrices.features[validation_index],
        validation_targets=matrices.targets[validation_index],
        validation_observed_masks=matrices.observed_masks[validation_index],
        validation_vega_weights=matrices.vega_weights[validation_index],
        training_profile=training_profile,
        trial=trial,
        trial_step_offset=trial_step_offset,
    )
    return model.predict(matrices.features[predict_index])
