from __future__ import annotations

from typing import cast

import numpy as np
import optuna

from ivsurf.config import NeuralModelConfig, TrainingProfileConfig
from ivsurf.models.lightgbm_model import LightGBMSurfaceModel
from ivsurf.models.neural_surface import NeuralSurfaceRegressor
from ivsurf.training.model_factory import suggest_model_from_trial


def test_neural_training_uses_validation_early_stopping() -> None:
    features = np.zeros((12, 4), dtype=np.float64)
    targets = np.zeros((12, 2), dtype=np.float64)
    observed_masks = np.ones((12, 2), dtype=np.float64)
    vega_weights = np.ones((12, 2), dtype=np.float64)
    config = NeuralModelConfig(
        hidden_width=8,
        depth=1,
        dropout=0.0,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        epochs=50,
        batch_size=4,
        seed=7,
        device="cpu",
    )
    training_profile = TrainingProfileConfig(
        profile_name="test_train_profile",
        epochs=10,
        neural_early_stopping_patience=1,
        neural_early_stopping_min_delta=1.0,
        lightgbm_early_stopping_rounds=3,
        lightgbm_early_stopping_min_delta=0.0,
    )
    model = NeuralSurfaceRegressor(config=config, grid_shape=(1, 2))

    model.fit(
        features=features[:8],
        targets=targets[:8],
        observed_masks=observed_masks[:8],
        vega_weights=vega_weights[:8],
        validation_features=features[8:],
        validation_targets=targets[8:],
        validation_observed_masks=observed_masks[8:],
        validation_vega_weights=vega_weights[8:],
        training_profile=training_profile,
    )

    assert model.best_epoch == 1
    assert model.epochs_completed == 2
    assert model.best_validation_score is not None
    assert model.predict(features[8:]).shape == (4, 2)


def test_lightgbm_training_uses_validation_early_stopping() -> None:
    train_features = np.asarray(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        dtype=np.float64,
    )
    train_targets = np.asarray(
        [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9], [1.0, 1.1]],
        dtype=np.float64,
    )
    validation_features = np.asarray([[1.5], [2.5], [3.5]], dtype=np.float64)
    validation_targets = np.asarray(
        [[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
        dtype=np.float64,
    )
    training_profile = TrainingProfileConfig(
        profile_name="test_train_profile",
        epochs=10,
        neural_early_stopping_patience=1,
        neural_early_stopping_min_delta=0.0,
        lightgbm_early_stopping_rounds=3,
        lightgbm_early_stopping_min_delta=10.0,
    )
    model = LightGBMSurfaceModel(
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=7,
        max_depth=3,
        min_child_samples=1,
        feature_fraction=1.0,
        lambda_l2=0.0,
        objective="regression",
        metric="l2",
        verbosity=-1,
    )

    model.fit(
        features=train_features,
        targets=train_targets,
        validation_features=validation_features,
        validation_targets=validation_targets,
        training_profile=training_profile,
    )

    assert model.best_iterations
    assert all(best_iteration < 50 for best_iteration in model.best_iterations)
    assert model.predict(validation_features).shape == (3, 2)


def test_lightgbm_tuning_respects_configured_device_type() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "feature_fraction": 0.9,
            "lambda_l2": 1.0,
        }
    )

    model = suggest_model_from_trial(
        model_name="lightgbm",
        trial=cast(optuna.Trial, trial),
        target_dim=4,
        grid_shape=(2, 2),
        base_neural_config=NeuralModelConfig(device="cpu"),
        base_lightgbm_params={
            "device_type": "cpu",
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": 7,
        },
    )

    assert isinstance(model, LightGBMSurfaceModel)
    assert model.params["device_type"] == "cpu"
