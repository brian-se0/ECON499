from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import optuna
import pytest

from ivsurf.config import NeuralModelConfig, TrainingProfileConfig
from ivsurf.models.lightgbm_model import LightGBMSurfaceModel
from ivsurf.models.neural_surface import NeuralSurfaceRegressor
from ivsurf.models.no_change import validate_no_change_feature_layout
from ivsurf.training.model_factory import suggest_model_from_trial


def _fit_small_lightgbm_model() -> tuple[LightGBMSurfaceModel, np.ndarray]:
    train_features = np.asarray(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        dtype=np.float64,
    )
    train_targets = np.asarray(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
        dtype=np.float64,
    )
    validation_features = np.asarray([[1.5], [2.5], [3.5]], dtype=np.float64)
    validation_targets = np.asarray(
        [[0.4, 0.5], [0.6, 0.7], [0.8, 0.9]],
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
    return model, validation_features


def test_neural_training_uses_validation_early_stopping() -> None:
    features = np.zeros((12, 4), dtype=np.float64)
    targets = np.zeros((12, 2), dtype=np.float64)
    observed_masks = np.ones((12, 2), dtype=np.float64)
    vega_weights = np.ones((12, 2), dtype=np.float64)
    training_weights = np.ones((12, 2), dtype=np.float64)
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
        training_weights=training_weights[:8],
        validation_features=features[8:],
        validation_targets=targets[8:],
        validation_observed_masks=observed_masks[8:],
        validation_vega_weights=vega_weights[8:],
        training_profile=training_profile,
    )

    assert model.best_epoch == 1
    assert model.epochs_completed == 2
    assert model.best_validation_score is not None
    predictions = model.predict(features[8:])
    assert predictions.shape == (4, 2)
    assert (predictions > 0.0).all()


def test_lightgbm_training_uses_validation_early_stopping() -> None:
    model, validation_features = _fit_small_lightgbm_model()

    assert model.best_iterations
    assert all(best_iteration < 50 for best_iteration in model.best_iterations)
    predictions = model.predict(validation_features)
    assert predictions.shape == (3, 2)
    assert (predictions > 0.0).all()


def test_lightgbm_sklearn_wrapper_predict_warns_on_numpy_inputs() -> None:
    model, validation_features = _fit_small_lightgbm_model()

    estimator = model.estimators[0]
    best_iteration = model.best_iterations[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw_predictions = estimator.predict(
            validation_features,
            num_iteration=best_iteration,
        )

    assert raw_predictions.shape == (3,)
    assert any(
        "X does not have valid feature names" in str(item.message)
        for item in caught
    )


def test_lightgbm_model_predict_uses_booster_without_warning_and_matches_wrapper() -> None:
    model, validation_features = _fit_small_lightgbm_model()

    wrapper_predictions: list[np.ndarray] = []
    with warnings.catch_warnings(record=True) as wrapper_caught:
        warnings.simplefilter("always")
        for estimator, best_iteration in zip(
            model.estimators,
            model.best_iterations,
            strict=True,
        ):
            wrapper_predictions.append(
                np.asarray(
                    estimator.predict(
                        validation_features,
                        num_iteration=best_iteration,
                    ),
                    dtype=np.float64,
                )
            )
    expected_predictions = model.target_adapter.inverse_predictions(
        np.column_stack(wrapper_predictions)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions = model.predict(validation_features)

    assert predictions.shape == (3, 2)
    assert np.allclose(predictions, expected_predictions, rtol=1.0e-12, atol=1.0e-12)
    assert any(
        "X does not have valid feature names" in str(item.message)
        for item in wrapper_caught
    )
    assert not any(
        "X does not have valid feature names" in str(item.message)
        for item in caught
    )


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


def test_no_change_feature_layout_guard_accepts_aligned_lag1_surface_columns() -> None:
    target_columns = ("target_total_variance_0000", "target_total_variance_0001")
    feature_columns = (
        "feature_surface_mean_01_0000",
        "feature_surface_mean_01_0001",
        "feature_surface_mean_05_0000",
    )

    validate_no_change_feature_layout(
        feature_columns=feature_columns,
        target_columns=target_columns,
    )


def test_no_change_feature_layout_guard_rejects_missing_or_reordered_lag1_columns() -> None:
    target_columns = ("target_total_variance_0000", "target_total_variance_0001")
    feature_columns = (
        "feature_surface_mean_01_0001",
        "feature_surface_mean_05_0000",
        "feature_surface_mean_01_0000",
    )

    with pytest.raises(ValueError, match="lag-1 surface features"):
        validate_no_change_feature_layout(
            feature_columns=feature_columns,
            target_columns=target_columns,
        )
