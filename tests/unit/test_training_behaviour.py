from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import optuna
import pytest
import torch
from sklearn.exceptions import ConvergenceWarning

from ivsurf.config import NeuralModelConfig, TrainingProfileConfig
from ivsurf.exceptions import ModelConvergenceError
from ivsurf.models.elasticnet import ElasticNetSurfaceModel
from ivsurf.models.lightgbm_model import LightGBMSurfaceModel
from ivsurf.models.neural_surface import NeuralSurfaceRegressor
from ivsurf.models.no_change import validate_no_change_feature_layout
from ivsurf.models.ridge import RidgeSurfaceModel
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
        n_factors=2,
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
    features = np.arange(48, dtype=np.float64).reshape(12, 4)
    targets = np.full((12, 2), 0.1, dtype=np.float64)
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
        neural_min_epochs_before_early_stop=3,
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
    assert model.epochs_completed == 3
    assert model.best_validation_score is not None
    assert model.feature_mean is not None
    assert np.allclose(model.feature_mean, features[:8].mean(axis=0))
    predictions = model.predict(features[8:])
    assert predictions.shape == (4, 2)
    assert (predictions > 0.0).all()
    assert np.isfinite(predictions).all()
    assert model.training_log_target_min is not None
    assert model.training_log_target_max is not None


def test_neural_prediction_is_bounded_to_training_log_target_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
    targets = np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype=np.float64)
    observed_masks = np.ones((3, 2), dtype=np.float64)
    vega_weights = np.ones((3, 2), dtype=np.float64)
    training_weights = np.ones((3, 2), dtype=np.float64)
    model = NeuralSurfaceRegressor(
        config=NeuralModelConfig(
            hidden_width=4,
            depth=1,
            dropout=0.0,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            epochs=2,
            batch_size=2,
            seed=7,
            device="cpu",
        ),
        grid_shape=(1, 2),
    )

    model.fit(
        features=features,
        targets=targets,
        observed_masks=observed_masks,
        vega_weights=vega_weights,
        training_weights=training_weights,
    )

    def fake_forward(batch_features: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (batch_features.shape[0], targets.shape[1]),
            1.0e9,
            dtype=batch_features.dtype,
            device=batch_features.device,
        )

    assert model.model is not None
    monkeypatch.setattr(model.model, "forward", fake_forward)
    predictions = model.predict(np.asarray([[10.0]], dtype=np.float64))

    assert np.isfinite(predictions).all()
    assert model.training_log_target_max is not None
    assert np.allclose(predictions[0], np.exp(model.training_log_target_max))


def test_lightgbm_training_uses_validation_early_stopping() -> None:
    model, validation_features = _fit_small_lightgbm_model()

    assert model.best_iterations
    assert all(best_iteration < 50 for best_iteration in model.best_iterations)
    assert model.inner_fit_count == model.n_factors
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
    assert model.factorizer is not None
    expected_predictions = model.target_adapter.inverse_predictions(
        model.factorizer.inverse_transform(np.column_stack(wrapper_predictions))
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
            "n_factors": 3,
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
    assert model.n_factors == 3


def test_elasticnet_fit_raises_typed_convergence_error_on_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ElasticNetSurfaceModel(alpha=0.1, l1_ratio=0.5, max_iter=50_000, tol=1.0e-4)

    def fake_fit(features: np.ndarray, targets: np.ndarray) -> object:
        warnings.warn("objective did not converge", ConvergenceWarning, stacklevel=2)
        return model.pipeline

    monkeypatch.setattr(model.pipeline, "fit", fake_fit)

    with pytest.raises(ModelConvergenceError, match="convergence warning"):
        model.fit(
            np.ones((4, 2), dtype=np.float64),
            np.full((4, 2), 0.1, dtype=np.float64),
        )


def test_ridge_prediction_clipping_stays_within_training_log_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RidgeSurfaceModel(alpha=1.0, clip_predictions_to_train_log_range=True)
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
    targets = np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype=np.float64)
    model.fit(features, targets)

    def fake_predict(features: np.ndarray) -> np.ndarray:
        return np.full((features.shape[0], 2), 100.0, dtype=np.float64)

    monkeypatch.setattr(model.pipeline, "predict", fake_predict)
    predictions = model.predict(np.asarray([[10.0], [11.0]], dtype=np.float64))

    assert np.isfinite(predictions).all()
    assert model.training_log_target_max is not None
    assert np.allclose(predictions[0], np.exp(model.training_log_target_max))
    assert model.last_clipped_prediction_count == 4


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
