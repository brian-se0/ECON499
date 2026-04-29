from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import optuna
import pytest
from sklearn.exceptions import ConvergenceWarning

from ivsurf.config import NeuralModelConfig, TrainingProfileConfig
from ivsurf.exceptions import ModelConvergenceError
from ivsurf.models import neural_surface as neural_surface_module
from ivsurf.models.elasticnet import ElasticNetSurfaceModel
from ivsurf.models.har_factor import validate_har_feature_layout
from ivsurf.models.lightgbm_model import LightGBMSurfaceModel
from ivsurf.models.naive import validate_naive_feature_layout
from ivsurf.models.neural_surface import NeuralSurfaceRegressor
from ivsurf.models.ridge import RidgeSurfaceModel
from ivsurf.training.model_factory import make_model_from_params, suggest_model_from_trial


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
        n_jobs=1,
        random_state=7,
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
        calendar_penalty_weight=0.0,
        convexity_penalty_weight=0.0,
        roughness_penalty_weight=0.0,
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
    model = NeuralSurfaceRegressor(
        config=config,
        grid_shape=(1, 2),
        moneyness_points=(-0.1, 0.0),
    )

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


def test_neural_prediction_reuses_train_window_standardization_and_stays_positive() -> None:
    features = np.asarray(
        [
            [1.0e-3, 1.0e3, 7.0],
            [2.0e-3, 2.0e3, 7.0],
            [3.0e-3, 3.0e3, 7.0],
            [4.0e-3, 4.0e3, 7.0],
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [[0.10, 0.20], [0.12, 0.22], [0.14, 0.24], [0.16, 0.26]],
        dtype=np.float64,
    )
    observed_masks = np.ones((4, 2), dtype=np.float64)
    vega_weights = np.ones((4, 2), dtype=np.float64)
    training_weights = np.ones((4, 2), dtype=np.float64)
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
            calendar_penalty_weight=0.0,
            convexity_penalty_weight=0.0,
            roughness_penalty_weight=0.0,
        ),
        grid_shape=(1, 2),
        moneyness_points=(-0.1, 0.0),
    )

    model.fit(
        features=features,
        targets=targets,
        observed_masks=observed_masks,
        vega_weights=vega_weights,
        training_weights=training_weights,
    )
    expected_feature_mean = features.mean(axis=0)
    expected_feature_scale = np.where(features.std(axis=0) > 0.0, features.std(axis=0), 1.0)
    predictions = model.predict(
        np.asarray(
            [
                [5.0e-3, 5.0e3, 7.0],
                [6.0e-3, 6.0e3, 7.0],
            ],
            dtype=np.float64,
        )
    )

    assert model.feature_mean is not None
    assert model.feature_scale is not None
    assert np.allclose(model.feature_mean, expected_feature_mean)
    assert np.allclose(model.feature_scale, expected_feature_scale)
    assert predictions.shape == (2, 2)
    assert np.isfinite(predictions).all()
    assert (predictions > 0.0).all()


def test_neural_regressor_rejects_enabled_roughness_penalty_on_too_small_grid() -> None:
    with pytest.raises(ValueError, match="roughness_penalty_weight"):
        NeuralSurfaceRegressor(
            config=NeuralModelConfig(
                device="cpu",
                calendar_penalty_weight=0.0,
                convexity_penalty_weight=0.0,
                roughness_penalty_weight=0.005,
            ),
            grid_shape=(2, 2),
            moneyness_points=(-0.1, 0.0),
        )


def test_neural_training_rejects_nonfinite_penalty_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonfinite_roughness_penalty(
        predictions: neural_surface_module.torch.Tensor,
        grid_shape: tuple[int, int],
    ) -> neural_surface_module.torch.Tensor:
        return predictions.new_tensor(float("nan"))

    monkeypatch.setattr(neural_surface_module, "roughness_penalty", nonfinite_roughness_penalty)
    model = NeuralSurfaceRegressor(
        config=NeuralModelConfig(
            hidden_width=4,
            depth=1,
            dropout=0.0,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            epochs=1,
            batch_size=2,
            seed=7,
            device="cpu",
            calendar_penalty_weight=0.0,
            convexity_penalty_weight=0.0,
            roughness_penalty_weight=0.005,
        ),
        grid_shape=(3, 3),
        moneyness_points=(-0.1, 0.0, 0.1),
    )

    with pytest.raises(RuntimeError, match="non-finite scalar loss"):
        model.fit(
            features=np.ones((2, 2), dtype=np.float64),
            targets=np.full((2, 9), 0.1, dtype=np.float64),
            observed_masks=np.ones((2, 9), dtype=np.float64),
            vega_weights=np.ones((2, 9), dtype=np.float64),
            training_weights=np.ones((2, 9), dtype=np.float64),
        )


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
        moneyness_points=(-0.1, 0.0),
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


def test_factor_tuning_rejects_trials_above_training_rank_cap() -> None:
    trial = optuna.trial.FixedTrial({"n_factors": 4, "alpha": 1.0})

    with pytest.warns(UserWarning, match="out of the range"), pytest.raises(
        ValueError,
        match="n_factors",
    ):
        suggest_model_from_trial(
            model_name="har_factor",
            trial=cast(optuna.Trial, trial),
            target_dim=9,
            max_factor_count=3,
            grid_shape=(3, 3),
            moneyness_points=(-0.1, 0.0, 0.1),
            base_neural_config=NeuralModelConfig(device="cpu"),
        )


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


def test_ridge_prediction_returns_unclipped_positive_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RidgeSurfaceModel(alpha=1.0)
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
    targets = np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype=np.float64)
    model.fit(features, targets)

    def fake_predict(features: np.ndarray) -> np.ndarray:
        return np.full((features.shape[0], 2), 100.0, dtype=np.float64)

    monkeypatch.setattr(model.pipeline, "predict", fake_predict)
    predictions = model.predict(np.asarray([[10.0], [11.0]], dtype=np.float64))

    assert np.isfinite(predictions).all()
    assert (predictions > 0.0).all()
    assert np.allclose(predictions, np.full((2, 2), np.exp(100.0), dtype=np.float64))


def test_naive_feature_layout_guard_accepts_aligned_lag1_surface_columns() -> None:
    target_columns = ("target_total_variance_0000", "target_total_variance_0001")
    feature_columns = (
        "feature_surface_mean_01_0000",
        "feature_surface_mean_01_0001",
        "feature_surface_mean_05_0000",
    )

    validate_naive_feature_layout(
        feature_columns=feature_columns,
        target_columns=target_columns,
    )


def test_naive_feature_layout_guard_rejects_missing_or_reordered_lag1_columns() -> None:
    target_columns = ("target_total_variance_0000", "target_total_variance_0001")
    feature_columns = (
        "feature_surface_mean_01_0001",
        "feature_surface_mean_05_0000",
        "feature_surface_mean_01_0000",
    )

    with pytest.raises(ValueError, match="lag-1 surface features"):
        validate_naive_feature_layout(
            feature_columns=feature_columns,
            target_columns=target_columns,
        )


def test_har_feature_layout_guard_accepts_aligned_har_lag_blocks() -> None:
    target_columns = ("target_total_variance_0000", "target_total_variance_0001")
    feature_columns = (
        "feature_surface_mean_01_0000",
        "feature_surface_mean_01_0001",
        "feature_surface_mean_05_0000",
        "feature_surface_mean_05_0001",
        "feature_surface_mean_22_0000",
        "feature_surface_mean_22_0001",
        "feature_surface_change_01_0000",
    )

    validate_har_feature_layout(
        feature_columns=feature_columns,
        target_columns=target_columns,
    )


def test_har_feature_layout_guard_rejects_missing_or_reordered_lag_blocks() -> None:
    target_columns = ("target_total_variance_0000", "target_total_variance_0001")
    feature_columns = (
        "feature_surface_mean_01_0000",
        "feature_surface_mean_01_0001",
        "feature_surface_mean_22_0000",
        "feature_surface_mean_22_0001",
        "feature_surface_mean_05_0000",
        "feature_surface_mean_05_0001",
    )

    with pytest.raises(ValueError, match="lag-1, lag-5, and lag-22"):
        validate_har_feature_layout(
            feature_columns=feature_columns,
            target_columns=target_columns,
        )


def test_model_factory_rejects_string_numeric_hyperparameters() -> None:
    with pytest.raises(TypeError, match="alpha"):
        make_model_from_params(
            model_name="ridge",
            params={"alpha": "1.0"},
            target_dim=2,
            grid_shape=(1, 2),
            moneyness_points=(-0.1, 0.0),
            base_neural_config=NeuralModelConfig(device="cpu"),
        )


def test_model_factory_rejects_float_integer_hyperparameters() -> None:
    with pytest.raises(TypeError, match="max_iter"):
        make_model_from_params(
            model_name="elasticnet",
            params={"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 50_000.5, "tol": 1.0e-4},
            target_dim=2,
            grid_shape=(1, 2),
            moneyness_points=(-0.1, 0.0),
            base_neural_config=NeuralModelConfig(device="cpu"),
        )


def test_model_factory_rejects_string_lightgbm_integer_params() -> None:
    params: dict[str, object] = {
        "n_factors": 2,
        "device_type": "cpu",
        "n_estimators": "100",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "lambda_l2": 1.0,
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 7,
    }

    with pytest.raises(TypeError, match="n_estimators"):
        make_model_from_params(
            model_name="lightgbm",
            params=params,
            target_dim=2,
            grid_shape=(1, 2),
            moneyness_points=(-0.1, 0.0),
            base_neural_config=NeuralModelConfig(device="cpu"),
        )


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("naive", {"unused": 1}),
        ("ridge", {"alpha": 1.0, "unused": 1}),
        (
            "elasticnet",
            {
                "alpha": 0.1,
                "l1_ratio": 0.5,
                "max_iter": 50_000,
                "tol": 1.0e-4,
                "unused": 1,
            },
        ),
        ("har_factor", {"n_factors": 2, "alpha": 1.0, "unused": 1}),
        (
            "random_forest",
            {
                "n_estimators": 100,
                "max_depth": 4,
                "min_samples_leaf": 1,
                "random_state": 7,
                "n_jobs": -1,
                "unused": 1,
            },
        ),
        (
            "lightgbm",
            {
                "n_factors": 2,
                "device_type": "cpu",
                "n_estimators": 100,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 20,
                "feature_fraction": 0.9,
                "lambda_l2": 1.0,
                "objective": "regression",
                "metric": "l2",
                "verbosity": -1,
                "n_jobs": -1,
                "random_state": 7,
                "unused": 1,
            },
        ),
        ("neural_surface", {"hidden_width": 64, "depth": 2, "unused": 1}),
    ],
)
def test_model_factory_rejects_unexpected_persisted_params(
    model_name: str,
    params: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=f"Unexpected {model_name} parameters"):
        make_model_from_params(
            model_name=model_name,
            params=params,
            target_dim=2,
            grid_shape=(1, 2),
            moneyness_points=(-0.1, 0.0),
            base_neural_config=NeuralModelConfig(device="cpu"),
        )
