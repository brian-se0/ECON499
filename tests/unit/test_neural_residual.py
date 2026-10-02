from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from ivcast.config import NeuralModelConfig, TrainingProfileConfig
from ivcast.models.neural_residual import (
    NeuralResidualMLP,
    NeuralResidualSurfaceRegressor,
    reconstruct_total_variance_from_residual,
)

GRID_SHAPE = (3, 3)
MONEYNESS_POINTS = (-0.1, 0.0, 0.1)
OUTPUT_DIM = GRID_SHAPE[0] * GRID_SHAPE[1]
FLOOR = 1.0e-8


def _cpu_config(**overrides: object) -> NeuralModelConfig:
    payload: dict[str, object] = {
        "model_name": "neural_surface_residual",
        "hidden_width": 16,
        "depth": 2,
        "dropout": 0.1,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "epochs": 2,
        "batch_size": 4,
        "seed": 7,
        "calendar_penalty_weight": 0.01,
        "convexity_penalty_weight": 0.01,
        "roughness_penalty_weight": 0.001,
        "output_total_variance_floor": FLOOR,
        "device": "cpu",
    }
    payload.update(overrides)
    return NeuralModelConfig.model_validate(payload)


def _training_profile() -> TrainingProfileConfig:
    return TrainingProfileConfig.model_validate(
        {
            "profile_name": "test_profile",
            "epochs": 2,
            "neural_early_stopping_patience": 2,
            "neural_early_stopping_min_delta": 0.0,
            "neural_min_epochs_before_early_stop": 1,
            "lightgbm_early_stopping_rounds": 5,
            "lightgbm_early_stopping_min_delta": 0.0,
            "lightgbm_first_metric_only": True,
        }
    )


def _synthetic_dataset(
    n_rows: int,
    n_extra_features: int = 5,
    seed: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lag1 = rng.uniform(1.0e-4, 0.2, size=(n_rows, OUTPUT_DIM))
    extra = rng.normal(size=(n_rows, n_extra_features))
    features = np.concatenate([lag1, extra], axis=1)
    targets = np.clip(
        lag1 * rng.uniform(0.9, 1.1, size=lag1.shape),
        FLOOR,
        None,
    )
    observed_masks = np.ones_like(targets)
    vega_weights = np.ones_like(targets)
    training_weights = np.ones_like(targets)
    return features, targets, observed_masks, vega_weights, training_weights


def test_zero_initialized_mlp_outputs_exactly_zero_residual() -> None:
    torch.manual_seed(11)
    model = NeuralResidualMLP(
        input_dim=OUTPUT_DIM + 5,
        output_dim=OUTPUT_DIM,
        hidden_width=16,
        depth=2,
        dropout=0.1,
    ).eval()
    features = torch.randn(7, OUTPUT_DIM + 5)
    residual = model(features)
    assert residual.dtype == torch.float64
    assert torch.all(residual == 0.0)


@settings(max_examples=50, deadline=None)
@given(
    lag1_values=st.lists(
        st.floats(min_value=1.0e-8, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=OUTPUT_DIM,
        max_size=OUTPUT_DIM,
    )
)
def test_zero_residual_reconstruction_is_persistence(lag1_values: list[float]) -> None:
    lag1 = torch.tensor([lag1_values], dtype=torch.float64)
    log_shifted = torch.log(lag1 + FLOOR)
    reconstructed = reconstruct_total_variance_from_residual(
        torch.zeros_like(lag1),
        log_shifted,
        epsilon=FLOOR,
        floor=FLOOR,
    )
    np.testing.assert_allclose(
        reconstructed.numpy(),
        lag1.numpy(),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


@settings(max_examples=50, deadline=None)
@given(
    lag1_value=st.floats(
        min_value=1.0e-6, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    residual_value=st.floats(
        min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False
    ),
)
def test_reconstruction_round_trips_the_log_innovation(
    lag1_value: float,
    residual_value: float,
) -> None:
    lag1 = torch.tensor([[lag1_value]], dtype=torch.float64)
    residual = torch.tensor([[residual_value]], dtype=torch.float64)
    log_shifted = torch.log(lag1 + FLOOR)
    reconstructed = reconstruct_total_variance_from_residual(
        residual,
        log_shifted,
        epsilon=FLOOR,
        floor=FLOOR,
    )
    assert float(reconstructed) >= FLOOR
    if float(reconstructed) > FLOOR:
        recovered = float(torch.log(reconstructed + FLOOR) - log_shifted)
        assert recovered == pytest.approx(residual_value, rel=1.0e-6, abs=1.0e-9)


def test_reconstruction_clamps_extreme_and_infinite_residuals_to_finite_surfaces() -> None:
    lag1 = torch.full((1, 4), 0.02, dtype=torch.float64)
    log_shifted = torch.log(lag1 + FLOOR)
    extreme = torch.tensor(
        [[500.0, -500.0, float("inf"), -float("inf")]],
        dtype=torch.float64,
    )
    reconstructed = reconstruct_total_variance_from_residual(
        extreme,
        log_shifted,
        epsilon=FLOOR,
        floor=FLOOR,
    )
    assert torch.isfinite(reconstructed).all()
    assert bool((reconstructed >= FLOOR).all())
    # The clamp binds exactly at the documented log-innovation bound.
    import math

    expected_upper = 0.02 * math.exp(20.0)
    assert float(reconstructed[0, 0]) == pytest.approx(expected_upper, rel=1.0e-6)
    assert float(reconstructed[0, 2]) == pytest.approx(expected_upper, rel=1.0e-6)


def test_reconstruction_rejects_nonpositive_epsilon_or_floor() -> None:
    values = torch.ones((1, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="strictly positive epsilon"):
        reconstruct_total_variance_from_residual(values, values, epsilon=0.0, floor=FLOOR)
    with pytest.raises(ValueError, match="strictly positive epsilon"):
        reconstruct_total_variance_from_residual(values, values, epsilon=FLOOR, floor=0.0)


def test_reconstruction_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share one shape"):
        reconstruct_total_variance_from_residual(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.zeros((1, 2), dtype=torch.float64),
            epsilon=FLOOR,
            floor=FLOOR,
        )


def test_untrained_regressor_prediction_path_reproduces_lag1_surface() -> None:
    features, _targets, _, _, _ = _synthetic_dataset(6)
    config = _cpu_config()
    regressor = NeuralResidualSurfaceRegressor(
        config=config,
        grid_shape=GRID_SHAPE,
        moneyness_points=MONEYNESS_POINTS,
    )
    torch.manual_seed(config.seed)
    model = NeuralResidualMLP(
        input_dim=features.shape[1],
        output_dim=OUTPUT_DIM,
        hidden_width=config.hidden_width,
        depth=config.depth,
        dropout=config.dropout,
    ).eval()
    mean = features.mean(axis=0)
    scale = np.where(features.std(axis=0) > 0.0, features.std(axis=0), 1.0)
    predictions = regressor._predict_from_arrays(
        model,
        device=torch.device("cpu"),
        standardized_features=(features - mean) / scale,
        raw_features=features,
        output_dim=OUTPUT_DIM,
    )
    np.testing.assert_allclose(predictions, features[:, :OUTPUT_DIM], rtol=1.0e-10)


def test_fit_and_predict_returns_positive_floored_surfaces() -> None:
    features, targets, observed, vega, training_weights = _synthetic_dataset(24)
    regressor = NeuralResidualSurfaceRegressor(
        config=_cpu_config(),
        grid_shape=GRID_SHAPE,
        moneyness_points=MONEYNESS_POINTS,
    )
    regressor.fit(
        features=features,
        targets=targets,
        observed_masks=observed,
        vega_weights=vega,
        training_weights=training_weights,
        validation_features=features,
        validation_targets=targets,
        validation_observed_masks=observed,
        validation_vega_weights=vega,
        training_profile=_training_profile(),
    )
    predictions = regressor.predict(features)
    assert predictions.shape == targets.shape
    assert np.isfinite(predictions).all()
    assert np.all(predictions >= FLOOR)
    assert regressor.best_epoch is not None
    assert regressor.validation_diagnostics is not None
    # Persistence-anchored training must not collapse to near-zero surfaces.
    assert regressor.validation_diagnostics.prediction_below_1e_6_share < 0.5


def test_fit_requires_observed_masks_and_training_weights() -> None:
    features, targets, observed, _vega, training_weights = _synthetic_dataset(8)
    regressor = NeuralResidualSurfaceRegressor(
        config=_cpu_config(),
        grid_shape=GRID_SHAPE,
        moneyness_points=MONEYNESS_POINTS,
    )
    with pytest.raises(ValueError, match="requires observed_masks"):
        regressor.fit(features, targets, training_weights=training_weights)
    with pytest.raises(ValueError, match="requires training_weights"):
        regressor.fit(features, targets, observed_masks=observed)


def test_fit_rejects_nonpositive_lag1_block() -> None:
    features, targets, observed, vega, training_weights = _synthetic_dataset(8)
    features[0, 0] = 0.0
    regressor = NeuralResidualSurfaceRegressor(
        config=_cpu_config(),
        grid_shape=GRID_SHAPE,
        moneyness_points=MONEYNESS_POINTS,
    )
    with pytest.raises(ValueError, match="strictly positive"):
        regressor.fit(
            features,
            targets,
            observed_masks=observed,
            vega_weights=vega,
            training_weights=training_weights,
        )


def test_predict_before_fit_raises() -> None:
    features, _, _, _, _ = _synthetic_dataset(4)
    regressor = NeuralResidualSurfaceRegressor(
        config=_cpu_config(),
        grid_shape=GRID_SHAPE,
        moneyness_points=MONEYNESS_POINTS,
    )
    with pytest.raises(ValueError, match="must be fit before predict"):
        regressor.predict(features)
