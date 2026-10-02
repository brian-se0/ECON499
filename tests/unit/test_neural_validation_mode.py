from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ivcast.config import NeuralModelConfig, TrainingProfileConfig
from ivcast.evaluation.loss_panels import mean_daily_loss_metric
from ivcast.models.neural_residual import NeuralResidualSurfaceRegressor
from ivcast.models.neural_surface import NeuralSurfaceRegressor


@pytest.mark.parametrize("regressor_type", [NeuralSurfaceRegressor, NeuralResidualSurfaceRegressor])
def test_dropout_validation_matches_inference_and_restores_training(
    regressor_type: type[NeuralSurfaceRegressor] | type[NeuralResidualSurfaceRegressor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = np.random.default_rng(17)
    features = generator.uniform(0.01, 0.2, size=(24, 6))
    targets = features[:, :3] * 1.15
    weights = np.ones_like(targets)
    config = NeuralModelConfig(
        device="cpu", hidden_width=16, depth=2, dropout=0.3,
        learning_rate=0.01, epochs=3, batch_size=4, seed=7,
        calendar_penalty_weight=0.0, convexity_penalty_weight=0.0,
        roughness_penalty_weight=0.0,
    )
    regressor = regressor_type(config, (1, 3), (-0.1, 0.0, 0.1))
    forward_modes: list[tuple[bool, bool]] = []
    original_forward = nn.Dropout.forward

    def observe_dropout(module: nn.Dropout, inputs: torch.Tensor) -> torch.Tensor:
        forward_modes.append((torch.is_inference_mode_enabled(), module.training))
        return original_forward(module, inputs)

    monkeypatch.setattr(nn.Dropout, "forward", observe_dropout)
    regressor.fit(
        features=features[:16], targets=targets[:16],
        observed_masks=weights[:16], training_weights=weights[:16],
        vega_weights=weights[:16],
        validation_features=features[16:], validation_targets=targets[16:],
        validation_observed_masks=weights[16:], validation_vega_weights=weights[16:],
        training_profile=TrainingProfileConfig(profile_name="test_validation", epochs=3),
    )

    first_validation = next(
        index for index, (inference, _) in enumerate(forward_modes) if inference
    )
    assert any(not inference for inference, _ in forward_modes[first_validation + 1:])
    assert all(not training for inference, training in forward_modes if inference)
    assert all(training for inference, training in forward_modes if not inference)
    assert regressor.model is not None
    assert not regressor.model.training
    predictions = regressor.predict(features[16:])
    np.testing.assert_array_equal(predictions, regressor.predict(features[16:]))
    inference_score = mean_daily_loss_metric(
        metric_name="observed_mse_total_variance", y_true=targets[16:],
        y_pred=predictions, observed_masks=weights[16:], vega_weights=weights[16:],
        positive_floor=1.0e-8,
    )
    assert regressor.best_validation_score == pytest.approx(inference_score, rel=1.0e-12)
    assert regressor.validation_diagnostics is not None
    assert regressor.validation_diagnostics.prediction_mean == pytest.approx(
        float(np.mean(predictions)), rel=1.0e-12,
    )
