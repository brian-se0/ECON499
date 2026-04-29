from __future__ import annotations

import pytest
import torch

from ivsurf.models.losses import weighted_surface_mse


def test_weighted_surface_mse_changes_when_imputed_loss_weight_changes() -> None:
    predictions = torch.tensor([[1.0, 4.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    observed_mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    training_weights = torch.tensor([[2.0, 1.0]], dtype=torch.float32)

    observed_only_loss = weighted_surface_mse(
        predictions=predictions,
        targets=targets,
        observed_mask=observed_mask,
        training_weights=training_weights,
        observed_loss_weight=1.0,
        imputed_loss_weight=0.0,
    )
    completed_surface_loss = weighted_surface_mse(
        predictions=predictions,
        targets=targets,
        observed_mask=observed_mask,
        training_weights=training_weights,
        observed_loss_weight=1.0,
        imputed_loss_weight=0.25,
    )

    assert float(observed_only_loss.item()) == 1.0
    assert float(completed_surface_loss.item()) == pytest.approx(4.25 / 2.25)


def test_weighted_surface_mse_rejects_negative_training_weights() -> None:
    with pytest.raises(ValueError, match="nonnegative training_weights"):
        weighted_surface_mse(
            predictions=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            targets=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            observed_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            training_weights=torch.tensor([[1.0, -1.0]], dtype=torch.float32),
            observed_loss_weight=1.0,
            imputed_loss_weight=0.25,
        )


def test_weighted_surface_mse_rejects_nonbinary_masks() -> None:
    with pytest.raises(ValueError, match="binary 0/1"):
        weighted_surface_mse(
            predictions=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            targets=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            observed_mask=torch.tensor([[1.0, 0.5]], dtype=torch.float32),
            training_weights=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            observed_loss_weight=1.0,
            imputed_loss_weight=0.25,
        )


def test_weighted_surface_mse_rejects_nonfinite_loss_weights() -> None:
    with pytest.raises(ValueError, match="observed_loss_weight"):
        weighted_surface_mse(
            predictions=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            targets=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            observed_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            training_weights=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            observed_loss_weight=float("nan"),
            imputed_loss_weight=0.25,
        )
