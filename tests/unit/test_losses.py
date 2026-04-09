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
