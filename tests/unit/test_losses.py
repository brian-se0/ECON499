from __future__ import annotations

import torch

from ivsurf.models.losses import weighted_surface_mse


def test_weighted_surface_mse_uses_vega_weighted_observed_cells() -> None:
    predictions = torch.tensor([[1.0, 4.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    observed_mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    vega_weights = torch.tensor([[2.0, 0.0]], dtype=torch.float32)

    loss = weighted_surface_mse(
        predictions=predictions,
        targets=targets,
        observed_mask=observed_mask,
        vega_weights=vega_weights,
        observed_loss_weight=1.0,
        imputed_loss_weight=0.25,
    )

    assert float(loss.item()) == 1.0
