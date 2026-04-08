"""Arbitrage-aware neural surface model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ivsurf.config import NeuralModelConfig
from ivsurf.models.base import SurfaceForecastModel
from ivsurf.models.losses import weighted_surface_mse
from ivsurf.models.penalties import (
    calendar_monotonicity_penalty,
    convexity_penalty,
    roughness_penalty,
)


class NeuralSurfaceMLP(nn.Module):
    """Compact MLP for joint surface prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_width
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(features))


@dataclass(slots=True)
class NeuralSurfaceRegressor(SurfaceForecastModel):
    """Torch regressor wrapper with explicit penalty weights."""

    config: NeuralModelConfig
    grid_shape: tuple[int, int]
    model: NeuralSurfaceMLP | None = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
    ) -> NeuralSurfaceRegressor:
        if observed_masks is None:
            message = "NeuralSurfaceRegressor requires observed_masks."
            raise ValueError(message)

        torch.manual_seed(self.config.seed)
        device = torch.device(self.config.device)
        input_dim = features.shape[1]
        output_dim = targets.shape[1]
        model = NeuralSurfaceMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_width=self.config.hidden_width,
            depth=self.config.depth,
            dropout=self.config.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        autocast_device = device.type if device.type != "cpu" else "cpu"
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        dataset = TensorDataset(
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(targets, dtype=torch.float32),
            torch.as_tensor(observed_masks, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        model.train()
        for _ in range(self.config.epochs):
            for batch_features, batch_targets, batch_masks in loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                batch_masks = batch_masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=autocast_device, enabled=device.type == "cuda"):
                    predictions = model(batch_features)
                    loss = weighted_surface_mse(
                        predictions=predictions,
                        targets=batch_targets,
                        observed_mask=batch_masks,
                        observed_loss_weight=self.config.observed_loss_weight,
                        imputed_loss_weight=self.config.imputed_loss_weight,
                    )
                    loss = loss + (
                        self.config.calendar_penalty_weight
                        * calendar_monotonicity_penalty(predictions, self.grid_shape)
                    )
                    loss = loss + (
                        self.config.convexity_penalty_weight
                        * convexity_penalty(predictions, self.grid_shape)
                    )
                    loss = loss + (
                        self.config.roughness_penalty_weight
                        * roughness_penalty(predictions, self.grid_shape)
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        self.model = model.eval()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            message = "NeuralSurfaceRegressor must be fit before predict."
            raise ValueError(message)
        device = torch.device(self.config.device)
        with torch.inference_mode():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            predictions = np.asarray(self.model(tensor).cpu().numpy(), dtype=np.float64)
        return predictions
