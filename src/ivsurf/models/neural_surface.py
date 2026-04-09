"""Arbitrage-aware neural surface model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ivsurf.config import NeuralModelConfig, TrainingProfileConfig
from ivsurf.evaluation.metrics import weighted_rmse
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
        self.output_activation = nn.Softplus()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.output_activation(self.network(features)))


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        message = "NeuralSurfaceRegressor requested CUDA, but torch.cuda.is_available() is False."
        raise RuntimeError(message)
    return device


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _validation_score(
    model: NeuralSurfaceMLP,
    *,
    device: torch.device,
    features: np.ndarray,
    targets: np.ndarray,
    observed_masks: np.ndarray,
    vega_weights: np.ndarray,
) -> float:
    with torch.inference_mode():
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        predictions = model(feature_tensor).cpu().numpy().astype(np.float64, copy=False)
    weights = observed_masks * np.maximum(vega_weights, 0.0)
    return weighted_rmse(
        y_true=targets.reshape(-1),
        y_pred=predictions.reshape(-1),
        weights=weights.reshape(-1),
    )


@dataclass(slots=True)
class NeuralSurfaceRegressor(SurfaceForecastModel):
    """Torch regressor wrapper with explicit penalty weights."""

    config: NeuralModelConfig
    grid_shape: tuple[int, int]
    model: NeuralSurfaceMLP | None = None
    best_epoch: int | None = None
    epochs_completed: int = 0
    best_validation_score: float | None = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
        *,
        validation_features: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
        validation_observed_masks: np.ndarray | None = None,
        validation_vega_weights: np.ndarray | None = None,
        training_profile: TrainingProfileConfig | None = None,
        trial: optuna.Trial | None = None,
        trial_step_offset: int = 0,
    ) -> NeuralSurfaceRegressor:
        if observed_masks is None:
            message = "NeuralSurfaceRegressor requires observed_masks."
            raise ValueError(message)
        if vega_weights is None:
            message = "NeuralSurfaceRegressor requires vega_weights."
            raise ValueError(message)
        if training_profile is not None and (
            validation_features is None
            or validation_targets is None
            or validation_observed_masks is None
            or validation_vega_weights is None
        ):
            message = "Validation arrays are required when a training_profile is provided."
            raise ValueError(message)

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        device = _resolve_device(self.config.device)
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
        use_cuda = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=True) if use_cuda else None
        dataset = TensorDataset(
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(targets, dtype=torch.float32),
            torch.as_tensor(observed_masks, dtype=torch.float32),
            torch.as_tensor(vega_weights, dtype=torch.float32),
        )
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=use_cuda,
        )

        best_state_dict: dict[str, torch.Tensor] | None = None
        best_epoch: int | None = None
        best_validation_score = float("inf")
        epochs_without_improvement = 0
        max_epochs = training_profile.epochs if training_profile is not None else self.config.epochs
        min_delta = (
            training_profile.neural_early_stopping_min_delta
            if training_profile is not None
            else 0.0
        )
        patience = (
            training_profile.neural_early_stopping_patience
            if training_profile is not None
            else max_epochs
        )

        model.train()
        for epoch in range(max_epochs):
            for batch_features, batch_targets, batch_masks, batch_vega_weights in loader:
                batch_features = batch_features.to(device, non_blocking=use_cuda)
                batch_targets = batch_targets.to(device, non_blocking=use_cuda)
                batch_masks = batch_masks.to(device, non_blocking=use_cuda)
                batch_vega_weights = batch_vega_weights.to(device, non_blocking=use_cuda)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_cuda):
                    predictions = model(batch_features)
                    loss = weighted_surface_mse(
                        predictions=predictions,
                        targets=batch_targets,
                        observed_mask=batch_masks,
                        vega_weights=batch_vega_weights,
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

                if use_cuda:
                    if scaler is None:
                        message = "CUDA training requires an initialized GradScaler."
                        raise RuntimeError(message)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            self.epochs_completed = epoch + 1
            if training_profile is None:
                continue

            if (
                validation_features is None
                or validation_targets is None
                or validation_observed_masks is None
                or validation_vega_weights is None
            ):
                message = "Validation arrays unexpectedly became unavailable during training."
                raise RuntimeError(message)
            validation_score = _validation_score(
                model,
                device=device,
                features=validation_features,
                targets=validation_targets,
                observed_masks=validation_observed_masks,
                vega_weights=validation_vega_weights,
            )
            if validation_score < (best_validation_score - min_delta):
                best_validation_score = validation_score
                best_epoch = epoch + 1
                best_state_dict = _clone_state_dict(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if trial is not None:
                trial.report(validation_score, trial_step_offset + epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if epochs_without_improvement >= patience:
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        elif training_profile is not None:
            if (
                validation_features is None
                or validation_targets is None
                or validation_observed_masks is None
                or validation_vega_weights is None
            ):
                message = "Validation arrays unexpectedly became unavailable after training."
                raise RuntimeError(message)
            best_validation_score = _validation_score(
                model,
                device=device,
                features=validation_features,
                targets=validation_targets,
                observed_masks=validation_observed_masks,
                vega_weights=validation_vega_weights,
            )
            best_epoch = self.epochs_completed

        self.best_epoch = best_epoch
        self.best_validation_score = (
            None if best_validation_score == float("inf") else float(best_validation_score)
        )
        self.model = model.eval()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            message = "NeuralSurfaceRegressor must be fit before predict."
            raise ValueError(message)
        device = _resolve_device(self.config.device)
        with torch.inference_mode():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            predictions = np.asarray(self.model(tensor).cpu().numpy(), dtype=np.float64)
        return predictions
