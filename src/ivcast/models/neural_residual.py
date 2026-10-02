"""Post-hoc residual reparameterization of the neural surface model.

This variant predicts the log-space innovation around the lag-1 completed
surface instead of the total-variance level. The output head is
zero-initialized, so an untrained model reproduces the lag-1 surface exactly
and training starts from the persistence benchmark rather than from a
near-zero surface. It exists only for the post-mortem diagnostic and is not
part of the official benchmark universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ivcast.config import NeuralModelConfig, TrainingProfileConfig
from ivcast.evaluation.loss_panels import mean_daily_loss_metric
from ivcast.models.base import SurfaceForecastModel
from ivcast.models.losses import weighted_surface_mse
from ivcast.models.neural_surface import (
    NEURAL_GRADIENT_CLIP_NORM,
    NeuralValidationDiagnostics,
    _clone_state_dict,
    _feature_standardization_stats,
    _require_finite_scalar_loss,
    _require_neural_penalty_grid_domain,
    _resolve_device,
    _standardize_features,
    _validated_total_variance,
)
from ivcast.models.penalties import (
    calendar_monotonicity_penalty,
    convexity_penalty,
    roughness_penalty,
)

RESIDUAL_MODEL_NAME = "neural_surface_residual"

# Explicit numerical guard on the one-session log-variance innovation. A clamp
# at +/-20 corresponds to an e^20 (~4.9e8x) one-day total-variance move, far
# beyond any economically meaningful innovation, so it binds only when the
# unbounded linear head emits transient extreme values early in training
# (including fp16 autocast overflow), where the unguarded exp would otherwise
# produce non-finite predictions and abort the fit.
RESIDUAL_LOG_INNOVATION_CLAMP = 20.0


def reconstruct_total_variance_from_residual(
    residual: torch.Tensor,
    log_shifted_lag1: torch.Tensor,
    *,
    epsilon: float,
    floor: float,
) -> torch.Tensor:
    """Invert the log-innovation parameterization back to floored total variance."""

    if epsilon <= 0.0 or floor <= 0.0:
        message = "Residual reconstruction requires strictly positive epsilon and floor."
        raise ValueError(message)
    if residual.shape != log_shifted_lag1.shape:
        message = (
            "Residual and lag-1 tensors must share one shape, found "
            f"{tuple(residual.shape)!r} != {tuple(log_shifted_lag1.shape)!r}."
        )
        raise ValueError(message)
    clamped_residual = torch.clamp(
        residual.to(dtype=torch.float64),
        min=-RESIDUAL_LOG_INNOVATION_CLAMP,
        max=RESIDUAL_LOG_INNOVATION_CLAMP,
    )
    reconstructed = torch.exp(log_shifted_lag1 + clamped_residual) - epsilon
    return torch.clamp(reconstructed, min=floor)


def _validated_positive_lag1_block(features: np.ndarray, output_dim: int) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] < output_dim:
        message = (
            "Residual model features must be rank-2 with at least "
            f"{output_dim} leading lag-1 surface columns, found shape {features.shape!r}."
        )
        raise ValueError(message)
    lag1_block = np.asarray(features[:, :output_dim], dtype=np.float64)
    if not np.isfinite(lag1_block).all():
        message = "Residual model lag-1 surface block must contain only finite values."
        raise ValueError(message)
    if not np.all(lag1_block > 0.0):
        message = "Residual model lag-1 surface block must be strictly positive."
        raise ValueError(message)
    return lag1_block


class NeuralResidualMLP(nn.Module):
    """MLP head predicting log-space surface innovations with a zero-initialized output."""

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
        output_layer = nn.Linear(current_dim, output_dim)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = cast(torch.Tensor, self.network(features))
        return residual.to(dtype=torch.float64)


@dataclass(slots=True)
class NeuralResidualSurfaceRegressor(SurfaceForecastModel):
    """Persistence-anchored torch regressor for the post-mortem diagnostic."""

    config: NeuralModelConfig
    grid_shape: tuple[int, int]
    moneyness_points: tuple[float, ...]
    model: NeuralResidualMLP | None = None
    best_epoch: int | None = None
    epochs_completed: int = 0
    best_validation_score: float | None = None
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    validation_diagnostics: NeuralValidationDiagnostics | None = None

    def __post_init__(self) -> None:
        _require_neural_penalty_grid_domain(grid_shape=self.grid_shape, config=self.config)
        if len(self.moneyness_points) != self.grid_shape[1]:
            message = (
                "NeuralResidualSurfaceRegressor moneyness_points length must match the "
                f"moneyness grid size: {len(self.moneyness_points)} != {self.grid_shape[1]}."
            )
            raise ValueError(message)
        moneyness_array = np.asarray(self.moneyness_points, dtype=np.float64)
        if not np.isfinite(moneyness_array).all():
            message = "NeuralResidualSurfaceRegressor moneyness_points must be finite."
            raise ValueError(message)
        if not np.all(np.diff(moneyness_array) > 0.0):
            message = "NeuralResidualSurfaceRegressor moneyness_points must be strictly increasing."
            raise ValueError(message)

    def _predict_from_arrays(
        self,
        model: NeuralResidualMLP,
        *,
        device: torch.device,
        standardized_features: np.ndarray,
        raw_features: np.ndarray,
        output_dim: int,
    ) -> np.ndarray:
        lag1_block = _validated_positive_lag1_block(raw_features, output_dim)
        epsilon = self.config.output_total_variance_floor
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                feature_tensor = torch.as_tensor(
                    standardized_features,
                    dtype=torch.float32,
                    device=device,
                )
                log_shifted_lag1 = torch.log(
                    torch.as_tensor(lag1_block + epsilon, dtype=torch.float64, device=device)
                )
                predictions = _validated_total_variance(
                    reconstruct_total_variance_from_residual(
                        model(feature_tensor),
                        log_shifted_lag1,
                        epsilon=epsilon,
                        floor=self.config.output_total_variance_floor,
                    ),
                    context="NeuralResidualSurfaceRegressor inference",
                )
        finally:
            model.train(was_training)
        return np.asarray(predictions.cpu().numpy(), dtype=np.float64)

    def _validation_diagnostics_from_predictions(
        self,
        predictions: np.ndarray,
        *,
        device: torch.device,
        targets: np.ndarray,
        observed_masks: np.ndarray,
        vega_weights: np.ndarray,
        metric_name: str,
        positive_floor: float,
    ) -> NeuralValidationDiagnostics:
        metric_value = mean_daily_loss_metric(
            metric_name=metric_name,
            y_true=targets,
            y_pred=predictions,
            observed_masks=observed_masks,
            vega_weights=vega_weights,
            positive_floor=positive_floor,
        )
        prediction_tensor = torch.as_tensor(predictions, dtype=torch.float32, device=device)
        target_mean = float(np.mean(targets))
        prediction_mean = float(np.mean(predictions))
        calendar_penalty_value = 0.0
        if self.config.calendar_penalty_weight > 0.0:
            calendar_penalty_value = float(
                calendar_monotonicity_penalty(prediction_tensor, self.grid_shape)
                .detach()
                .cpu()
                .item()
            )
        convexity_penalty_value = 0.0
        if self.config.convexity_penalty_weight > 0.0:
            convexity_penalty_value = float(
                convexity_penalty(
                    prediction_tensor,
                    self.grid_shape,
                    moneyness_points=self.moneyness_points,
                )
                .detach()
                .cpu()
                .item()
            )
        roughness_penalty_value = 0.0
        if self.config.roughness_penalty_weight > 0.0:
            roughness_penalty_value = float(
                roughness_penalty(prediction_tensor, self.grid_shape).detach().cpu().item()
            )
        return NeuralValidationDiagnostics(
            metric_name=metric_name,
            metric_value=metric_value,
            prediction_mean=prediction_mean,
            target_mean=target_mean,
            prediction_target_ratio=(
                prediction_mean / target_mean if target_mean > 0.0 else float("nan")
            ),
            prediction_below_1e_6_share=float(np.mean(predictions < 1.0e-6)),
            calendar_penalty=calendar_penalty_value,
            convexity_penalty=convexity_penalty_value,
            roughness_penalty=roughness_penalty_value,
        )

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
        training_weights: np.ndarray | None = None,
        *,
        validation_features: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
        validation_observed_masks: np.ndarray | None = None,
        validation_vega_weights: np.ndarray | None = None,
        training_profile: TrainingProfileConfig | None = None,
        trial: optuna.Trial | None = None,
        trial_step_offset: int = 0,
        validation_metric_name: str = "observed_mse_total_variance",
        validation_positive_floor: float = 1.0e-8,
    ) -> NeuralResidualSurfaceRegressor:
        if observed_masks is None:
            message = "NeuralResidualSurfaceRegressor requires observed_masks."
            raise ValueError(message)
        if training_weights is None:
            message = "NeuralResidualSurfaceRegressor requires training_weights."
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
        epsilon = self.config.output_total_variance_floor
        lag1_block = _validated_positive_lag1_block(features, output_dim)
        feature_mean, feature_scale = _feature_standardization_stats(features)
        standardized_features = _standardize_features(
            features,
            mean=feature_mean,
            scale=feature_scale,
        )
        standardized_validation_features = (
            None
            if validation_features is None
            else _standardize_features(
                validation_features,
                mean=feature_mean,
                scale=feature_scale,
            )
        )
        model = NeuralResidualMLP(
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
            torch.as_tensor(standardized_features, dtype=torch.float32),
            torch.log(torch.as_tensor(lag1_block + epsilon, dtype=torch.float64)),
            torch.as_tensor(targets, dtype=torch.float32),
            torch.as_tensor(observed_masks, dtype=torch.float32),
            torch.as_tensor(training_weights, dtype=torch.float32),
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
        best_validation_diagnostics: NeuralValidationDiagnostics | None = None
        epochs_without_improvement = 0
        max_epochs = training_profile.epochs if training_profile is not None else self.config.epochs
        min_delta = (
            training_profile.neural_early_stopping_min_delta
            if training_profile is not None
            else 0.0
        )
        min_epochs_before_early_stop = (
            training_profile.neural_min_epochs_before_early_stop
            if training_profile is not None
            else 1
        )
        patience = (
            training_profile.neural_early_stopping_patience
            if training_profile is not None
            else max_epochs
        )

        model.train()
        for epoch in range(max_epochs):
            for (
                batch_features,
                batch_log_shifted_lag1,
                batch_targets,
                batch_masks,
                batch_training_weights,
            ) in loader:
                batch_features = batch_features.to(device, non_blocking=use_cuda)
                batch_log_shifted_lag1 = batch_log_shifted_lag1.to(
                    device,
                    non_blocking=use_cuda,
                )
                batch_targets = batch_targets.to(device, non_blocking=use_cuda)
                batch_masks = batch_masks.to(device, non_blocking=use_cuda)
                batch_training_weights = batch_training_weights.to(
                    device,
                    non_blocking=use_cuda,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_cuda):
                    predictions = _validated_total_variance(
                        reconstruct_total_variance_from_residual(
                            model(batch_features),
                            batch_log_shifted_lag1,
                            epsilon=epsilon,
                            floor=self.config.output_total_variance_floor,
                        ),
                        context="NeuralResidualSurfaceRegressor training",
                    )
                    loss = weighted_surface_mse(
                        predictions=predictions,
                        targets=batch_targets,
                        observed_mask=batch_masks,
                        training_weights=batch_training_weights,
                        observed_loss_weight=self.config.observed_loss_weight,
                        imputed_loss_weight=self.config.imputed_loss_weight,
                    )
                    if self.config.calendar_penalty_weight > 0.0:
                        loss = loss + (
                            self.config.calendar_penalty_weight
                            * calendar_monotonicity_penalty(predictions, self.grid_shape)
                        )
                    if self.config.convexity_penalty_weight > 0.0:
                        loss = loss + (
                            self.config.convexity_penalty_weight
                            * convexity_penalty(
                                predictions,
                                self.grid_shape,
                                moneyness_points=self.moneyness_points,
                            )
                        )
                    if self.config.roughness_penalty_weight > 0.0:
                        loss = loss + (
                            self.config.roughness_penalty_weight
                            * roughness_penalty(predictions, self.grid_shape)
                        )
                    loss = _require_finite_scalar_loss(
                        loss,
                        context="NeuralResidualSurfaceRegressor training",
                    )

                if use_cuda:
                    if scaler is None:
                        message = "CUDA training requires an initialized GradScaler."
                        raise RuntimeError(message)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=NEURAL_GRADIENT_CLIP_NORM,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=NEURAL_GRADIENT_CLIP_NORM,
                    )
                    optimizer.step()

            self.epochs_completed = epoch + 1
            if training_profile is None:
                continue

            if (
                validation_features is None
                or validation_targets is None
                or validation_observed_masks is None
                or validation_vega_weights is None
                or standardized_validation_features is None
            ):
                message = "Validation arrays unexpectedly became unavailable during training."
                raise RuntimeError(message)
            validation_predictions = self._predict_from_arrays(
                model,
                device=device,
                standardized_features=standardized_validation_features,
                raw_features=validation_features,
                output_dim=output_dim,
            )
            validation_diagnostics = self._validation_diagnostics_from_predictions(
                validation_predictions,
                device=device,
                targets=validation_targets,
                observed_masks=validation_observed_masks,
                vega_weights=validation_vega_weights,
                metric_name=validation_metric_name,
                positive_floor=validation_positive_floor,
            )
            validation_score = validation_diagnostics.metric_value
            if validation_score < (best_validation_score - min_delta):
                best_validation_score = validation_score
                best_epoch = epoch + 1
                best_state_dict = _clone_state_dict(model)
                best_validation_diagnostics = validation_diagnostics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if trial is not None:
                trial.report(validation_score, trial_step_offset + epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if (
                (epoch + 1) >= min_epochs_before_early_stop
                and epochs_without_improvement >= patience
            ):
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        elif training_profile is not None:
            if (
                validation_features is None
                or validation_targets is None
                or validation_observed_masks is None
                or validation_vega_weights is None
                or standardized_validation_features is None
            ):
                message = "Validation arrays unexpectedly became unavailable after training."
                raise RuntimeError(message)
            fallback_predictions = self._predict_from_arrays(
                model,
                device=device,
                standardized_features=standardized_validation_features,
                raw_features=validation_features,
                output_dim=output_dim,
            )
            fallback_diagnostics = self._validation_diagnostics_from_predictions(
                fallback_predictions,
                device=device,
                targets=validation_targets,
                observed_masks=validation_observed_masks,
                vega_weights=validation_vega_weights,
                metric_name=validation_metric_name,
                positive_floor=validation_positive_floor,
            )
            best_validation_score = fallback_diagnostics.metric_value
            best_epoch = self.epochs_completed
            best_validation_diagnostics = fallback_diagnostics

        self.best_epoch = best_epoch
        self.best_validation_score = (
            None if best_validation_score == float("inf") else float(best_validation_score)
        )
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.validation_diagnostics = best_validation_diagnostics
        self.model = model.eval()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if (
            self.model is None
            or self.feature_mean is None
            or self.feature_scale is None
        ):
            message = "NeuralResidualSurfaceRegressor must be fit before predict."
            raise ValueError(message)
        device = _resolve_device(self.config.device)
        standardized_features = _standardize_features(
            features,
            mean=self.feature_mean,
            scale=self.feature_scale,
        )
        output_dim = self.grid_shape[0] * self.grid_shape[1]
        return self._predict_from_arrays(
            self.model,
            device=device,
            standardized_features=standardized_features,
            raw_features=features,
            output_dim=output_dim,
        )
