"""Arbitrage-aware neural surface model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import optuna
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from ivsurf.config import NeuralModelConfig, TrainingProfileConfig
from ivsurf.evaluation.loss_panels import mean_daily_loss_metric
from ivsurf.models.base import SurfaceForecastModel
from ivsurf.models.losses import weighted_surface_mse
from ivsurf.models.penalties import (
    calendar_monotonicity_penalty,
    convexity_penalty,
    roughness_penalty,
)

NEURAL_GRADIENT_CLIP_NORM = 5.0


class NeuralSurfaceMLP(nn.Module):
    """Compact MLP for joint surface prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int,
        depth: int,
        dropout: float,
        output_total_variance_floor: float,
    ) -> None:
        super().__init__()
        if output_total_variance_floor <= 0.0:
            message = "output_total_variance_floor must be strictly positive."
            raise ValueError(message)
        self.output_total_variance_floor = output_total_variance_floor
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
        raw_predictions = cast(torch.Tensor, self.network(features))
        return cast(
            torch.Tensor,
            functional.softplus(raw_predictions.to(dtype=torch.float64))
            + self.output_total_variance_floor,
        )


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        message = "NeuralSurfaceRegressor requested CUDA, but torch.cuda.is_available() is False."
        raise RuntimeError(message)
    return device


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _feature_standardization_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(features.mean(axis=0), dtype=np.float64)
    scale = np.asarray(features.std(axis=0), dtype=np.float64)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    return mean, safe_scale


def _standardize_features(
    features: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.asarray((features - mean) / scale, dtype=np.float64)


def _validated_total_variance(
    total_variance_predictions: torch.Tensor,
    *,
    context: str,
) -> torch.Tensor:
    if not bool(torch.isfinite(total_variance_predictions).all().detach().item()):
        message = f"{context} produced non-finite total-variance predictions."
        raise RuntimeError(message)
    if not bool((total_variance_predictions >= 0.0).all().detach().item()):
        message = f"{context} produced negative total-variance predictions."
        raise RuntimeError(message)
    return total_variance_predictions


def _require_neural_penalty_grid_domain(
    *,
    grid_shape: tuple[int, int],
    config: NeuralModelConfig,
) -> None:
    if config.calendar_penalty_weight > 0.0 and grid_shape[0] < 2:
        message = (
            "calendar_penalty_weight is positive, so the neural grid requires at "
            f"least 2 maturity points; found grid_shape={grid_shape!r}."
        )
        raise ValueError(message)
    if config.convexity_penalty_weight > 0.0 and grid_shape[1] < 3:
        message = (
            "convexity_penalty_weight is positive, so the neural grid requires at "
            f"least 3 moneyness points; found grid_shape={grid_shape!r}."
        )
        raise ValueError(message)
    if config.roughness_penalty_weight > 0.0 and (grid_shape[0] < 3 or grid_shape[1] < 3):
        message = (
            "roughness_penalty_weight is positive, so the neural grid requires at "
            f"least 3 maturity points and 3 moneyness points; found grid_shape={grid_shape!r}."
        )
        raise ValueError(message)


def _require_finite_scalar_loss(loss: torch.Tensor, *, context: str) -> torch.Tensor:
    if loss.ndim == 0 and bool(torch.isfinite(loss).detach().item()):
        return loss
    message = f"{context} produced a non-finite scalar loss."
    raise RuntimeError(message)


@dataclass(frozen=True, slots=True)
class NeuralValidationDiagnostics:
    """Validation diagnostics persisted for the selected neural checkpoint."""

    metric_name: str
    metric_value: float
    prediction_mean: float
    target_mean: float
    prediction_target_ratio: float
    prediction_below_1e_6_share: float
    calendar_penalty: float
    convexity_penalty: float
    roughness_penalty: float


def _predict_total_variance_array(
    model: NeuralSurfaceMLP,
    *,
    device: torch.device,
    standardized_features: np.ndarray,
) -> np.ndarray:
    with torch.inference_mode():
        feature_tensor = torch.as_tensor(
            standardized_features,
            dtype=torch.float32,
            device=device,
        )
        predictions = _validated_total_variance(
            model(feature_tensor),
            context="NeuralSurfaceRegressor inference",
        )
    return np.asarray(predictions.cpu().numpy(), dtype=np.float64)


def _validation_score_and_diagnostics(
    model: NeuralSurfaceMLP,
    *,
    device: torch.device,
    standardized_features: np.ndarray,
    targets: np.ndarray,
    observed_masks: np.ndarray,
    vega_weights: np.ndarray,
    grid_shape: tuple[int, int],
    moneyness_points: tuple[float, ...],
    calendar_penalty_weight: float,
    convexity_penalty_weight: float,
    roughness_penalty_weight: float,
    metric_name: str,
    positive_floor: float,
) -> NeuralValidationDiagnostics:
    predictions = _predict_total_variance_array(
        model,
        device=device,
        standardized_features=standardized_features,
    )
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
    if calendar_penalty_weight > 0.0:
        calendar_penalty_value = float(
            calendar_monotonicity_penalty(prediction_tensor, grid_shape).detach().cpu().item()
        )
    convexity_penalty_value = 0.0
    if convexity_penalty_weight > 0.0:
        convexity_penalty_value = float(
            convexity_penalty(
                prediction_tensor,
                grid_shape,
                moneyness_points=moneyness_points,
            )
            .detach()
            .cpu()
            .item()
        )
    roughness_penalty_value = 0.0
    if roughness_penalty_weight > 0.0:
        roughness_penalty_value = float(
            roughness_penalty(prediction_tensor, grid_shape).detach().cpu().item()
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


@dataclass(slots=True)
class NeuralSurfaceRegressor(SurfaceForecastModel):
    """Torch regressor wrapper with explicit penalty weights."""

    config: NeuralModelConfig
    grid_shape: tuple[int, int]
    moneyness_points: tuple[float, ...]
    model: NeuralSurfaceMLP | None = None
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
                "NeuralSurfaceRegressor moneyness_points length must match the "
                f"moneyness grid size: {len(self.moneyness_points)} != {self.grid_shape[1]}."
            )
            raise ValueError(message)
        moneyness_array = np.asarray(self.moneyness_points, dtype=np.float64)
        if not np.isfinite(moneyness_array).all():
            message = "NeuralSurfaceRegressor moneyness_points must be finite."
            raise ValueError(message)
        if not np.all(np.diff(moneyness_array) > 0.0):
            message = "NeuralSurfaceRegressor moneyness_points must be strictly increasing."
            raise ValueError(message)

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
    ) -> NeuralSurfaceRegressor:
        if observed_masks is None:
            message = "NeuralSurfaceRegressor requires observed_masks."
            raise ValueError(message)
        if training_weights is None:
            message = "NeuralSurfaceRegressor requires training_weights."
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
        model = NeuralSurfaceMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_width=self.config.hidden_width,
            depth=self.config.depth,
            dropout=self.config.dropout,
            output_total_variance_floor=self.config.output_total_variance_floor,
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
                batch_targets,
                batch_masks,
                batch_training_weights,
            ) in loader:
                batch_features = batch_features.to(device, non_blocking=use_cuda)
                batch_targets = batch_targets.to(device, non_blocking=use_cuda)
                batch_masks = batch_masks.to(device, non_blocking=use_cuda)
                batch_training_weights = batch_training_weights.to(
                    device,
                    non_blocking=use_cuda,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_cuda):
                    predictions = _validated_total_variance(
                        model(batch_features),
                        context="NeuralSurfaceRegressor training",
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
                        context="NeuralSurfaceRegressor training",
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
                validation_targets is None
                or validation_observed_masks is None
                or validation_vega_weights is None
                or standardized_validation_features is None
            ):
                message = "Validation arrays unexpectedly became unavailable during training."
                raise RuntimeError(message)
            validation_diagnostics = _validation_score_and_diagnostics(
                model,
                device=device,
                standardized_features=standardized_validation_features,
                targets=validation_targets,
                observed_masks=validation_observed_masks,
                vega_weights=validation_vega_weights,
                grid_shape=self.grid_shape,
                moneyness_points=self.moneyness_points,
                calendar_penalty_weight=self.config.calendar_penalty_weight,
                convexity_penalty_weight=self.config.convexity_penalty_weight,
                roughness_penalty_weight=self.config.roughness_penalty_weight,
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
                validation_targets is None
                or validation_observed_masks is None
                or validation_vega_weights is None
                or standardized_validation_features is None
            ):
                message = "Validation arrays unexpectedly became unavailable after training."
                raise RuntimeError(message)
            fallback_diagnostics = _validation_score_and_diagnostics(
                model,
                device=device,
                standardized_features=standardized_validation_features,
                targets=validation_targets,
                observed_masks=validation_observed_masks,
                vega_weights=validation_vega_weights,
                grid_shape=self.grid_shape,
                moneyness_points=self.moneyness_points,
                calendar_penalty_weight=self.config.calendar_penalty_weight,
                convexity_penalty_weight=self.config.convexity_penalty_weight,
                roughness_penalty_weight=self.config.roughness_penalty_weight,
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
            message = "NeuralSurfaceRegressor must be fit before predict."
            raise ValueError(message)
        device = _resolve_device(self.config.device)
        standardized_features = _standardize_features(
            features,
            mean=self.feature_mean,
            scale=self.feature_scale,
        )
        return _predict_total_variance_array(
            self.model,
            device=device,
            standardized_features=standardized_features,
        )
