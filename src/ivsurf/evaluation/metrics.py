"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if total <= 0.0:
        return np.asarray(np.full_like(weights, 1.0 / weights.size, dtype=np.float64))
    return np.asarray(weights / total, dtype=np.float64)


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    normalized = _normalize_weights(weights.astype(np.float64, copy=False))
    return float(np.sqrt(np.sum(normalized * np.square(y_pred - y_true))))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    normalized = _normalize_weights(weights.astype(np.float64, copy=False))
    return float(np.sum(normalized * np.abs(y_pred - y_true)))


def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    normalized = weights.astype(np.float64, copy=False)
    total = normalized.sum()
    if total <= 0.0:
        return float(np.mean(np.square(y_true - y_pred)))
    normalized = normalized / total
    return float(np.sum(normalized * np.square(y_true - y_pred)))


def validate_total_variance_array(
    total_variance: np.ndarray,
    *,
    context: str,
    allow_zero: bool = True,
) -> np.ndarray:
    values = np.asarray(total_variance, dtype=np.float64)
    if not np.isfinite(values).all():
        message = f"{context} contains non-finite total variance values."
        raise ValueError(message)
    invalid = values < 0.0 if allow_zero else values <= 0.0
    if invalid.any():
        relation = "negative" if allow_zero else "non-positive"
        minimum = float(values[invalid].min())
        message = (
            f"{context} contains {relation} total variance values; "
            f"minimum_invalid_value={minimum}."
        )
        raise ValueError(message)
    return values


def qlike(y_true: np.ndarray, y_pred: np.ndarray, positive_floor: float) -> float:
    true_values = validate_total_variance_array(
        y_true,
        context="QLIKE y_true",
        allow_zero=True,
    )
    pred_values = validate_total_variance_array(
        y_pred,
        context="QLIKE y_pred",
        allow_zero=True,
    )
    true_clipped = np.maximum(true_values, positive_floor)
    pred_clipped = np.maximum(pred_values, positive_floor)
    return float(np.mean((true_clipped / pred_clipped) - np.log(true_clipped / pred_clipped) - 1.0))


def total_variance_to_iv(
    total_variance: np.ndarray,
    maturity_years: np.ndarray,
    *,
    total_variance_floor: float = 0.0,
) -> np.ndarray:
    validated_total_variance = validate_total_variance_array(
        total_variance,
        context="IV conversion input",
        allow_zero=True,
    )
    maturity = np.maximum(maturity_years, 1.0e-12)
    floored_total_variance = np.maximum(validated_total_variance, total_variance_floor)
    return np.asarray(np.sqrt(floored_total_variance / maturity), dtype=np.float64)
