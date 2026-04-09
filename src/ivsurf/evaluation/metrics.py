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


def qlike(y_true: np.ndarray, y_pred: np.ndarray, positive_floor: float) -> float:
    true_clipped = np.maximum(y_true, positive_floor)
    pred_clipped = np.maximum(y_pred, positive_floor)
    return float(np.mean((true_clipped / pred_clipped) - np.log(true_clipped / pred_clipped) - 1.0))


def total_variance_to_iv(
    total_variance: np.ndarray,
    maturity_years: np.ndarray,
    *,
    total_variance_floor: float = 0.0,
) -> np.ndarray:
    maturity = np.maximum(maturity_years, 1.0e-12)
    floored_total_variance = np.maximum(total_variance, total_variance_floor)
    return np.asarray(np.sqrt(floored_total_variance / maturity), dtype=np.float64)
