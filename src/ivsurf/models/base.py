"""Shared model interfaces and dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class DatasetMatrices:
    """Dense matrices extracted from the daily feature dataset."""

    quote_dates: np.ndarray
    target_dates: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    observed_masks: np.ndarray
    vega_weights: np.ndarray
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


def columns_by_prefix(frame: pl.DataFrame, prefix: str) -> tuple[str, ...]:
    """Return sorted columns that share a prefix."""

    return tuple(sorted(name for name in frame.columns if name.startswith(prefix)))


def ordered_feature_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    """Return a stable feature order with lagged surfaces first."""

    ordered_prefixes = (
        "feature_surface_mean_01_",
        "feature_surface_mean_05_",
        "feature_surface_mean_22_",
        "feature_surface_change_01_",
        "feature_mask_mean_01_",
        "feature_mask_mean_05_",
        "feature_mask_mean_22_",
    )
    ordered: list[str] = []
    for prefix in ordered_prefixes:
        ordered.extend(columns_by_prefix(frame, prefix))
    remaining = sorted(
        name for name in frame.columns if name.startswith("feature_") and name not in set(ordered)
    )
    ordered.extend(remaining)
    return tuple(ordered)


def dataset_to_matrices(frame: pl.DataFrame) -> DatasetMatrices:
    """Convert the feature parquet layout to dense numpy matrices."""

    feature_columns = ordered_feature_columns(frame)
    target_columns = columns_by_prefix(frame, "target_total_variance_")
    observed_mask_columns = columns_by_prefix(frame, "target_observed_mask_")
    vega_weight_columns = columns_by_prefix(frame, "target_vega_weight_")
    if not target_columns:
        message = "Feature dataset must contain target_total_variance columns."
        raise ValueError(message)
    return DatasetMatrices(
        quote_dates=frame["quote_date"].to_numpy(),
        target_dates=frame["target_date"].to_numpy(),
        features=frame.select(feature_columns).to_numpy(),
        targets=frame.select(target_columns).to_numpy(),
        observed_masks=frame.select(observed_mask_columns).to_numpy(),
        vega_weights=frame.select(vega_weight_columns).to_numpy(),
        feature_columns=feature_columns,
        target_columns=target_columns,
    )


class SurfaceForecastModel:
    """Minimal model protocol."""

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
    ) -> SurfaceForecastModel:
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError
