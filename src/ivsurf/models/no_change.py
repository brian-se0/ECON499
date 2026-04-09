"""No-change benchmark."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ivsurf.models.base import SurfaceForecastModel


def validate_no_change_feature_layout(
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    lag_prefix: str = "feature_surface_mean_01_",
    target_prefix: str = "target_total_variance_",
) -> None:
    """Fail fast if lag-1 surface inputs do not exactly match the target grid layout."""

    target_suffixes = tuple(
        column.removeprefix(target_prefix)
        for column in target_columns
        if column.startswith(target_prefix)
    )
    if len(target_suffixes) != len(target_columns):
        message = (
            "No-change baseline requires target_total_variance columns with the expected "
            f"prefix {target_prefix!r}."
        )
        raise ValueError(message)
    expected_feature_columns = tuple(f"{lag_prefix}{suffix}" for suffix in target_suffixes)
    leading_feature_columns = tuple(feature_columns[: len(target_columns)])
    if leading_feature_columns != expected_feature_columns:
        message = (
            "No-change baseline requires lag-1 surface features to be present first and aligned "
            "one-to-one with the target surface columns. "
            f"Expected leading columns {expected_feature_columns!r}, "
            f"found {leading_feature_columns!r}."
        )
        raise ValueError(message)


class NoChangeSurfaceModel(SurfaceForecastModel):
    """Forecast tomorrow's surface as today's completed surface."""

    def __init__(self, lag_prefix: str = "feature_surface_mean_01_") -> None:
        self.lag_prefix = lag_prefix
        self._column_count: int | None = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
        training_weights: np.ndarray | None = None,
    ) -> NoChangeSurfaceModel:
        self._column_count = targets.shape[1]
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._column_count is None:
            message = "NoChangeSurfaceModel must be fit before predict."
            raise ValueError(message)
        return features[:, : self._column_count].copy()
