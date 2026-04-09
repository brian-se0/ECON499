"""Shared positive-target transforms for total-variance models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ivsurf.evaluation.metrics import validate_total_variance_array


@dataclass(slots=True)
class LogPositiveTargetAdapter:
    """Fit models on log total variance and invert predictions back to total variance."""

    context_label: str

    def transform_targets(self, values: np.ndarray, *, array_name: str) -> np.ndarray:
        validated = validate_total_variance_array(
            values,
            context=f"{self.context_label} {array_name}",
            allow_zero=False,
        )
        return np.asarray(np.log(validated), dtype=np.float64)

    def inverse_predictions(self, values: np.ndarray) -> np.ndarray:
        log_values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(log_values).all():
            message = f"{self.context_label} produced non-finite log-space predictions."
            raise ValueError(message)
        predictions = np.asarray(np.exp(log_values), dtype=np.float64)
        if not np.isfinite(predictions).all():
            message = f"{self.context_label} produced non-finite total-variance predictions."
            raise ValueError(message)
        return predictions
