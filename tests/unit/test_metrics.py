from __future__ import annotations

import numpy as np

from ivsurf.evaluation.metrics import total_variance_to_iv, weighted_mse


def test_weighted_mse_matches_manual_weighted_average() -> None:
    y_true = np.asarray([1.0, 3.0, 5.0], dtype=np.float64)
    y_pred = np.asarray([2.0, 1.0, 6.0], dtype=np.float64)
    weights = np.asarray([1.0, 2.0, 1.0], dtype=np.float64)

    result = weighted_mse(y_true, y_pred, weights)

    expected = ((1.0 * 1.0) + (2.0 * 4.0) + (1.0 * 1.0)) / 4.0
    assert result == expected


def test_total_variance_to_iv_applies_explicit_floor_before_square_root() -> None:
    total_variance = np.asarray([[-1.0e-4], [4.0e-4]], dtype=np.float64)
    maturity_years = np.asarray([[30.0 / 365.0], [30.0 / 365.0]], dtype=np.float64)

    result = total_variance_to_iv(
        total_variance=total_variance,
        maturity_years=maturity_years,
        total_variance_floor=1.0e-8,
    ).reshape(-1)

    assert np.isfinite(result).all()
    assert result[0] > 0.0
