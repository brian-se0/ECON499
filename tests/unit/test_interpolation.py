from __future__ import annotations

import numpy as np

from ivsurf.surfaces.interpolation import (
    COMPLETION_STATUS_EXTRAPOLATED,
    COMPLETION_STATUS_INTERPOLATED,
    COMPLETION_STATUS_OBSERVED,
    complete_surface,
)


def test_surface_completion_fills_all_cells() -> None:
    observed = np.asarray([[0.04, np.nan, 0.09], [np.nan, 0.07, np.nan]], dtype=np.float64)
    completed = complete_surface(
        observed_total_variance=observed,
        observed_mask=np.isfinite(observed),
        maturity_coordinates=np.asarray([7.0 / 365.0, 30.0 / 365.0]),
        moneyness_coordinates=np.asarray([-0.1, 0.0, 0.1]),
        interpolation_order=("maturity", "moneyness"),
        interpolation_cycles=2,
        total_variance_floor=1.0e-8,
    )
    assert np.isfinite(completed.completed_total_variance).all()
    assert completed.completed_total_variance[0, 0] == observed[0, 0]


def test_surface_completion_ignores_finite_values_without_observed_mask() -> None:
    observed = np.asarray([[0.01, 999.0, 0.03]], dtype=np.float64)
    completed = complete_surface(
        observed_total_variance=observed,
        observed_mask=np.asarray([[True, False, True]]),
        maturity_coordinates=np.asarray([30.0 / 365.0]),
        moneyness_coordinates=np.asarray([-0.1, 0.0, 0.1]),
        interpolation_order=("moneyness",),
        interpolation_cycles=1,
        total_variance_floor=1.0e-8,
    )

    np.testing.assert_allclose(completed.completed_total_variance[0], [0.01, 0.02, 0.03])
    assert bool(completed.observed_mask[0, 1]) is False


def test_surface_completion_marks_interpolation_and_extrapolation_status() -> None:
    observed = np.asarray([[np.nan, 0.02, np.nan, 0.04, np.nan]], dtype=np.float64)
    completed = complete_surface(
        observed_total_variance=observed,
        observed_mask=np.asarray([[False, True, False, True, False]]),
        maturity_coordinates=np.asarray([30.0 / 365.0]),
        moneyness_coordinates=np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2]),
        interpolation_order=("moneyness",),
        interpolation_cycles=1,
        total_variance_floor=1.0e-8,
    )

    np.testing.assert_allclose(
        completed.completed_total_variance[0],
        [0.02, 0.02, 0.03, 0.04, 0.04],
    )
    assert completed.completion_status[0].tolist() == [
        COMPLETION_STATUS_EXTRAPOLATED,
        COMPLETION_STATUS_OBSERVED,
        COMPLETION_STATUS_INTERPOLATED,
        COMPLETION_STATUS_OBSERVED,
        COMPLETION_STATUS_EXTRAPOLATED,
    ]
