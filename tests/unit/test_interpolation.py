from __future__ import annotations

import numpy as np

from ivsurf.surfaces.interpolation import complete_surface


def test_surface_completion_fills_all_cells() -> None:
    observed = np.asarray([[0.04, np.nan, 0.09], [np.nan, 0.07, np.nan]], dtype=np.float64)
    completed = complete_surface(
        observed_total_variance=observed,
        maturity_coordinates=np.asarray([7.0 / 365.0, 30.0 / 365.0]),
        moneyness_coordinates=np.asarray([-0.1, 0.0, 0.1]),
        interpolation_order=("maturity", "moneyness"),
        interpolation_cycles=2,
        total_variance_floor=1.0e-8,
    )
    assert np.isfinite(completed.completed_total_variance).all()
    assert completed.completed_total_variance[0, 0] == observed[0, 0]

