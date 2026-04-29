from __future__ import annotations

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from ivsurf.surfaces.grid import SurfaceGrid, assign_grid_indices


@given(
    st.lists(st.integers(min_value=1, max_value=1000), min_size=2, max_size=8, unique=True),
    st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=8, unique=True),
)
def test_valid_surface_grid_assignment_is_deterministic(
    maturity_days: list[int],
    moneyness_scaled: list[int],
) -> None:
    sorted_maturities = tuple(sorted(maturity_days))
    sorted_moneyness = tuple(value / 100.0 for value in sorted(moneyness_scaled))
    grid = SurfaceGrid(maturity_days=sorted_maturities, moneyness_points=sorted_moneyness)
    maturity_index = len(sorted_maturities) // 2
    moneyness_index = len(sorted_moneyness) // 2
    frame = pl.DataFrame(
        {
            "tau_years": [sorted_maturities[maturity_index] / 365.0],
            "log_moneyness": [sorted_moneyness[moneyness_index]],
        }
    )

    first = assign_grid_indices(frame, grid)
    second = assign_grid_indices(frame, grid)

    assert first["inside_grid_domain"].to_list() == [True]
    assert first["maturity_index"].to_list() == [maturity_index]
    assert first["moneyness_index"].to_list() == [moneyness_index]
    assert first.to_dict(as_series=False) == second.to_dict(as_series=False)
