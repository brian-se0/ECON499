from __future__ import annotations

import torch
from hypothesis import given
from hypothesis import strategies as st

from ivsurf.models.penalties import calendar_monotonicity_penalty, convexity_penalty


@given(
    st.floats(min_value=0.01, max_value=0.20),
    st.floats(min_value=0.001, max_value=0.05),
    st.floats(min_value=0.001, max_value=0.05),
)
def test_quadratic_surface_has_no_penalty(base: float, slope: float, curvature: float) -> None:
    money = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    maturity = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32).unsqueeze(1)
    surface = base + (slope * maturity) + (curvature * money.square())
    flattened = surface.reshape(1, -1)
    assert float(calendar_monotonicity_penalty(flattened, (3, 3))) == 0.0
    assert float(convexity_penalty(flattened, (3, 3))) == 0.0

