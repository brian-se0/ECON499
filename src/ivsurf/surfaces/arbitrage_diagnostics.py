"""Surface-shape diagnostics for calendar monotonicity and convexity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ArbitrageDiagnostics:
    """Violation counts and magnitudes."""

    calendar_violation_count: int
    calendar_violation_magnitude: float
    convexity_violation_count: int
    convexity_violation_magnitude: float


def calendar_monotonicity_violations(surface: np.ndarray) -> tuple[int, float]:
    """Count decreases across maturity."""

    diffs = np.diff(surface, axis=0)
    violations = np.minimum(diffs, 0.0)
    return int(np.count_nonzero(violations < 0.0)), float(np.abs(violations).sum())


def convexity_violations(surface: np.ndarray) -> tuple[int, float]:
    """Count negative second differences across moneyness."""

    second_diffs = surface[:, 2:] - (2.0 * surface[:, 1:-1]) + surface[:, :-2]
    violations = np.minimum(second_diffs, 0.0)
    return int(np.count_nonzero(violations < 0.0)), float(np.abs(violations).sum())


def summarize_diagnostics(surface: np.ndarray) -> ArbitrageDiagnostics:
    """Summarize daily arbitrage-aware diagnostics."""

    calendar_count, calendar_magnitude = calendar_monotonicity_violations(surface)
    convexity_count, convexity_magnitude = convexity_violations(surface)
    return ArbitrageDiagnostics(
        calendar_violation_count=calendar_count,
        calendar_violation_magnitude=calendar_magnitude,
        convexity_violation_count=convexity_count,
        convexity_violation_magnitude=convexity_magnitude,
    )

