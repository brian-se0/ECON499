"""Fail-fast validation for fixed surface-grid coordinates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise


def validate_moneyness_points(values: Sequence[float]) -> tuple[float, ...]:
    """Require finite, unique, strictly increasing moneyness coordinates."""

    if not values:
        message = "moneyness_points must contain at least one coordinate."
        raise ValueError(message)
    normalized_values: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            message = f"moneyness_points must contain numeric coordinates; got {value!r}."
            raise ValueError(message)
        normalized_values.append(float(value))
    normalized = tuple(normalized_values)
    for value in normalized:
        if not math.isfinite(value):
            message = f"moneyness_points must be finite; got {value!r}."
            raise ValueError(message)
    if len(set(normalized)) != len(normalized):
        message = f"moneyness_points must be unique; got {normalized!r}."
        raise ValueError(message)
    if any(left >= right for left, right in pairwise(normalized)):
        message = f"moneyness_points must be strictly increasing; got {normalized!r}."
        raise ValueError(message)
    return normalized


def validate_maturity_days(values: Sequence[int]) -> tuple[int, ...]:
    """Require positive, unique, strictly increasing integer maturities."""

    if not values:
        message = "maturity_days must contain at least one coordinate."
        raise ValueError(message)
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            message = f"maturity_days must contain integer day counts; got {value!r}."
            raise ValueError(message)
        if value <= 0:
            message = f"maturity_days must be strictly positive; got {value!r}."
            raise ValueError(message)
        normalized.append(value)
    coordinates = tuple(normalized)
    if len(set(coordinates)) != len(coordinates):
        message = f"maturity_days must be unique; got {coordinates!r}."
        raise ValueError(message)
    if any(left >= right for left, right in pairwise(coordinates)):
        message = f"maturity_days must be strictly increasing; got {coordinates!r}."
        raise ValueError(message)
    return coordinates
