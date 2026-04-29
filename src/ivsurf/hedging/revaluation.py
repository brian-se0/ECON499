"""Book revaluation and option pricing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

from ivsurf.evaluation.metrics import validate_total_variance_array
from ivsurf.hedging.book import BookInstrument
from ivsurf.surfaces.grid import (
    SurfaceGrid,
    require_complete_unique_surface_grid,
)


@dataclass(frozen=True, slots=True)
class OptionValuation:
    """Price and sensitivities for one option."""

    price: float
    delta: float
    vega: float
    sigma: float
    remaining_days: int
    log_moneyness: float


@dataclass(frozen=True, slots=True)
class BookValuation:
    """Aggregated book valuation."""

    total_value: float
    total_delta: float
    total_vega: float


class SurfaceDomainError(ValueError):
    """Raised when hedging valuation requests a surface coordinate outside the grid."""


class SurfaceInterpolator:
    """Bilinear interpolator over a completed total-variance surface."""

    def __init__(
        self,
        maturity_days: np.ndarray,
        moneyness_points: np.ndarray,
        total_variance_grid: np.ndarray,
    ) -> None:
        self.maturity_days = maturity_days.astype(np.float64)
        self.moneyness_points = moneyness_points.astype(np.float64)
        self.total_variance_grid = validate_total_variance_array(
            total_variance_grid.astype(np.float64),
            context="SurfaceInterpolator total_variance_grid",
            allow_zero=True,
        )
        self._interpolator = RegularGridInterpolator(
            (self.maturity_days, self.moneyness_points),
            self.total_variance_grid,
            bounds_error=True,
        )

    def _require_in_domain(
        self,
        *,
        remaining_days: int,
        log_moneyness: float,
        instrument_label: str | None,
        valuation_date: date | None,
        model_name: str | None,
    ) -> tuple[float, float]:
        query_days = float(remaining_days)
        query_moneyness = float(log_moneyness)
        maturity_min = float(self.maturity_days.min())
        maturity_max = float(self.maturity_days.max())
        moneyness_min = float(self.moneyness_points.min())
        moneyness_max = float(self.moneyness_points.max())
        if (
            np.isfinite(query_days)
            and np.isfinite(query_moneyness)
            and maturity_min <= query_days <= maturity_max
            and moneyness_min <= query_moneyness <= moneyness_max
        ):
            return query_days, query_moneyness
        context_parts = []
        if model_name is not None:
            context_parts.append(f"model_name={model_name}")
        if instrument_label is not None:
            context_parts.append(f"instrument_label={instrument_label}")
        if valuation_date is not None:
            context_parts.append(f"valuation_date={valuation_date.isoformat()}")
        context = "unknown context" if not context_parts else ", ".join(context_parts)
        message = (
            "Surface interpolation query is outside the configured grid domain: "
            f"remaining_days={query_days!r}, maturity_bounds=({maturity_min!r}, "
            f"{maturity_max!r}), log_moneyness={query_moneyness!r}, "
            f"moneyness_bounds=({moneyness_min!r}, {moneyness_max!r}), {context}."
        )
        raise SurfaceDomainError(message)

    def sigma(
        self,
        remaining_days: int,
        log_moneyness: float,
        *,
        instrument_label: str | None = None,
        valuation_date: date | None = None,
        model_name: str | None = None,
    ) -> float:
        query_days, query_moneyness = self._require_in_domain(
            remaining_days=remaining_days,
            log_moneyness=log_moneyness,
            instrument_label=instrument_label,
            valuation_date=valuation_date,
            model_name=model_name,
        )
        total_variance = float(
            self._interpolator(np.asarray([[query_days, query_moneyness]])).item()
        )
        validate_total_variance_array(
            np.asarray([total_variance], dtype=np.float64),
            context="SurfaceInterpolator interpolated total variance",
            allow_zero=True,
        )
        tau_years = max(query_days / 365.0, 1.0e-12)
        return float(np.sqrt(total_variance / tau_years))


def surface_interpolator_from_frame(
    frame: pl.DataFrame,
    total_variance_column: str,
    grid: SurfaceGrid,
) -> SurfaceInterpolator:
    """Build a surface interpolator from one long-form daily surface frame."""

    require_complete_unique_surface_grid(
        frame,
        grid,
        dataset_name="Surface interpolator frame",
    )
    ordered = frame.sort(["maturity_index", "moneyness_index"])
    maturity_days = ordered["maturity_days"].unique().sort().to_numpy()
    moneyness_points = ordered["moneyness_point"].unique().sort().to_numpy()
    grid = ordered[total_variance_column].to_numpy().reshape(
        maturity_days.shape[0],
        moneyness_points.shape[0],
    )
    return SurfaceInterpolator(
        maturity_days=maturity_days,
        moneyness_points=moneyness_points,
        total_variance_grid=grid,
    )


def require_positive_spot(
    spot: float,
    *,
    context: str,
    valuation_date: date | None = None,
) -> float:
    """Validate that a chosen spot is finite and strictly positive."""

    spot_value = float(spot)
    if np.isfinite(spot_value) and spot_value > 0.0:
        return spot_value
    date_fragment = "" if valuation_date is None else f" on {valuation_date.isoformat()}"
    message = (
        f"{context} requires a finite strictly positive spot{date_fragment}; "
        f"got {spot_value!r}."
    )
    raise ValueError(message)


def black_scholes_value(
    spot: float,
    strike: float,
    tau_years: float,
    sigma: float,
    option_type: str,
    rate: float,
) -> tuple[float, float, float]:
    """Return Black-Scholes price, delta, and vega."""

    if tau_years <= 0.0 or sigma <= 0.0:
        intrinsic = max(spot - strike, 0.0) if option_type == "C" else max(strike - spot, 0.0)
        if option_type == "C":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return float(intrinsic), float(delta), 0.0

    sqrt_tau = np.sqrt(tau_years)
    d1 = (np.log(spot / strike) + ((rate + 0.5 * sigma * sigma) * tau_years)) / (
        sigma * sqrt_tau
    )
    d2 = d1 - (sigma * sqrt_tau)
    discount = np.exp(-rate * tau_years)
    if option_type == "C":
        price = (spot * norm.cdf(d1)) - (strike * discount * norm.cdf(d2))
        delta = norm.cdf(d1)
    else:
        price = (strike * discount * norm.cdf(-d2)) - (spot * norm.cdf(-d1))
        delta = norm.cdf(d1) - 1.0
    vega = spot * norm.pdf(d1) * sqrt_tau
    return float(price), float(delta), float(vega)


def value_instrument(
    instrument: BookInstrument,
    valuation_date: date,
    spot: float,
    surface: SurfaceInterpolator,
    rate: float,
    model_name: str | None = None,
) -> OptionValuation:
    """Value one carried option under the provided date/spot/surface state."""

    validated_spot = require_positive_spot(
        spot,
        context=f"Option valuation for {instrument.label}",
        valuation_date=valuation_date,
    )
    elapsed_days = (valuation_date - instrument.trade_date).days
    remaining_days = instrument.initial_maturity_days - elapsed_days
    tau_years = max(remaining_days / 365.0, 0.0)
    log_moneyness = float(np.log(instrument.strike / validated_spot))
    sigma = surface.sigma(
        remaining_days,
        log_moneyness,
        instrument_label=instrument.label,
        valuation_date=valuation_date,
        model_name=model_name,
    )
    price, delta, vega = black_scholes_value(
        spot=validated_spot,
        strike=instrument.strike,
        tau_years=tau_years,
        sigma=sigma,
        option_type=instrument.option_type,
        rate=rate,
    )
    return OptionValuation(
        price=price * instrument.quantity,
        delta=delta * instrument.quantity,
        vega=vega * instrument.quantity,
        sigma=sigma,
        remaining_days=remaining_days,
        log_moneyness=log_moneyness,
    )


def value_book(
    instruments: list[BookInstrument],
    valuation_date: date,
    spot: float,
    surface: SurfaceInterpolator,
    rate: float,
    model_name: str | None = None,
) -> BookValuation:
    """Aggregate book value, delta, and vega."""

    values = [
        value_instrument(
            instrument=instrument,
            valuation_date=valuation_date,
            spot=spot,
            surface=surface,
            rate=rate,
            model_name=model_name,
        )
        for instrument in instruments
    ]
    return BookValuation(
        total_value=float(sum(item.price for item in values)),
        total_delta=float(sum(item.delta for item in values)),
        total_vega=float(sum(item.vega for item in values)),
    )
