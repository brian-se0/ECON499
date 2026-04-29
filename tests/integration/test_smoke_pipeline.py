from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from ivsurf.cleaning.derived_fields import (
    add_derived_fields,
    add_effective_decision_timestamp,
    build_tau_lookup,
)
from ivsurf.cleaning.option_filters import apply_option_quality_flags, valid_option_rows
from ivsurf.config import (
    CleaningConfig,
    FeatureConfig,
    MarketCalendarConfig,
    SurfaceGridConfig,
    WalkforwardConfig,
)
from ivsurf.features.tabular_dataset import build_daily_feature_dataset
from ivsurf.models.base import dataset_to_matrices
from ivsurf.models.naive import NaiveSurfaceModel
from ivsurf.splits.walkforward import build_walkforward_splits
from ivsurf.surfaces.aggregation import aggregate_daily_surface
from ivsurf.surfaces.grid import (
    MATURITY_COORDINATE,
    MONEYNESS_COORDINATE,
    SURFACE_GRID_SCHEMA_VERSION,
    SurfaceGrid,
    assign_grid_indices,
)
from ivsurf.surfaces.interpolation import COMPLETED_SURFACE_SCHEMA_VERSION, complete_surface

SURFACE_CONFIG_HASH = "surface-hash"


def _make_daily_option_rows(quote_date: date, spot: float) -> pl.DataFrame:
    moneyness_points = (-0.1, 0.0, 0.1)
    expiries = [date(2021, 1, 11), date(2021, 2, 5)]
    roots = ["SPXW", "SPXW"]
    rows: list[dict[str, object]] = []
    for expiry, root in zip(expiries, roots, strict=True):
        for option_type in ("C", "P"):
            for point in moneyness_points:
                strike = spot * np.exp(point)
                iv = 0.18 + (0.02 * abs(point)) + (0.01 if expiry == expiries[1] else 0.0)
                rows.append(
                    {
                        "underlying_symbol": "^SPX",
                        "quote_date": quote_date,
                        "root": root,
                        "expiration": expiry,
                        "strike": float(strike),
                        "option_type": option_type,
                        "trade_volume": 10,
                        "bid_size_1545": 5,
                        "bid_1545": 1.0,
                        "ask_size_1545": 5,
                        "ask_1545": 1.2,
                        "underlying_bid_1545": spot - 0.1,
                        "underlying_ask_1545": spot + 0.1,
                        "implied_underlying_price_1545": spot,
                        "active_underlying_price_1545": spot,
                        "implied_volatility_1545": iv,
                        "delta_1545": 0.5,
                        "gamma_1545": 0.1,
                        "theta_1545": -0.01,
                        "vega_1545": 1.0,
                        "rho_1545": 0.01,
                        "open_interest": 100,
                    }
                )
    return pl.DataFrame(rows)


def test_end_to_end_smoke_pipeline() -> None:
    calendar_config = MarketCalendarConfig()
    cleaning_config = CleaningConfig()
    surface_config = SurfaceGridConfig(
        moneyness_points=(-0.1, 0.0, 0.1),
        maturity_days=(5, 40),
    )
    grid = SurfaceGrid.from_config(surface_config)

    daily_surfaces: list[pl.DataFrame] = []
    for quote_date, spot in zip(
        [
            date(2021, 1, 4),
            date(2021, 1, 5),
            date(2021, 1, 6),
            date(2021, 1, 7),
            date(2021, 1, 8),
            date(2021, 1, 11),
        ],
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        strict=True,
    ):
        bronze = _make_daily_option_rows(quote_date=quote_date, spot=spot)
        tau_lookup = build_tau_lookup(frame=bronze, calendar_config=calendar_config)
        silver = add_derived_fields(frame=bronze, tau_lookup=tau_lookup)
        silver = add_effective_decision_timestamp(frame=silver, calendar_config=calendar_config)
        silver = apply_option_quality_flags(frame=silver, config=cleaning_config)
        valid = valid_option_rows(silver)
        assigned = assign_grid_indices(frame=valid, grid=grid).filter(pl.col("inside_grid_domain"))
        observed = aggregate_daily_surface(frame=assigned, grid=grid, config=surface_config).sort(
            ["quote_date", "maturity_index", "moneyness_index"]
        )
        observed_matrix = (
            observed["observed_total_variance"].fill_null(np.nan).to_numpy().reshape(grid.shape)
        )
        completed = complete_surface(
            observed_total_variance=observed_matrix,
            observed_mask=np.isfinite(observed_matrix),
            maturity_coordinates=grid.maturity_years,
            moneyness_coordinates=np.asarray(grid.moneyness_points, dtype=np.float64),
            interpolation_order=("maturity", "moneyness"),
            interpolation_cycles=2,
            total_variance_floor=1.0e-8,
        )
        daily_surfaces.append(
            observed.with_columns(
                pl.Series(
                    "completed_total_variance",
                    completed.completed_total_variance.reshape(-1),
                ),
                pl.lit(silver["effective_decision_timestamp"][0]).alias(
                    "effective_decision_timestamp"
                ),
                pl.lit(SURFACE_GRID_SCHEMA_VERSION).alias("surface_grid_schema_version"),
                pl.lit(grid.grid_hash).alias("surface_grid_hash"),
                pl.lit(MATURITY_COORDINATE).alias("maturity_coordinate"),
                pl.lit(MONEYNESS_COORDINATE).alias("moneyness_coordinate"),
                pl.lit(COMPLETED_SURFACE_SCHEMA_VERSION).alias("target_surface_version"),
                pl.lit(SURFACE_CONFIG_HASH).alias("surface_config_hash"),
            )
        )

    feature_frame = build_daily_feature_dataset(
        surface_frame=pl.concat(daily_surfaces),
        grid=grid,
        feature_config=FeatureConfig(lag_windows=(1, 2, 3)),
        calendar_config=calendar_config,
    ).feature_frame
    matrices = dataset_to_matrices(feature_frame)
    splits = build_walkforward_splits(
        dates=feature_frame["quote_date"].to_list(),
        config=WalkforwardConfig(
            train_size=1,
            validation_size=1,
            test_size=1,
            step_size=1,
            expanding_train=True,
        ),
    )
    model = NaiveSurfaceModel().fit(
        features=matrices.features[:2],
        targets=matrices.targets[:2],
        observed_masks=matrices.observed_masks[:2],
        vega_weights=matrices.vega_weights[:2],
    )
    predictions = model.predict(matrices.features[2:3])
    assert predictions.shape == (1, matrices.targets.shape[1])
    assert len(splits) >= 1
