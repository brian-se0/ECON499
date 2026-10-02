"""Vega-weighted daily surface aggregation."""

from __future__ import annotations

import polars as pl

from ivcast.config import SurfaceGridConfig
from ivcast.surfaces.grid import SurfaceGrid


def aggregate_daily_surface(
    frame: pl.DataFrame,
    grid: SurfaceGrid,
    config: SurfaceGridConfig,
) -> pl.DataFrame:
    """Aggregate valid option rows to daily observed grid cells.

    Rows are assigned to the nearest grid node, but a node's value must be the total
    variance at the node maturity. Each row's total variance is therefore expressed at
    the node maturity under a locally constant implied volatility,
    `total_variance * node_tau / tau`, before vega weighting. Averaging raw total
    variance across contracts whose maturities span the whole bin would mix maturity
    with volatility (the 30-day node bins contracts from 22 to 45 days).
    """

    node_years = pl.col("maturity_index").replace_strict(
        list(range(len(grid.maturity_days))),
        [float(value) for value in grid.maturity_years],
        return_dtype=pl.Float64,
    )
    implied_variance_rate = pl.col("total_variance") / pl.col("tau_years")
    node_total_variance = implied_variance_rate * node_years
    grouped = (
        frame.group_by(["quote_date", "maturity_index", "moneyness_index"])
        .agg(
            (
                (node_total_variance * pl.col("vega_1545")).sum() / pl.col("vega_1545").sum()
            ).alias("observed_total_variance"),
            (
                (implied_variance_rate.sqrt() * pl.col("vega_1545")).sum()
                / pl.col("vega_1545").sum()
            ).alias("observed_iv"),
            (pl.col("spread_1545") * pl.col("vega_1545")).sum().alias("vega_weighted_spread_sum"),
            pl.col("vega_1545").sum().alias("vega_sum"),
            pl.len().alias("observation_count"),
        )
        .with_columns(
            (pl.col("vega_weighted_spread_sum") / pl.col("vega_sum")).alias("weighted_spread_1545"),
            (pl.col("observation_count") >= config.observed_cell_min_count).alias("observed_mask"),
        )
        .drop("vega_weighted_spread_sum")
    )

    date_values = grouped.select(pl.col("quote_date").unique()).to_series().to_list()
    rows: list[dict[str, object]] = []
    for quote_date in date_values:
        for maturity_index, maturity_day in enumerate(grid.maturity_days):
            for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
                rows.append(
                    {
                        "quote_date": quote_date,
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_day,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                    }
                )
    dense_grid = pl.DataFrame(rows)
    return dense_grid.join(
        grouped,
        on=["quote_date", "maturity_index", "moneyness_index"],
        how="left",
        validate="m:1",
    ).with_columns(
        pl.col("observed_mask").fill_null(False),
        pl.col("observation_count").fill_null(0),
        pl.col("vega_sum").fill_null(0.0),
        pl.col("weighted_spread_1545").fill_null(0.0),
    )
