"""Daily liquidity summaries derived from surface cells."""

from __future__ import annotations

import polars as pl


def build_daily_liquidity_features(surface_frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate daily liquidity and coverage statistics."""

    return (
        surface_frame.group_by("quote_date")
        .agg(
            pl.col("observed_mask").sum().alias("observed_cell_count"),
            pl.len().alias("grid_cell_count"),
            pl.col("vega_sum").sum().alias("daily_vega_sum"),
            pl.col("observation_count").sum().alias("daily_option_count"),
            (pl.col("weighted_spread_1545") * pl.col("vega_sum")).sum().alias(
                "weighted_spread_num"
            ),
        )
        .with_columns(
            (pl.col("observed_cell_count") / pl.col("grid_cell_count")).alias("coverage_ratio"),
            (
                pl.when(pl.col("daily_vega_sum") > 0.0)
                .then(pl.col("weighted_spread_num") / pl.col("daily_vega_sum"))
                .otherwise(0.0)
            ).alias("daily_weighted_spread_1545"),
        )
        .drop("weighted_spread_num")
        .sort("quote_date")
    )

