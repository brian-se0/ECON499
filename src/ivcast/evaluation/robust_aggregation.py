"""Weighting-robustness aggregations over saved forecast-realization panels.

The official primary loss weights observed cells by target-day vega. These
helpers recompute observed-cell MSE under two alternative weighting schemes so
model rankings can be checked for weighting sensitivity:

- maturity-balanced: vega-weighted MSE within each observed maturity slice,
  then an equal-weight average across the slices observed that day;
- equal-cell: an unweighted MSE across all observed cells.

Both are post-hoc diagnostics computed from the saved Stage-07 panel; they
never feed back into model training or the official artifacts.
"""

from __future__ import annotations

import polars as pl

REQUIRED_PANEL_COLUMNS: tuple[str, ...] = (
    "model_name",
    "quote_date",
    "target_date",
    "maturity_days",
    "actual_observed_mask",
    "observed_weight",
    "actual_completed_total_variance",
    "predicted_total_variance",
)

MATURITY_BALANCED_METRIC = "maturity_balanced_observed_mse_total_variance"
EQUAL_CELL_METRIC = "equal_cell_observed_mse_total_variance"

_GROUP_KEYS: tuple[str, ...] = ("model_name", "quote_date", "target_date")


def _require_observed_panel_contract(panel: pl.DataFrame) -> pl.DataFrame:
    """Validate the aligned panel and return its observed-cell rows."""

    missing_columns = [name for name in REQUIRED_PANEL_COLUMNS if name not in panel.columns]
    if missing_columns:
        message = f"Forecast-realization panel is missing required columns: {missing_columns!r}."
        raise ValueError(message)
    if panel.height == 0:
        message = "Forecast-realization panel must contain at least one row."
        raise ValueError(message)
    if panel.schema["actual_observed_mask"] != pl.Boolean:
        message = (
            "Forecast-realization panel actual_observed_mask must be Boolean, found "
            f"{panel.schema['actual_observed_mask']!r}."
        )
        raise ValueError(message)
    null_counts = panel.select(
        pl.col(name).null_count().alias(name) for name in REQUIRED_PANEL_COLUMNS
    ).row(0, named=True)
    null_columns = [name for name, count in null_counts.items() if count > 0]
    if null_columns:
        message = (
            f"Forecast-realization panel contains nulls in required columns: {null_columns!r}."
        )
        raise ValueError(message)

    observed = panel.filter(pl.col("actual_observed_mask"))
    if observed.height == 0:
        message = "Forecast-realization panel contains no observed target cells."
        raise ValueError(message)
    for name in (
        "observed_weight",
        "actual_completed_total_variance",
        "predicted_total_variance",
    ):
        if not observed[name].is_finite().all():
            message = f"Observed panel column {name!r} must contain only finite values."
            raise ValueError(message)
    nonpositive_weights = int(observed.filter(pl.col("observed_weight") <= 0.0).height)
    if nonpositive_weights > 0:
        message = (
            "Observed target cells must carry strictly positive observed_weight; "
            f"invalid_count={nonpositive_weights}."
        )
        raise ValueError(message)

    coverage = observed.group_by("model_name").agg(
        pl.col("target_date").n_unique().alias("n_target_dates")
    )
    if coverage["n_target_dates"].n_unique() != 1:
        message = (
            "All models must share the same observed target-date coverage; found "
            f"{coverage.sort('model_name').to_dicts()!r}."
        )
        raise ValueError(message)
    return observed


def build_weighting_robustness_daily_losses(panel: pl.DataFrame) -> pl.DataFrame:
    """Compute maturity-balanced and equal-cell observed MSE per model and day."""

    observed = _require_observed_panel_contract(panel)
    squared_error = (
        pl.col("actual_completed_total_variance") - pl.col("predicted_total_variance")
    ) ** 2

    slice_losses = observed.group_by([*_GROUP_KEYS, "maturity_days"]).agg(
        (
            (pl.col("observed_weight") * squared_error).sum() / pl.col("observed_weight").sum()
        ).alias("slice_weighted_mse"),
        pl.len().alias("slice_observed_cell_count"),
    )
    maturity_daily = slice_losses.group_by(_GROUP_KEYS).agg(
        pl.col("slice_weighted_mse").mean().alias(MATURITY_BALANCED_METRIC),
        pl.len().alias("observed_maturity_slice_count"),
    )
    equal_daily = observed.group_by(_GROUP_KEYS).agg(
        squared_error.mean().alias(EQUAL_CELL_METRIC),
        pl.len().alias("observed_cell_count"),
    )
    joined = maturity_daily.join(equal_daily, on=list(_GROUP_KEYS), how="inner", validate="1:1")
    if joined.height != maturity_daily.height or joined.height != equal_daily.height:
        message = (
            "Maturity-balanced and equal-cell daily losses must align one-to-one; found "
            f"{maturity_daily.height} vs {equal_daily.height} vs joined {joined.height}."
        )
        raise ValueError(message)
    return joined.sort(_GROUP_KEYS)


def summarize_weighting_robustness(
    robust_daily_losses: pl.DataFrame,
    official_daily_loss_frame: pl.DataFrame,
) -> pl.DataFrame:
    """Rank models under official, maturity-balanced, and equal-cell weighting."""

    if "observed_mse_total_variance" not in official_daily_loss_frame.columns:
        message = (
            "Official daily loss frame must contain observed_mse_total_variance for the "
            "weighting-robustness comparison."
        )
        raise ValueError(message)
    official_means = official_daily_loss_frame.group_by("model_name").agg(
        pl.col("observed_mse_total_variance").mean().alias("mean_official_vega_weighted_mse")
    )
    robust_means = robust_daily_losses.group_by("model_name").agg(
        pl.col(MATURITY_BALANCED_METRIC).mean().alias(f"mean_{MATURITY_BALANCED_METRIC}"),
        pl.col(EQUAL_CELL_METRIC).mean().alias(f"mean_{EQUAL_CELL_METRIC}"),
        pl.len().alias("n_target_dates"),
    )
    summary = official_means.join(robust_means, on="model_name", how="inner", validate="1:1")
    if summary.height != official_means.height or summary.height != robust_means.height:
        message = (
            "Official and robust loss frames must cover the same model set; found "
            f"{sorted(official_means['model_name'].to_list())!r} vs "
            f"{sorted(robust_means['model_name'].to_list())!r}."
        )
        raise ValueError(message)
    return (
        summary.with_columns(
            pl.col("mean_official_vega_weighted_mse")
            .rank("ordinal")
            .cast(pl.Int64)
            .alias("rank_official"),
            pl.col(f"mean_{MATURITY_BALANCED_METRIC}")
            .rank("ordinal")
            .cast(pl.Int64)
            .alias("rank_maturity_balanced"),
            pl.col(f"mean_{EQUAL_CELL_METRIC}")
            .rank("ordinal")
            .cast(pl.Int64)
            .alias("rank_equal_cell"),
        )
        .sort("rank_official")
    )
