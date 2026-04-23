"""Tabular report artifacts built from saved pipeline outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from ivsurf.io.atomic import write_text_atomic
from ivsurf.io.parquet import write_csv_frame


def _relative_improvement(best_value: float, benchmark_value: float) -> float:
    if benchmark_value == 0.0:
        return float("nan")
    return ((benchmark_value - best_value) / benchmark_value) * 100.0


def build_ranked_loss_table(
    loss_summary: pl.DataFrame,
    *,
    benchmark_model: str,
    metric_column: str,
) -> pl.DataFrame:
    """Add rank and benchmark-relative improvement to the loss summary."""

    if metric_column not in loss_summary.columns:
        message = f"Loss summary does not contain required metric column {metric_column}."
        raise ValueError(message)
    if benchmark_model not in loss_summary["model_name"].to_list():
        message = f"Benchmark model {benchmark_model} not found in loss summary."
        raise ValueError(message)
    selected_columns = ["model_name", metric_column]
    if metric_column.startswith("mean_"):
        std_column = metric_column.replace("mean_", "std_", 1)
        if std_column in loss_summary.columns:
            selected_columns.append(std_column)
    if "n_target_dates" in loss_summary.columns:
        selected_columns.append("n_target_dates")
    selected_summary = loss_summary.select(selected_columns)
    benchmark_value = float(
        selected_summary.filter(pl.col("model_name") == benchmark_model)[metric_column][0]
    )
    ranked = selected_summary.sort(metric_column).with_row_index("rank", offset=1)
    return ranked.with_columns(
        (
            ((pl.lit(benchmark_value) - pl.col(metric_column)) / pl.lit(benchmark_value)) * 100.0
        ).alias("improvement_vs_benchmark_pct")
    )


def build_tail_risk_table(
    daily_loss_frame: pl.DataFrame,
    *,
    benchmark_model: str,
    metric_column: str,
) -> pl.DataFrame:
    """Summarize tail-risk behaviour for one daily loss metric."""

    if metric_column not in daily_loss_frame.columns:
        message = f"Daily loss frame does not contain required metric column {metric_column}."
        raise ValueError(message)
    summary = (
        daily_loss_frame.group_by("model_name")
        .agg(
            pl.col(metric_column).mean().alias("mean_loss"),
            pl.col(metric_column).quantile(0.90).alias("p90_loss"),
            pl.col(metric_column).quantile(0.95).alias("p95_loss"),
            pl.col(metric_column).quantile(0.99).alias("p99_loss"),
            pl.col(metric_column).max().alias("max_loss"),
            pl.len().alias("n_target_dates"),
        )
        .sort(["p95_loss", "max_loss", "mean_loss"])
    )
    if benchmark_model not in summary["model_name"].to_list():
        message = f"Benchmark model {benchmark_model} not found in daily loss frame."
        raise ValueError(message)
    benchmark_row = summary.filter(pl.col("model_name") == benchmark_model).row(0, named=True)
    benchmark_p95 = float(benchmark_row["p95_loss"])
    benchmark_max = float(benchmark_row["max_loss"])
    return summary.with_columns(
        pl.lit(metric_column).alias("loss_metric"),
        (
            ((pl.lit(benchmark_p95) - pl.col("p95_loss")) / pl.lit(benchmark_p95)) * 100.0
        ).alias("p95_improvement_vs_benchmark_pct"),
        (
            ((pl.lit(benchmark_max) - pl.col("max_loss")) / pl.lit(benchmark_max)) * 100.0
        ).alias("max_improvement_vs_benchmark_pct"),
    ).select(
        "loss_metric",
        "model_name",
        "mean_loss",
        "p90_loss",
        "p95_loss",
        "p99_loss",
        "max_loss",
        "p95_improvement_vs_benchmark_pct",
        "max_improvement_vs_benchmark_pct",
        "n_target_dates",
    )


def build_worst_day_drilldown_table(
    daily_loss_frame: pl.DataFrame,
    *,
    benchmark_model: str,
    metric_column: str,
    top_n_per_model: int = 5,
) -> pl.DataFrame:
    """Return the worst target days for each model under one daily loss metric."""

    if metric_column not in daily_loss_frame.columns:
        message = f"Daily loss frame does not contain required metric column {metric_column}."
        raise ValueError(message)
    losses = daily_loss_frame.select(
        "model_name",
        "quote_date",
        "target_date",
        pl.col(metric_column).alias("loss_value"),
    )
    benchmark_losses = losses.filter(pl.col("model_name") == benchmark_model).select(
        "quote_date",
        "target_date",
        pl.col("loss_value").alias("benchmark_loss_value"),
    )
    if benchmark_losses.is_empty():
        message = f"Benchmark model {benchmark_model} not found in daily loss frame."
        raise ValueError(message)
    return (
        losses.join(
            benchmark_losses,
            on=["quote_date", "target_date"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.lit(metric_column).alias("loss_metric"),
            pl.lit(benchmark_model).alias("benchmark_model"),
            (pl.col("loss_value") - pl.col("benchmark_loss_value")).alias(
                "excess_loss_vs_benchmark"
            ),
            pl.when(pl.col("benchmark_loss_value") > 0.0)
            .then(pl.col("loss_value") / pl.col("benchmark_loss_value"))
            .otherwise(None)
            .alias("loss_ratio_vs_benchmark"),
            pl.col("loss_value")
            .rank(method="ordinal", descending=True)
            .over("model_name")
            .alias("rank_within_model"),
        )
        .filter(pl.col("rank_within_model") <= top_n_per_model)
        .sort(["model_name", "rank_within_model"])
        .select(
            "loss_metric",
            "model_name",
            "rank_within_model",
            "quote_date",
            "target_date",
            "loss_value",
            "benchmark_model",
            "benchmark_loss_value",
            "excess_loss_vs_benchmark",
            "loss_ratio_vs_benchmark",
        )
    )


def build_ranked_hedging_table(
    hedging_summary: pl.DataFrame,
    *,
    benchmark_model: str,
    metric_column: str = "mean_abs_revaluation_error",
) -> pl.DataFrame:
    """Rank hedging summaries against the benchmark model."""

    if benchmark_model not in hedging_summary["model_name"].to_list():
        message = f"Benchmark model {benchmark_model} not found in hedging summary."
        raise ValueError(message)
    benchmark_value = float(
        hedging_summary.filter(pl.col("model_name") == benchmark_model)[metric_column][0]
    )
    ranked = hedging_summary.sort(metric_column).with_row_index("rank", offset=1)
    return ranked.with_columns(
        (
            ((pl.lit(benchmark_value) - pl.col(metric_column)) / pl.lit(benchmark_value)) * 100.0
        ).alias("improvement_vs_benchmark_pct")
    )


def build_dm_results_table(dm_results: list[Mapping[str, Any]]) -> pl.DataFrame:
    """Normalize DM JSON payloads into a sorted table."""

    if not dm_results:
        return pl.DataFrame(
            {
                "model_a": [],
                "model_b": [],
                "test_statistic": [],
                "p_value": [],
                "alternative": [],
                "loss_mean_a": [],
                "loss_mean_b": [],
            }
        )
    return pl.DataFrame(dm_results).sort("p_value")


def build_spa_table(spa_result: Mapping[str, Any]) -> pl.DataFrame:
    """Normalize the SPA JSON payload into a one-row table."""

    return pl.DataFrame([dict(spa_result)])


def build_mcs_table(
    mcs_result: Mapping[str, Any],
    *,
    all_models: Sequence[str],
) -> pl.DataFrame:
    """Normalize the simplified Tmax-set JSON payload into one row per model."""

    superior_models = set(mcs_result["superior_models"])
    all_model_names = sorted(set(all_models))
    unknown_models = superior_models.difference(all_model_names)
    if unknown_models:
        message = (
            "MCS result contains superior_models missing from loss_summary: "
            f"{sorted(unknown_models)}."
        )
        raise ValueError(message)
    return pl.DataFrame(
        {
            "model_name": all_model_names,
            "included_in_simplified_tmax_set": [
                model_name in superior_models for model_name in all_model_names
            ],
            "simplified_tmax_alpha": [float(mcs_result["alpha"])] * len(all_model_names),
        }
    ).sort(["included_in_simplified_tmax_set", "model_name"], descending=[True, False])


def build_slice_leader_table(
    slice_metric_frame: pl.DataFrame,
    *,
    benchmark_model: str,
    metric_column: str,
) -> pl.DataFrame:
    """Pick the best model for each slice and compare it to the benchmark."""

    rows: list[dict[str, Any]] = []
    grouped = slice_metric_frame.partition_by(
        ["slice_family", "slice_label", "evaluation_scope"],
        maintain_order=True,
    )
    for group in grouped:
        benchmark_rows = group.filter(pl.col("model_name") == benchmark_model)
        if benchmark_rows.is_empty():
            message = (
                "Slice metric frame does not contain the benchmark model for slice "
                f"{group['slice_label'][0]}."
            )
            raise ValueError(message)
        best = group.sort(metric_column).row(0, named=True)
        benchmark = benchmark_rows.row(0, named=True)
        best_value = float(best[metric_column])
        benchmark_value = float(benchmark[metric_column])
        rows.append(
            {
                "slice_family": str(best["slice_family"]),
                "slice_label": str(best["slice_label"]),
                "evaluation_scope": str(best["evaluation_scope"]),
                "best_model_name": str(best["model_name"]),
                "best_metric_value": best_value,
                "benchmark_model": benchmark_model,
                "benchmark_metric_value": benchmark_value,
                "improvement_vs_benchmark_pct": _relative_improvement(best_value, benchmark_value),
            }
        )
    return pl.DataFrame(rows).sort(
        ["slice_family", "evaluation_scope", "improvement_vs_benchmark_pct"],
        descending=[False, False, True],
    )


def build_surface_cell_leader_table(
    panel: pl.DataFrame,
    *,
    benchmark_model: str,
    actual_column: str = "actual_completed_total_variance",
    predicted_column: str = "predicted_total_variance",
    weight_column: str = "observed_weight",
) -> pl.DataFrame:
    """Summarize the best-performing model at each surface cell under observed-scope MSE."""

    required_columns = (
        "model_name",
        "maturity_days",
        "moneyness_point",
        "actual_observed_mask",
        actual_column,
        predicted_column,
        weight_column,
    )
    missing_columns = [column for column in required_columns if column not in panel.columns]
    if missing_columns:
        message = f"Forecast-realization panel is missing required columns: {missing_columns}."
        raise ValueError(message)

    observed_panel = panel.filter(pl.col("actual_observed_mask"))
    if observed_panel.is_empty():
        message = "Surface cell leader table requires at least one observed target cell."
        raise ValueError(message)

    per_model_cell = (
        observed_panel.with_columns(
            (
                pl.col(weight_column)
                * (pl.col(predicted_column) - pl.col(actual_column)).pow(2)
            ).alias("weighted_squared_error")
        )
        .group_by(["model_name", "maturity_days", "moneyness_point"])
        .agg(
            pl.col("weighted_squared_error").sum().alias("weighted_squared_error_sum"),
            pl.col(weight_column).sum().alias("weight_sum"),
            pl.len().alias("n_observed_days"),
        )
        .with_columns(
            pl.when(pl.col("weight_sum") > 0.0)
            .then(pl.col("weighted_squared_error_sum") / pl.col("weight_sum"))
            .otherwise(None)
            .alias("cell_mse")
        )
        .sort(["maturity_days", "moneyness_point", "cell_mse", "model_name"])
    )

    benchmark_rows = (
        per_model_cell.filter(pl.col("model_name") == benchmark_model)
        .select(
            "maturity_days",
            "moneyness_point",
            pl.col("cell_mse").alias("benchmark_cell_mse"),
            pl.col("n_observed_days").alias("benchmark_observed_days"),
        )
        .sort(["maturity_days", "moneyness_point"])
    )
    if benchmark_rows.is_empty():
        message = f"Benchmark model {benchmark_model!r} not found in observed cell panel."
        raise ValueError(message)

    leaders = (
        per_model_cell.group_by(["maturity_days", "moneyness_point"], maintain_order=True)
        .agg(
            pl.col("model_name").first().alias("best_model_name"),
            pl.col("cell_mse").first().alias("best_cell_mse"),
            pl.col("n_observed_days").first().alias("n_observed_days"),
        )
        .join(
            benchmark_rows,
            on=["maturity_days", "moneyness_point"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.lit(benchmark_model).alias("benchmark_model"),
            pl.when(pl.col("benchmark_cell_mse") > 0.0)
            .then(
                (
                    (pl.col("benchmark_cell_mse") - pl.col("best_cell_mse"))
                    / pl.col("benchmark_cell_mse")
                )
                * 100.0
            )
            .otherwise(0.0)
            .alias("improvement_vs_benchmark_pct"),
        )
        .sort(["maturity_days", "moneyness_point"])
    )
    return leaders.select(
        "maturity_days",
        "moneyness_point",
        "best_model_name",
        "best_cell_mse",
        "benchmark_model",
        "benchmark_cell_mse",
        "improvement_vs_benchmark_pct",
        "n_observed_days",
        "benchmark_observed_days",
    )


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def frame_to_markdown(frame: pl.DataFrame) -> str:
    """Render a compact GitHub-flavored Markdown table."""

    if frame.is_empty():
        return "_No rows_"
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body_lines = [
        "| " + " | ".join(_format_markdown_value(row[column]) for column in frame.columns) + " |"
        for row in frame.iter_rows(named=True)
    ]
    return "\n".join([header, divider, *body_lines])


def write_table_artifacts(
    output_dir: Path,
    *,
    tables: Mapping[str, pl.DataFrame],
) -> list[Path]:
    """Write CSV and Markdown versions of each table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for name, frame in tables.items():
        csv_path = output_dir / f"{name}.csv"
        md_path = output_dir / f"{name}.md"
        if frame.is_empty():
            csv_frame = pl.DataFrame(
                {column: pl.Series([], dtype=pl.String) for column in frame.columns}
            )
        else:
            csv_frame = pl.DataFrame(
                [
                    {column: _format_markdown_value(row[column]) for column in frame.columns}
                    for row in frame.iter_rows(named=True)
                ]
            )
        write_csv_frame(csv_frame, csv_path)
        write_text_atomic(md_path, frame_to_markdown(frame), encoding="utf-8")
        written_paths.extend([csv_path, md_path])
    return written_paths


def build_report_overview_markdown(
    *,
    benchmark_model: str,
    official_loss_metrics: Sequence[str],
    primary_loss_metric: str,
    best_loss_by_metric_rows: Sequence[Mapping[str, Any]],
    summary_metric_column: str,
    ranked_loss_table: pl.DataFrame,
    tail_risk_table: pl.DataFrame,
    worst_day_drilldown: pl.DataFrame,
    ranked_hedging_table: pl.DataFrame,
    mcs_table: pl.DataFrame,
    slice_leader_table: pl.DataFrame,
    interpolation_summary: pl.DataFrame,
) -> str:
    """Create a concise report index for the generated artifacts."""

    best_loss = ranked_loss_table.row(0, named=True)
    tail_risk_leader = tail_risk_table.row(0, named=True)
    best_hedging = ranked_hedging_table.row(0, named=True)
    included_models = ", ".join(
        mcs_table.filter(pl.col("included_in_simplified_tmax_set"))["model_name"].to_list()
    )
    best_slice_gains = slice_leader_table.head(5)
    interpolation_row = interpolation_summary.row(0, named=True)
    official_metrics_rendered = ", ".join(f"`{metric}`" for metric in official_loss_metrics)

    lines = [
        "# Report Artifacts",
        "",
        f"- Benchmark model: `{benchmark_model}`",
        f"- Official loss metrics: {official_metrics_rendered}",
        f"- Primary loss metric: `{primary_loss_metric}`",
        (
            f"- Best full-sample loss model: `{best_loss['model_name']}` "
            f"({best_loss[summary_metric_column]:.6f})"
        ),
        (
            f"- Best primary tail-risk model by 95th percentile: "
            f"`{tail_risk_leader['model_name']}` ({tail_risk_leader['p95_loss']:.6f})"
        ),
        (
            f"- Best hedging revaluation model: `{best_hedging['model_name']}` "
            f"({best_hedging['mean_abs_revaluation_error']:.6f})"
        ),
        (
            "- Simplified Tmax-set included models: "
            f"{included_models if included_models else 'none'}"
        ),
        (
            "- Interpolation sensitivity summary: "
            f"mean RMSE diff {interpolation_row['mean_rmse_diff']:.6f}, "
            f"max abs diff {interpolation_row['max_max_abs_diff']:.6f}"
        ),
    ]
    for row in best_loss_by_metric_rows:
        lines.append(
            f"- Best model by `{row['loss_metric']}`: `{row['model_name']}` "
            f"({float(row['metric_value']):.6f})"
        )
    lines.extend(
        [
            "",
            "## Strongest Slice Gains",
            "",
            frame_to_markdown(best_slice_gains),
            "",
            "## Tail Risk",
            "",
            frame_to_markdown(tail_risk_table.head(5)),
            "",
            "## Worst Primary-Loss Days",
            "",
            frame_to_markdown(worst_day_drilldown.head(10)),
        ]
    )
    return "\n".join(lines) + "\n"
