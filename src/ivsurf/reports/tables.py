"""Tabular report artifacts built from saved pipeline outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import orjson
import polars as pl


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

    if benchmark_model not in loss_summary["model_name"].to_list():
        message = f"Benchmark model {benchmark_model} not found in loss summary."
        raise ValueError(message)
    benchmark_value = float(
        loss_summary.filter(pl.col("model_name") == benchmark_model)[metric_column][0]
    )
    ranked = loss_summary.sort(metric_column).with_row_index("rank", offset=1)
    return ranked.with_columns(
        (
            ((pl.lit(benchmark_value) - pl.col(metric_column)) / pl.lit(benchmark_value)) * 100.0
        ).alias("improvement_vs_benchmark_pct")
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


def build_mcs_table(mcs_result: Mapping[str, Any]) -> pl.DataFrame:
    """Normalize the MCS JSON payload into one row per model."""

    included_models = set(mcs_result["included_models"])
    excluded_models = set(mcs_result["excluded_models"])
    all_models = sorted(included_models | excluded_models)
    return pl.DataFrame(
        {
            "model_name": all_models,
            "included_in_mcs": [model_name in included_models for model_name in all_models],
            "mcs_alpha": [float(mcs_result["alpha"])] * len(all_models),
        }
    ).sort(["included_in_mcs", "model_name"], descending=[True, False])


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
        csv_frame.write_csv(csv_path)
        md_path.write_text(frame_to_markdown(frame), encoding="utf-8")
        written_paths.extend([csv_path, md_path])
    return written_paths


def build_report_overview_markdown(
    *,
    benchmark_model: str,
    metric_column: str,
    ranked_loss_table: pl.DataFrame,
    ranked_hedging_table: pl.DataFrame,
    mcs_table: pl.DataFrame,
    slice_leader_table: pl.DataFrame,
    interpolation_summary: pl.DataFrame,
) -> str:
    """Create a concise report index for the generated artifacts."""

    best_loss = ranked_loss_table.row(0, named=True)
    best_hedging = ranked_hedging_table.row(0, named=True)
    included_models = ", ".join(
        mcs_table.filter(pl.col("included_in_mcs"))["model_name"].to_list()
    )
    best_slice_gains = slice_leader_table.head(5)
    interpolation_row = interpolation_summary.row(0, named=True)

    lines = [
        "# Report Artifacts",
        "",
        f"- Benchmark model: `{benchmark_model}`",
        f"- Primary loss metric: `{metric_column}`",
        (
            f"- Best full-sample loss model: `{best_loss['model_name']}` "
            f"({best_loss[metric_column]:.6f})"
        ),
        (
            f"- Best hedging revaluation model: `{best_hedging['model_name']}` "
            f"({best_hedging['mean_abs_revaluation_error']:.6f})"
        ),
        f"- MCS included models: {included_models if included_models else 'none'}",
        (
            "- Interpolation sensitivity summary: "
            f"mean RMSE diff {interpolation_row['mean_rmse_diff']:.6f}, "
            f"max abs diff {interpolation_row['max_max_abs_diff']:.6f}"
        ),
        "",
        "## Strongest Slice Gains",
        "",
        frame_to_markdown(best_slice_gains),
    ]
    return "\n".join(lines) + "\n"
