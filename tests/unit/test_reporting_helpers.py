from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import orjson
import polars as pl

from ivsurf.config import SurfaceGridConfig
from ivsurf.evaluation.diagnostics import (
    build_actual_diagnostic_frame,
    build_forecast_diagnostic_frame,
    summarize_diagnostic_frame,
)
from ivsurf.evaluation.interpolation_sensitivity import (
    build_interpolation_sensitivity_frame,
    summarize_interpolation_sensitivity,
)
from ivsurf.reports.figures import (
    write_ecdf_chart,
    write_multi_line_chart,
    write_normalized_log_ranking_chart,
    write_ranked_bar_chart,
    write_surface_heatmap,
)
from ivsurf.reports.tables import (
    build_mcs_table,
    build_ranked_hedging_table,
    build_ranked_loss_table,
    build_slice_leader_table,
    build_surface_cell_leader_table,
    frame_to_markdown,
)
from ivsurf.reproducibility import sha256_file, write_run_manifest
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.surfaces.interpolation import complete_surface


def _actual_surface_frame(grid: SurfaceGrid) -> pl.DataFrame:
    observed = [
        [0.0100, float("nan"), 0.0115],
        [0.0110, 0.0120, float("nan")],
        [float("nan"), 0.0130, 0.0140],
    ]
    completed = complete_surface(
        observed_total_variance=pl.DataFrame(observed).to_numpy(),
        observed_mask=np.isfinite(pl.DataFrame(observed).to_numpy()),
        maturity_coordinates=grid.maturity_years,
        moneyness_coordinates=pl.Series(grid.moneyness_points).to_numpy(),
        interpolation_order=("maturity", "moneyness"),
        interpolation_cycles=2,
        total_variance_floor=1.0e-8,
    ).completed_total_variance

    rows: list[dict[str, object]] = []
    for maturity_index, maturity_days in enumerate(grid.maturity_days):
        for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
            rows.append(
                {
                    "quote_date": date(2021, 1, 5),
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "observed_total_variance": observed[maturity_index][moneyness_index],
                    "completed_total_variance": float(completed[maturity_index, moneyness_index]),
                    "observed_mask": observed[maturity_index][moneyness_index] is not None,
                }
            )
    return pl.DataFrame(rows)


def _forecast_frame(grid: SurfaceGrid) -> pl.DataFrame:
    good_surface = [
        [0.0105, 0.0100, 0.0106],
        [0.0115, 0.0110, 0.0116],
        [0.0125, 0.0120, 0.0126],
    ]
    bad_surface = [
        [0.0120, 0.0140, 0.0120],
        [0.0110, 0.0130, 0.0110],
        [0.0100, 0.0120, 0.0100],
    ]
    rows: list[dict[str, object]] = []
    for model_name, surface in (("good", good_surface), ("bad", bad_surface)):
        for maturity_index, maturity_days in enumerate(grid.maturity_days):
            for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
                rows.append(
                    {
                        "model_name": model_name,
                        "quote_date": date(2021, 1, 4),
                        "target_date": date(2021, 1, 5),
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "predicted_total_variance": float(surface[maturity_index][moneyness_index]),
                    }
                )
    return pl.DataFrame(rows)


def test_diagnostics_and_interpolation_sensitivity(tmp_path: Path) -> None:
    grid = SurfaceGrid.from_config(
        SurfaceGridConfig(
            moneyness_points=(-0.1, 0.0, 0.1),
            maturity_days=(30, 60, 90),
        )
    )
    actual_surface_frame = _actual_surface_frame(grid)
    forecast_frame = _forecast_frame(grid)

    forecast_diagnostics = build_forecast_diagnostic_frame(forecast_frame, grid)
    actual_diagnostics = build_actual_diagnostic_frame(actual_surface_frame, grid)
    summary = summarize_diagnostic_frame(pl.concat([forecast_diagnostics, actual_diagnostics]))

    good_row = summary.filter(pl.col("model_name") == "good").row(0, named=True)
    bad_row = summary.filter(pl.col("model_name") == "bad").row(0, named=True)
    assert good_row["mean_calendar_violation_count"] < bad_row["mean_calendar_violation_count"]
    assert good_row["mean_convexity_violation_count"] < bad_row["mean_convexity_violation_count"]

    sensitivity = build_interpolation_sensitivity_frame(
        actual_surface_frame,
        grid=grid,
        surface_config=SurfaceGridConfig(
            moneyness_points=(-0.1, 0.0, 0.1),
            maturity_days=(30, 60, 90),
        ),
        alternate_order=("moneyness", "maturity"),
    )
    assert sensitivity.height == 1
    assert sensitivity["max_abs_diff"][0] >= 0.0

    sensitivity_summary = summarize_interpolation_sensitivity(sensitivity)
    assert sensitivity_summary["n_quote_dates"][0] == 1

    loss_summary = pl.DataFrame(
        {
            "model_name": ["naive", "good", "bad"],
            "mean_observed_wrmse_total_variance": [0.0500, 0.0400, 0.0800],
            "std_observed_wrmse_total_variance": [0.0050, 0.0040, 0.0060],
            "n_target_dates": [10, 10, 10],
        }
    )
    ranked_loss = build_ranked_loss_table(
        loss_summary=loss_summary,
        benchmark_model="naive",
        metric_column="mean_observed_wrmse_total_variance",
    )
    assert ranked_loss["model_name"][0] == "good"

    mcs_table = build_mcs_table(
        {
            "superior_models": ["good", "naive"],
            "iterations": [],
            "alpha": 0.10,
            "block_size": 5,
            "bootstrap_reps": 100,
            "procedure_name": "simplified_tmax_elimination",
        },
        all_models=["naive", "good", "bad"],
    )
    assert mcs_table["model_name"].to_list() == ["good", "naive", "bad"]
    assert mcs_table["included_in_simplified_tmax_set"].to_list() == [True, True, False]

    hedging_summary = pl.DataFrame(
        {
            "model_name": ["naive", "good", "bad"],
            "mean_abs_revaluation_error": [1.20, 0.90, 1.80],
            "mean_squared_revaluation_error": [2.0, 1.5, 3.0],
            "mean_abs_hedged_pnl": [0.60, 0.50, 0.70],
            "mean_squared_hedged_pnl": [0.45, 0.30, 0.65],
            "hedged_pnl_variance": [0.20, 0.15, 0.30],
            "n_trades": [10, 10, 10],
        }
    )
    ranked_hedging = build_ranked_hedging_table(
        hedging_summary=hedging_summary,
        benchmark_model="naive",
    )
    assert ranked_hedging["model_name"][0] == "good"

    slice_metric_frame = pl.DataFrame(
        {
            "model_name": ["naive", "good", "bad", "naive", "good", "bad"],
            "slice_family": ["maturity"] * 3 + ["moneyness"] * 3,
            "slice_label": ["30d"] * 3 + ["+0.00"] * 3,
            "evaluation_scope": ["observed"] * 6,
            "wrmse_total_variance": [0.0500, 0.0400, 0.0700, 0.0600, 0.0500, 0.0900],
        }
    )
    leaders = build_slice_leader_table(
        slice_metric_frame=slice_metric_frame,
        benchmark_model="naive",
        metric_column="wrmse_total_variance",
    )
    assert leaders["best_model_name"].to_list() == ["good", "good"]
    assert "best_model_name" in frame_to_markdown(leaders)

    surface_panel = pl.DataFrame(
        {
            "model_name": [
                "naive",
                "challenger",
                "naive",
                "challenger",
            ],
            "maturity_days": [30, 30, 60, 60],
            "moneyness_point": [0.0, 0.0, 0.1, 0.1],
            "actual_observed_mask": [True, True, True, True],
            "actual_completed_total_variance": [0.0100, 0.0100, 0.0200, 0.0200],
            "predicted_total_variance": [0.0110, 0.0102, 0.0201, 0.0212],
            "observed_weight": [2.0, 2.0, 1.0, 1.0],
        }
    )
    surface_leaders = build_surface_cell_leader_table(
        surface_panel,
        benchmark_model="naive",
    )
    assert surface_leaders["best_model_name"].to_list() == ["challenger", "naive"]
    assert surface_leaders["improvement_vs_benchmark_pct"][0] > 0.0
    assert surface_leaders["improvement_vs_benchmark_pct"][1] == 0.0

    bar_chart_path = write_ranked_bar_chart(
        ranked_loss,
        label_column="model_name",
        value_column="mean_observed_wrmse_total_variance",
        output_path=tmp_path / "loss.svg",
        title="Loss Ranking",
    )
    line_chart_path = write_multi_line_chart(
        slice_metric_frame.with_columns(
            pl.Series("slice_value_float", [30.0, 30.0, 30.0, 0.0, 0.0, 0.0])
        ),
        x_column="slice_value_float",
        y_column="wrmse_total_variance",
        series_column="model_name",
        output_path=tmp_path / "slice.svg",
        title="Slice Metrics",
        x_label="Slice",
        y_label="WRMSE",
        include_series=("naive", "good"),
    )
    normalized_ranking_path = write_normalized_log_ranking_chart(
        ranked_loss,
        label_column="model_name",
        value_column="mean_observed_wrmse_total_variance",
        benchmark_label="naive",
        output_path=tmp_path / "loss_log.svg",
        title="Loss Ranking Relative to Naive",
    )
    heatmap_path = write_surface_heatmap(
        surface_leaders,
        x_column="moneyness_point",
        y_column="maturity_days",
        winner_column="best_model_name",
        improvement_column="improvement_vs_benchmark_pct",
        output_path=tmp_path / "surface_heatmap.svg",
        title="Best Model by Surface Cell",
        x_label="Log-moneyness",
        y_label="Maturity (days)",
    )
    ecdf_path = write_ecdf_chart(
        sensitivity,
        value_column="rmse_diff",
        output_path=tmp_path / "sensitivity_ecdf.svg",
        title="Interpolation Sensitivity ECDF",
        x_label="RMSE difference",
        y_label="Empirical CDF",
    )
    assert "<svg" in bar_chart_path.read_text(encoding="utf-8")
    assert "<svg" in line_chart_path.read_text(encoding="utf-8")
    assert "naive = 1" in normalized_ranking_path.read_text(encoding="utf-8")
    assert "challenger" in heatmap_path.read_text(encoding="utf-8")
    assert "Empirical CDF" in ecdf_path.read_text(encoding="utf-8")


def test_write_run_manifest_records_required_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("foo: 1\nbar: true\n", encoding="utf-8")
    input_artifact = tmp_path / "input.txt"
    input_artifact.write_text("input\n", encoding="utf-8")
    output_artifact = tmp_path / "output.txt"
    output_artifact.write_text("output\n", encoding="utf-8")
    split_manifest = tmp_path / "splits.json"
    split_manifest.write_bytes(orjson.dumps([{"split_id": "s1"}]))

    manifest_path = write_run_manifest(
        manifests_dir=tmp_path / "manifests",
        repo_root=Path.cwd(),
        script_name="unit_test_stage",
        started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
        config_paths=[config_path],
        input_artifact_paths=[input_artifact],
        output_artifact_paths=[output_artifact],
        split_manifest_path=split_manifest,
        random_seed=7,
        extra_metadata={"stage": "unit"},
    )
    payload = orjson.loads(manifest_path.read_bytes())

    assert payload["script_name"] == "unit_test_stage"
    assert payload["random_seed"] == 7
    assert payload["split_manifest_hash"] == sha256_file(split_manifest)
    assert payload["data_manifest_hash"] is not None
    assert payload["config_snapshots"][0]["content"] == "foo: 1\nbar: true\n"
    assert "package_versions" in payload
    assert "hardware_metadata" in payload
