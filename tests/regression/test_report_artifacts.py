from __future__ import annotations

from dataclasses import asdict
from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import orjson
import polars as pl

from ivsurf.evaluation.alignment import build_forecast_realization_panel
from ivsurf.evaluation.loss_panels import build_daily_loss_frame
from ivsurf.hedging.pnl import evaluate_model_hedging, summarize_hedging_results
from ivsurf.hedging.revaluation import surface_interpolator_from_frame
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.surfaces.interpolation import complete_surface

GOLDEN_DIR = Path(__file__).with_name("golden")


def _load_script_module(script_path: Path, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, payload: str) -> Path:
    path.write_text(payload, encoding="utf-8")
    return path


def _surface_rows(
    quote_date: date,
    sigma_level: float,
    grid: SurfaceGrid,
) -> list[dict[str, object]]:
    observed = []
    for maturity_index, maturity_days in enumerate(grid.maturity_days):
        maturity_years = maturity_days / 365.0
        row = []
        for moneyness_index, _moneyness_point in enumerate(grid.moneyness_points):
            value = (
                sigma_level * maturity_years
                + (0.0005 * maturity_index)
                + (0.0004 * moneyness_index)
            )
            row.append(value)
        observed.append(row)
    observed[0][1] = float("nan")
    observed[2][0] = float("nan")
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
        maturity_years = maturity_days / 365.0
        for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
            observed_value = observed[maturity_index][moneyness_index]
            completed_value = float(completed[maturity_index, moneyness_index])
            rows.append(
                {
                    "quote_date": quote_date,
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "observed_total_variance": (
                        None if observed_value != observed_value else observed_value
                    ),
                    "observed_iv": None
                    if observed_value != observed_value
                    else float((observed_value / maturity_years) ** 0.5),
                    "completed_total_variance": completed_value,
                    "completed_iv": float((completed_value / maturity_years) ** 0.5),
                    "observed_mask": observed_value == observed_value,
                    "vega_sum": 1.0 if observed_value == observed_value else 0.0,
                }
            )
    return rows


def _forecast_rows(
    quote_date: date,
    target_date: date,
    sigma_level: float,
    grid: SurfaceGrid,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model_scales = {
        "no_change": 0.96,
        "ridge": 0.995,
        "neural_surface": 1.0,
    }
    for model_name, scale in model_scales.items():
        for maturity_index, maturity_days in enumerate(grid.maturity_days):
            maturity_years = maturity_days / 365.0
            for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
                total_variance = (
                    sigma_level * scale * maturity_years
                    + (0.0005 * maturity_index)
                    + (0.0004 * moneyness_index)
                )
                if (
                    model_name == "no_change"
                    and quote_date == date(2021, 1, 6)
                    and maturity_index == 0
                    and moneyness_index == 0
                ):
                    total_variance = 1.0e-8
                rows.append(
                    {
                        "model_name": model_name,
                        "quote_date": quote_date,
                        "target_date": target_date,
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "predicted_total_variance": float(total_variance),
                    }
                )
    return rows


def test_report_artifact_bundle_regression(tmp_path: Path) -> None:
    grid = SurfaceGrid(
        maturity_days=(30, 60, 90),
        moneyness_points=(-0.1, 0.0, 0.1),
    )
    workflow_label = "hpo_30_trials__train_30_epochs"
    gold_dir = tmp_path / "data" / "gold"
    gold_year = gold_dir / "year=2021"
    gold_year.mkdir(parents=True)
    forecast_dir = gold_dir / "forecasts" / workflow_label
    forecast_dir.mkdir(parents=True)
    silver_dir = tmp_path / "data" / "silver" / "year=2021"
    silver_dir.mkdir(parents=True)
    manifests_dir = tmp_path / "data" / "manifests"
    (manifests_dir / "stats" / workflow_label).mkdir(parents=True)
    (manifests_dir / "hedging" / workflow_label).mkdir(parents=True)

    quote_dates = [
        date(2021, 1, 4),
        date(2021, 1, 5),
        date(2021, 1, 6),
        date(2021, 1, 7),
    ]
    sigma_levels = [0.040, 0.044, 0.049, 0.053]
    gold_rows: list[dict[str, object]] = []
    gold_summary_rows: list[dict[str, object]] = []
    for quote_date, sigma_level in zip(quote_dates, sigma_levels, strict=True):
        day_rows = _surface_rows(quote_date, sigma_level, grid)
        day_path = gold_year / f"{quote_date.isoformat()}.parquet"
        pl.DataFrame(day_rows).write_parquet(day_path)
        gold_rows.extend(day_rows)
        gold_summary_rows.append(
            {
                "gold_path": str(day_path),
                "quote_date": quote_date.isoformat(),
                "observed_cells": sum(1 for row in day_rows if row["observed_mask"]),
            }
        )
    (manifests_dir / "gold_surface_summary.json").write_bytes(
        orjson.dumps(gold_summary_rows, option=orjson.OPT_INDENT_2)
    )

    forecast_rows: list[dict[str, object]] = []
    for quote_date, target_date, sigma_level in zip(
        quote_dates[:-1], quote_dates[1:], sigma_levels[1:], strict=True
    ):
        forecast_rows.extend(_forecast_rows(quote_date, target_date, sigma_level, grid))
    pl.DataFrame(forecast_rows).write_parquet(forecast_dir / "forecasts.parquet")

    silver_rows = [
        {"quote_date": quote_date, "active_underlying_price_1545": spot}
        for quote_date, spot in zip(quote_dates, [100.0, 101.0, 102.0, 103.0], strict=True)
    ]
    pl.DataFrame(silver_rows).write_parquet(silver_dir / "spots.parquet")

    actual_surface_frame = pl.DataFrame(gold_rows).sort(
        ["quote_date", "maturity_index", "moneyness_index"]
    )
    forecast_frame = pl.DataFrame(forecast_rows).sort(
        ["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"]
    )
    panel = build_forecast_realization_panel(
        actual_surface_frame=actual_surface_frame,
        forecast_frame=forecast_frame,
        total_variance_floor=1.0e-8,
    )
    panel.write_parquet(
        manifests_dir / "stats" / workflow_label / "forecast_realization_panel.parquet"
    )

    daily_loss_frame = build_daily_loss_frame(
        panel=panel,
        positive_floor=1.0e-8,
        full_grid_weighting="uniform",
    )
    daily_loss_frame.write_parquet(
        manifests_dir / "stats" / workflow_label / "daily_loss_frame.parquet"
    )

    loss_metrics = (
        "observed_mse_total_variance",
        "observed_qlike_total_variance",
    )
    summary_expressions: list[pl.Expr] = []
    for loss_metric in loss_metrics:
        summary_expressions.extend(
            (
                pl.col(loss_metric).mean().alias(f"mean_{loss_metric}"),
                pl.col(loss_metric).std(ddof=1).alias(f"std_{loss_metric}"),
            )
        )
    loss_summary = (
        daily_loss_frame.group_by("model_name")
        .agg(*summary_expressions, pl.len().alias("n_target_dates"))
        .sort("mean_observed_mse_total_variance")
    )
    loss_summary.write_parquet(manifests_dir / "stats" / workflow_label / "loss_summary.parquet")

    dm_results = [
        {
            "loss_metric": "observed_mse_total_variance",
            "model_a": "no_change",
            "model_b": "ridge",
            "n_obs": 3,
            "mean_loss_a": 0.0024,
            "mean_loss_b": 0.0008,
            "mean_differential": 0.0016,
            "statistic": 2.1,
            "p_value": 0.08,
            "alternative": "greater",
            "max_lag": 0,
        },
        {
            "loss_metric": "observed_mse_total_variance",
            "model_a": "no_change",
            "model_b": "neural_surface",
            "n_obs": 3,
            "mean_loss_a": 0.0024,
            "mean_loss_b": 0.0,
            "mean_differential": 0.0024,
            "statistic": 3.4,
            "p_value": 0.02,
            "alternative": "greater",
            "max_lag": 0,
        },
        {
            "loss_metric": "observed_qlike_total_variance",
            "model_a": "no_change",
            "model_b": "ridge",
            "n_obs": 3,
            "mean_loss_a": 0.029,
            "mean_loss_b": 0.011,
            "mean_differential": 0.018,
            "statistic": 2.0,
            "p_value": 0.08,
            "alternative": "greater",
            "max_lag": 0,
        },
        {
            "loss_metric": "observed_qlike_total_variance",
            "model_a": "no_change",
            "model_b": "neural_surface",
            "n_obs": 3,
            "mean_loss_a": 0.029,
            "mean_loss_b": 0.0,
            "mean_differential": 0.029,
            "statistic": 3.1,
            "p_value": 0.02,
            "alternative": "greater",
            "max_lag": 0,
        },
    ]
    (manifests_dir / "stats" / workflow_label / "dm_results.json").write_bytes(
        orjson.dumps(dm_results, option=orjson.OPT_INDENT_2)
    )
    (manifests_dir / "stats" / workflow_label / "spa_result.json").write_bytes(
        orjson.dumps(
            [
                {
                    "loss_metric": "observed_mse_total_variance",
                    "benchmark_model": "no_change",
                    "candidate_models": ["ridge", "neural_surface"],
                    "observed_statistic": 2.5,
                    "p_value": 0.04,
                    "mean_differentials": [0.0007, 0.0024],
                    "superior_models_by_mean": ["ridge", "neural_surface"],
                    "block_size": 3,
                    "bootstrap_reps": 50,
                },
                {
                    "loss_metric": "observed_qlike_total_variance",
                    "benchmark_model": "no_change",
                    "candidate_models": ["ridge", "neural_surface"],
                    "observed_statistic": 2.3,
                    "p_value": 0.05,
                    "mean_differentials": [0.018, 0.029],
                    "superior_models_by_mean": ["ridge", "neural_surface"],
                    "block_size": 3,
                    "bootstrap_reps": 50,
                },
            ],
            option=orjson.OPT_INDENT_2,
        )
    )
    (manifests_dir / "stats" / workflow_label / "mcs_result.json").write_bytes(
        orjson.dumps(
            [
                {
                    "loss_metric": "observed_mse_total_variance",
                    "superior_models": ["ridge", "neural_surface"],
                    "iterations": [
                        {
                            "included_models": ["no_change", "ridge", "neural_surface"],
                            "test_statistic": 2.4,
                            "p_value": 0.03,
                            "eliminated_model": "no_change",
                        },
                        {
                            "included_models": ["ridge", "neural_surface"],
                            "test_statistic": 1.1,
                            "p_value": 0.22,
                            "eliminated_model": None,
                        },
                    ],
                    "alpha": 0.10,
                    "block_size": 3,
                    "bootstrap_reps": 50,
                    "procedure_name": "simplified_tmax_elimination",
                },
                {
                    "loss_metric": "observed_qlike_total_variance",
                    "superior_models": ["ridge", "neural_surface"],
                    "iterations": [
                        {
                            "included_models": ["no_change", "ridge", "neural_surface"],
                            "test_statistic": 2.3,
                            "p_value": 0.04,
                            "eliminated_model": "no_change",
                        },
                        {
                            "included_models": ["ridge", "neural_surface"],
                            "test_statistic": 1.0,
                            "p_value": 0.24,
                            "eliminated_model": None,
                        },
                    ],
                    "alpha": 0.10,
                    "block_size": 3,
                    "bootstrap_reps": 50,
                    "procedure_name": "simplified_tmax_elimination",
                },
            ],
            option=orjson.OPT_INDENT_2,
        )
    )

    actual_lookup = {
        key[0]: value
        for key, value in actual_surface_frame.partition_by("quote_date", as_dict=True).items()
    }
    spot_lookup = {
        cast(date, row["quote_date"]): float(cast(float, row["active_underlying_price_1545"]))
        for row in silver_rows
    }
    hedging_rows = []
    for group in forecast_frame.partition_by(
        ["model_name", "quote_date", "target_date"],
        maintain_order=True,
    ):
        hedging_rows.append(
            asdict(
                evaluate_model_hedging(
                    model_name=str(group["model_name"][0]),
                    quote_date=group["quote_date"][0],
                    target_date=group["target_date"][0],
                    trade_spot=spot_lookup[group["quote_date"][0]],
                    target_spot=spot_lookup[group["target_date"][0]],
                    actual_surface_t=surface_interpolator_from_frame(
                        actual_lookup[group["quote_date"][0]],
                        total_variance_column="completed_total_variance",
                    ),
                    actual_surface_t1=surface_interpolator_from_frame(
                        actual_lookup[group["target_date"][0]],
                        total_variance_column="completed_total_variance",
                    ),
                    predicted_surface_t1=surface_interpolator_from_frame(
                        group,
                        total_variance_column="predicted_total_variance",
                    ),
                    rate=0.0,
                    level_notional=1.0,
                    skew_notional=1.0,
                    calendar_notional=0.5,
                    skew_moneyness_abs=0.1,
                    short_maturity_days=30,
                    long_maturity_days=90,
                    hedge_maturity_days=30,
                    hedge_straddle_moneyness=0.0,
                )
            )
        )
    hedging_results = pl.DataFrame(hedging_rows).sort(["model_name", "quote_date", "target_date"])
    hedging_results.write_parquet(
        manifests_dir / "hedging" / workflow_label / "hedging_results.parquet"
    )
    summarize_hedging_results(hedging_results).write_parquet(
        manifests_dir / "hedging" / workflow_label / "hedging_summary.parquet"
    )

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{gold_dir.as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
        ),
    )
    surface_config_path = _write_yaml(
        tmp_path / "surface.yaml",
        (
            "moneyness_points: [-0.1, 0.0, 0.1]\n"
            "maturity_days: [30, 60, 90]\n"
            'interpolation_order: ["maturity", "moneyness"]\n'
            "interpolation_cycles: 2\n"
            "total_variance_floor: 1.0e-8\n"
            "observed_cell_min_count: 1\n"
        ),
    )
    metrics_config_path = _write_yaml(tmp_path / "metrics.yaml", "positive_floor: 1.0e-8\n")
    stats_config_path = _write_yaml(
        tmp_path / "stats.yaml",
        (
            "loss_metrics:\n"
            '  - "observed_mse_total_variance"\n'
            '  - "observed_qlike_total_variance"\n'
            'benchmark_model: "no_change"\n'
            'dm_alternative: "greater"\n'
            "dm_max_lag: 0\n"
            "spa_block_size: 3\n"
            "spa_bootstrap_reps: 50\n"
            "mcs_block_size: 3\n"
            "mcs_bootstrap_reps: 50\n"
            "mcs_alpha: 0.10\n"
            "bootstrap_seed: 7\n"
            'full_grid_weighting: "uniform"\n'
        ),
    )
    report_config_path = _write_yaml(
        tmp_path / "report.yaml",
        (
            'benchmark_model: "no_change"\n'
            "official_loss_metrics:\n"
            '  - "observed_mse_total_variance"\n'
            '  - "observed_qlike_total_variance"\n'
            'primary_loss_metric: "observed_mse_total_variance"\n'
            'interpolation_comparison_order: ["moneyness", "maturity"]\n'
            "top_models_per_figure: 3\n"
            "stress_windows:\n"
            '  - label: "sample_window"\n'
            '    start_date: "2021-01-05"\n'
            '    end_date: "2021-01-07"\n'
        ),
    )

    script_module = _load_script_module(
        Path.cwd() / "scripts" / "09_make_report_artifacts.py",
        "report_artifacts_script",
    )
    script_module.main(
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
        metrics_config_path=metrics_config_path,
        stats_config_path=stats_config_path,
        report_config_path=report_config_path,
    )

    report_dir = manifests_dir / "report_artifacts" / workflow_label
    ranked_loss_csv = (report_dir / "tables" / "ranked_loss_summary.csv").read_text(
        encoding="utf-8"
    )
    slice_leaders_csv = (report_dir / "tables" / "slice_leaders.csv").read_text(encoding="utf-8")
    index_md = (report_dir / "index.md").read_text(encoding="utf-8")
    qlike_ranked_loss_csv = (
        report_dir / "tables" / "ranked_loss_summary__observed_qlike_total_variance.csv"
    ).read_text(encoding="utf-8")

    assert ranked_loss_csv == (GOLDEN_DIR / "ranked_loss_summary.csv").read_text(encoding="utf-8")
    assert slice_leaders_csv == (GOLDEN_DIR / "slice_leaders.csv").read_text(encoding="utf-8")
    assert index_md == (GOLDEN_DIR / "report_index.md").read_text(encoding="utf-8")
    assert "nan" not in ranked_loss_csv.lower()
    assert "nan" not in slice_leaders_csv.lower()
    assert "nan" not in index_md.lower()
    assert "mean_observed_qlike_total_variance" in qlike_ranked_loss_csv

    run_manifest_files = sorted(
        (manifests_dir / "runs" / "09_make_report_artifacts").glob("*.json")
    )
    assert len(run_manifest_files) == 1
