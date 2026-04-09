from __future__ import annotations

from dataclasses import asdict
from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import cast

import orjson
import polars as pl

from ivsurf.config import SurfaceGridConfig, load_yaml_config
from ivsurf.evaluation.alignment import build_forecast_realization_panel
from ivsurf.evaluation.loss_panels import build_daily_loss_frame
from ivsurf.hedging.pnl import evaluate_model_hedging, summarize_hedging_results
from ivsurf.hedging.revaluation import surface_interpolator_from_frame
from ivsurf.surfaces.grid import SurfaceGrid


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    rows: list[dict[str, object]] = []
    for maturity_index, maturity_days in enumerate(grid.maturity_days):
        maturity_years = maturity_days / 365.0
        for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
            total_variance = (
                sigma_level * maturity_years
                + (0.00015 * maturity_index)
                + (0.00008 * abs(moneyness_point))
            )
            rows.append(
                {
                    "quote_date": quote_date,
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "observed_total_variance": total_variance,
                    "observed_iv": float((total_variance / maturity_years) ** 0.5),
                    "completed_total_variance": total_variance,
                    "completed_iv": float((total_variance / maturity_years) ** 0.5),
                    "observed_mask": True,
                    "vega_sum": 1.0,
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
        "no_change": 1.02,
        "ridge": 0.995,
        "neural_surface": 1.0,
    }
    for model_name, scale in model_scales.items():
        for maturity_index, maturity_days in enumerate(grid.maturity_days):
            maturity_years = maturity_days / 365.0
            for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
                total_variance = (
                    sigma_level * scale * maturity_years
                    + (0.00015 * maturity_index)
                    + (0.00008 * abs(moneyness_point))
                )
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


def test_report_stage_consumes_real_stage07_contracts(tmp_path: Path) -> None:
    repo_root = _repo_root()
    surface_config = SurfaceGridConfig.model_validate(
        load_yaml_config(repo_root / "configs" / "data" / "surface.yaml")
    )
    grid = SurfaceGrid.from_config(surface_config)
    workflow_label = "hpo_30_trials__train_30_epochs"

    gold_dir = tmp_path / "data" / "gold"
    gold_year = gold_dir / "year=2021"
    gold_year.mkdir(parents=True)
    forecast_dir = gold_dir / "forecasts" / workflow_label
    forecast_dir.mkdir(parents=True)
    silver_dir = tmp_path / "data" / "silver" / "year=2021"
    silver_dir.mkdir(parents=True)
    manifests_dir = tmp_path / "data" / "manifests"
    stats_dir = manifests_dir / "stats" / workflow_label
    hedging_dir = manifests_dir / "hedging" / workflow_label
    stats_dir.mkdir(parents=True)
    hedging_dir.mkdir(parents=True)

    quote_dates = [
        date(2021, 1, 4),
        date(2021, 1, 5),
        date(2021, 1, 6),
        date(2021, 1, 7),
    ]
    sigma_levels = [0.040, 0.044, 0.048, 0.051]

    gold_rows: list[dict[str, object]] = []
    gold_summary_rows: list[dict[str, object]] = []
    for quote_date_value, sigma_level in zip(quote_dates, sigma_levels, strict=True):
        day_rows = _surface_rows(quote_date_value, sigma_level, grid)
        day_path = gold_year / f"{quote_date_value.isoformat()}.parquet"
        pl.DataFrame(day_rows).write_parquet(day_path)
        gold_rows.extend(day_rows)
        gold_summary_rows.append(
            {
                "gold_path": str(day_path),
                "quote_date": quote_date_value.isoformat(),
                "observed_cells": sum(1 for row in day_rows if row["observed_mask"]),
            }
        )
    (manifests_dir / "gold_surface_summary.json").write_bytes(
        orjson.dumps(gold_summary_rows, option=orjson.OPT_INDENT_2)
    )

    forecast_rows: list[dict[str, object]] = []
    for quote_date_value, target_date_value, sigma_level in zip(
        quote_dates[:-1], quote_dates[1:], sigma_levels[1:], strict=True
    ):
        forecast_rows.extend(_forecast_rows(quote_date_value, target_date_value, sigma_level, grid))
    forecast_frame = pl.DataFrame(forecast_rows).sort(
        ["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"]
    )
    forecast_frame.write_parquet(forecast_dir / "forecasts.parquet")

    silver_rows = [
        {"quote_date": quote_date_value, "active_underlying_price_1545": spot}
        for quote_date_value, spot in zip(
            quote_dates,
            [3700.0, 3712.0, 3724.0, 3738.0],
            strict=True,
        )
    ]
    pl.DataFrame(silver_rows).write_parquet(silver_dir / "spots.parquet")

    actual_surface_frame = pl.DataFrame(gold_rows).sort(
        ["quote_date", "maturity_index", "moneyness_index"]
    )
    panel = build_forecast_realization_panel(
        actual_surface_frame=actual_surface_frame,
        forecast_frame=forecast_frame,
        total_variance_floor=1.0e-8,
    )
    panel.write_parquet(stats_dir / "forecast_realization_panel.parquet")

    daily_loss_frame = build_daily_loss_frame(
        panel=panel,
        positive_floor=1.0e-8,
        full_grid_weighting="uniform",
    )
    daily_loss_frame.write_parquet(stats_dir / "daily_loss_frame.parquet")

    loss_metric = "observed_mse_total_variance"
    loss_summary = (
        daily_loss_frame.group_by("model_name")
        .agg(
            pl.col(loss_metric).mean().alias(f"mean_{loss_metric}"),
            pl.col(loss_metric).std(ddof=1).alias(f"std_{loss_metric}"),
            pl.len().alias("n_target_dates"),
        )
        .sort(f"mean_{loss_metric}")
    )
    loss_summary.write_parquet(stats_dir / "loss_summary.parquet")

    dm_results = [
        {
            "model_a": "no_change",
            "model_b": "ridge",
            "n_obs": 3,
            "mean_loss_a": 0.0022,
            "mean_loss_b": 0.0011,
            "mean_differential": 0.0011,
            "statistic": 2.4,
            "p_value": 0.05,
            "alternative": "greater",
            "max_lag": 0,
        },
        {
            "model_a": "no_change",
            "model_b": "neural_surface",
            "n_obs": 3,
            "mean_loss_a": 0.0022,
            "mean_loss_b": 0.0009,
            "mean_differential": 0.0013,
            "statistic": 2.8,
            "p_value": 0.03,
            "alternative": "greater",
            "max_lag": 0,
        },
    ]
    (stats_dir / "dm_results.json").write_bytes(
        orjson.dumps(dm_results, option=orjson.OPT_INDENT_2)
    )
    (stats_dir / "spa_result.json").write_bytes(
        orjson.dumps(
            {
                "benchmark_model": "no_change",
                "candidate_models": ["ridge", "neural_surface"],
                "observed_statistic": 2.7,
                "p_value": 0.04,
                "mean_differentials": [0.0011, 0.0013],
                "superior_models_by_mean": ["ridge", "neural_surface"],
                "block_size": 5,
                "bootstrap_reps": 500,
            },
            option=orjson.OPT_INDENT_2,
        )
    )
    (stats_dir / "mcs_result.json").write_bytes(
        orjson.dumps(
            {
                "superior_models": ["ridge", "neural_surface"],
                "iterations": [
                    {
                        "included_models": ["no_change", "ridge", "neural_surface"],
                        "test_statistic": 2.7,
                        "p_value": 0.04,
                        "eliminated_model": "no_change",
                    },
                    {
                        "included_models": ["ridge", "neural_surface"],
                        "test_statistic": 1.0,
                        "p_value": 0.22,
                        "eliminated_model": None,
                    },
                ],
                "alpha": 0.10,
                "block_size": 5,
                "bootstrap_reps": 500,
            },
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
    hedging_results.write_parquet(hedging_dir / "hedging_results.parquet")
    summarize_hedging_results(hedging_results).write_parquet(
        hedging_dir / "hedging_summary.parquet"
    )

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{gold_dir.as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'sample_start_date: "2021-01-04"\n'
            'sample_end_date: "2021-01-07"\n'
        ),
    )

    script_module = _load_script_module(
        repo_root / "scripts" / "09_make_report_artifacts.py",
        "report_stage_contract_script",
    )
    script_module.main(
        raw_config_path=raw_config_path,
        surface_config_path=repo_root / "configs" / "data" / "surface.yaml",
        metrics_config_path=repo_root / "configs" / "eval" / "metrics.yaml",
        stats_config_path=repo_root / "configs" / "eval" / "stats_tests.yaml",
        report_config_path=repo_root / "configs" / "eval" / "report_artifacts.yaml",
    )

    report_dir = manifests_dir / "report_artifacts" / workflow_label
    ranked_loss_summary = (report_dir / "tables" / "ranked_loss_summary.csv").read_text(
        encoding="utf-8"
    )
    mcs_table = (report_dir / "tables" / "mcs_result.csv").read_text(encoding="utf-8")
    report_index = (report_dir / "index.md").read_text(encoding="utf-8")

    assert "mean_observed_mse_total_variance" in ranked_loss_summary
    assert "included_in_mcs" in mcs_table
    assert "no_change,false" in mcs_table
    assert "ridge,true" in mcs_table
    assert "Primary loss metric: `observed_mse_total_variance`" in report_index
